using NNlib
import Zygote

function cross_last(a::AbstractArray, b::AbstractArray)
    a1 = _view_last1(a, 1); a2 = _view_last1(a, 2); a3 = _view_last1(a, 3)
    b1 = _view_last1(b, 1); b2 = _view_last1(b, 2); b3 = _view_last1(b, 3)
    c1 = a2 .* b3 .- a3 .* b2
    c2 = a3 .* b1 .- a1 .* b3
    c3 = a1 .* b2 .- a2 .* b1
    return cat(c1, c2, c3; dims=ndims(c1) + 1)
end

struct FoldingTrunkConfig
    num_blocks::Int
    sequence_state_dim::Int
    pairwise_state_dim::Int
    sequence_head_width::Int
    pairwise_head_width::Int
    position_bins::Int
    dropout::Float32
    layer_drop::Float32
    cpu_grad_checkpoint::Bool
    max_recycles::Int
    chunk_size::Union{Nothing,Int}
    structure_module::StructureModuleConfig
end

function FoldingTrunkConfig()
    return FoldingTrunkConfig(
        48,
        1024,
        128,
        32,
        32,
        32,
        0f0,
        0f0,
        false,
        4,
        nothing,
        StructureModuleConfig(),
    )
end

@concrete struct RelativePosition <: Onion.Layer
    bins::Int
    embedding
end

@layer RelativePosition

function RelativePosition(bins::Int, pairwise_state_dim::Int)
    embedding = Flux.Embedding(2 * bins + 2, pairwise_state_dim)
    return RelativePosition(bins, embedding)
end

function (m::RelativePosition)(residue_index::AbstractArray; mask=nothing)
    diff = reshape(residue_index, size(residue_index, 1), 1, size(residue_index, 2)) .-
           reshape(residue_index, size(residue_index, 1), size(residue_index, 2), 1)
    diff = clamp.(diff, -m.bins, m.bins)
    diff = diff .+ m.bins .+ 1

    if mask !== nothing
        pair_mask = reshape(mask, size(mask, 1), 1, size(mask, 2)) .* reshape(mask, size(mask, 1), size(mask, 2), 1)
        diff = ifelse.(pair_mask .== 1, diff, 0)
    end

    diff = diff .+ 1 # shift to 1-based indexing for Flux.Embedding
    emb = m.embedding(diff)
    # embedding returns (c_z, B, L, L); trunk expects (B, L, L, c_z)
    return permutedims(emb, (2, 3, 4, 1))
end

@concrete struct RelativePositionJL <: Onion.Layer
    bins::Int
    embedding
end

@layer RelativePositionJL

function RelativePositionJL(bins::Int, pairwise_state_dim::Int)
    embedding = Flux.Embedding(2 * bins + 2, pairwise_state_dim)
    return RelativePositionJL(bins, embedding)
end

function (m::RelativePositionJL)(residue_index::AbstractArray; mask=nothing)
    # residue_index: (L, B)
    diff = reshape(residue_index, 1, size(residue_index, 1), size(residue_index, 2)) .-
           reshape(residue_index, size(residue_index, 1), 1, size(residue_index, 2))
    diff = clamp.(diff, -m.bins, m.bins)
    diff = diff .+ m.bins .+ 1

    if mask !== nothing
        pair_mask = reshape(mask, size(mask, 1), 1, size(mask, 2)) .* reshape(mask, 1, size(mask, 1), size(mask, 2))
        diff = ifelse.(pair_mask .== 1, diff, 0)
    end

    diff = diff .+ 1 # shift to 1-based indexing for Flux.Embedding
    emb = m.embedding(diff)
    # embedding returns (c_z, L, L, B)
    return emb
end

@concrete struct FoldingTrunk <: Onion.Layer
    cfg::FoldingTrunkConfig
    pairwise_positional_embedding
    blocks
    recycle_bins::Int
    recycle_s_norm
    recycle_z_norm
    recycle_disto
    structure_module
    trunk2sm_s
    trunk2sm_z
    chunk_size
end

@layer FoldingTrunk

function FoldingTrunk(; cfg::FoldingTrunkConfig=FoldingTrunkConfig())
    c_s = cfg.sequence_state_dim
    c_z = cfg.pairwise_state_dim

    pairwise_positional_embedding = RelativePosition(cfg.position_bins, c_z)
    blocks = [
        TriangularSelfAttentionBlock(
            c_s,
            c_z,
            cfg.sequence_head_width,
            cfg.pairwise_head_width;
            dropout = cfg.dropout,
        )
        for _ in 1:cfg.num_blocks
    ]

    recycle_bins = 15
    recycle_s_norm = LayerNormLast(c_s)
    recycle_z_norm = LayerNormLast(c_z)
    recycle_disto = Flux.Embedding(recycle_bins, c_z)
    recycle_disto.weight[:, 1] .= 0f0

    structure_module = StructureModule(cfg=cfg.structure_module)
    trunk2sm_s = LinearLast(c_s, structure_module.cfg.c_s)
    trunk2sm_z = LinearLast(c_z, structure_module.cfg.c_z)

    return FoldingTrunk(
        cfg,
        pairwise_positional_embedding,
        blocks,
        recycle_bins,
        recycle_s_norm,
        recycle_z_norm,
        recycle_disto,
        structure_module,
        trunk2sm_s,
        trunk2sm_z,
        cfg.chunk_size,
    )
end

function set_chunk_size!(m::FoldingTrunk, chunk_size::Union{Nothing,Int})
    m.chunk_size = chunk_size
    return m
end

function _trunk_iter(m::FoldingTrunk, s, z, residx, mask)
    z = z .+ m.pairwise_positional_embedding(residx, mask=mask)
    for block in m.blocks
        s, z = block(s, z; mask=mask, residue_index=residx, chunk_size=m.chunk_size)
    end
    return s, z
end

function (m::FoldingTrunk)(
    seq_feats,
    pair_feats,
    true_aa,
    residx,
    mask;
    no_recycles=nothing,
)
    s_s_0 = seq_feats
    s_z_0 = pair_feats

    if no_recycles === nothing
        no_recycles = m.cfg.max_recycles
    else
        no_recycles += 1
    end
    no_recycles < 1 && (no_recycles = 1)

    recycle_s = zeros_like(s_s_0, size(s_s_0)...)
    recycle_z = zeros_like(s_z_0, size(s_z_0)...)
    # recycle bins are per (B, L, L), not per channel
    recycle_bins = zeros_like(s_z_0, Int, size(s_z_0)[1:3]...)

    s_s = s_s_0
    s_z = s_z_0

    mask_f = mask === nothing ? nothing : convert.(eltype(s_s_0), mask)

    structure = nothing
    for _ in 1:no_recycles
        recycle_s = m.recycle_s_norm(recycle_s)
        recycle_z = m.recycle_z_norm(recycle_z)
        # embedding returns (c_z, B, L, L) -> (B, L, L, c_z)
        recycle_z = recycle_z .+ permutedims(m.recycle_disto(recycle_bins .+ 1), (2, 3, 4, 1))

        s_s, s_z = _trunk_iter(m, s_s_0 .+ recycle_s, s_z_0 .+ recycle_z, residx, mask)

        structure = m.structure_module(
            Dict(:single => m.trunk2sm_s(s_s), :pair => m.trunk2sm_z(s_z)),
            true_aa,
            mask_f,
        )

        recycle_s = s_s
        recycle_z = s_z

        coords = structure[:positions][end, :, :, 1:3, :]
        recycle_bins = Zygote.ignore() do
            distogram(coords, 3.375f0, 21.375f0, m.recycle_bins)
        end
    end

    structure === nothing && error("FoldingTrunk: no recycle iterations ran.")
    structure[:s_s] = s_s
    structure[:s_z] = s_z
    return structure
end

function distogram(coords, min_bin::Real, max_bin::Real, num_bins::Int)
    boundaries = range(Float32(min_bin), Float32(max_bin); length=num_bins - 1)
    boundaries = collect(boundaries)
    boundaries = to_device(boundaries, coords, Float32) .^ 2

    N = coords[:, :, 1, :]
    CA = coords[:, :, 2, :]
    C = coords[:, :, 3, :]

    b = CA .- N
    c = C .- CA
    a = cross_last(b, c)
    CB = (-0.58273431f0 .* a) .+ (0.56802827f0 .* b) .- (0.54067466f0 .* c) .+ CA

    diff = reshape(CB, size(CB, 1), 1, size(CB, 2), size(CB, 3)) .-
           reshape(CB, size(CB, 1), size(CB, 2), 1, size(CB, 3))
    dists = sum(diff .^ 2; dims=4)

    bview = reshape(boundaries, 1, 1, 1, :)
    bins = sum(dists .> bview; dims=4)
    bins = dropdims(bins; dims=4)
    return Int.(bins)
end

@concrete struct FoldingTrunkJL <: Onion.Layer
    cfg::FoldingTrunkConfig
    pairwise_positional_embedding
    blocks
    recycle_bins::Int
    recycle_s_norm
    recycle_z_norm
    recycle_disto
    structure_module
    trunk2sm_s
    trunk2sm_z
    chunk_size
end

@layer FoldingTrunkJL

function FoldingTrunkJL(; cfg::FoldingTrunkConfig=FoldingTrunkConfig())
    c_s = cfg.sequence_state_dim
    c_z = cfg.pairwise_state_dim

    pairwise_positional_embedding = RelativePositionJL(cfg.position_bins, c_z)
    blocks = [
        TriangularSelfAttentionBlockJL(
            c_s,
            c_z,
            cfg.sequence_head_width,
            cfg.pairwise_head_width;
            dropout = cfg.dropout,
        )
        for _ in 1:cfg.num_blocks
    ]

    recycle_bins = 15
    recycle_s_norm = LayerNormFirst(c_s)
    recycle_z_norm = LayerNormFirst(c_z)
    recycle_disto = Flux.Embedding(recycle_bins, c_z)
    recycle_disto.weight[:, 1] .= 0f0

    structure_module = StructureModuleJL(cfg=cfg.structure_module)
    trunk2sm_s = LinearFirst(c_s, structure_module.inner.cfg.c_s)
    trunk2sm_z = LinearFirst(c_z, structure_module.inner.cfg.c_z)

    return FoldingTrunkJL(
        cfg,
        pairwise_positional_embedding,
        blocks,
        recycle_bins,
        recycle_s_norm,
        recycle_z_norm,
        recycle_disto,
        structure_module,
        trunk2sm_s,
        trunk2sm_z,
        cfg.chunk_size,
    )
end

function set_chunk_size!(m::FoldingTrunkJL, chunk_size::Union{Nothing,Int})
    m.chunk_size = chunk_size
    return m
end

function _trunk_iter_jl(m::FoldingTrunkJL, s, z, residx, mask)
    z = z .+ m.pairwise_positional_embedding(residx, mask=mask)
    for block in m.blocks
        s, z = block(s, z; mask=mask, residue_index=residx, chunk_size=m.chunk_size)
    end
    return s, z
end

function (m::FoldingTrunkJL)(
    seq_feats,
    pair_feats,
    true_aa,
    residx,
    mask;
    no_recycles=nothing,
)
    s_s_0 = seq_feats
    s_z_0 = pair_feats

    if no_recycles === nothing
        no_recycles = m.cfg.max_recycles
    else
        no_recycles += 1
    end
    no_recycles < 1 && (no_recycles = 1)

    recycle_s = zeros_like(s_s_0, size(s_s_0)...)
    recycle_z = zeros_like(s_z_0, size(s_z_0)...)
    recycle_bins = zeros_like(s_z_0, Int, size(s_z_0, 2), size(s_z_0, 3), size(s_z_0, 4))

    s_s = s_s_0
    s_z = s_z_0

    mask_f = mask === nothing ? nothing : convert.(eltype(s_s_0), mask)

    structure = nothing
    for _ in 1:no_recycles
        recycle_s = m.recycle_s_norm(recycle_s)
        recycle_z = m.recycle_z_norm(recycle_z)
        recycle_z = recycle_z .+ m.recycle_disto(recycle_bins .+ 1)

        s_s, s_z = _trunk_iter_jl(m, s_s_0 .+ recycle_s, s_z_0 .+ recycle_z, residx, mask)

        structure = m.structure_module(
            Dict(:single => m.trunk2sm_s(s_s), :pair => m.trunk2sm_z(s_z)),
            true_aa,
            mask_f,
        )

        recycle_s = s_s
        recycle_z = s_z

        coords = structure[:positions][end, :, 1:3, :, :]
        recycle_bins = Zygote.ignore() do
            distogram_jl(coords, 3.375f0, 21.375f0, m.recycle_bins)
        end
    end

    structure === nothing && error("FoldingTrunkJL: no recycle iterations ran.")
    structure[:s_s] = s_s
    structure[:s_z] = s_z
    return structure
end

function cross_first(a::AbstractArray, b::AbstractArray)
    a1 = _view_first1(a, 1); a2 = _view_first1(a, 2); a3 = _view_first1(a, 3)
    b1 = _view_first1(b, 1); b2 = _view_first1(b, 2); b3 = _view_first1(b, 3)
    c1 = a2 .* b3 .- a3 .* b2
    c2 = a3 .* b1 .- a1 .* b3
    c3 = a1 .* b2 .- a2 .* b1
    c1 = reshape(c1, 1, size(c1)...)
    c2 = reshape(c2, 1, size(c2)...)
    c3 = reshape(c3, 1, size(c3)...)
    return cat(c1, c2, c3; dims=1)
end

function distogram_jl(coords, min_bin::Real, max_bin::Real, num_bins::Int)
    # coords: (3, 3, L, B) for N, CA, C
    boundaries = range(Float32(min_bin), Float32(max_bin); length=num_bins - 1)
    boundaries = collect(boundaries)
    boundaries = to_device(boundaries, coords, Float32) .^ 2

    N = view(coords, :, 1, :, :)
    CA = view(coords, :, 2, :, :)
    C = view(coords, :, 3, :, :)

    b = CA .- N
    c = C .- CA
    a = cross_first(b, c)
    CB = (-0.58273431f0 .* a) .+ (0.56802827f0 .* b) .- (0.54067466f0 .* c) .+ CA

    diff = reshape(CB, 3, size(CB, 2), 1, size(CB, 3)) .-
           reshape(CB, 3, 1, size(CB, 2), size(CB, 3))
    dists = sum(diff .^ 2; dims=1)
    dists = dropdims(dists; dims=1) # (L, L, B)

    dists_exp = reshape(dists, size(dists)..., 1)
    bview = reshape(boundaries, 1, 1, 1, :)
    bins = sum(dists_exp .> bview; dims=4)
    bins = dropdims(bins; dims=4)
    return Int.(bins)
end

@concrete struct FoldingTrunkJLCore <: Onion.Layer
    cfg::FoldingTrunkConfig
    pairwise_positional_embedding
    blocks
    recycle_bins::Int
    recycle_s_norm
    recycle_z_norm
    recycle_disto
    structure_module
    trunk2sm_s
    trunk2sm_z
    chunk_size
end

@layer FoldingTrunkJLCore

function FoldingTrunkJLCore(; cfg::FoldingTrunkConfig=FoldingTrunkConfig())
    c_s = cfg.sequence_state_dim
    c_z = cfg.pairwise_state_dim

    pairwise_positional_embedding = RelativePositionJL(cfg.position_bins, c_z)
    blocks = [
        TriangularSelfAttentionBlockJL(
            c_s,
            c_z,
            cfg.sequence_head_width,
            cfg.pairwise_head_width;
            dropout = cfg.dropout,
        )
        for _ in 1:cfg.num_blocks
    ]

    recycle_bins = 15
    recycle_s_norm = LayerNormFirst(c_s)
    recycle_z_norm = LayerNormFirst(c_z)
    recycle_disto = Flux.Embedding(recycle_bins, c_z)
    recycle_disto.weight[:, 1] .= 0f0

    structure_module = StructureModuleJLCore(cfg=cfg.structure_module)
    trunk2sm_s = LinearFirst(c_s, structure_module.cfg.c_s)
    trunk2sm_z = LinearFirst(c_z, structure_module.cfg.c_z)

    return FoldingTrunkJLCore(
        cfg,
        pairwise_positional_embedding,
        blocks,
        recycle_bins,
        recycle_s_norm,
        recycle_z_norm,
        recycle_disto,
        structure_module,
        trunk2sm_s,
        trunk2sm_z,
        cfg.chunk_size,
    )
end

function set_chunk_size!(m::FoldingTrunkJLCore, chunk_size::Union{Nothing,Int})
    m.chunk_size = chunk_size
    return m
end

function _trunk_iter_jl_core(m::FoldingTrunkJLCore, s, z, residx, mask)
    z = z .+ m.pairwise_positional_embedding(residx, mask=mask)
    for block in m.blocks
        s, z = block(s, z; mask=mask, residue_index=residx, chunk_size=m.chunk_size)
    end
    return s, z
end

function (m::FoldingTrunkJLCore)(
    seq_feats,
    pair_feats,
    true_aa,
    residx,
    mask;
    no_recycles=nothing,
)
    s_s_0 = seq_feats
    s_z_0 = pair_feats

    if no_recycles === nothing
        no_recycles = m.cfg.max_recycles
    else
        no_recycles += 1
    end
    no_recycles < 1 && (no_recycles = 1)

    recycle_s = zeros_like(s_s_0, size(s_s_0)...)
    recycle_z = zeros_like(s_z_0, size(s_z_0)...)
    recycle_bins = zeros_like(s_z_0, Int, size(s_z_0, 2), size(s_z_0, 3), size(s_z_0, 4))

    s_s = s_s_0
    s_z = s_z_0

    mask_f = mask === nothing ? nothing : convert.(eltype(s_s_0), mask)

    structure = nothing
    for _ in 1:no_recycles
        recycle_s = m.recycle_s_norm(recycle_s)
        recycle_z = m.recycle_z_norm(recycle_z)
        recycle_z = recycle_z .+ m.recycle_disto(recycle_bins .+ 1)

        s_s, s_z = _trunk_iter_jl_core(m, s_s_0 .+ recycle_s, s_z_0 .+ recycle_z, residx, mask)

        structure = m.structure_module(
            Dict(:single => m.trunk2sm_s(s_s), :pair => m.trunk2sm_z(s_z)),
            true_aa,
            mask_f,
        )

        recycle_s = s_s
        recycle_z = s_z

        coords = structure[:positions][end, :, 1:3, :, :]
        recycle_bins = Zygote.ignore() do
            distogram_jl(coords, 3.375f0, 21.375f0, m.recycle_bins)
        end
    end

    structure === nothing && error("FoldingTrunkJLCore: no recycle iterations ran.")
    structure[:s_s] = s_s
    structure[:s_z] = s_z
    return structure
end
