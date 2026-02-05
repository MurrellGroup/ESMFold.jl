using NNlib

struct ESMFoldConfig
    trunk::FoldingTrunkConfig
    lddt_head_hid_dim::Int
    use_esm_attn_map::Bool
end

function ESMFoldConfig(; trunk::FoldingTrunkConfig=FoldingTrunkConfig(), lddt_head_hid_dim::Int=128, use_esm_attn_map::Bool=false)
    return ESMFoldConfig(trunk, lddt_head_hid_dim, use_esm_attn_map)
end

@concrete struct ESMFoldLDDTHead <: Onion.Layer
    norm
    linear_1
    linear_2
    linear_3
end

@layer ESMFoldLDDTHead

function ESMFoldLDDTHead(c_in::Int, c_hidden::Int, c_out::Int)
    norm = LayerNormLast(c_in)
    linear_1 = LinearLast(c_in, c_hidden)
    linear_2 = LinearLast(c_hidden, c_hidden)
    linear_3 = LinearLast(c_hidden, c_out)
    return ESMFoldLDDTHead(norm, linear_1, linear_2, linear_3)
end

function (m::ESMFoldLDDTHead)(x)
    x = m.norm(x)
    x = m.linear_1(x)
    x = m.linear_2(x)
    x = m.linear_3(x)
    return x
end

@concrete struct ESMFoldLDDTHeadJL <: Onion.Layer
    norm
    linear_1
    linear_2
    linear_3
end

@layer ESMFoldLDDTHeadJL

function ESMFoldLDDTHeadJL(c_in::Int, c_hidden::Int, c_out::Int)
    norm = LayerNormFirst(c_in)
    linear_1 = LinearFirst(c_in, c_hidden)
    linear_2 = LinearFirst(c_hidden, c_hidden)
    linear_3 = LinearFirst(c_hidden, c_out)
    return ESMFoldLDDTHeadJL(norm, linear_1, linear_2, linear_3)
end

function (m::ESMFoldLDDTHeadJL)(x)
    x = m.norm(x)
    x = m.linear_1(x)
    x = m.linear_2(x)
    x = m.linear_3(x)
    return x
end

@concrete struct ESMFoldModel <: Onion.Layer
    cfg::ESMFoldConfig
    embed::ESMFoldEmbed
    trunk::FoldingTrunk
    distogram_bins::Int
    distogram_head
    ptm_head
    lm_head
    lddt_bins::Int
    lddt_head
end

@concrete struct ESMFoldModelJL <: Onion.Layer
    cfg::ESMFoldConfig
    embed::ESMFoldEmbed
    trunk::FoldingTrunkJLCore
    distogram_bins::Int
    distogram_head
    ptm_head
    lm_head
    lddt_bins::Int
    lddt_head
end

@layer ESMFoldModel

function ESMFoldModel(
    esm::ESM2;
    cfg::ESMFoldConfig=ESMFoldConfig(),
)
    c_s = cfg.trunk.sequence_state_dim
    c_z = cfg.trunk.pairwise_state_dim

    embed = ESMFoldEmbed(
        esm;
        c_s = c_s,
        c_z = c_z,
        use_esm_attn_map = cfg.use_esm_attn_map,
    )
    trunk = FoldingTrunk(cfg=cfg.trunk)

    distogram_bins = 64
    distogram_head = LinearLast(c_z, distogram_bins)
    ptm_head = LinearLast(c_z, distogram_bins)
    lm_head = LinearLast(c_s, embed.n_tokens_embed)
    lddt_bins = 50
    lddt_head = ESMFoldLDDTHead(cfg.trunk.structure_module.c_s, cfg.lddt_head_hid_dim, 37 * lddt_bins)

    return ESMFoldModel(
        cfg,
        embed,
        trunk,
        distogram_bins,
        distogram_head,
        ptm_head,
        lm_head,
        lddt_bins,
        lddt_head,
    )
end

@layer ESMFoldModelJL

function ESMFoldModelJL(
    esm::ESM2;
    cfg::ESMFoldConfig=ESMFoldConfig(),
)
    c_s = cfg.trunk.sequence_state_dim
    c_z = cfg.trunk.pairwise_state_dim

    embed = ESMFoldEmbed(
        esm;
        c_s = c_s,
        c_z = c_z,
        use_esm_attn_map = cfg.use_esm_attn_map,
    )
    trunk = FoldingTrunkJLCore(cfg=cfg.trunk)

    distogram_bins = 64
    distogram_head = LinearFirst(c_z, distogram_bins)
    ptm_head = LinearFirst(c_z, distogram_bins)
    lm_head = LinearFirst(c_s, embed.n_tokens_embed)
    lddt_bins = 50
    lddt_head = ESMFoldLDDTHeadJL(cfg.trunk.structure_module.c_s, cfg.lddt_head_hid_dim, 37 * lddt_bins)

    return ESMFoldModelJL(
        cfg,
        embed,
        trunk,
        distogram_bins,
        distogram_head,
        ptm_head,
        lm_head,
        lddt_bins,
        lddt_head,
    )
end

function _default_residx(aa::AbstractArray)
    L = size(aa, 2)
    residx = collect(0:(L - 1))
    residx = reshape(residx, 1, L)
    residx = repeat(residx, size(aa, 1), 1)
    return to_device(residx, aa, eltype(residx))
end

function _default_residx_jl(aa::AbstractArray)
    L = size(aa, 1)
    residx = collect(0:(L - 1))
    residx = reshape(residx, L, 1)
    residx = repeat(residx, 1, size(aa, 2))
    return to_device(residx, aa, eltype(residx))
end

function (m::ESMFoldModel)(
    aa::AbstractArray{Int,2};
    mask = nothing,
    residx = nothing,
    masking_pattern = nothing,
    num_recycles = nothing,
)
    if mask === nothing
        mask = ones_like(aa, size(aa)...)
    end

    if residx === nothing
        residx = _default_residx(aa)
    end

    if masking_pattern !== nothing
        masking_pattern = to_device(masking_pattern, aa, eltype(aa))
    end

    embed_out = m.embed(
        aa;
        mask = mask,
        masking_pattern = masking_pattern,
        return_pair = m.cfg.use_esm_attn_map,
    )

    if m.cfg.use_esm_attn_map
        s_s_0_cf = embed_out.sequence
        s_z_0_cf = embed_out.pair
    else
        s_s_0_cf = embed_out
        s_z_0_cf = nothing
    end

    s_s_0 = permutedims(s_s_0_cf, (3, 2, 1))
    s_z_0 = if s_z_0_cf === nothing
        # (B, L, L, c_z)
        zeros_like(
            s_s_0,
            size(s_s_0, 1),
            size(s_s_0, 2),
            size(s_s_0, 2),
            m.cfg.trunk.pairwise_state_dim,
        )
    else
        permutedims(s_z_0_cf, (4, 2, 3, 1))
    end

    structure = m.trunk(
        s_s_0,
        s_z_0,
        aa,
        residx,
        mask;
        no_recycles = num_recycles,
    )

    disto_logits = m.distogram_head(structure[:s_z])
    disto_logits = (disto_logits .+ permutedims(disto_logits, (1, 3, 2, 4))) ./ 2
    structure[:distogram_logits] = disto_logits

    structure[:lm_logits] = m.lm_head(structure[:s_s])
    structure[:aatype] = aa

    make_atom14_masks!(structure)

    for k in (:atom14_atom_exists, :atom37_atom_exists)
        structure[k] .*= reshape(mask, size(mask, 1), size(mask, 2), 1)
    end
    structure[:residue_index] = residx

    lddt_logits = m.lddt_head(structure[:states])
    lddt_head = _reshape_last_corder(lddt_logits, 37, m.lddt_bins)
    structure[:lddt_head] = lddt_head

    plddt = categorical_lddt(lddt_head[end, :, :, :, :], bins=m.lddt_bins)
    structure[:plddt] = 100f0 .* plddt

    ptm_logits = m.ptm_head(structure[:s_z])
    structure[:ptm_logits] = ptm_logits

    seqlen = sum(mask .== 1; dims=2)
    ptm_vals = Vector{eltype(ptm_logits)}(undef, size(ptm_logits, 1))
    for b in 1:size(ptm_logits, 1)
        sl = Int(seqlen[b])
        ptm_vals[b] = compute_tm(ptm_logits[b, 1:sl, 1:sl, :]; max_bin=31, no_bins=m.distogram_bins)
    end
    structure[:ptm] = to_device(reshape(collect(ptm_vals), size(ptm_logits, 1)), ptm_logits, eltype(ptm_logits))

    structure_update = compute_predicted_aligned_error(ptm_logits; max_bin=31, no_bins=m.distogram_bins)
    for (k, v) in structure_update
        structure[k] = v
    end

    return structure
end

function (m::ESMFoldModelJL)(
    aa::AbstractArray{Int,2};
    mask = nothing,
    residx = nothing,
    masking_pattern = nothing,
    num_recycles = nothing,
)
    if mask === nothing
        mask = ones_like(aa, size(aa)...)
    end

    if residx === nothing
        residx = _default_residx_jl(aa)
    end

    if masking_pattern !== nothing
        masking_pattern = to_device(masking_pattern, aa, eltype(aa))
    end

    aa_fl = permutedims(aa, (2, 1))
    mask_fl = permutedims(mask, (2, 1))
    masking_pattern_fl = masking_pattern === nothing ? nothing : permutedims(masking_pattern, (2, 1))

    embed_out = m.embed(
        aa_fl;
        mask = mask_fl,
        masking_pattern = masking_pattern_fl,
        return_pair = m.cfg.use_esm_attn_map,
    )

    if m.cfg.use_esm_attn_map
        s_s_0 = embed_out.sequence
        s_z_0 = embed_out.pair
    else
        s_s_0 = embed_out
        s_z_0 = nothing
    end

    s_z_0 = if s_z_0 === nothing
        zeros_like(
            s_s_0,
            m.cfg.trunk.pairwise_state_dim,
            size(s_s_0, 2),
            size(s_s_0, 2),
            size(s_s_0, 3),
        )
    else
        s_z_0
    end

    structure = m.trunk(
        s_s_0,
        s_z_0,
        aa,
        residx,
        mask;
        no_recycles = num_recycles,
    )

    disto_logits = m.distogram_head(structure[:s_z])
    disto_logits = (disto_logits .+ permutedims(disto_logits, (1, 3, 2, 4))) ./ 2
    structure[:distogram_logits] = disto_logits

    structure[:lm_logits] = m.lm_head(structure[:s_s])
    structure[:aatype] = aa

    make_atom14_masks_jl!(structure)

    for k in (:atom14_atom_exists, :atom37_atom_exists)
        structure[k] .*= reshape(mask, 1, size(mask, 1), size(mask, 2))
    end
    structure[:residue_index] = residx

    states = structure[:states]
    states_cfirst = permutedims(states, (2, 1, 3, 4))
    lddt_logits = m.lddt_head(states_cfirst)
    lddt_tmp = _reshape_first_corder(lddt_logits, 37, m.lddt_bins)
    lddt_head = permutedims(lddt_tmp, (3, 4, 5, 2, 1))
    structure[:lddt_head] = lddt_head

    plddt = categorical_lddt_jl(lddt_head[end, :, :, :, :], bins=m.lddt_bins)
    structure[:plddt] = 100f0 .* plddt

    ptm_logits = m.ptm_head(structure[:s_z])
    structure[:ptm_logits] = ptm_logits

    seqlen = sum(mask .== 1; dims=1)
    ptm_vals = Vector{eltype(ptm_logits)}(undef, size(ptm_logits, 4))
    for b in 1:size(ptm_logits, 4)
        sl = Int(seqlen[1, b])
        ptm_vals[b] = compute_tm_jl(ptm_logits[:, 1:sl, 1:sl, b]; max_bin=31, no_bins=m.distogram_bins)
    end
    structure[:ptm] = to_device(reshape(collect(ptm_vals), size(ptm_logits, 4)), ptm_logits, eltype(ptm_logits))

    structure_update = compute_predicted_aligned_error_jl(ptm_logits; max_bin=31, no_bins=m.distogram_bins)
    for (k, v) in structure_update
        structure[k] = v
    end

    return structure
end

function infer(
    m::ESMFoldModel,
    sequences::Union{AbstractString,AbstractVector{<:AbstractString}};
    residx = nothing,
    masking_pattern = nothing,
    num_recycles = nothing,
    residue_index_offset::Int = 512,
    chain_linker::AbstractString = "G"^25,
)
    seqs = isa(sequences, AbstractString) ? [sequences] : sequences

    aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(
        seqs;
        residue_index_offset = residue_index_offset,
        chain_linker = chain_linker,
    )

    if residx === nothing
        residx = _residx
    elseif !isa(residx, AbstractArray)
        residx = collate_dense_tensors(residx)
    end

    like = m.embed.esm.embed_tokens.weight
    aatype = to_device(aatype, like, Int)
    mask = to_device(mask, like, eltype(like))
    residx = to_device(residx, like, Int)
    linker_mask = to_device(linker_mask, like, eltype(like))
    if masking_pattern !== nothing
        masking_pattern = to_device(masking_pattern, like, eltype(aatype))
    end

    output = m(
        aatype;
        mask = mask,
        residx = residx,
        masking_pattern = masking_pattern,
        num_recycles = num_recycles,
    )

    output[:atom37_atom_exists] .*= reshape(linker_mask, size(linker_mask, 1), size(linker_mask, 2), 1)

    weighted_plddt = output[:plddt] .* output[:atom37_atom_exists]
    numerator = sum(weighted_plddt; dims=(2, 3))
    denom = sum(output[:atom37_atom_exists]; dims=(2, 3))
    output[:mean_plddt] = numerator ./ denom
    output[:chain_index] = chain_index

    return output
end

function output_to_pdb(m::ESMFoldModel, output::AbstractDict)
    return output_to_pdb(output)
end

function infer_pdbs(m::ESMFoldModel, seqs::AbstractVector{<:AbstractString}; kwargs...)
    output = infer(m, seqs; kwargs...)
    return output_to_pdb(output)
end

function infer_pdb(m::ESMFoldModel, seq::AbstractString; kwargs...)
    return infer_pdbs(m, [seq]; kwargs...)[1]
end

function confidence_metrics(output::AbstractDict)
    return (
        plddt = output[:plddt],
        mean_plddt = output[:mean_plddt],
        ptm = output[:ptm],
        predicted_aligned_error = output[:predicted_aligned_error],
        aligned_confidence_probs = output[:aligned_confidence_probs],
        max_predicted_aligned_error = output[:max_predicted_aligned_error],
    )
end

function set_chunk_size!(m::ESMFoldModel, chunk_size::Union{Nothing,Int})
    set_chunk_size!(m.trunk, chunk_size)
    return m
end

function device_ref(m::ESMFoldModel)
    return m.embed.esm.embed_tokens.weight
end
