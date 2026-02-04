using Pkg
Pkg.activate("/Users/benmurrell/JuliaM3/juliaESM"; io=devnull)

using NPZ
using Statistics
using ESMEmbed
using NNlib

ref = NPZ.npzread("/Users/benmurrell/JuliaM3/juliaESM/esmfold_ipa_debug.npz")

path = "weights/esm.safetensors"
reader = ESMEmbed.SafeTensors.Reader(path)
(
    num_layers,
    embed_dim,
    attention_heads,
    c_s,
    c_z,
    sequence_head_width,
    pairwise_head_width,
    position_bins,
    num_blocks,
    lddt_head_hid_dim,
) = ESMEmbed._infer_esmfold_full_config(reader)

alphabet = ESMEmbed.Alphabet_from_architecture("ESM-1b")
esm = ESMEmbed.ESM2(
    num_layers,
    embed_dim,
    attention_heads;
    alphabet = alphabet,
    token_dropout = true,
)

trunk_cfg = ESMEmbed.FoldingTrunkConfig(
    num_blocks,
    c_s,
    c_z,
    sequence_head_width,
    pairwise_head_width,
    position_bins,
    0f0,
    0f0,
    false,
    4,
    nothing,
    ESMEmbed.StructureModuleConfig(),
)

cfg = ESMEmbed.ESMFoldConfig(; trunk=trunk_cfg, lddt_head_hid_dim=lddt_head_hid_dim, use_esm_attn_map=false)
model = ESMEmbed.ESMFold(esm; cfg=cfg)
ESMEmbed.load_esmfold_safetensors!(model, reader)

seq = "ELLKKLLEELKG"
out = ESMEmbed.infer(model, seq; num_recycles=0)
s_s = model.trunk.trunk2sm_s(out[:s_s])
s_z = model.trunk.trunk2sm_z(out[:s_z])

sm = model.trunk.structure_module
s = sm.layer_norm_s(s_s)
z = sm.layer_norm_z(s_z)
s = sm.linear_in(s)
mask = ones(Float32, size(s, 1), size(s, 2))
r = ESMEmbed.rigid_identity(size(s)[1:end-1], s; fmt=:quat)

function ipa_internals(ipa::ESMEmbed.InvariantPointAttention, s, z, r::ESMEmbed.Rigid, mask)
    q = ipa.linear_q(s)
    q = reshape(q, size(q)[1:end-1]..., ipa.c_hidden, ipa.no_heads)
    n = ndims(q)
    q = permutedims(q, vcat(collect(1:(n - 2)), n, n - 1)) # (..., L, H, C)

    q_pts = ipa.linear_q_points(s, r)

    kv = ipa.linear_kv(s)
    kv = reshape(kv, size(kv)[1:end-1]..., 2 * ipa.c_hidden, ipa.no_heads)
    n = ndims(kv)
    kv = permutedims(kv, vcat(collect(1:(n - 2)), n, n - 1)) # (..., L, H, 2C)
    k = ESMEmbed._view_last1(kv, 1:ipa.c_hidden)
    v = ESMEmbed._view_last1(kv, (ipa.c_hidden + 1):(2 * ipa.c_hidden))

    kv_pts = ipa.linear_kv_points(s, r)
    k_pts = ESMEmbed._view_last2(kv_pts, 1:ipa.no_qk_points, Colon())
    v_pts = ESMEmbed._view_last2(kv_pts, (ipa.no_qk_points + 1):(ipa.no_qk_points + ipa.no_v_points), Colon())

    b = ipa.linear_b(z)

    q_bhlc = permutedims(q, (1, 3, 2, 4)) # (B, H, L, C)
    k_bhcl = permutedims(k, (1, 3, 4, 2)) # (B, H, C, L)
    B = size(q_bhlc, 1)
    H = size(q_bhlc, 2)
    L = size(q_bhlc, 3)
    C = size(q_bhlc, 4)
    q3 = permutedims(reshape(q_bhlc, B * H, L, C), (2, 3, 1))
    k3 = permutedims(reshape(k_bhcl, B * H, C, L), (2, 3, 1))
    a3 = NNlib.batched_mul(q3, k3)
    a = reshape(a3, L, L, B, H)
    a = permutedims(a, (3, 4, 1, 2))

    a .*= sqrt(1f0 / (3f0 * ipa.c_hidden))
    b_perm = permutedims(b, (1, 4, 2, 3))
    a .+= sqrt(1f0 / 3f0) .* b_perm

    q_exp = reshape(q_pts, size(q_pts, 1), size(q_pts, 2), 1, size(q_pts, 3), size(q_pts, 4), size(q_pts, 5))
    k_exp = reshape(k_pts, size(k_pts, 1), 1, size(k_pts, 2), size(k_pts, 3), size(k_pts, 4), size(k_pts, 5))
    pt_att = q_exp .- k_exp
    pt_att = sum(pt_att .^ 2; dims=6)

    head_weights = NNlib.softplus.(ipa.head_weights)
    head_weights = head_weights .* sqrt(1f0 / (3f0 * (ipa.no_qk_points * 9f0 / 2f0)))
    hw = reshape(head_weights, 1, 1, 1, ipa.no_heads, 1, 1)
    pt_att = sum(pt_att .* hw; dims=5) .* (-0.5f0)
    pt_att = dropdims(pt_att; dims=(5, 6))
    pt_att = permutedims(pt_att, (1, 4, 2, 3))

    square_mask = reshape(mask, size(mask, 1), 1, size(mask, 2)) .* reshape(mask, size(mask, 1), size(mask, 2), 1)
    square_mask = ipa.inf .* (square_mask .- 1)
    a .+= pt_att
    a .+= reshape(square_mask, size(square_mask, 1), 1, size(square_mask, 2), size(square_mask, 3))

    attn = NNlib.softmax(a; dims=4)

    v_bhlc = permutedims(v, (1, 3, 2, 4))
    a3 = reshape(permutedims(attn, (3, 4, 1, 2)), L, L, :)
    v3 = permutedims(reshape(v_bhlc, B * H, L, C), (2, 3, 1))
    o3 = NNlib.batched_mul(a3, v3)
    o = reshape(o3, L, C, B, H)
    o = permutedims(o, (3, 1, 4, 2)) # (B, L, H, C)
    o = permutedims(o, (1, 2, 4, 3)) # (B, L, C, H)
    o_flat = reshape(o, B, L, H * C)

    v_pts_x = ESMEmbed._view_last1(v_pts, 1)
    v_pts_y = ESMEmbed._view_last1(v_pts, 2)
    v_pts_z = ESMEmbed._view_last1(v_pts, 3)

    vpx = permutedims(v_pts_x, (2, 4, 1, 3))
    vpy = permutedims(v_pts_y, (2, 4, 1, 3))
    vpz = permutedims(v_pts_z, (2, 4, 1, 3))

    vpx3 = reshape(vpx, L, ipa.no_v_points, :)
    vpy3 = reshape(vpy, L, ipa.no_v_points, :)
    vpz3 = reshape(vpz, L, ipa.no_v_points, :)

    o_px = NNlib.batched_mul(a3, vpx3)
    o_py = NNlib.batched_mul(a3, vpy3)
    o_pz = NNlib.batched_mul(a3, vpz3)

    o_px = reshape(o_px, L, ipa.no_v_points, size(s, 1), ipa.no_heads)
    o_py = reshape(o_py, L, ipa.no_v_points, size(s, 1), ipa.no_heads)
    o_pz = reshape(o_pz, L, ipa.no_v_points, size(s, 1), ipa.no_heads)

    o_px = permutedims(o_px, (3, 1, 4, 2))
    o_py = permutedims(o_py, (3, 1, 4, 2))
    o_pz = permutedims(o_pz, (3, 1, 4, 2))

    o_pt = cat(o_px, o_py, o_pz; dims=5)
    o_pt = ESMEmbed.invert_apply_rigid(r, o_pt)

    o_pt_norm = sqrt.(sum(o_pt .^ 2; dims=5) .+ ipa.eps)
    o_pt_norm = dropdims(o_pt_norm; dims=5)
    o_pt_norm = permutedims(o_pt_norm, (1, 2, 4, 3))
    o_pt_norm = reshape(o_pt_norm, size(o_pt_norm, 1), size(o_pt_norm, 2), ipa.no_heads * ipa.no_v_points)

    o_pt = permutedims(o_pt, (1, 2, 4, 3, 5))
    o_pt = reshape(o_pt, size(o_pt)[1:end-3]..., ipa.no_heads * ipa.no_v_points, 3)
    o_pt_x = ESMEmbed._view_last1(o_pt, 1)
    o_pt_y = ESMEmbed._view_last1(o_pt, 2)
    o_pt_z = ESMEmbed._view_last1(o_pt, 3)

    a_t = permutedims(attn, (1, 2, 4, 3))
    a_exp = reshape(a_t, size(a_t, 1), size(a_t, 2), size(a_t, 3), size(a_t, 4), 1)
    z_swap = permutedims(z, (1, 3, 2, 4))
    z_exp = reshape(z_swap, size(z_swap, 1), 1, size(z_swap, 2), size(z_swap, 3), size(z_swap, 4))
    o_pair = sum(a_exp .* z_exp; dims=3)
    o_pair = dropdims(o_pair; dims=3)
    o_pair = permutedims(o_pair, (1, 3, 2, 4)) # (B, L, H, C_z)
    o_pair = permutedims(o_pair, (1, 2, 4, 3))
    o_pair = reshape(o_pair, size(o_pair, 1), size(o_pair, 2), ipa.no_heads * ipa.c_z)

    concat = cat(o_flat, o_pt_x, o_pt_y, o_pt_z, o_pt_norm, o_pair; dims=ndims(o_flat))
    out = ipa.linear_out(concat)

    return Dict(
        :q => q,
        :k => k,
        :v => v,
        :q_pts => q_pts,
        :k_pts => k_pts,
        :v_pts => v_pts,
        :b => b,
        :attn_logits => a,
        :attn => attn,
        :o_flat => o_flat,
        :o_pt_norm => o_pt_norm,
        :o_pair => o_pair,
        :out => out,
        :s => s,
        :z => z,
    )
end

dbg = ipa_internals(sm.ipa, s, z, r, mask)

function point_proj_local(m::ESMEmbed.PointProjection, activations)
    raw = m.linear(activations)
    B = size(raw, 1)
    L = size(raw, 2)
    H = m.no_heads
    P = m.num_points
    @assert size(raw, 3) == H * P * 3
    out = similar(raw, B, L, H, P, 3)
    @inbounds for axis in 1:3
        base = (axis - 1) * H * P
        for h in 1:H
            for p in 1:P
                idx = base + (h - 1) * P + p
                out[:, :, h, p, axis] .= raw[:, :, idx]
            end
        end
    end
    return out
end

function point_proj_local_interleaved(m::ESMEmbed.PointProjection, activations)
    raw = m.linear(activations)
    B = size(raw, 1)
    L = size(raw, 2)
    H = m.no_heads
    P = m.num_points
    @assert size(raw, 3) == H * P * 3
    out = similar(raw, B, L, H, P, 3)
    @inbounds for h in 1:H
        for p in 1:P
            base = (h - 1) * P * 3 + (p - 1) * 3
            out[:, :, h, p, 1] .= raw[:, :, base + 1]
            out[:, :, h, p, 2] .= raw[:, :, base + 2]
            out[:, :, h, p, 3] .= raw[:, :, base + 3]
        end
    end
    return out
end

q_pts_local = point_proj_local(sm.ipa.linear_q_points, s)
kv_pts_local = point_proj_local(sm.ipa.linear_kv_points, s)
k_pts_local = kv_pts_local[:, :, :, 1:sm.ipa.no_qk_points, :]
v_pts_local = kv_pts_local[:, :, :, (sm.ipa.no_qk_points + 1):end, :]
q_pts_linear = sm.ipa.linear_q_points.linear(s)
kv_pts_linear = sm.ipa.linear_kv_points.linear(s)

q_pts_local_alt = point_proj_local_interleaved(sm.ipa.linear_q_points, s)
kv_pts_local_alt = point_proj_local_interleaved(sm.ipa.linear_kv_points, s)
k_pts_local_alt = kv_pts_local_alt[:, :, :, 1:sm.ipa.no_qk_points, :]
v_pts_local_alt = kv_pts_local_alt[:, :, :, (sm.ipa.no_qk_points + 1):end, :]

s_ref = Float32.(ref["s"])
q_pts_linear_from_ref_s = sm.ipa.linear_q_points.linear(s_ref)
kv_pts_linear_from_ref_s = sm.ipa.linear_kv_points.linear(s_ref)

function diff_stats(a, b)
    max_abs = maximum(abs.(a .- b))
    mean_abs = mean(abs.(a .- b))
    return max_abs, mean_abs
end

pairs = Dict(
    "s" => dbg[:s],
    "z" => dbg[:z],
    "q" => dbg[:q],
    "k" => dbg[:k],
    "v" => dbg[:v],
    "q_pts" => dbg[:q_pts],
    "k_pts" => dbg[:k_pts],
    "v_pts" => dbg[:v_pts],
    "q_pts_local" => q_pts_local,
    "k_pts_local" => k_pts_local,
    "v_pts_local" => v_pts_local,
    "q_pts_linear" => q_pts_linear,
    "kv_pts_linear" => kv_pts_linear,
    "b" => dbg[:b],
    "attn_logits" => dbg[:attn_logits],
    "attn" => dbg[:attn],
    "o_flat" => dbg[:o_flat],
    "o_pt_norm" => dbg[:o_pt_norm],
    "o_pair" => dbg[:o_pair],
    "out" => dbg[:out],
)

for (k, v) in pairs
    a = Float32.(Array(v))
    b = Float32.(ref[k])
    max_abs, mean_abs = diff_stats(a, b)
    println(k, " max_abs=", max_abs, " mean_abs=", mean_abs)
end

let
    a = Float32.(Array(q_pts_local_alt))
    b = Float32.(ref["q_pts_local"])
    max_abs, mean_abs = diff_stats(a, b)
    println("q_pts_local_alt_vs_ref max_abs=", max_abs, " mean_abs=", mean_abs)
end

let
    a = Float32.(Array(k_pts_local_alt))
    b = Float32.(ref["k_pts_local"])
    max_abs, mean_abs = diff_stats(a, b)
    println("k_pts_local_alt_vs_ref max_abs=", max_abs, " mean_abs=", mean_abs)
end

let
    a = Float32.(Array(v_pts_local_alt))
    b = Float32.(ref["v_pts_local"])
    max_abs, mean_abs = diff_stats(a, b)
    println("v_pts_local_alt_vs_ref max_abs=", max_abs, " mean_abs=", mean_abs)
end

let
    a = Float32.(Array(q_pts_linear_from_ref_s))
    b = Float32.(ref["q_pts_linear"])
    max_abs, mean_abs = diff_stats(a, b)
    println("q_pts_linear_from_ref_s_vs_ref max_abs=", max_abs, " mean_abs=", mean_abs)
end

let
    a = Float32.(Array(kv_pts_linear_from_ref_s))
    b = Float32.(ref["kv_pts_linear"])
    max_abs, mean_abs = diff_stats(a, b)
    println("kv_pts_linear_from_ref_s_vs_ref max_abs=", max_abs, " mean_abs=", mean_abs)
end

function diff_with_perm(label, a, b, perm)
    b_perm = permutedims(b, perm)
    if size(a) != size(b_perm)
        println(label, " perm=", perm, " skipped size(a)=", size(a), " size(b_perm)=", size(b_perm))
        return
    end
    max_abs, mean_abs = diff_stats(a, b_perm)
    println(label, " perm=", perm, " max_abs=", max_abs, " mean_abs=", mean_abs)
end

let
    a = Float32.(dbg[:q_pts])
    b = Float32.(ref["q_pts"])
    # Only permute last 3 dims (H, P, 3)
    nd = ndims(a)
    base = collect(1:(nd - 3))
    perms = [
        (nd - 2, nd - 1, nd),
        (nd - 2, nd, nd - 1),
        (nd - 1, nd - 2, nd),
        (nd - 1, nd, nd - 2),
        (nd, nd - 2, nd - 1),
        (nd, nd - 1, nd - 2),
    ]
    for p in perms
        perm = vcat(base, collect(p))
        diff_with_perm("q_pts_perm", a, b, Tuple(perm))
    end
end
