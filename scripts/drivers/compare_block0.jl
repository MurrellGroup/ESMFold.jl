using Pkg
Pkg.activate("/Users/benmurrell/JuliaM3/juliaESM"; io=devnull)

using NPZ
using Statistics
using ESMEmbed
using NNlib

function tri_attn_internals(ta::ESMEmbed.TriangleAttention, x, mask; python_order::Bool=true)
    if !ta.starting
        n = ndims(x)
        x = permutedims(x, vcat(collect(1:(n - 3)), [n - 1, n - 2, n]))
        nmask = ndims(mask)
        mask = permutedims(mask, vcat(collect(1:(nmask - 2)), [nmask, nmask - 1]))
    end

    x_ln = ta.layer_norm(x)
    mask_bias = ta.inf .* (mask .- 1)
    mask_bias = reshape(mask_bias, size(mask_bias, 1), size(mask_bias, 2), 1, 1, size(mask_bias, 3))
    triangle_bias = ESMEmbed.permute_final_dims(ta.linear(x_ln), (2, 0, 1)) # (B, H, I, J)
    triangle_bias = reshape(
        triangle_bias,
        size(triangle_bias, 1),
        1,
        size(triangle_bias, 2),
        size(triangle_bias, 3),
        size(triangle_bias, 4),
    )

    q = ta.mha.linear_q(x_ln)
    k = ta.mha.linear_k(x_ln)
    v = ta.mha.linear_v(x_ln)
    q = reshape(q, size(q)[1:end-1]..., ta.mha.c_hidden, ta.mha.no_heads)
    k = reshape(k, size(k)[1:end-1]..., ta.mha.c_hidden, ta.mha.no_heads)
    v = reshape(v, size(v)[1:end-1]..., ta.mha.c_hidden, ta.mha.no_heads)

    # permute to (..., Q, H, C)
    n = ndims(q)
    q = permutedims(q, vcat(collect(1:(n - 2)), n, n - 1))
    k = permutedims(k, vcat(collect(1:(n - 2)), n, n - 1))
    v = permutedims(v, vcat(collect(1:(n - 2)), n, n - 1))

    # transpose to (..., H, Q, C)
    n = ndims(q)
    q = permutedims(q, vcat(collect(1:(n - 3)), [n - 1, n - 2, n]))
    k = permutedims(k, vcat(collect(1:(n - 3)), [n - 1, n - 2, n]))
    v = permutedims(v, vcat(collect(1:(n - 3)), [n - 1, n - 2, n]))
    q ./= sqrt(Float32(ta.mha.c_hidden))

    batch_shape = size(x_ln)[1:end-2]
    Q = size(x_ln, ndims(x_ln) - 1)
    K = size(x_ln, ndims(x_ln) - 1)
    H = ta.mha.no_heads
    C = ta.mha.c_hidden

    q3 = reshape(q, prod(batch_shape) * H, Q, C)
    k3 = reshape(k, prod(batch_shape) * H, K, C)
    q3 = permutedims(q3, (2, 3, 1))
    k3 = permutedims(k3, (2, 3, 1))
    logits3 = NNlib.batched_mul(q3, permutedims(k3, (2, 1, 3))) # (Q, K, B*H)

    logits = reshape(logits3, Q, K, batch_shape..., H)
    logits = permutedims(logits, (3:(2 + length(batch_shape))..., ndims(logits), 1, 2)) # (batch..., H, Q, K)
    logits .+= mask_bias .+ triangle_bias

    attn = NNlib.softmax(logits; dims=ndims(logits))

    attn_qk = permutedims(attn, vcat((ndims(attn) - 1), ndims(attn), collect(1:(ndims(attn) - 2))...))
    attn3 = reshape(attn_qk, Q, K, :)
    v3 = reshape(v, prod(batch_shape) * H, K, C)
    v3 = permutedims(v3, (2, 3, 1)) # (K, C, B*H)
    o3 = NNlib.batched_mul(attn3, v3) # (Q, C, B*H)
    o = reshape(o3, Q, C, batch_shape..., H)
    o = permutedims(o, (3:(2 + length(batch_shape))..., ndims(o), 1, 2)) # (batch..., H, Q, C)
    o = permutedims(o, vcat(collect(1:(ndims(o) - 3)), [ndims(o) - 1, ndims(o) - 2, ndims(o)])) # (batch..., Q, H, C)

    if ta.mha.linear_g !== nothing
        g = NNlib.sigmoid.(ta.mha.linear_g(x_ln))
        g = reshape(g, size(g)[1:end-1]..., C, H)
        n = ndims(g)
        g = permutedims(g, vcat(collect(1:(n - 2)), n, n - 1)) # (..., Q, H, C)
        o .*= g
    end

    if python_order
        n = ndims(o)
        o = permutedims(o, vcat(collect(1:(n - 2)), n, n - 1)) # (..., Q, C, H)
    end
    o_flat = reshape(o, size(o)[1:end-2]..., H * C)
    return logits, attn, o_flat, mask_bias, triangle_bias, x_ln, q, k, v
end



ref = NPZ.npzread("/Users/benmurrell/JuliaM3/juliaESM/esmfold_block0_debug.npz")

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
aa = ESMEmbed.sequence_to_af2_indices(seq)
aa = reshape(aa, 1, :)
mask = ones(Int, size(aa))

embed_out = model.embed(aa; mask=mask, return_pair=false)
s_s_0 = permutedims(embed_out, (3,2,1))
B, L, _ = size(s_s_0)
s_z_0 = zeros(Float32, B, L, L, model.cfg.trunk.pairwise_state_dim)

block = model.trunk.blocks[1]

bias = block.pair_to_sequence(s_z_0)
seq_ln = block.layernorm_1(s_s_0)
proj_out = block.seq_attention.proj(seq_ln)
# q,k,v using same split as ESMFoldAttention (head-major qkv)
H = block.seq_attention.num_heads
head_width = block.seq_attention.head_width
t = reshape(proj_out, B, L, head_width, 3, H)
t = permutedims(t, (1,2,5,4,3)) # (B, L, H, 3, head_width)
q = view(t, :, :, :, 1, :)
k = view(t, :, :, :, 2, :)
v = view(t, :, :, :, 3, :)

g_proj_out = block.seq_attention.g_proj(seq_ln)

seq_attn_out, _ = block.seq_attention(seq_ln; mask=mask, bias=bias)
seq_state_attn = s_s_0 .+ seq_attn_out
seq_state_mlp = block.mlp_seq(seq_state_attn)

pair_state = s_z_0 .+ block.sequence_to_pair(seq_state_mlp)
tri_mask = reshape(mask, size(mask,1), size(mask,2), 1) .* reshape(mask, size(mask,1), 1, size(mask,2))

tri_mul_out = block.tri_mul_out(pair_state; mask=tri_mask)
pair_state = pair_state .+ tri_mul_out
tri_mul_in = block.tri_mul_in(pair_state; mask=tri_mask)
pair_state = pair_state .+ tri_mul_in
pair_state_before_start = pair_state
tri_att_start = block.tri_att_start(pair_state_before_start; mask=tri_mask)
pair_state = pair_state_before_start .+ tri_att_start
pair_state_before_end = pair_state
tri_att_end = block.tri_att_end(pair_state_before_end; mask=tri_mask)
pair_state = pair_state_before_end .+ tri_att_end
pair_state_mlp = block.mlp_pair(pair_state)

tri_att_start_logits, tri_att_start_attn, tri_att_start_pre_o, tri_att_start_mask_bias, tri_att_start_triangle_bias, tri_att_start_ln, tri_att_start_q, tri_att_start_k, tri_att_start_v = tri_attn_internals(
    block.tri_att_start,
    pair_state_before_start,
    tri_mask;
    python_order=true,
)
tri_att_end_logits, tri_att_end_attn, tri_att_end_pre_o, tri_att_end_mask_bias, tri_att_end_triangle_bias, tri_att_end_ln, tri_att_end_q, tri_att_end_k, tri_att_end_v = tri_attn_internals(
    block.tri_att_end,
    pair_state_before_end,
    tri_mask;
    python_order=true,
)

# Compute attention weights following ESMFoldAttention implementation
function attn_weights(m::ESMEmbed.ESMFoldAttention, x, mask, bias)
    t = m.proj(x)
    t = reshape(t, size(t)[1:end-1]..., m.head_width, 3, m.num_heads)
    b = ndims(x) - 2
    perm = vcat(collect(1:(b + 1)), b + 4, b + 3, b + 2)
    t = permutedims(t, perm)
    q = view(t, ntuple(_ -> :, ndims(t) - 2)..., 1, :)
    k = view(t, ntuple(_ -> :, ndims(t) - 2)..., 2, :)
    q .*= m.rescale_factor

    q_bh = permutedims(q, (1, 3, 2, 4))
    k_bh = permutedims(k, (1, 3, 2, 4))
    B = size(q_bh, 1)
    H = size(q_bh, 2)
    L = size(q_bh, 3)
    D = size(q_bh, 4)
    q3 = permutedims(reshape(q_bh, B * H, L, D), (2, 3, 1))
    k3 = permutedims(reshape(k_bh, B * H, L, D), (2, 3, 1))

    a = NNlib.batched_mul(q3, permutedims(k3, (2, 1, 3)))
    a_raw = a

    if bias !== nothing
        bias_h = permutedims(bias, (1:ndims(bias)-3..., ndims(bias), ndims(bias) - 2, ndims(bias) - 1))
        b_perm = permutedims(bias_h, (ndims(bias_h) - 1, ndims(bias_h), 1:(ndims(bias_h) - 2)...))
        b3 = reshape(b_perm, size(b_perm, 1), size(b_perm, 2), :)
        a .+= b3
    end

    if mask !== nothing
        B, L = size(mask, 1), size(mask, 2)
        mask_k = reshape(mask, B, 1, 1, L)
        mask_k = repeat(mask_k, 1, m.num_heads, L, 1)
        neg_inf = oftype(zero(eltype(a)), -Inf)
        mask_bias = ifelse.(mask_k .== 1, zero(eltype(a)), neg_inf)
        mb_perm = permutedims(mask_bias, (3, 4, 1, 2))
        mb3 = reshape(mb_perm, size(mb_perm, 1), size(mb_perm, 2), :)
        a .+= mb3
    end

    a = NNlib.softmax(a; dims=2)

    # reshape to (B, H, L, L)
    a4 = reshape(a, L, L, B, m.num_heads)
    a4 = permutedims(a4, (3, 4, 1, 2))
    # reshape raw logits to (B, H, L, L)
    a_raw4 = reshape(a_raw, L, L, B, m.num_heads)
    a_raw4 = permutedims(a_raw4, (3, 4, 1, 2))
    return a4, a_raw4
end

function attn_pre_outputs(m::ESMEmbed.ESMFoldAttention, x, mask, bias)
    t = m.proj(x)
    t = reshape(t, size(t)[1:end-1]..., m.head_width, 3, m.num_heads)
    b = ndims(x) - 2
    perm = vcat(collect(1:(b + 1)), b + 4, b + 3, b + 2)
    t = permutedims(t, perm)
    q = view(t, ntuple(_ -> :, ndims(t) - 2)..., 1, :)
    k = view(t, ntuple(_ -> :, ndims(t) - 2)..., 2, :)
    v = view(t, ntuple(_ -> :, ndims(t) - 2)..., 3, :)
    q .*= m.rescale_factor

    q_bh = permutedims(q, (1, 3, 2, 4))
    k_bh = permutedims(k, (1, 3, 2, 4))
    v_bh = permutedims(v, (1, 3, 2, 4))
    B = size(q_bh, 1)
    H = size(q_bh, 2)
    L = size(q_bh, 3)
    D = size(q_bh, 4)

    q3 = permutedims(reshape(q_bh, B * H, L, D), (2, 3, 1))
    k3 = permutedims(reshape(k_bh, B * H, L, D), (2, 3, 1))
    v3 = permutedims(reshape(v_bh, B * H, L, D), (2, 3, 1))

    a = NNlib.batched_mul(q3, permutedims(k3, (2, 1, 3)))

    if bias !== nothing
        bias_h = permutedims(bias, (1:ndims(bias)-3..., ndims(bias), ndims(bias) - 2, ndims(bias) - 1))
        b_perm = permutedims(bias_h, (ndims(bias_h) - 1, ndims(bias_h), 1:(ndims(bias_h) - 2)...))
        b3 = reshape(b_perm, size(b_perm, 1), size(b_perm, 2), :)
        a .+= b3
    end

    if mask !== nothing
        Bm, Lm = size(mask, 1), size(mask, 2)
        mask_k = reshape(mask, Bm, 1, 1, Lm)
        mask_k = repeat(mask_k, 1, m.num_heads, Lm, 1)
        neg_inf = oftype(zero(eltype(a)), -Inf)
        mask_bias = ifelse.(mask_k .== 1, zero(eltype(a)), neg_inf)
        mb_perm = permutedims(mask_bias, (3, 4, 1, 2))
        mb3 = reshape(mb_perm, size(mb_perm, 1), size(mb_perm, 2), :)
        a .+= mb3
    end

    a = NNlib.softmax(a; dims=2)
    o = NNlib.batched_mul(a, v3)
    o = reshape(o, L, D, B, H)
    o = permutedims(o, (3, 1, 2, 4))
    o = reshape(o, B, L, H * D)

    o_pre_gate = o
    if m.gated
        g = NNlib.sigmoid.(m.g_proj(x))
        o = g .* o
    end
    return o_pre_gate, o
end

attn, attn_logits_model = attn_weights(block.seq_attention, seq_ln, mask, bias) # (B, H, L, L)
attn_perm = permutedims(attn, (1, 3, 2, 4)) # (B, L, H, L) to match python output
ref_attn = Float32.(ref["attn"])
ref_attn_bhlk = permutedims(ref_attn, (1, 3, 2, 4)) # (B, H, L, L)

seq_attn_pre_gate, seq_attn_pre_o_proj = attn_pre_outputs(block.seq_attention, seq_ln, mask, bias)

# logits to match python debug (B, H, L, L)
q_py = permutedims(q, (1, 3, 2, 4)) # (B, H, L, D)
k_py = permutedims(k, (1, 3, 2, 4)) # (B, H, L, D)
Bq, Hq, Lq, Dq = size(q_py)
q_bh = reshape(q_py, Bq * Hq, Lq, Dq)
k_bh = reshape(k_py, Bq * Hq, Lq, Dq)
q3 = permutedims(q_bh, (2, 3, 1)) # (L, D, B*H)
k3 = permutedims(k_bh, (3, 2, 1)) # (D, L, B*H)
logits3 = NNlib.batched_mul(q3, k3) # (L, L, B*H)
logits = reshape(logits3, Lq, Lq, Bq, Hq)
logits = permutedims(logits, (3, 4, 1, 2)) # (B, H, L, L)

function diff_stats(a, b)
    max_abs = maximum(abs.(a .- b))
    mean_abs = mean(abs.(a .- b))
    return max_abs, mean_abs
end

pairs = Dict(
    "s_s_0" => s_s_0,
    "s_z_0" => s_z_0,
    "bias" => bias,
    "seq_ln" => seq_ln,
    "proj_out" => proj_out,
    "q" => q,
    "k" => k,
    "v" => v,
    "g_proj_out" => g_proj_out,
    "seq_attn_out" => seq_attn_out,
    "seq_attn_pre_gate" => seq_attn_pre_gate,
    "seq_attn_pre_o_proj" => seq_attn_pre_o_proj,
    "attn" => attn_perm,
    "logits" => logits,
    "seq_state_attn" => seq_state_attn,
    "seq_state_mlp" => seq_state_mlp,
    "pair_state_seq2pair" => (s_z_0 .+ block.sequence_to_pair(seq_state_mlp)),
    "tri_mul_out" => tri_mul_out,
    "tri_mul_in" => tri_mul_in,
    "pair_state_before_start" => pair_state_before_start,
    "pair_state_before_end" => pair_state_before_end,
    "tri_att_start" => tri_att_start,
    "tri_att_end" => tri_att_end,
    "tri_att_start_logits" => tri_att_start_logits,
    "tri_att_start_attn" => tri_att_start_attn,
    "tri_att_start_pre_o" => tri_att_start_pre_o,
    "tri_att_start_mask_bias" => tri_att_start_mask_bias,
    "tri_att_start_triangle_bias" => tri_att_start_triangle_bias,
    "tri_att_start_ln" => tri_att_start_ln,
    "tri_att_start_q" => tri_att_start_q,
    "tri_att_start_k" => tri_att_start_k,
    "tri_att_start_v" => tri_att_start_v,
    "tri_att_end_logits" => tri_att_end_logits,
    "tri_att_end_attn" => tri_att_end_attn,
    "tri_att_end_pre_o" => tri_att_end_pre_o,
    "tri_att_end_mask_bias" => tri_att_end_mask_bias,
    "tri_att_end_triangle_bias" => tri_att_end_triangle_bias,
    "tri_att_end_ln" => tri_att_end_ln,
    "tri_att_end_q" => tri_att_end_q,
    "tri_att_end_k" => tri_att_end_k,
    "tri_att_end_v" => tri_att_end_v,
    "pair_state_final" => pair_state_mlp,
)

for (k, v) in pairs
    a = Float32.(Array(v))
    b = Float32.(ref[k])
    max_abs, mean_abs = diff_stats(a, b)
    println(k, " max_abs=", max_abs, " mean_abs=", mean_abs)
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
    a = Float32.(tri_att_start_q)
    b = Float32.(ref["tri_att_start_q"])
    perms = [
        (1, 2, 3, 4, 5),
        (1, 3, 2, 4, 5),
        (1, 2, 4, 3, 5),
        (1, 4, 3, 2, 5),
        (1, 3, 4, 2, 5),
        (1, 4, 2, 3, 5),
    ]
    for perm in perms
        diff_with_perm("tri_att_start_q_perm", a, b, perm)
    end
end

let stats = diff_stats(Float32.(attn), ref_attn_bhlk)
    max_abs, mean_abs = stats
    println("attn_bhlk_vs_ref max_abs=", max_abs, " mean_abs=", mean_abs)
end

let bias_bhlk = permutedims(Float32.(bias), (1, 4, 2, 3))
    attn_from_logits = NNlib.softmax(Float32.(attn_logits_model) .+ bias_bhlk; dims=4)
    stats = diff_stats(attn_from_logits, ref_attn_bhlk)
    max_abs, mean_abs = stats
    println("attn_from_logits_vs_ref max_abs=", max_abs, " mean_abs=", mean_abs)
end

let bias_bhlk = permutedims(Float32.(bias), (1, 4, 2, 3))
    logits = Float32.(attn_logits_model) .+ bias_bhlk
    max_logits = maximum(logits; dims=4)
    exp_logits = exp.(logits .- max_logits)
    attn_stable = exp_logits ./ sum(exp_logits; dims=4)
    stats = diff_stats(attn_stable, ref_attn_bhlk)
    max_abs, mean_abs = stats
    println("attn_from_logits_stable_vs_ref max_abs=", max_abs, " mean_abs=", mean_abs)
end

# Manual attention output using python logits/v + Julia projections.
let
    logits_scaled = Float32.(ref["logits"]) .* block.seq_attention.rescale_factor
    bias_bhlk = permutedims(Float32.(bias), (1, 4, 2, 3))
    attn_bhlk = NNlib.softmax(logits_scaled .+ bias_bhlk; dims=4) # (B,H,L,L)

    v_py = Float32.(ref["v"]) # (B,L,H,D)
    v_bhlk = permutedims(v_py, (1, 3, 2, 4)) # (B,H,L,D)

    Bv, Hv, Lv, Dv = size(v_bhlk)
    a3 = reshape(permutedims(attn_bhlk, (3, 4, 1, 2)), Lv, Lv, Bv * Hv)
    v3 = reshape(permutedims(v_bhlk, (3, 4, 1, 2)), Lv, Dv, Bv * Hv)
    o3 = NNlib.batched_mul(a3, v3) # (L, D, B*H)
    o = reshape(o3, Lv, Dv, Bv, Hv)
    o = permutedims(o, (3, 1, 2, 4)) # (B, L, D, H)
    o = reshape(o, Bv, Lv, Hv * Dv) # (B, L, C)

    if block.seq_attention.gated
        g = NNlib.sigmoid.(g_proj_out)
        o = g .* o
    end
    seq_attn_manual = block.seq_attention.o_proj(o)
    stats = diff_stats(seq_attn_manual, Float32.(ref["seq_attn_out"]))
    max_abs, mean_abs = stats
    println("seq_attn_manual_vs_ref max_abs=", max_abs, " mean_abs=", mean_abs)
end

let stats = diff_stats(Float32.(attn_logits_model), Float32.(ref["logits"]))
    max_abs, mean_abs = stats
    println("attn_logits_model_vs_ref max_abs=", max_abs, " mean_abs=", mean_abs)
end

let stats = diff_stats(Float32.(attn_logits_model) ./ block.seq_attention.rescale_factor, Float32.(ref["logits"]))
    max_abs, mean_abs = stats
    println("attn_logits_model_unscaled_vs_ref max_abs=", max_abs, " mean_abs=", mean_abs)
end

let stats = diff_stats(Float32.(attn_logits_model), Float32.(ref["logits"]) .* block.seq_attention.rescale_factor)
    max_abs, mean_abs = stats
    println("attn_logits_model_vs_ref_scaled max_abs=", max_abs, " mean_abs=", mean_abs)
end

let ref_logits_scaled = Float32.(ref["logits"]) .* block.seq_attention.rescale_factor
    ref_logits_scaled_t = permutedims(ref_logits_scaled, (1, 2, 4, 3))
    stats = diff_stats(Float32.(attn_logits_model), ref_logits_scaled_t)
    max_abs, mean_abs = stats
    println("attn_logits_model_vs_ref_scaled_T max_abs=", max_abs, " mean_abs=", mean_abs)
end
