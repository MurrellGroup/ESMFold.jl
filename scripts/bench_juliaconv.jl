using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using BenchmarkTools
using Random
using Statistics
using ESMFold

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0
BenchmarkTools.DEFAULT_PARAMETERS.samples = 50
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1

Random.seed!(42)

function _report(name::AbstractString, trial::BenchmarkTools.Trial)
    tmin = minimum(trial).time
    tmed = median(trial).time
    mem = median(trial).memory
    allocs = median(trial).allocs
    println("\n=== ", name, " ===")
    println("min:  ", BenchmarkTools.prettytime(tmin))
    println("median:", BenchmarkTools.prettytime(tmed))
    println("memory:", Base.format_bytes(mem), "  allocs: ", allocs)
    return nothing
end

function _bench_esmfold_attention(; B=1, L=256, C=1024, num_heads=32, head_width=32)
    x = rand(Float32, B, L, C)
    mask = ones(Float32, B, L)
    bias = nothing
    attn = ESMFold.ESMFoldAttention(C, num_heads, head_width; gated=true)

    # Warmup to avoid compilation time.
    attn(x; mask=mask, bias=bias)
    GC.gc()

    trial = @benchmark $attn($x; mask=$mask, bias=$bias)
    _report("ESMFoldAttention (feature-last)", trial)
    return trial
end

function _bench_esmfold_attention_jl(; B=1, L=256, C=1024, num_heads=32, head_width=32)
    x = rand(Float32, C, L, B)
    mask = ones(Float32, L, B)
    bias = nothing

    attn_fl = ESMFold.ESMFoldAttention(C, num_heads, head_width; gated=true)
    attn_jl = ESMFold.ESMFoldAttentionJL(C, num_heads, head_width; gated=true)

    attn_jl.proj.weight .= attn_fl.proj.weight
    attn_jl.o_proj.weight .= attn_fl.o_proj.weight
    attn_jl.o_proj.bias .= attn_fl.o_proj.bias
    attn_jl.g_proj.weight .= attn_fl.g_proj.weight
    attn_jl.g_proj.bias .= attn_fl.g_proj.bias

    if get(ENV, "CHECK", "0") == "1"
        x_fl = permutedims(x, (3, 2, 1))
        mask_fl = permutedims(mask, (2, 1))
        y_fl, _ = attn_fl(x_fl; mask=mask_fl, bias=bias)
        y_jl, _ = attn_jl(x; mask=mask, bias=bias)
        y_fl_jl = permutedims(y_fl, (3, 2, 1))
        max_diff = maximum(abs.(y_fl_jl .- y_jl))
        println("Parity check max diff: ", max_diff)
    end

    attn_jl(x; mask=mask, bias=bias)
    GC.gc()

    trial = @benchmark $attn_jl($x; mask=$mask, bias=$bias)
    _report("ESMFoldAttentionJL (feature-first)", trial)
    return trial
end

function _bench_sequence_to_pair(; B=1, L=256, C_s=1024, C_z=128)
    x = rand(Float32, B, L, C_s)
    layer = ESMFold.SequenceToPair(C_s, C_z ÷ 2, C_z)

    layer(x)
    GC.gc()

    trial = @benchmark $layer($x)
    _report("SequenceToPair (feature-last)", trial)
    return trial
end

function _bench_sequence_to_pair_jl(; B=1, L=256, C_s=1024, C_z=128)
    x = rand(Float32, C_s, L, B)
    layer_fl = ESMFold.SequenceToPair(C_s, C_z ÷ 2, C_z)
    layer_jl = ESMFold.SequenceToPairJL(C_s, C_z ÷ 2, C_z)

    layer_jl.proj.weight .= layer_fl.proj.weight
    layer_jl.proj.bias .= layer_fl.proj.bias
    layer_jl.o_proj.weight .= layer_fl.o_proj.weight
    layer_jl.o_proj.bias .= layer_fl.o_proj.bias
    layer_jl.layernorm.w .= layer_fl.layernorm.w
    layer_jl.layernorm.b .= layer_fl.layernorm.b

    if get(ENV, "CHECK", "0") == "1"
        x_fl = permutedims(x, (3, 2, 1))
        y_fl = layer_fl(x_fl)
        y_jl = layer_jl(x)
        y_fl_jl = permutedims(y_fl, (4, 2, 3, 1))
        max_diff = maximum(abs.(y_fl_jl .- y_jl))
        println("Parity check max diff: ", max_diff)
    end

    layer_jl(x)
    GC.gc()

    trial = @benchmark $layer_jl($x)
    _report("SequenceToPairJL (feature-first)", trial)
    return trial
end

function _bench_pair_to_sequence(; B=1, L=256, C_z=128, num_heads=32)
    z = rand(Float32, B, L, L, C_z)
    layer = ESMFold.PairToSequence(C_z, num_heads)

    layer(z)
    GC.gc()

    trial = @benchmark $layer($z)
    _report("PairToSequence (feature-last)", trial)
    return trial
end

function _bench_pair_to_sequence_jl(; B=1, L=256, C_z=128, num_heads=32)
    z = rand(Float32, C_z, L, L, B)
    layer_fl = ESMFold.PairToSequence(C_z, num_heads)
    layer_jl = ESMFold.PairToSequenceJL(C_z, num_heads)

    layer_jl.linear.weight .= layer_fl.linear.weight
    layer_jl.layernorm.w .= layer_fl.layernorm.w
    layer_jl.layernorm.b .= layer_fl.layernorm.b

    if get(ENV, "CHECK", "0") == "1"
        z_fl = permutedims(z, (4, 2, 3, 1))
        y_fl = layer_fl(z_fl)
        y_jl = layer_jl(z)
        y_fl_jl = permutedims(y_fl, (4, 2, 3, 1))
        max_diff = maximum(abs.(y_fl_jl .- y_jl))
        println("Parity check max diff: ", max_diff)
    end

    layer_jl(z)
    GC.gc()

    trial = @benchmark $layer_jl($z)
    _report("PairToSequenceJL (feature-first)", trial)
    return trial
end

function _bench_triangle_attention_start(; B=1, L=128, C_z=128, head_width=32)
    num_heads = C_z ÷ head_width
    x = rand(Float32, B, L, L, C_z)
    mask = ones(Float32, B, L, L)
    layer = ESMFold.TriangleAttention(C_z, head_width, num_heads; starting=true, inf=1e9)

    layer(x; mask=mask)
    GC.gc()

    trial = @benchmark $layer($x; mask=$mask)
    _report("TriangleAttention (feature-last, start)", trial)
    return trial
end

function _bench_triangle_attention_start_jl(; B=1, L=128, C_z=128, head_width=32)
    num_heads = C_z ÷ head_width
    x = rand(Float32, C_z, L, L, B)
    mask = ones(Float32, L, L, B)

    layer_fl = ESMFold.TriangleAttention(C_z, head_width, num_heads; starting=true, inf=1e9)
    layer_jl = ESMFold.TriangleAttentionJL(C_z, head_width, num_heads; starting=true, inf=1e9)

    layer_jl.linear.weight .= layer_fl.linear.weight
    layer_jl.layer_norm.w .= layer_fl.layer_norm.w
    layer_jl.layer_norm.b .= layer_fl.layer_norm.b
    layer_jl.mha.linear_q.weight .= layer_fl.mha.linear_q.weight
    layer_jl.mha.linear_k.weight .= layer_fl.mha.linear_k.weight
    layer_jl.mha.linear_v.weight .= layer_fl.mha.linear_v.weight
    layer_jl.mha.linear_o.weight .= layer_fl.mha.linear_o.weight
    layer_jl.mha.linear_o.bias .= layer_fl.mha.linear_o.bias
    if layer_fl.mha.linear_g !== nothing
        layer_jl.mha.linear_g.weight .= layer_fl.mha.linear_g.weight
        layer_jl.mha.linear_g.bias .= layer_fl.mha.linear_g.bias
    end

    if get(ENV, "CHECK", "0") == "1"
        x_fl = permutedims(x, (4, 2, 3, 1))
        mask_fl = permutedims(mask, (3, 1, 2))
        y_fl = layer_fl(x_fl; mask=mask_fl)
        y_jl = layer_jl(x; mask=mask)
        y_fl_jl = permutedims(y_fl, (4, 2, 3, 1))
        max_diff = maximum(abs.(y_fl_jl .- y_jl))
        println("Parity check max diff: ", max_diff)
    end

    layer_jl(x; mask=mask)
    GC.gc()

    trial = @benchmark $layer_jl($x; mask=$mask)
    _report("TriangleAttentionJL (feature-first, start)", trial)
    return trial
end

function _bench_triangle_attention_end(; B=1, L=128, C_z=128, head_width=32)
    num_heads = C_z ÷ head_width
    x = rand(Float32, B, L, L, C_z)
    mask = ones(Float32, B, L, L)
    layer = ESMFold.TriangleAttention(C_z, head_width, num_heads; starting=false, inf=1e9)

    layer(x; mask=mask)
    GC.gc()

    trial = @benchmark $layer($x; mask=$mask)
    _report("TriangleAttention (feature-last, end)", trial)
    return trial
end

function _bench_triangle_attention_end_jl(; B=1, L=128, C_z=128, head_width=32)
    num_heads = C_z ÷ head_width
    x = rand(Float32, C_z, L, L, B)
    mask = ones(Float32, L, L, B)

    layer_fl = ESMFold.TriangleAttention(C_z, head_width, num_heads; starting=false, inf=1e9)
    layer_jl = ESMFold.TriangleAttentionJL(C_z, head_width, num_heads; starting=false, inf=1e9)

    layer_jl.linear.weight .= layer_fl.linear.weight
    layer_jl.layer_norm.w .= layer_fl.layer_norm.w
    layer_jl.layer_norm.b .= layer_fl.layer_norm.b
    layer_jl.mha.linear_q.weight .= layer_fl.mha.linear_q.weight
    layer_jl.mha.linear_k.weight .= layer_fl.mha.linear_k.weight
    layer_jl.mha.linear_v.weight .= layer_fl.mha.linear_v.weight
    layer_jl.mha.linear_o.weight .= layer_fl.mha.linear_o.weight
    layer_jl.mha.linear_o.bias .= layer_fl.mha.linear_o.bias
    if layer_fl.mha.linear_g !== nothing
        layer_jl.mha.linear_g.weight .= layer_fl.mha.linear_g.weight
        layer_jl.mha.linear_g.bias .= layer_fl.mha.linear_g.bias
    end

    if get(ENV, "CHECK", "0") == "1"
        x_fl = permutedims(x, (4, 2, 3, 1))
        mask_fl = permutedims(mask, (3, 1, 2))
        y_fl = layer_fl(x_fl; mask=mask_fl)
        y_jl = layer_jl(x; mask=mask)
        y_fl_jl = permutedims(y_fl, (4, 2, 3, 1))
        max_diff = maximum(abs.(y_fl_jl .- y_jl))
        println("Parity check max diff: ", max_diff)
    end

    layer_jl(x; mask=mask)
    GC.gc()

    trial = @benchmark $layer_jl($x; mask=$mask)
    _report("TriangleAttentionJL (feature-first, end)", trial)
    return trial
end

function _bench_triangle_mul_out(; B=1, L=128, C_z=128)
    x = rand(Float32, B, L, L, C_z)
    mask = ones(Float32, B, L, L)
    layer = ESMFold.TriangleMultiplicationOutgoing(C_z, C_z)

    layer(x; mask=mask)
    GC.gc()

    trial = @benchmark $layer($x; mask=$mask)
    _report("TriangleMultiplicationOutgoing (feature-last)", trial)
    return trial
end

function _bench_triangle_mul_out_jl(; B=1, L=128, C_z=128)
    x = rand(Float32, C_z, L, L, B)
    mask = ones(Float32, L, L, B)

    layer_fl = ESMFold.TriangleMultiplicationOutgoing(C_z, C_z)
    layer_jl = ESMFold.TriangleMultiplicationOutgoingJL(C_z, C_z)

    layer_jl.inner.linear_a_p.weight .= layer_fl.inner.linear_a_p.weight
    layer_jl.inner.linear_a_p.bias .= layer_fl.inner.linear_a_p.bias
    layer_jl.inner.linear_a_g.weight .= layer_fl.inner.linear_a_g.weight
    layer_jl.inner.linear_a_g.bias .= layer_fl.inner.linear_a_g.bias
    layer_jl.inner.linear_b_p.weight .= layer_fl.inner.linear_b_p.weight
    layer_jl.inner.linear_b_p.bias .= layer_fl.inner.linear_b_p.bias
    layer_jl.inner.linear_b_g.weight .= layer_fl.inner.linear_b_g.weight
    layer_jl.inner.linear_b_g.bias .= layer_fl.inner.linear_b_g.bias
    layer_jl.inner.linear_g.weight .= layer_fl.inner.linear_g.weight
    layer_jl.inner.linear_g.bias .= layer_fl.inner.linear_g.bias
    layer_jl.inner.linear_z.weight .= layer_fl.inner.linear_z.weight
    layer_jl.inner.linear_z.bias .= layer_fl.inner.linear_z.bias
    layer_jl.inner.layer_norm_in.w .= layer_fl.inner.layer_norm_in.w
    layer_jl.inner.layer_norm_in.b .= layer_fl.inner.layer_norm_in.b
    layer_jl.inner.layer_norm_out.w .= layer_fl.inner.layer_norm_out.w
    layer_jl.inner.layer_norm_out.b .= layer_fl.inner.layer_norm_out.b

    if get(ENV, "CHECK", "0") == "1"
        x_fl = permutedims(x, (4, 2, 3, 1))
        mask_fl = permutedims(mask, (3, 1, 2))
        y_fl = layer_fl(x_fl; mask=mask_fl)
        y_jl = layer_jl(x; mask=mask)
        y_fl_jl = permutedims(y_fl, (4, 2, 3, 1))
        max_diff = maximum(abs.(y_fl_jl .- y_jl))
        println("Parity check max diff: ", max_diff)
    end

    layer_jl(x; mask=mask)
    GC.gc()

    trial = @benchmark $layer_jl($x; mask=$mask)
    _report("TriangleMultiplicationOutgoingJL (feature-first)", trial)
    return trial
end

function _bench_triangle_mul_in(; B=1, L=128, C_z=128)
    x = rand(Float32, B, L, L, C_z)
    mask = ones(Float32, B, L, L)
    layer = ESMFold.TriangleMultiplicationIncoming(C_z, C_z)

    layer(x; mask=mask)
    GC.gc()

    trial = @benchmark $layer($x; mask=$mask)
    _report("TriangleMultiplicationIncoming (feature-last)", trial)
    return trial
end

function _bench_triangle_mul_in_jl(; B=1, L=128, C_z=128)
    x = rand(Float32, C_z, L, L, B)
    mask = ones(Float32, L, L, B)

    layer_fl = ESMFold.TriangleMultiplicationIncoming(C_z, C_z)
    layer_jl = ESMFold.TriangleMultiplicationIncomingJL(C_z, C_z)

    layer_jl.inner.linear_a_p.weight .= layer_fl.inner.linear_a_p.weight
    layer_jl.inner.linear_a_p.bias .= layer_fl.inner.linear_a_p.bias
    layer_jl.inner.linear_a_g.weight .= layer_fl.inner.linear_a_g.weight
    layer_jl.inner.linear_a_g.bias .= layer_fl.inner.linear_a_g.bias
    layer_jl.inner.linear_b_p.weight .= layer_fl.inner.linear_b_p.weight
    layer_jl.inner.linear_b_p.bias .= layer_fl.inner.linear_b_p.bias
    layer_jl.inner.linear_b_g.weight .= layer_fl.inner.linear_b_g.weight
    layer_jl.inner.linear_b_g.bias .= layer_fl.inner.linear_b_g.bias
    layer_jl.inner.linear_g.weight .= layer_fl.inner.linear_g.weight
    layer_jl.inner.linear_g.bias .= layer_fl.inner.linear_g.bias
    layer_jl.inner.linear_z.weight .= layer_fl.inner.linear_z.weight
    layer_jl.inner.linear_z.bias .= layer_fl.inner.linear_z.bias
    layer_jl.inner.layer_norm_in.w .= layer_fl.inner.layer_norm_in.w
    layer_jl.inner.layer_norm_in.b .= layer_fl.inner.layer_norm_in.b
    layer_jl.inner.layer_norm_out.w .= layer_fl.inner.layer_norm_out.w
    layer_jl.inner.layer_norm_out.b .= layer_fl.inner.layer_norm_out.b

    if get(ENV, "CHECK", "0") == "1"
        x_fl = permutedims(x, (4, 2, 3, 1))
        mask_fl = permutedims(mask, (3, 1, 2))
        y_fl = layer_fl(x_fl; mask=mask_fl)
        y_jl = layer_jl(x; mask=mask)
        y_fl_jl = permutedims(y_fl, (4, 2, 3, 1))
        max_diff = maximum(abs.(y_fl_jl .- y_jl))
        println("Parity check max diff: ", max_diff)
    end

    layer_jl(x; mask=mask)
    GC.gc()

    trial = @benchmark $layer_jl($x; mask=$mask)
    _report("TriangleMultiplicationIncomingJL (feature-first)", trial)
    return trial
end

function _bench_triangular_block(; B=1, L=128, C_s=1024, C_z=128, seq_head_width=32, pair_head_width=32)
    x = rand(Float32, B, L, C_s)
    z = rand(Float32, B, L, L, C_z)
    mask = ones(Float32, B, L)
    layer = ESMFold.TriangularSelfAttentionBlock(C_s, C_z, seq_head_width, pair_head_width; dropout=0f0)

    layer(x, z; mask=mask)
    GC.gc()

    trial = @benchmark $layer($x, $z; mask=$mask)
    _report("TriangularSelfAttentionBlock (feature-last)", trial)
    return trial
end

function _bench_triangular_block_jl(; B=1, L=128, C_s=1024, C_z=128, seq_head_width=32, pair_head_width=32)
    x = rand(Float32, C_s, L, B)
    z = rand(Float32, C_z, L, L, B)
    mask = ones(Float32, L, B)

    layer_fl = ESMFold.TriangularSelfAttentionBlock(C_s, C_z, seq_head_width, pair_head_width; dropout=0f0)
    layer_jl = ESMFold.TriangularSelfAttentionBlockJL(C_s, C_z, seq_head_width, pair_head_width; dropout=0f0)

    layer_jl.layernorm_1.w .= layer_fl.layernorm_1.w
    layer_jl.layernorm_1.b .= layer_fl.layernorm_1.b

    # SequenceToPair weights
    layer_jl.sequence_to_pair.proj.weight .= layer_fl.sequence_to_pair.proj.weight
    layer_jl.sequence_to_pair.proj.bias .= layer_fl.sequence_to_pair.proj.bias
    layer_jl.sequence_to_pair.o_proj.weight .= layer_fl.sequence_to_pair.o_proj.weight
    layer_jl.sequence_to_pair.o_proj.bias .= layer_fl.sequence_to_pair.o_proj.bias
    layer_jl.sequence_to_pair.layernorm.w .= layer_fl.sequence_to_pair.layernorm.w
    layer_jl.sequence_to_pair.layernorm.b .= layer_fl.sequence_to_pair.layernorm.b

    # PairToSequence weights
    layer_jl.pair_to_sequence.linear.weight .= layer_fl.pair_to_sequence.linear.weight
    layer_jl.pair_to_sequence.layernorm.w .= layer_fl.pair_to_sequence.layernorm.w
    layer_jl.pair_to_sequence.layernorm.b .= layer_fl.pair_to_sequence.layernorm.b

    # Seq attention weights
    layer_jl.seq_attention.proj.weight .= layer_fl.seq_attention.proj.weight
    layer_jl.seq_attention.o_proj.weight .= layer_fl.seq_attention.o_proj.weight
    layer_jl.seq_attention.o_proj.bias .= layer_fl.seq_attention.o_proj.bias
    layer_jl.seq_attention.g_proj.weight .= layer_fl.seq_attention.g_proj.weight
    layer_jl.seq_attention.g_proj.bias .= layer_fl.seq_attention.g_proj.bias

    # Triangle mul/att weights
    for (jl, fl) in (
        (layer_jl.tri_mul_out.inner, layer_fl.tri_mul_out.inner),
        (layer_jl.tri_mul_in.inner, layer_fl.tri_mul_in.inner),
    )
        jl.linear_a_p.weight .= fl.linear_a_p.weight
        jl.linear_a_p.bias .= fl.linear_a_p.bias
        jl.linear_a_g.weight .= fl.linear_a_g.weight
        jl.linear_a_g.bias .= fl.linear_a_g.bias
        jl.linear_b_p.weight .= fl.linear_b_p.weight
        jl.linear_b_p.bias .= fl.linear_b_p.bias
        jl.linear_b_g.weight .= fl.linear_b_g.weight
        jl.linear_b_g.bias .= fl.linear_b_g.bias
        jl.linear_g.weight .= fl.linear_g.weight
        jl.linear_g.bias .= fl.linear_g.bias
        jl.linear_z.weight .= fl.linear_z.weight
        jl.linear_z.bias .= fl.linear_z.bias
        jl.layer_norm_in.w .= fl.layer_norm_in.w
        jl.layer_norm_in.b .= fl.layer_norm_in.b
        jl.layer_norm_out.w .= fl.layer_norm_out.w
        jl.layer_norm_out.b .= fl.layer_norm_out.b
    end

    for (jl, fl) in (
        (layer_jl.tri_att_start, layer_fl.tri_att_start),
        (layer_jl.tri_att_end, layer_fl.tri_att_end),
    )
        jl.linear.weight .= fl.linear.weight
        jl.layer_norm.w .= fl.layer_norm.w
        jl.layer_norm.b .= fl.layer_norm.b
        jl.mha.linear_q.weight .= fl.mha.linear_q.weight
        jl.mha.linear_k.weight .= fl.mha.linear_k.weight
        jl.mha.linear_v.weight .= fl.mha.linear_v.weight
        jl.mha.linear_o.weight .= fl.mha.linear_o.weight
        jl.mha.linear_o.bias .= fl.mha.linear_o.bias
        if fl.mha.linear_g !== nothing
            jl.mha.linear_g.weight .= fl.mha.linear_g.weight
            jl.mha.linear_g.bias .= fl.mha.linear_g.bias
        end
    end

    # MLP weights
    layer_jl.mlp_seq.fc1.weight .= layer_fl.mlp_seq.fc1.weight
    layer_jl.mlp_seq.fc1.bias .= layer_fl.mlp_seq.fc1.bias
    layer_jl.mlp_seq.fc2.weight .= layer_fl.mlp_seq.fc2.weight
    layer_jl.mlp_seq.fc2.bias .= layer_fl.mlp_seq.fc2.bias
    layer_jl.mlp_seq.norm.w .= layer_fl.mlp_seq.norm.w
    layer_jl.mlp_seq.norm.b .= layer_fl.mlp_seq.norm.b

    layer_jl.mlp_pair.fc1.weight .= layer_fl.mlp_pair.fc1.weight
    layer_jl.mlp_pair.fc1.bias .= layer_fl.mlp_pair.fc1.bias
    layer_jl.mlp_pair.fc2.weight .= layer_fl.mlp_pair.fc2.weight
    layer_jl.mlp_pair.fc2.bias .= layer_fl.mlp_pair.fc2.bias
    layer_jl.mlp_pair.norm.w .= layer_fl.mlp_pair.norm.w
    layer_jl.mlp_pair.norm.b .= layer_fl.mlp_pair.norm.b

    if get(ENV, "CHECK", "0") == "1"
        x_fl = permutedims(x, (3, 2, 1))
        z_fl = permutedims(z, (4, 2, 3, 1))
        mask_fl = permutedims(mask, (2, 1))
        y_fl_s, y_fl_z = layer_fl(x_fl, z_fl; mask=mask_fl)
        y_jl_s, y_jl_z = layer_jl(x, z; mask=mask)
        y_fl_s_jl = permutedims(y_fl_s, (3, 2, 1))
        y_fl_z_jl = permutedims(y_fl_z, (4, 2, 3, 1))
        max_diff_s = maximum(abs.(y_fl_s_jl .- y_jl_s))
        max_diff_z = maximum(abs.(y_fl_z_jl .- y_jl_z))
        println("Parity check max diff s: ", max_diff_s)
        println("Parity check max diff z: ", max_diff_z)
    end

    layer_jl(x, z; mask=mask)
    GC.gc()

    trial = @benchmark $layer_jl($x, $z; mask=$mask)
    _report("TriangularSelfAttentionBlockJL (feature-first)", trial)
    return trial
end

function _bench_structure_module(; B=1, L=128)
    cfg = ESMFold.StructureModuleConfig()
    s = rand(Float32, B, L, cfg.c_s)
    z = rand(Float32, B, L, L, cfg.c_z)
    aatype = rand(0:20, B, L)
    mask = ones(Float32, B, L)

    sm = ESMFold.StructureModule(cfg=cfg)

    sm(Dict(:single => s, :pair => z), aatype, mask)
    GC.gc()

    trial = @benchmark $sm(Dict(:single => $s, :pair => $z), $aatype, $mask)
    _report("StructureModule (feature-last)", trial)
    return trial
end

function _bench_structure_module_jl(; B=1, L=128)
    cfg = ESMFold.StructureModuleConfig()
    s = rand(Float32, cfg.c_s, L, B)
    z = rand(Float32, cfg.c_z, L, L, B)
    aatype = rand(0:20, L, B)
    mask = ones(Float32, L, B)

    sm = ESMFold.StructureModuleJL(cfg=cfg)

    sm(Dict(:single => s, :pair => z), aatype, mask)
    GC.gc()

    trial = @benchmark $sm(Dict(:single => $s, :pair => $z), $aatype, $mask)
    _report("StructureModuleJL (feature-first)", trial)
    return trial
end

function _bench_structure_module_jl_core(; B=1, L=128)
    cfg = ESMFold.StructureModuleConfig()
    s = rand(Float32, cfg.c_s, L, B)
    z = rand(Float32, cfg.c_z, L, L, B)
    aatype = rand(0:20, L, B)
    mask = ones(Float32, L, B)

    sm = ESMFold.StructureModuleJLCore(cfg=cfg)

    sm(Dict(:single => s, :pair => z), aatype, mask)
    GC.gc()

    trial = @benchmark $sm(Dict(:single => $s, :pair => $z), $aatype, $mask)
    _report("StructureModuleJLCore (feature-first)", trial)
    return trial
end

function _bench_folding_trunk(; B=1, L=64, num_blocks=4)
    cfg0 = ESMFold.FoldingTrunkConfig()
    cfg = ESMFold.FoldingTrunkConfig(
        num_blocks,
        cfg0.sequence_state_dim,
        cfg0.pairwise_state_dim,
        cfg0.sequence_head_width,
        cfg0.pairwise_head_width,
        cfg0.position_bins,
        cfg0.dropout,
        cfg0.layer_drop,
        cfg0.cpu_grad_checkpoint,
        cfg0.max_recycles,
        cfg0.chunk_size,
        cfg0.structure_module,
    )
    s = rand(Float32, B, L, cfg.sequence_state_dim)
    z = rand(Float32, B, L, L, cfg.pairwise_state_dim)
    aatype = rand(0:20, B, L)
    residx = reshape(collect(0:(L - 1)), 1, L)
    residx = repeat(residx, B, 1)
    mask = ones(Float32, B, L)

    trunk = ESMFold.FoldingTrunk(cfg=cfg)

    trunk(s, z, aatype, residx, mask; no_recycles=0)
    GC.gc()

    trial = @benchmark $trunk($s, $z, $aatype, $residx, $mask; no_recycles=0)
    _report("FoldingTrunk (feature-last, no_recycles=0, num_blocks=$(num_blocks))", trial)
    return trial
end

function _bench_folding_trunk_jl(; B=1, L=64, num_blocks=4)
    cfg0 = ESMFold.FoldingTrunkConfig()
    cfg = ESMFold.FoldingTrunkConfig(
        num_blocks,
        cfg0.sequence_state_dim,
        cfg0.pairwise_state_dim,
        cfg0.sequence_head_width,
        cfg0.pairwise_head_width,
        cfg0.position_bins,
        cfg0.dropout,
        cfg0.layer_drop,
        cfg0.cpu_grad_checkpoint,
        cfg0.max_recycles,
        cfg0.chunk_size,
        cfg0.structure_module,
    )
    s = rand(Float32, cfg.sequence_state_dim, L, B)
    z = rand(Float32, cfg.pairwise_state_dim, L, L, B)
    aatype = rand(0:20, L, B)
    residx = reshape(collect(0:(L - 1)), L, 1)
    residx = repeat(residx, 1, B)
    mask = ones(Float32, L, B)

    trunk = ESMFold.FoldingTrunkJL(cfg=cfg)

    trunk(s, z, aatype, residx, mask; no_recycles=0)
    GC.gc()

    trial = @benchmark $trunk($s, $z, $aatype, $residx, $mask; no_recycles=0)
    _report("FoldingTrunkJL (feature-first, no_recycles=0, num_blocks=$(num_blocks))", trial)
    return trial
end

function _bench_folding_trunk_jl_core(; B=1, L=64, num_blocks=4)
    cfg0 = ESMFold.FoldingTrunkConfig()
    cfg = ESMFold.FoldingTrunkConfig(
        num_blocks,
        cfg0.sequence_state_dim,
        cfg0.pairwise_state_dim,
        cfg0.sequence_head_width,
        cfg0.pairwise_head_width,
        cfg0.position_bins,
        cfg0.dropout,
        cfg0.layer_drop,
        cfg0.cpu_grad_checkpoint,
        cfg0.max_recycles,
        cfg0.chunk_size,
        cfg0.structure_module,
    )
    s = rand(Float32, cfg.sequence_state_dim, L, B)
    z = rand(Float32, cfg.pairwise_state_dim, L, L, B)
    aatype = rand(0:20, L, B)
    residx = reshape(collect(0:(L - 1)), L, 1)
    residx = repeat(residx, 1, B)
    mask = ones(Float32, L, B)

    trunk = ESMFold.FoldingTrunkJLCore(cfg=cfg)

    trunk(s, z, aatype, residx, mask; no_recycles=0)
    GC.gc()

    trial = @benchmark $trunk($s, $z, $aatype, $residx, $mask; no_recycles=0)
    _report("FoldingTrunkJLCore (feature-first, no_recycles=0, num_blocks=$(num_blocks))", trial)
    return trial
end
const CASES = Dict(
    "esmfold_attention" => _bench_esmfold_attention,
    "esmfold_attention_jl" => _bench_esmfold_attention_jl,
    "sequence_to_pair" => _bench_sequence_to_pair,
    "sequence_to_pair_jl" => _bench_sequence_to_pair_jl,
    "pair_to_sequence" => _bench_pair_to_sequence,
    "pair_to_sequence_jl" => _bench_pair_to_sequence_jl,
    "triangle_attention_start" => _bench_triangle_attention_start,
    "triangle_attention_start_jl" => _bench_triangle_attention_start_jl,
    "triangle_attention_end" => _bench_triangle_attention_end,
    "triangle_attention_end_jl" => _bench_triangle_attention_end_jl,
    "triangle_mul_out" => _bench_triangle_mul_out,
    "triangle_mul_out_jl" => _bench_triangle_mul_out_jl,
    "triangle_mul_in" => _bench_triangle_mul_in,
    "triangle_mul_in_jl" => _bench_triangle_mul_in_jl,
    "triangular_block" => _bench_triangular_block,
    "triangular_block_jl" => _bench_triangular_block_jl,
    "structure_module" => _bench_structure_module,
    "structure_module_jl" => _bench_structure_module_jl,
    "structure_module_jl_core" => _bench_structure_module_jl_core,
    "folding_trunk" => _bench_folding_trunk,
    "folding_trunk_jl" => _bench_folding_trunk_jl,
    "folding_trunk_jl_core" => _bench_folding_trunk_jl_core,
)

function main()
    case = get(ENV, "CASE", "esmfold_attention")
    if !haskey(CASES, case)
        println("Unknown CASE=", case)
        println("Available cases: ", join(sort(collect(keys(CASES))), ", "))
        return
    end
    CASES[case]()
end

main()
