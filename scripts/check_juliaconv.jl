using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using BenchmarkTools
using Random
using Statistics
using Zygote
using ESMFold

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.05
BenchmarkTools.DEFAULT_PARAMETERS.samples = 3
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1

Random.seed!(42)
ESMFold.set_training!(false)

const OUT_PATH = joinpath(@__DIR__, "..", "docs", "juliaconv_checks.md")
const SKIP_ZYGOTE = get(ENV, "SKIP_ZYGOTE", "0") == "1"

function _copy_params!(dst, src)
    if dst === nothing || src === nothing
        return
    end
    if dst isa AbstractDict || src isa AbstractDict
        return
    end
    if dst isa AbstractVector && src isa AbstractVector
        n = min(length(dst), length(src))
        if eltype(dst) <: AbstractString
            return
        end
        if eltype(dst) <: Number || eltype(dst) <: AbstractArray
            size(dst) == size(src) || return
            dst .= src
        else
            for i in 1:n
                _copy_params!(dst[i], src[i])
            end
        end
        return
    end
    if dst isa AbstractArray && src isa AbstractArray
        size(dst) == size(src) || return
        dst .= src
        return
    end
    if dst isa Number || dst isa Symbol || dst isa Bool
        return
    end
    dst_fields = fieldnames(typeof(dst))
    src_fields = fieldnames(typeof(src))
    if :inner in dst_fields && !(:inner in src_fields)
        inner = getfield(dst, :inner)
        if src isa typeof(inner)
            _copy_params!(inner, src)
        end
    end
    for name in dst_fields
        name in src_fields || continue
        _copy_params!(getfield(dst, name), getfield(src, name))
    end
    return
end

function _scalarize(x)
    if x isa Number
        return x
    elseif x isa AbstractArray
        return sum(x)
    elseif x isa Tuple
        s = zero(Float32)
        for v in x
            s += _scalarize(v)
        end
        return s
    elseif x isa Dict
        s = zero(Float32)
        for v in values(x)
            if v isa AbstractArray || v isa Number || v isa Tuple
                s += _scalarize(v)
            end
        end
        return s
    else
        return zero(Float32)
    end
end

function _maxdiff(a, b)
    return maximum(abs.(a .- b))
end

function _zygote_check_input(f, x)
    SKIP_ZYGOTE && return true, "skipped"
    if eltype(x) <: Integer
        return true, "input integer; skipped"
    end
    try
        g = Base.invokelatest(Zygote.gradient, x -> _scalarize(f(x)), x)[1]
        ok = g !== nothing && all(isfinite, g)
        return ok, ""
    catch err
        return false, sprint(showerror, err)
    end
end

function _zygote_check_params(f, params)
    SKIP_ZYGOTE && return true, "skipped"
    try
        gs = Base.invokelatest(Zygote.gradient, () -> _scalarize(f()), params)
        ok = true
        for p in params
            g = gs[p]
            g === nothing && (ok = false)
        end
        return ok, ""
    catch err
        return false, sprint(showerror, err)
    end
end

function _bench_pair(name, f_base, f_jl)
    f_base()
    f_jl()
    GC.gc()
    trial_base = @benchmark $f_base()
    trial_jl = @benchmark $f_jl()
    base_med = median(trial_base).time
    jl_med = median(trial_jl).time
    ratio = jl_med / base_med
    return base_med, jl_med, ratio
end

function _gpu_sniff(f, args...; window=80)
    loc = Base.functionloc(f, Tuple{map(typeof, args)...})
    loc === nothing && return ["source unavailable"]
    file, line = loc
    !isfile(file) && return ["source unavailable"]
    lines = readlines(file)
    start_line = max(1, line)
    stop_line = min(length(lines), line + window - 1)
    snippet = lines[start_line:stop_line]
    notes = String[]
    for (i, l) in enumerate(snippet)
        if occursin(r"^\\s*for\\s", l) || occursin(r"^\\s*while\\s", l)
            push!(notes, "loop at +$(i - 1)")
        elseif occursin("setindex!", l)
            push!(notes, "setindex! at +$(i - 1)")
        elseif occursin("@inbounds", l)
            push!(notes, "@inbounds at +$(i - 1)")
        end
    end
    return isempty(notes) ? ["no obvious scalar indexing"] : notes
end

results = Vector{Dict{String,Any}}()

function _push_result!(name; parity=nothing, z_in=nothing, z_param=nothing, z_err="", gpu=String[], base_time=nothing, jl_time=nothing, ratio=nothing)
    push!(results, Dict(
        "name" => name,
        "parity" => parity,
        "zygote_input" => z_in,
        "zygote_params" => z_param,
        "zygote_err" => z_err,
        "gpu" => gpu,
        "base_time" => base_time,
        "jl_time" => jl_time,
        "ratio" => ratio,
    ))
end

function _run_linear_first()
    C = 16; L = 8; B = 2; Cout = 12
    x_jl = rand(Float32, C, L, B)
    x_fl = permutedims(x_jl, (3, 2, 1))
    base = ESMFold.LinearLast(C, Cout)
    jl = ESMFold.LinearFirst(C, Cout)
    _copy_params!(jl, base)
    y_fl = base(x_fl)
    y_jl = jl(x_jl)
    diff = _maxdiff(permutedims(y_fl, (3, 2, 1)), y_jl)
    z_in, z_err = _zygote_check_input(x -> jl(x), x_jl)
    z_param, z_err2 = _zygote_check_params(() -> jl(x_jl), Zygote.Params([jl.weight, jl.bias]))
    z_err = z_err == "" ? z_err2 : z_err
    gpu = _gpu_sniff(jl, x_jl)
    base_t, jl_t, ratio = _bench_pair("LinearFirst", () -> base(x_fl), () -> jl(x_jl))
    _push_result!("LinearFirst", parity=diff, z_in=z_in, z_param=z_param, z_err=z_err, gpu=gpu, base_time=base_t, jl_time=jl_t, ratio=ratio)
end

function _run_layernorm_first()
    C = 16; L = 8; B = 2
    x_jl = rand(Float32, C, L, B)
    x_fl = permutedims(x_jl, (3, 2, 1))
    base = ESMFold.LayerNormLast(C)
    jl = ESMFold.LayerNormFirst(C)
    _copy_params!(jl, base)
    y_fl = base(x_fl)
    y_jl = jl(x_jl)
    diff = _maxdiff(permutedims(y_fl, (3, 2, 1)), y_jl)
    z_in, z_err = _zygote_check_input(x -> jl(x), x_jl)
    z_param, z_err2 = _zygote_check_params(() -> jl(x_jl), Zygote.Params([jl.w, jl.b]))
    z_err = z_err == "" ? z_err2 : z_err
    gpu = _gpu_sniff(jl, x_jl)
    base_t, jl_t, ratio = _bench_pair("LayerNormFirst", () -> base(x_fl), () -> jl(x_jl))
    _push_result!("LayerNormFirst", parity=diff, z_in=z_in, z_param=z_param, z_err=z_err, gpu=gpu, base_time=base_t, jl_time=jl_t, ratio=ratio)
end

function _run_esmfold_attention()
    C = 32; L = 16; B = 2; H = 4; D = 8
    x_jl = rand(Float32, C, L, B)
    mask_jl = ones(Float32, L, B)
    x_fl = permutedims(x_jl, (3, 2, 1))
    mask_fl = permutedims(mask_jl, (2, 1))
    base = ESMFold.ESMFoldAttention(C, H, D; gated=true)
    jl = ESMFold.ESMFoldAttentionJL(C, H, D; gated=true)
    _copy_params!(jl, base)
    y_fl, _ = base(x_fl; mask=mask_fl, bias=nothing)
    y_jl, _ = jl(x_jl; mask=mask_jl, bias=nothing)
    diff = _maxdiff(permutedims(y_fl, (3, 2, 1)), y_jl)
    z_in, z_err = _zygote_check_input(x -> first(jl(x; mask=mask_jl, bias=nothing)), x_jl)
    z_param, z_err2 = _zygote_check_params(() -> first(jl(x_jl; mask=mask_jl, bias=nothing)), Zygote.Params([jl.proj.weight, jl.o_proj.weight, jl.o_proj.bias, jl.g_proj.weight, jl.g_proj.bias]))
    z_err = z_err == "" ? z_err2 : z_err
    gpu = _gpu_sniff(jl, x_jl)
    base_t, jl_t, ratio = _bench_pair("ESMFoldAttentionJL", () -> base(x_fl; mask=mask_fl, bias=nothing), () -> jl(x_jl; mask=mask_jl, bias=nothing))
    _push_result!("ESMFoldAttentionJL", parity=diff, z_in=z_in, z_param=z_param, z_err=z_err, gpu=gpu, base_time=base_t, jl_time=jl_t, ratio=ratio)
end

function _run_sequence_to_pair()
    C_s = 32; C_z = 16; L = 16; B = 2
    x_jl = rand(Float32, C_s, L, B)
    x_fl = permutedims(x_jl, (3, 2, 1))
    base = ESMFold.SequenceToPair(C_s, C_z ÷ 2, C_z)
    jl = ESMFold.SequenceToPairJL(C_s, C_z ÷ 2, C_z)
    _copy_params!(jl, base)
    y_fl = base(x_fl)
    y_jl = jl(x_jl)
    diff = _maxdiff(permutedims(y_fl, (4, 2, 3, 1)), y_jl)
    z_in, z_err = _zygote_check_input(x -> jl(x), x_jl)
    z_param, z_err2 = _zygote_check_params(() -> jl(x_jl), Zygote.Params([jl.layernorm.w, jl.layernorm.b, jl.proj.weight, jl.proj.bias, jl.o_proj.weight, jl.o_proj.bias]))
    z_err = z_err == "" ? z_err2 : z_err
    gpu = _gpu_sniff(jl, x_jl)
    base_t, jl_t, ratio = _bench_pair("SequenceToPairJL", () -> base(x_fl), () -> jl(x_jl))
    _push_result!("SequenceToPairJL", parity=diff, z_in=z_in, z_param=z_param, z_err=z_err, gpu=gpu, base_time=base_t, jl_time=jl_t, ratio=ratio)
end

function _run_pair_to_sequence()
    C_z = 16; L = 16; B = 2; H = 4
    z_jl = rand(Float32, C_z, L, L, B)
    z_fl = permutedims(z_jl, (4, 2, 3, 1))
    base = ESMFold.PairToSequence(C_z, H)
    jl = ESMFold.PairToSequenceJL(C_z, H)
    _copy_params!(jl, base)
    y_fl = base(z_fl)
    y_jl = jl(z_jl)
    diff = _maxdiff(permutedims(y_fl, (4, 2, 3, 1)), y_jl)
    z_in, z_err = _zygote_check_input(x -> jl(x), z_jl)
    z_param, z_err2 = _zygote_check_params(() -> jl(z_jl), Zygote.Params([jl.layernorm.w, jl.layernorm.b, jl.linear.weight]))
    z_err = z_err == "" ? z_err2 : z_err
    gpu = _gpu_sniff(jl, z_jl)
    base_t, jl_t, ratio = _bench_pair("PairToSequenceJL", () -> base(z_fl), () -> jl(z_jl))
    _push_result!("PairToSequenceJL", parity=diff, z_in=z_in, z_param=z_param, z_err=z_err, gpu=gpu, base_time=base_t, jl_time=jl_t, ratio=ratio)
end

function _run_residue_mlp()
    C = 32; L = 16; B = 2
    x_jl = rand(Float32, C, L, B)
    x_fl = permutedims(x_jl, (3, 2, 1))
    base = ESMFold.ResidueMLP(C, 4 * C; dropout=0f0)
    jl = ESMFold.ResidueMLPJL(C, 4 * C; dropout=0f0)
    _copy_params!(jl, base)
    y_fl = base(x_fl)
    y_jl = jl(x_jl)
    diff = _maxdiff(permutedims(y_fl, (3, 2, 1)), y_jl)
    z_in, z_err = _zygote_check_input(x -> jl(x), x_jl)
    z_param, z_err2 = _zygote_check_params(() -> jl(x_jl), Zygote.Params([jl.norm.w, jl.norm.b, jl.fc1.weight, jl.fc1.bias, jl.fc2.weight, jl.fc2.bias]))
    z_err = z_err == "" ? z_err2 : z_err
    gpu = _gpu_sniff(jl, x_jl)
    base_t, jl_t, ratio = _bench_pair("ResidueMLPJL", () -> base(x_fl), () -> jl(x_jl))
    _push_result!("ResidueMLPJL", parity=diff, z_in=z_in, z_param=z_param, z_err=z_err, gpu=gpu, base_time=base_t, jl_time=jl_t, ratio=ratio)
end

function _run_ofmha()
    C = 16; L = 16; B = 2; H = 4; C_hidden = 4
    q_jl = rand(Float32, C, L, B)
    kv_jl = rand(Float32, C, L, B)
    q_fl = permutedims(q_jl, (3, 2, 1))
    kv_fl = permutedims(kv_jl, (3, 2, 1))
    base = ESMFold.OFMultiheadAttention(C, C, C, C_hidden, H; gating=true, inf=1e9)
    jl = ESMFold.OFMultiheadAttentionJL(C, C, C, C_hidden, H; gating=true, inf=1e9)
    _copy_params!(jl, base)
    y_fl = base(q_fl, kv_fl; biases=Any[])
    y_jl = jl(q_jl, kv_jl; biases=Any[])
    diff = _maxdiff(permutedims(y_fl, (3, 2, 1)), y_jl)
    z_in, z_err = _zygote_check_input(x -> jl(x, kv_jl; biases=Any[]), q_jl)
    z_param, z_err2 = _zygote_check_params(() -> jl(q_jl, kv_jl; biases=Any[]), Zygote.Params([jl.linear_q.weight, jl.linear_k.weight, jl.linear_v.weight, jl.linear_o.weight, jl.linear_o.bias, jl.linear_g.weight, jl.linear_g.bias]))
    z_err = z_err == "" ? z_err2 : z_err
    gpu = _gpu_sniff(jl, q_jl, kv_jl)
    base_t, jl_t, ratio = _bench_pair("OFMultiheadAttentionJL", () -> base(q_fl, kv_fl; biases=Any[]), () -> jl(q_jl, kv_jl; biases=Any[]))
    _push_result!("OFMultiheadAttentionJL", parity=diff, z_in=z_in, z_param=z_param, z_err=z_err, gpu=gpu, base_time=base_t, jl_time=jl_t, ratio=ratio)
end

function _run_triangle_attention(starting::Bool)
    C_z = 16; L = 16; B = 2; head_width = 4; H = C_z ÷ head_width
    x_jl = rand(Float32, C_z, L, L, B)
    mask_jl = ones(Float32, L, L, B)
    x_fl = permutedims(x_jl, (4, 2, 3, 1))
    mask_fl = permutedims(mask_jl, (3, 1, 2))
    base = ESMFold.TriangleAttention(C_z, head_width, H; starting=starting, inf=1e9)
    jl = ESMFold.TriangleAttentionJL(C_z, head_width, H; starting=starting, inf=1e9)
    _copy_params!(jl, base)
    y_fl = base(x_fl; mask=mask_fl)
    y_jl = jl(x_jl; mask=mask_jl)
    diff = _maxdiff(permutedims(y_fl, (4, 2, 3, 1)), y_jl)
    z_in, z_err = _zygote_check_input(x -> jl(x; mask=mask_jl), x_jl)
    z_param, z_err2 = _zygote_check_params(() -> jl(x_jl; mask=mask_jl), Zygote.Params([jl.layer_norm.w, jl.layer_norm.b, jl.linear.weight, jl.mha.linear_q.weight, jl.mha.linear_k.weight, jl.mha.linear_v.weight, jl.mha.linear_o.weight, jl.mha.linear_o.bias, jl.mha.linear_g.weight, jl.mha.linear_g.bias]))
    z_err = z_err == "" ? z_err2 : z_err
    gpu = _gpu_sniff(jl, x_jl)
    label = starting ? "TriangleAttentionJL(start)" : "TriangleAttentionJL(end)"
    base_t, jl_t, ratio = _bench_pair(label, () -> base(x_fl; mask=mask_fl), () -> jl(x_jl; mask=mask_jl))
    _push_result!(label, parity=diff, z_in=z_in, z_param=z_param, z_err=z_err, gpu=gpu, base_time=base_t, jl_time=jl_t, ratio=ratio)
end

function _run_triangle_mul(outgoing::Bool)
    C_z = 16; L = 16; B = 2; C_hidden = 8
    z_jl = rand(Float32, C_z, L, L, B)
    mask_jl = ones(Float32, L, L, B)
    z_fl = permutedims(z_jl, (4, 2, 3, 1))
    mask_fl = permutedims(mask_jl, (3, 1, 2))
    base = ESMFold.TriangleMultiplicativeUpdate(C_z, C_hidden; outgoing=outgoing)
    jl = ESMFold.TriangleMultiplicativeUpdateJL(C_z, C_hidden; outgoing=outgoing)
    _copy_params!(jl, base)
    y_fl = base(z_fl; mask=mask_fl)
    y_jl = jl(z_jl; mask=mask_jl)
    diff = _maxdiff(permutedims(y_fl, (4, 2, 3, 1)), y_jl)
    z_in, z_err = _zygote_check_input(x -> jl(x; mask=mask_jl), z_jl)
    z_param, z_err2 = _zygote_check_params(() -> jl(z_jl; mask=mask_jl), Zygote.Params([jl.linear_a_p.weight, jl.linear_a_p.bias, jl.linear_a_g.weight, jl.linear_a_g.bias, jl.linear_b_p.weight, jl.linear_b_p.bias, jl.linear_b_g.weight, jl.linear_b_g.bias, jl.linear_g.weight, jl.linear_g.bias, jl.linear_z.weight, jl.linear_z.bias, jl.layer_norm_in.w, jl.layer_norm_in.b, jl.layer_norm_out.w, jl.layer_norm_out.b]))
    z_err = z_err == "" ? z_err2 : z_err
    gpu = _gpu_sniff(jl, z_jl)
    label = outgoing ? "TriangleMultiplicativeUpdateJL(out)" : "TriangleMultiplicativeUpdateJL(in)"
    base_t, jl_t, ratio = _bench_pair(label, () -> base(z_fl; mask=mask_fl), () -> jl(z_jl; mask=mask_jl))
    _push_result!(label, parity=diff, z_in=z_in, z_param=z_param, z_err=z_err, gpu=gpu, base_time=base_t, jl_time=jl_t, ratio=ratio)
end

function _run_triangular_block()
    C_s = 32; C_z = 16; L = 12; B = 2; seq_head = 8; pair_head = 4
    s_jl = rand(Float32, C_s, L, B)
    z_jl = rand(Float32, C_z, L, L, B)
    mask_jl = ones(Float32, L, B)
    residx_jl = reshape(collect(0:(L - 1)), L, 1)
    residx_jl = repeat(residx_jl, 1, B)
    residx_fl = permutedims(residx_jl, (2, 1))
    s_fl = permutedims(s_jl, (3, 2, 1))
    z_fl = permutedims(z_jl, (4, 2, 3, 1))
    mask_fl = permutedims(mask_jl, (2, 1))
    base = ESMFold.TriangularSelfAttentionBlock(C_s, C_z, seq_head, pair_head; dropout=0f0)
    jl = ESMFold.TriangularSelfAttentionBlockJL(C_s, C_z, seq_head, pair_head; dropout=0f0)
    _copy_params!(jl, base)
    s2_fl, z2_fl = base(s_fl, z_fl; mask=mask_fl, residue_index=residx_fl, chunk_size=nothing)
    s2_jl, z2_jl = jl(s_jl, z_jl; mask=mask_jl, residue_index=residx_jl, chunk_size=nothing)
    diff = max(_maxdiff(permutedims(s2_fl, (3, 2, 1)), s2_jl), _maxdiff(permutedims(z2_fl, (4, 2, 3, 1)), z2_jl))
    z_in, z_err = _zygote_check_input(x -> first(jl(x, z_jl; mask=mask_jl, residue_index=residx_jl, chunk_size=nothing)), s_jl)
    z_param, z_err2 = _zygote_check_params(() -> _scalarize(jl(s_jl, z_jl; mask=mask_jl, residue_index=residx_jl, chunk_size=nothing)), Zygote.Params([jl]))
    z_err = z_err == "" ? z_err2 : z_err
    gpu = _gpu_sniff(jl, s_jl, z_jl)
    base_t, jl_t, ratio = _bench_pair("TriangularSelfAttentionBlockJL", () -> base(s_fl, z_fl; mask=mask_fl, residue_index=residx_fl, chunk_size=nothing), () -> jl(s_jl, z_jl; mask=mask_jl, residue_index=residx_jl, chunk_size=nothing))
    _push_result!("TriangularSelfAttentionBlockJL", parity=diff, z_in=z_in, z_param=z_param, z_err=z_err, gpu=gpu, base_time=base_t, jl_time=jl_t, ratio=ratio)
end

function _run_relative_position()
    C_z = 16; bins = 16; L = 16; B = 2
    residx_jl = reshape(collect(0:(L - 1)), L, 1)
    residx_jl = repeat(residx_jl, 1, B)
    residx_fl = permutedims(residx_jl, (2, 1))
    base = ESMFold.RelativePosition(bins, C_z)
    jl = ESMFold.RelativePositionJL(bins, C_z)
    _copy_params!(jl, base)
    y_fl = base(residx_fl; mask=nothing)
    y_jl = jl(residx_jl; mask=nothing)
    diff = _maxdiff(permutedims(y_fl, (4, 2, 3, 1)), y_jl)
    z_in, z_err = _zygote_check_input(x -> jl(x; mask=nothing), residx_jl)
    z_param, z_err2 = _zygote_check_params(() -> jl(residx_jl; mask=nothing), Zygote.Params([jl.embedding.weight]))
    z_err = z_err == "" ? z_err2 : z_err
    gpu = _gpu_sniff(jl, residx_jl)
    base_t, jl_t, ratio = _bench_pair("RelativePositionJL", () -> base(residx_fl; mask=nothing), () -> jl(residx_jl; mask=nothing))
    _push_result!("RelativePositionJL", parity=diff, z_in=z_in, z_param=z_param, z_err=z_err, gpu=gpu, base_time=base_t, jl_time=jl_t, ratio=ratio)
end

function _run_point_projection()
    C = 16; L = 16; B = 2; H = 2; P = 2
    s_jl = rand(Float32, C, L, B)
    s_fl = permutedims(s_jl, (3, 2, 1))
    r_jl = ESMFold.rigid_identity_jl((L, B), s_jl)
    r_fl = ESMFold.rigid_identity((B, L), s_fl)
    base = ESMFold.PointProjection(C, P, H)
    jl = ESMFold.PointProjectionJL(C, P, H)
    _copy_params!(jl, base)
    y_fl = base(s_fl, r_fl)
    y_jl = jl(s_jl, r_jl)
    diff = _maxdiff(permutedims(y_fl, (5, 4, 3, 2, 1)), y_jl)
    z_in, z_err = _zygote_check_input(x -> jl(x, r_jl), s_jl)
    z_param, z_err2 = _zygote_check_params(() -> jl(s_jl, r_jl), Zygote.Params([jl.linear.weight, jl.linear.bias]))
    z_err = z_err == "" ? z_err2 : z_err
    gpu = _gpu_sniff(jl, s_jl, r_jl)
    base_t, jl_t, ratio = _bench_pair("PointProjectionJL", () -> base(s_fl, r_fl), () -> jl(s_jl, r_jl))
    _push_result!("PointProjectionJL", parity=diff, z_in=z_in, z_param=z_param, z_err=z_err, gpu=gpu, base_time=base_t, jl_time=jl_t, ratio=ratio)
end

function _run_invariant_point_attention()
    C_s = 16; C_z = 8; C_hidden = 4; L = 12; B = 2; H = 2; Pq = 2; Pv = 2
    s_jl = rand(Float32, C_s, L, B)
    z_jl = rand(Float32, C_z, L, L, B)
    mask_jl = ones(Float32, L, B)
    s_fl = permutedims(s_jl, (3, 2, 1))
    z_fl = permutedims(z_jl, (4, 2, 3, 1))
    mask_fl = permutedims(mask_jl, (2, 1))
    r_jl = ESMFold.rigid_identity_jl((L, B), s_jl)
    r_fl = ESMFold.rigid_identity((B, L), s_fl)
    base = ESMFold.InvariantPointAttention(C_s, C_z, C_hidden, H, Pq, Pv; inf=1e5, eps=1e-8)
    jl = ESMFold.InvariantPointAttentionJL(C_s, C_z, C_hidden, H, Pq, Pv; inf=1e5, eps=1e-8)
    _copy_params!(jl, base)
    y_fl = base(s_fl, z_fl, r_fl, mask_fl)
    y_jl = jl(s_jl, z_jl, r_jl, mask_jl)
    diff = _maxdiff(permutedims(y_fl, (3, 2, 1)), y_jl)
    z_in, z_err = _zygote_check_input(x -> jl(x, z_jl, r_jl, mask_jl), s_jl)
    z_param, z_err2 = _zygote_check_params(() -> jl(s_jl, z_jl, r_jl, mask_jl), Zygote.Params([jl.linear_q.weight, jl.linear_q_points.linear.weight, jl.linear_kv.weight, jl.linear_kv_points.linear.weight, jl.linear_b.weight, jl.linear_out.weight, jl.linear_out.bias]))
    z_err = z_err == "" ? z_err2 : z_err
    gpu = _gpu_sniff(jl, s_jl, z_jl, r_jl, mask_jl)
    base_t, jl_t, ratio = _bench_pair("InvariantPointAttentionJL", () -> base(s_fl, z_fl, r_fl, mask_fl), () -> jl(s_jl, z_jl, r_jl, mask_jl))
    _push_result!("InvariantPointAttentionJL", parity=diff, z_in=z_in, z_param=z_param, z_err=z_err, gpu=gpu, base_time=base_t, jl_time=jl_t, ratio=ratio)
end

function _run_backbone_update()
    C = 16; L = 16; B = 2
    s_jl = rand(Float32, C, L, B)
    s_fl = permutedims(s_jl, (3, 2, 1))
    base = ESMFold.BackboneUpdate(C)
    jl = ESMFold.BackboneUpdateJL(C)
    _copy_params!(jl, base)
    y_fl = base(s_fl)
    y_jl = jl(s_jl)
    diff = _maxdiff(permutedims(y_fl, (3, 2, 1)), y_jl)
    z_in, z_err = _zygote_check_input(x -> jl(x), s_jl)
    z_param, z_err2 = _zygote_check_params(() -> jl(s_jl), Zygote.Params([jl.linear.weight, jl.linear.bias]))
    z_err = z_err == "" ? z_err2 : z_err
    gpu = _gpu_sniff(jl, s_jl)
    base_t, jl_t, ratio = _bench_pair("BackboneUpdateJL", () -> base(s_fl), () -> jl(s_jl))
    _push_result!("BackboneUpdateJL", parity=diff, z_in=z_in, z_param=z_param, z_err=z_err, gpu=gpu, base_time=base_t, jl_time=jl_t, ratio=ratio)
end

function _run_transition()
    C = 16; L = 16; B = 2; layers = 2
    s_jl = rand(Float32, C, L, B)
    s_fl = permutedims(s_jl, (3, 2, 1))
    base = ESMFold.StructureModuleTransition(C, layers, 0f0)
    jl = ESMFold.StructureModuleTransitionJL(C, layers, 0f0)
    _copy_params!(jl, base)
    y_fl = base(s_fl)
    y_jl = jl(s_jl)
    diff = _maxdiff(permutedims(y_fl, (3, 2, 1)), y_jl)
    z_in, z_err = _zygote_check_input(x -> jl(x), s_jl)
    z_param, z_err2 = _zygote_check_params(() -> jl(s_jl), Zygote.Params([jl]))
    z_err = z_err == "" ? z_err2 : z_err
    gpu = _gpu_sniff(jl, s_jl)
    base_t, jl_t, ratio = _bench_pair("StructureModuleTransitionJL", () -> base(s_fl), () -> jl(s_jl))
    _push_result!("StructureModuleTransitionJL", parity=diff, z_in=z_in, z_param=z_param, z_err=z_err, gpu=gpu, base_time=base_t, jl_time=jl_t, ratio=ratio)
end

function _run_angle_resnet()
    C = 16; C_hidden = 8; L = 12; B = 2; blocks = 1; angles = 7
    s_jl = rand(Float32, C, L, B)
    s0_jl = rand(Float32, C, L, B)
    s_fl = permutedims(s_jl, (3, 2, 1))
    s0_fl = permutedims(s0_jl, (3, 2, 1))
    base = ESMFold.AngleResnet(C, C_hidden, blocks, angles, 1f-8)
    jl = ESMFold.AngleResnetJL(C, C_hidden, blocks, angles, 1f-8)
    _copy_params!(jl, base)
    un_fl, ang_fl = base(s_fl, s0_fl)
    un_jl, ang_jl = jl(s_jl, s0_jl)
    diff = max(_maxdiff(permutedims(un_fl, (4, 3, 2, 1)), un_jl), _maxdiff(permutedims(ang_fl, (4, 3, 2, 1)), ang_jl))
    z_in, z_err = _zygote_check_input(x -> first(jl(x, s0_jl)), s_jl)
    z_param, z_err2 = _zygote_check_params(() -> _scalarize(jl(s_jl, s0_jl)), Zygote.Params([jl]))
    z_err = z_err == "" ? z_err2 : z_err
    gpu = _gpu_sniff(jl, s_jl, s0_jl)
    base_t, jl_t, ratio = _bench_pair("AngleResnetJL", () -> base(s_fl, s0_fl), () -> jl(s_jl, s0_jl))
    _push_result!("AngleResnetJL", parity=diff, z_in=z_in, z_param=z_param, z_err=z_err, gpu=gpu, base_time=base_t, jl_time=jl_t, ratio=ratio)
end

function _run_structure_module()
    cfg0 = ESMFold.StructureModuleConfig()
    cfg = ESMFold.StructureModuleConfig(
        4,
        2,
        1,
        4,
        1,
        1,
        1,
        0f0,
        1,
        1,
        1,
        cfg0.no_angles,
        cfg0.trans_scale_factor,
        cfg0.epsilon,
        cfg0.inf,
    )
    C_s = cfg.c_s; C_z = cfg.c_z; L = 3; B = 1
    s_jl = rand(Float32, C_s, L, B)
    z_jl = rand(Float32, C_z, L, L, B)
    aatype_jl = rand(0:20, L, B)
    aatype_fl = permutedims(aatype_jl, (2, 1))
    mask_jl = ones(Float32, L, B)
    s_fl = permutedims(s_jl, (3, 2, 1))
    z_fl = permutedims(z_jl, (4, 2, 3, 1))
    mask_fl = permutedims(mask_jl, (2, 1))
    base = ESMFold.StructureModule(cfg=cfg)
    jl = ESMFold.StructureModuleJL(cfg=cfg)
    _copy_params!(jl.inner, base)
    out_fl = base(Dict(:single => s_fl, :pair => z_fl), aatype_fl, mask_fl)
    out_jl = jl(Dict(:single => s_jl, :pair => z_jl), aatype_jl, mask_jl)
    diff = max(_maxdiff(permutedims(out_fl[:single], (3, 2, 1)), out_jl[:single]), _maxdiff(permutedims(out_fl[:positions], (1, 5, 4, 3, 2)), out_jl[:positions]))
    z_in, z_err = _zygote_check_input(x -> jl(Dict(:single => x, :pair => z_jl), aatype_jl, mask_jl), s_jl)
    z_param, z_err2 = _zygote_check_params(() -> _scalarize(jl(Dict(:single => s_jl, :pair => z_jl), aatype_jl, mask_jl)), Zygote.Params([jl]))
    z_err = z_err == "" ? z_err2 : z_err
    gpu = _gpu_sniff(jl, Dict(:single => s_jl, :pair => z_jl), aatype_jl, mask_jl)
    base_t, jl_t, ratio = _bench_pair("StructureModuleJL", () -> base(Dict(:single => s_fl, :pair => z_fl), aatype_fl, mask_fl), () -> jl(Dict(:single => s_jl, :pair => z_jl), aatype_jl, mask_jl))
    _push_result!("StructureModuleJL", parity=diff, z_in=z_in, z_param=z_param, z_err=z_err, gpu=gpu, base_time=base_t, jl_time=jl_t, ratio=ratio)
end

function _run_structure_module_core()
    cfg0 = ESMFold.StructureModuleConfig()
    cfg = ESMFold.StructureModuleConfig(
        4,
        2,
        1,
        4,
        1,
        1,
        1,
        0f0,
        1,
        1,
        1,
        cfg0.no_angles,
        cfg0.trans_scale_factor,
        cfg0.epsilon,
        cfg0.inf,
    )
    C_s = cfg.c_s; C_z = cfg.c_z; L = 3; B = 1
    s_jl = rand(Float32, C_s, L, B)
    z_jl = rand(Float32, C_z, L, L, B)
    aatype_jl = rand(0:20, L, B)
    aatype_fl = permutedims(aatype_jl, (2, 1))
    mask_jl = ones(Float32, L, B)
    s_fl = permutedims(s_jl, (3, 2, 1))
    z_fl = permutedims(z_jl, (4, 2, 3, 1))
    mask_fl = permutedims(mask_jl, (2, 1))
    base = ESMFold.StructureModule(cfg=cfg)
    jl = ESMFold.StructureModuleJLCore(cfg=cfg)
    _copy_params!(jl, base)
    out_fl = base(Dict(:single => s_fl, :pair => z_fl), aatype_fl, mask_fl)
    out_jl = jl(Dict(:single => s_jl, :pair => z_jl), aatype_jl, mask_jl)
    diff = max(_maxdiff(permutedims(out_fl[:single], (3, 2, 1)), out_jl[:single]), _maxdiff(permutedims(out_fl[:positions], (1, 5, 4, 3, 2)), out_jl[:positions]))
    z_in, z_err = _zygote_check_input(x -> jl(Dict(:single => x, :pair => z_jl), aatype_jl, mask_jl), s_jl)
    z_param, z_err2 = _zygote_check_params(() -> _scalarize(jl(Dict(:single => s_jl, :pair => z_jl), aatype_jl, mask_jl)), Zygote.Params([jl]))
    z_err = z_err == "" ? z_err2 : z_err
    gpu = _gpu_sniff(jl, Dict(:single => s_jl, :pair => z_jl), aatype_jl, mask_jl)
    base_t, jl_t, ratio = _bench_pair("StructureModuleJLCore", () -> base(Dict(:single => s_fl, :pair => z_fl), aatype_fl, mask_fl), () -> jl(Dict(:single => s_jl, :pair => z_jl), aatype_jl, mask_jl))
    _push_result!("StructureModuleJLCore", parity=diff, z_in=z_in, z_param=z_param, z_err=z_err, gpu=gpu, base_time=base_t, jl_time=jl_t, ratio=ratio)
end

function _run_folding_trunk()
    cfg0 = ESMFold.FoldingTrunkConfig()
    cfg = ESMFold.FoldingTrunkConfig(
        1,
        8,
        2,
        2,
        2,
        cfg0.position_bins,
        0f0,
        cfg0.layer_drop,
        cfg0.cpu_grad_checkpoint,
        0,
        cfg0.chunk_size,
        ESMFold.StructureModuleConfig(
            4,
            2,
            1,
            4,
            1,
            1,
            1,
            0f0,
            1,
            1,
            1,
            cfg0.structure_module.no_angles,
            cfg0.structure_module.trans_scale_factor,
            cfg0.structure_module.epsilon,
            cfg0.structure_module.inf,
        ),
    )
    L = 3; B = 1
    s_jl = rand(Float32, cfg.sequence_state_dim, L, B)
    z_jl = rand(Float32, cfg.pairwise_state_dim, L, L, B)
    aatype_jl = rand(0:20, L, B)
    aatype_fl = permutedims(aatype_jl, (2, 1))
    residx_jl = reshape(collect(0:(L - 1)), L, 1)
    residx_jl = repeat(residx_jl, 1, B)
    residx_fl = permutedims(residx_jl, (2, 1))
    mask_jl = ones(Float32, L, B)
    s_fl = permutedims(s_jl, (3, 2, 1))
    z_fl = permutedims(z_jl, (4, 2, 3, 1))
    mask_fl = permutedims(mask_jl, (2, 1))
    base = ESMFold.FoldingTrunk(cfg=cfg)
    jl = ESMFold.FoldingTrunkJL(cfg=cfg)
    _copy_params!(jl, base)
    out_fl = base(s_fl, z_fl, aatype_fl, residx_fl, mask_fl; no_recycles=0)
    out_jl = jl(s_jl, z_jl, aatype_jl, residx_jl, mask_jl; no_recycles=0)
    diff = _maxdiff(permutedims(out_fl[:positions], (1, 5, 4, 3, 2)), out_jl[:positions])
    z_in, z_err = _zygote_check_input(x -> jl(x, z_jl, aatype_jl, residx_jl, mask_jl; no_recycles=0), s_jl)
    z_param, z_err2 = _zygote_check_params(() -> _scalarize(jl(s_jl, z_jl, aatype_jl, residx_jl, mask_jl; no_recycles=0)), Zygote.Params([jl]))
    z_err = z_err == "" ? z_err2 : z_err
    gpu = _gpu_sniff(jl, s_jl, z_jl, aatype_jl, residx_jl, mask_jl)
    base_t, jl_t, ratio = _bench_pair("FoldingTrunkJL", () -> base(s_fl, z_fl, aatype_fl, residx_fl, mask_fl; no_recycles=0), () -> jl(s_jl, z_jl, aatype_jl, residx_jl, mask_jl; no_recycles=0))
    _push_result!("FoldingTrunkJL", parity=diff, z_in=z_in, z_param=z_param, z_err=z_err, gpu=gpu, base_time=base_t, jl_time=jl_t, ratio=ratio)
end

function _run_folding_trunk_core()
    cfg0 = ESMFold.FoldingTrunkConfig()
    cfg = ESMFold.FoldingTrunkConfig(
        1,
        8,
        2,
        2,
        2,
        cfg0.position_bins,
        0f0,
        cfg0.layer_drop,
        cfg0.cpu_grad_checkpoint,
        0,
        cfg0.chunk_size,
        ESMFold.StructureModuleConfig(
            4,
            2,
            1,
            4,
            1,
            1,
            1,
            0f0,
            1,
            1,
            1,
            cfg0.structure_module.no_angles,
            cfg0.structure_module.trans_scale_factor,
            cfg0.structure_module.epsilon,
            cfg0.structure_module.inf,
        ),
    )
    L = 3; B = 1
    s_jl = rand(Float32, cfg.sequence_state_dim, L, B)
    z_jl = rand(Float32, cfg.pairwise_state_dim, L, L, B)
    aatype_jl = rand(0:20, L, B)
    aatype_fl = permutedims(aatype_jl, (2, 1))
    residx_jl = reshape(collect(0:(L - 1)), L, 1)
    residx_jl = repeat(residx_jl, 1, B)
    residx_fl = permutedims(residx_jl, (2, 1))
    mask_jl = ones(Float32, L, B)
    s_fl = permutedims(s_jl, (3, 2, 1))
    z_fl = permutedims(z_jl, (4, 2, 3, 1))
    mask_fl = permutedims(mask_jl, (2, 1))
    base = ESMFold.FoldingTrunk(cfg=cfg)
    jl = ESMFold.FoldingTrunkJLCore(cfg=cfg)
    _copy_params!(jl, base)
    out_fl = base(s_fl, z_fl, aatype_fl, residx_fl, mask_fl; no_recycles=0)
    out_jl = jl(s_jl, z_jl, aatype_jl, residx_jl, mask_jl; no_recycles=0)
    diff = _maxdiff(permutedims(out_fl[:positions], (1, 5, 4, 3, 2)), out_jl[:positions])
    z_in, z_err = _zygote_check_input(x -> jl(x, z_jl, aatype_jl, residx_jl, mask_jl; no_recycles=0), s_jl)
    z_param, z_err2 = _zygote_check_params(() -> _scalarize(jl(s_jl, z_jl, aatype_jl, residx_jl, mask_jl; no_recycles=0)), Zygote.Params([jl]))
    z_err = z_err == "" ? z_err2 : z_err
    gpu = _gpu_sniff(jl, s_jl, z_jl, aatype_jl, residx_jl, mask_jl)
    base_t, jl_t, ratio = _bench_pair("FoldingTrunkJLCore", () -> base(s_fl, z_fl, aatype_fl, residx_fl, mask_fl; no_recycles=0), () -> jl(s_jl, z_jl, aatype_jl, residx_jl, mask_jl; no_recycles=0))
    _push_result!("FoldingTrunkJLCore", parity=diff, z_in=z_in, z_param=z_param, z_err=z_err, gpu=gpu, base_time=base_t, jl_time=jl_t, ratio=ratio)
end

function _run_esmfold_model_jl()
    cfg0 = ESMFold.FoldingTrunkConfig()
    sm_cfg = ESMFold.StructureModuleConfig(
        4,
        2,
        2,
        4,
        1,
        1,
        1,
        0f0,
        1,
        1,
        1,
        cfg0.structure_module.no_angles,
        cfg0.structure_module.trans_scale_factor,
        cfg0.structure_module.epsilon,
        cfg0.structure_module.inf,
    )
    trunk_cfg = ESMFold.FoldingTrunkConfig(
        1,
        8,
        4,
        2,
        2,
        cfg0.position_bins,
        0f0,
        cfg0.layer_drop,
        cfg0.cpu_grad_checkpoint,
        0,
        cfg0.chunk_size,
        sm_cfg,
    )
    cfg = ESMFold.ESMFoldConfig(; trunk=trunk_cfg, lddt_head_hid_dim=8, use_esm_attn_map=false)

    alphabet = ESMFold.Alphabet_from_architecture("ESM-1b")
    esm_base = ESMFold.ESM2(1, 16, 2; alphabet=alphabet, token_dropout=false)
    esm_jl = ESMFold.ESM2(1, 16, 2; alphabet=alphabet, token_dropout=false)
    base = ESMFold.ESMFoldModel(esm_base; cfg=cfg)
    jl = ESMFold.ESMFoldModelJL(esm_jl; cfg=cfg)
    _copy_params!(jl, base)

    L = 4; B = 2
    aa_jl = rand(0:20, L, B)
    aa_fl = permutedims(aa_jl, (2, 1))
    mask_jl = ones(Int, L, B)
    mask_fl = permutedims(mask_jl, (2, 1))
    residx_jl = reshape(collect(0:(L - 1)), L, 1)
    residx_jl = repeat(residx_jl, 1, B)
    residx_fl = permutedims(residx_jl, (2, 1))

    out_fl = base(aa_fl; mask=mask_fl, residx=residx_fl, num_recycles=0)
    out_jl = jl(aa_jl; mask=mask_jl, residx=residx_jl, num_recycles=0)

    diff_s = _maxdiff(permutedims(out_fl[:s_s], (3, 2, 1)), out_jl[:s_s])
    diff_pos = _maxdiff(permutedims(out_fl[:positions], (1, 5, 4, 3, 2)), out_jl[:positions])
    diff_disto = _maxdiff(permutedims(out_fl[:distogram_logits], (4, 2, 3, 1)), out_jl[:distogram_logits])
    diff_lm = _maxdiff(permutedims(out_fl[:lm_logits], (3, 2, 1)), out_jl[:lm_logits])
    diff = max(diff_s, diff_pos, diff_disto, diff_lm)

    z_in, z_err = _zygote_check_input(x -> jl(x; mask=mask_jl, residx=residx_jl, num_recycles=0), aa_jl)
    z_param, z_err2 = _zygote_check_params(() -> _scalarize(jl(aa_jl; mask=mask_jl, residx=residx_jl, num_recycles=0)), Zygote.Params([jl]))
    z_err = z_err == "" ? z_err2 : z_err
    gpu = _gpu_sniff(jl, aa_jl)
    base_t, jl_t, ratio = _bench_pair("ESMFoldModelJL", () -> base(aa_fl; mask=mask_fl, residx=residx_fl, num_recycles=0), () -> jl(aa_jl; mask=mask_jl, residx=residx_jl, num_recycles=0))
    _push_result!("ESMFoldModelJL", parity=diff, z_in=z_in, z_param=z_param, z_err=z_err, gpu=gpu, base_time=base_t, jl_time=jl_t, ratio=ratio)
end

const CASES = Dict(
    "linear_first" => _run_linear_first,
    "layernorm_first" => _run_layernorm_first,
    "esmfold_attention" => _run_esmfold_attention,
    "sequence_to_pair" => _run_sequence_to_pair,
    "pair_to_sequence" => _run_pair_to_sequence,
    "residue_mlp" => _run_residue_mlp,
    "ofmha" => _run_ofmha,
    "triangle_attention_start" => () -> _run_triangle_attention(true),
    "triangle_attention_end" => () -> _run_triangle_attention(false),
    "triangle_mul_out" => () -> _run_triangle_mul(true),
    "triangle_mul_in" => () -> _run_triangle_mul(false),
    "triangular_block" => _run_triangular_block,
    "relative_position" => _run_relative_position,
    "point_projection" => _run_point_projection,
    "ipa" => _run_invariant_point_attention,
    "backbone_update" => _run_backbone_update,
    "transition" => _run_transition,
    "angle_resnet" => _run_angle_resnet,
    "structure_module" => _run_structure_module,
    "structure_module_core" => _run_structure_module_core,
    "folding_trunk" => _run_folding_trunk,
    "folding_trunk_core" => _run_folding_trunk_core,
    "esmfold_model_jl" => _run_esmfold_model_jl,
)

function _write_results(; append::Bool=false)
    mode = append ? "a" : "w"
    open(OUT_PATH, mode) do io
        if !append
            println(io, "# Julia-Convention Checks")
            println(io, "")
            println(io, "Heuristic GPU sniff: static scan for loops, @inbounds, or setindex! in method source; no GPU execution.")
            println(io, "")
            println(io, "| Layer | Parity Max Diff | Zygote Input | Zygote Params | GPU Sniff | Base Median | JL Median | Ratio | Notes |")
            println(io, "| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        end
        for r in results
            parity = r["parity"]
            parity_s = parity === nothing ? "" : string(parity)
            z_in = r["zygote_input"]
            z_param = r["zygote_params"]
            z_in_s = z_in === nothing ? "" : (z_in ? "ok" : "fail")
            z_param_s = z_param === nothing ? "" : (z_param ? "ok" : "fail")
            gpu = join(r["gpu"], "; ")
            base_t = r["base_time"]
            jl_t = r["jl_time"]
            ratio = r["ratio"]
            base_s = base_t === nothing ? "" : BenchmarkTools.prettytime(base_t)
            jl_s = jl_t === nothing ? "" : BenchmarkTools.prettytime(jl_t)
            ratio_s = ratio === nothing ? "" : string(round(ratio; digits=3))
            note = r["zygote_err"]
            println(io, "| ", r["name"], " | ", parity_s, " | ", z_in_s, " | ", z_param_s, " | ", gpu, " | ", base_s, " | ", jl_s, " | ", ratio_s, " | ", note, " |")
        end
    end
end

function main()
    case_env = get(ENV, "CASE", "all")
    append = get(ENV, "APPEND", "0") == "1"
    targets = case_env == "all" ? sort(collect(keys(CASES))) : split(case_env, ",")
    for name in targets
        haskey(CASES, name) || error("Unknown CASE=$(name). Available: $(join(sort(collect(keys(CASES))), ", "))")
        CASES[name]()
    end
    _write_results(append=append)
end

main()
