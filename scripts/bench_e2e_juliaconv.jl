using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using ESMFold

ESMFold.set_training!(false)

seq = "ELLKKLLEELKG"
weights_path = joinpath(@__DIR__, "..", "weights", "esm.safetensors")

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

function _load_base_model()
    if isfile(weights_path)
        reader = ESMFold.SafeTensors.Reader(weights_path)
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
        ) = ESMFold._infer_esmfold_full_config(reader)

        alphabet = ESMFold.Alphabet_from_architecture("ESM-1b")
        esm = ESMFold.ESM2(
            num_layers,
            embed_dim,
            attention_heads;
            alphabet = alphabet,
            token_dropout = true,
        )

        trunk_cfg = ESMFold.FoldingTrunkConfig(
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
            ESMFold.StructureModuleConfig(),
        )

        cfg = ESMFold.ESMFoldConfig(; trunk=trunk_cfg, lddt_head_hid_dim=lddt_head_hid_dim, use_esm_attn_map=false)
        model = ESMFold.ESMFoldModel(esm; cfg=cfg)
        ESMFold.load_esmfold_safetensors!(model, reader)
        return model
    end

    return ESMFold.load_ESMFold()
end

function _load_jl_model(base::ESMFold.ESMFoldModel)
    esm0 = base.embed.esm
    esm = ESMFold.ESM2(
        esm0.num_layers,
        esm0.embed_dim,
        esm0.attention_heads;
        alphabet = esm0.alphabet,
        token_dropout = esm0.token_dropout,
    )
    model = ESMFold.ESMFoldModelJL(esm; cfg=base.cfg)
    _copy_params!(model, base)
    return model
end

function _jl_to_base_output(out)
    base = Dict{Symbol,Any}()
    base[:positions] = permutedims(out[:positions], (1, 5, 4, 3, 2))
    base[:aatype] = permutedims(out[:aatype], (2, 1))
    base[:residue_index] = permutedims(out[:residue_index], (2, 1))
    base[:atom14_atom_exists] = permutedims(out[:atom14_atom_exists], (3, 2, 1))
    base[:atom37_atom_exists] = permutedims(out[:atom37_atom_exists], (3, 2, 1))
    base[:residx_atom37_to_atom14] = permutedims(out[:residx_atom37_to_atom14], (3, 2, 1))
    base[:plddt] = permutedims(out[:plddt], (2, 1, 3))
    return base
end

function infer_jl(
    m::ESMFold.ESMFoldModelJL,
    sequences::Union{AbstractString,AbstractVector{<:AbstractString}};
    residx = nothing,
    masking_pattern = nothing,
    num_recycles = nothing,
    residue_index_offset::Int = 512,
    chain_linker::AbstractString = "G"^25,
)
    seqs = isa(sequences, AbstractString) ? [sequences] : sequences

    aatype, mask, _residx, linker_mask, chain_index = ESMFold.batch_encode_sequences(
        seqs;
        residue_index_offset = residue_index_offset,
        chain_linker = chain_linker,
    )

    if residx === nothing
        residx = _residx
    elseif !isa(residx, AbstractArray)
        residx = ESMFold.collate_dense_tensors(residx)
    end

    like = m.embed.esm.embed_tokens.weight
    aatype = ESMFold.to_device(aatype, like, Int)
    mask = ESMFold.to_device(mask, like, eltype(like))
    residx = ESMFold.to_device(residx, like, Int)
    linker_mask = ESMFold.to_device(linker_mask, like, eltype(like))
    if masking_pattern !== nothing
        masking_pattern = ESMFold.to_device(masking_pattern, like, eltype(aatype))
    end

    aatype_jl = permutedims(aatype, (2, 1))
    mask_jl = permutedims(mask, (2, 1))
    residx_jl = permutedims(residx, (2, 1))
    masking_pattern_jl = masking_pattern === nothing ? nothing : permutedims(masking_pattern, (2, 1))

    output = m(
        aatype_jl;
        mask = mask_jl,
        residx = residx_jl,
        masking_pattern = masking_pattern_jl,
        num_recycles = num_recycles,
    )

    # Apply linker mask to atom37 exists.
    output[:atom37_atom_exists] .*= reshape(permutedims(linker_mask, (2, 1)), 1, size(linker_mask, 2), size(linker_mask, 1))

    atom37 = permutedims(output[:atom37_atom_exists], (2, 3, 1)) # (L, B, 37)
    weighted_plddt = output[:plddt] .* atom37
    numerator = sum(weighted_plddt; dims=(1, 3))
    denom = sum(atom37; dims=(1, 3))
    output[:mean_plddt] = numerator ./ denom
    output[:chain_index] = chain_index

    return output
end

base_model = _load_base_model()
jl_model = _load_jl_model(base_model)

function _run_base_once(m, seq)
    output = ESMFold.infer(m, seq)
    pdb = ESMFold.output_to_pdb(output)[1]
    return output, pdb
end

function _run_jl_once(m, seq)
    out_jl = infer_jl(m, seq)
    out_jl_base = _jl_to_base_output(out_jl)
    pdb = ESMFold.output_to_pdb(out_jl_base)[1]
    return out_jl, out_jl_base, pdb
end

println("Warming up base...")
out_base_1, pdb_base_1 = _run_base_once(base_model, seq)
GC.gc()
println("Timing base (second run)...")
t_base = @elapsed (out_base_2, pdb_base) = _run_base_once(base_model, seq)

println("Warming up JL...")
out_jl_1, out_jl_base_1, pdb_jl_1 = _run_jl_once(jl_model, seq)
GC.gc()
println("Timing JL (second run)...")
t_jl = @elapsed (out_jl_2, out_jl_base, pdb_jl) = _run_jl_once(jl_model, seq)

pos_diff = maximum(abs.(out_base_2[:positions] .- out_jl_base[:positions]))
println("plddt size base: ", size(out_base_2[:plddt]), " jl: ", size(out_jl_base[:plddt]))
plddt_diff = size(out_base_2[:plddt]) == size(out_jl_base[:plddt]) ?
    maximum(abs.(out_base_2[:plddt] .- out_jl_base[:plddt])) : NaN

pdb_match = pdb_base == pdb_jl

println("Base second-run time (s): ", t_base)
println("JL second-run time (s): ", t_jl)
println("Speed ratio (JL/Base): ", t_jl / t_base)
println("Parity max diff: positions=", pos_diff, " plddt=", plddt_diff, " PDB match=", pdb_match)
if !pdb_match
    base_lines = split(pdb_base, '\n')
    jl_lines = split(pdb_jl, '\n')
    max_lines = min(length(base_lines), length(jl_lines))
    for i in 1:max_lines
        if base_lines[i] != jl_lines[i]
            println("First PDB diff at line ", i)
            println("base: ", base_lines[i])
            println("jl:   ", jl_lines[i])
            break
        end
    end
end
