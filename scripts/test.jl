using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using ESMFold

seq = "ELLKKLLEELKG"
expected_path = joinpath(@__DIR__, "output_ELLKKLLEELKG.pdb")
generated_path = joinpath(@__DIR__, "output_ELLKKLLEELKG.generated.pdb")

function _load_model()
    weights_path = joinpath(@__DIR__, "..", "weights", "esm.safetensors")
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

model = _load_model()
output = ESMFold.infer(model, seq)
pdb = ESMFold.output_to_pdb(output)[1]
open(generated_path, "w") do io
    write(io, pdb)
end

expected = read(expected_path, String)
function _parse_float_field(line::AbstractString, r::UnitRange{Int})
    last(r) > lastindex(line) && return nothing
    s = strip(line[r])
    isempty(s) && return nothing
    return tryparse(Float64, s)
end

function _pdb_line_match(expected::AbstractString, got::AbstractString; tol::Float64=0.02)
    if startswith(expected, "ATOM") || startswith(expected, "HETATM")
        startswith(got, "ATOM") || startswith(got, "HETATM") || return false
        # Compare non-float fields with fixed-column slices.
        slices = (
            1:6,   # record
            7:11,  # serial
            13:16, # atom name
            17:17, # alt loc
            18:20, # res name
            22:22, # chain id
            23:26, # res seq
            27:27, # insertion code
        )
        for r in slices
            last(r) <= lastindex(expected) || return expected == got
            last(r) <= lastindex(got) || return false
            if expected[r] != got[r]
                return false
            end
        end
        # Float fields: x,y,z, occupancy, tempFactor
        fields = (31:38, 39:46, 47:54, 55:60, 61:66)
        for r in fields
            e = _parse_float_field(expected, r)
            g = _parse_float_field(got, r)
            if e === nothing || g === nothing
                return expected == got
            end
            if abs(e - g) > tol
                return false
            end
        end
        # Element symbol (optional)
        if lastindex(expected) >= 78 && lastindex(got) >= 78
            if expected[77:78] != got[77:78]
                return false
            end
        end
        return true
    end
    return expected == got
end

if expected != pdb
    println("PDB mismatch.")
    exp_lines = split(expected, '\n')
    got_lines = split(pdb, '\n')
    max_lines = min(length(exp_lines), length(got_lines))
    matched = true
    for i in 1:max_lines
        if !_pdb_line_match(exp_lines[i], got_lines[i])
            matched = false
            println("First diff at line ", i)
            println("expected: ", exp_lines[i])
            println("got:      ", got_lines[i])
            break
        end
    end
    if !matched || length(exp_lines) != length(got_lines)
        error("PDB does not match expected output.")
    else
        println("PDB matches within tolerance.")
    end
end

metrics = ESMFold.confidence_metrics(output)
println("PDB matches: ", expected_path)
println("mean_plddt: ", metrics.mean_plddt)
println("ptm: ", metrics.ptm)
println("max_predicted_aligned_error: ", metrics.max_predicted_aligned_error)
