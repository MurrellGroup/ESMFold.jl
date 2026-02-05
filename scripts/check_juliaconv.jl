using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Zygote
using Flux
using Statistics
using ESMFold

ESMFold.set_training!(false)

const OUT_PATH = joinpath(@__DIR__, "..", "docs", "juliaconv_checks.md")
const SKIP_ZYGOTE = get(ENV, "SKIP_ZYGOTE", "0") == "1"

seq = "ELLKKLLEELKG"
weights_path = joinpath(@__DIR__, "..", "weights", "esm.safetensors")

function _load_model()
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

function _run_once(m, seq)
    output = ESMFold.infer(m, seq)
    pdb = ESMFold.output_to_pdb(output)[1]
    return output, pdb
end

model = _load_model()

println("Warming up...")
out_1, pdb_1 = _run_once(model, seq)
GC.gc()
println("Timing (second run)...")
t = @elapsed (out_2, pdb_2) = _run_once(model, seq)

zygote_ok = true
zygote_err = ""
if !SKIP_ZYGOTE
    try
        params = Flux.params(model)
        gs = Zygote.gradient(() -> sum(ESMFold.infer(model, seq)[:plddt]), params)
        for p in params
            if gs[p] === nothing
                zygote_ok = false
                zygote_err = "missing gradient"
                break
            end
        end
    catch err
        zygote_ok = false
        zygote_err = sprint(showerror, err)
    end
else
    zygote_err = "skipped"
end

open(OUT_PATH, "w") do io
    println(io, "# Julia-Convention Checks")
    println(io)
    println(io, "- Second-run time (s): `", t, "`")
    println(io, "- PDB match warm vs second: `", pdb_1 == pdb_2, "`")
    println(io, "- Positions shape: `", size(out_2[:positions]), "`")
    println(io, "- PLDDT shape: `", size(out_2[:plddt]), "`")
    println(io, "- Zygote ok: `", zygote_ok, "`")
    if !zygote_ok || zygote_err == "skipped"
        println(io, "- Zygote note: `", zygote_err, "`")
    end
end

println("Wrote ", OUT_PATH)
