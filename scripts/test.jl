using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using ESMEmbed

seq = "ELLKKLLEELKG"
expected_path = joinpath(@__DIR__, "output_ELLKKLLEELKG.pdb")
generated_path = joinpath(@__DIR__, "output_ELLKKLLEELKG.generated.pdb")

function _load_model()
    weights_path = joinpath(@__DIR__, "..", "weights", "esm.safetensors")
    if isfile(weights_path)
        reader = ESMEmbed.SafeTensors.Reader(weights_path)
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
        return model
    end

    return ESMEmbed.load_ESMFold()
end

model = _load_model()
output = ESMEmbed.infer(model, seq)
pdb = ESMEmbed.output_to_pdb(output)[1]
open(generated_path, "w") do io
    write(io, pdb)
end

expected = read(expected_path, String)
if expected != pdb
    println("PDB mismatch.")
    exp_lines = split(expected, '\n')
    got_lines = split(pdb, '\n')
    max_lines = min(length(exp_lines), length(got_lines))
    for i in 1:max_lines
        if exp_lines[i] != got_lines[i]
            println("First diff at line ", i)
            println("expected: ", exp_lines[i])
            println("got:      ", got_lines[i])
            break
        end
    end
    error("PDB does not match expected output.")
end

metrics = ESMEmbed.confidence_metrics(output)
println("PDB matches: ", expected_path)
println("mean_plddt: ", metrics.mean_plddt)
println("ptm: ", metrics.ptm)
println("max_predicted_aligned_error: ", metrics.max_predicted_aligned_error)
