using Pkg
Pkg.activate("/Users/benmurrell/JuliaM3/juliaESM"; io=devnull)

using ESMEmbed

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
output = ESMEmbed.infer(model, seq)
atom37 = ESMEmbed.atom14_to_atom37(output[:positions][end, :, :, :, :], output)
mask = output[:atom37_atom_exists]
if ndims(mask) == 2
    mask = reshape(mask, 1, size(mask, 1), size(mask, 2))
end

let zero_atoms = 0, existing_atoms = 0
    for b in 1:size(mask, 1), l in 1:size(mask, 2), a in 1:size(mask, 3)
        if mask[b, l, a] > 0.5
            existing_atoms += 1
            if atom37[b, l, a, 1] == 0 && atom37[b, l, a, 2] == 0 && atom37[b, l, a, 3] == 0
                zero_atoms += 1
            end
        end
    end
    println("existing atoms with zero coords: ", zero_atoms, " / ", existing_atoms)
end

pdb = ESMEmbed.output_to_pdb(output)[1]
open("output_ELLKKLLEELKG.pdb", "w") do io
    write(io, pdb)
end

println("wrote output_ELLKKLLEELKG.pdb")
