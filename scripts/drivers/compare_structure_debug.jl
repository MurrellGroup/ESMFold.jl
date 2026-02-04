using Pkg
Pkg.activate("/Users/benmurrell/JuliaM3/juliaESM"; io=devnull)

using NPZ
using Statistics
using ESMEmbed

ref = NPZ.npzread("/Users/benmurrell/JuliaM3/juliaESM/esmfold_structure_debug.npz")

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

single = Float32.(ref["single"])
pair = Float32.(ref["pair"])
aatype = Int.(ref["aatype"])
mask = Float32.(ref["mask"])

sm = model.trunk.structure_module
out = sm(Dict(:single => single, :pair => pair), aatype, mask)

function diff_stats(a, b)
    max_abs = maximum(abs.(a .- b))
    mean_abs = mean(abs.(a .- b))
    return max_abs, mean_abs
end

keys = (
    :frames,
    :sidechain_frames,
    :unnormalized_angles,
    :angles,
    :positions,
    :states,
)

for k in keys
    ref_key = String(k)
    haskey(ref, ref_key) || continue
    a = Float32.(Array(out[k]))
    b = Float32.(ref[ref_key])
    max_abs, mean_abs = diff_stats(a, b)
    println("sm_", ref_key, " max_abs=", max_abs, " mean_abs=", mean_abs)
end
