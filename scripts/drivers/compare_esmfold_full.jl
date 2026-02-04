using Pkg
Pkg.activate("/Users/benmurrell/JuliaM3/juliaESM"; io=devnull)

using NPZ
using Statistics
using ESMEmbed

ref = NPZ.npzread("/Users/benmurrell/JuliaM3/juliaESM/esmfold_full_ref_nr0.npz")

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
output = ESMEmbed.infer(model, seq; num_recycles=0)

function diff_stats(a, b)
    max_abs = maximum(abs.(a .- b))
    mean_abs = mean(abs.(a .- b))
    return max_abs, mean_abs
end

# keys to compare
float_keys = (
    :s_s_0, :s_z_0, :s_s, :s_z, :positions, :frames, :sidechain_frames,
    :unnormalized_angles, :angles, :states, :distogram_logits, :lm_logits,
    :lddt_head, :plddt, :ptm_logits, :ptm, :predicted_aligned_error,
    :aligned_confidence_probs, :max_predicted_aligned_error, :mean_plddt,
    :atom14_atom_exists, :atom37_atom_exists,
)

int_keys = (
    :aatype, :residue_index, :residx_atom14_to_atom37, :residx_atom37_to_atom14, :chain_index,
)

println("=== Float tensor diffs ===")
for k in float_keys
    haskey(output, k) || continue
    haskey(ref, String(k)) || continue
    aval = output[k]
    bval = ref[String(k)]
    a = aval isa Number ? [aval] : Array(aval)
    b = bval isa Number ? [bval] : bval
    # ensure Float32 for comparison
    a = Float32.(a)
    b = Float32.(b)
    max_abs, mean_abs = diff_stats(a, b)
    println(String(k), " max_abs=", max_abs, " mean_abs=", mean_abs)
end

println("=== Int tensor diffs ===")
for k in int_keys
    haskey(output, k) || continue
    haskey(ref, String(k)) || continue
    aval = output[k]
    bval = ref[String(k)]
    a = aval isa Number ? [aval] : Array(aval)
    b = bval isa Number ? [bval] : bval
    eq = all(a .== b)
    println(String(k), " equal=", eq, " mismatches=", count(identity, a .!= b))
end
