using Pkg
Pkg.activate("/Users/benmurrell/JuliaM3/juliaESM"; io=devnull)

using NPZ
using Statistics
using ESMEmbed
using NNlib

ref = NPZ.npzread("/Users/benmurrell/JuliaM3/juliaESM/esmfold_structure_intermediate.npz")

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

s_ln_s = sm.layer_norm_s(single)
z_ln = sm.layer_norm_z(pair)
s_initial = s_ln_s
s_in = sm.linear_in(s_ln_s)

rigids = ESMEmbed.rigid_identity(size(s_in)[1:end-1], s_in; fmt=:quat)
ipa_out = sm.ipa(s_in, z_ln, rigids, mask)
s_after_ipa = s_in .+ ipa_out
s_after_ipa = sm.ipa_dropout(s_after_ipa)
s_after_ln = sm.layer_norm_ipa(s_after_ipa)
s_after_transition = sm.transition(s_after_ln)

bb_update = sm.bb_update(s_after_transition)
rigids1 = ESMEmbed.compose_q_update_vec(rigids, bb_update)

backb_to_global = ESMEmbed.Rigid(
    ESMEmbed.Rotation(rot_mats=ESMEmbed.get_rot_mats(rigids1.rots)),
    rigids1.trans,
)
backb_to_global = ESMEmbed.scale_translation(backb_to_global, sm.cfg.trans_scale_factor)

unnormalized_angles, angles = sm.angle_resnet(s_after_transition, s_initial)

default_frames, group_idx, atom_mask, lit_positions = ESMEmbed._init_residue_constants!(sm, angles)
all_frames_to_global = ESMEmbed.torsion_angles_to_frames(backb_to_global, angles, aatype, default_frames)
pred_xyz = ESMEmbed.frames_and_literature_positions_to_atom14_pos(
    all_frames_to_global,
    aatype,
    default_frames,
    group_idx,
    atom_mask,
    lit_positions,
)

scaled_rigids = ESMEmbed.scale_translation(rigids1, sm.cfg.trans_scale_factor)

function diff_stats(a, b)
    max_abs = maximum(abs.(a .- b))
    mean_abs = mean(abs.(a .- b))
    return max_abs, mean_abs
end

function report(name, a, b)
    max_abs, mean_abs = diff_stats(Float32.(a), Float32.(b))
    println(name, " max_abs=", max_abs, " mean_abs=", mean_abs)
end

report("s_ln_s", s_ln_s, ref["s_ln_s"])
report("z_ln", z_ln, ref["z_ln"])
report("s_initial", s_initial, ref["s_initial"])
report("s_in", s_in, ref["s_in"])
report("ipa_out", ipa_out, ref["ipa_out"])
report("s_after_ipa", s_after_ipa, ref["s_after_ipa"])
report("s_after_ln", s_after_ln, ref["s_after_ln"])
report("s_after_transition", s_after_transition, ref["s_after_transition"])
report("bb_update", bb_update, ref["bb_update"])
report("rigids_t7", ESMEmbed.to_tensor_7(rigids1), ref["rigids_t7"])
report("backb_t7", ESMEmbed.to_tensor_7(backb_to_global), ref["backb_t7"])
report("unnormalized_angles", unnormalized_angles, ref["unnormalized_angles"])
report("angles", angles, ref["angles"])
report("frames", ESMEmbed.to_tensor_7(scaled_rigids), ref["frames"])
report("sidechain_frames", ESMEmbed.to_tensor_4x4(all_frames_to_global), ref["sidechain_frames"])
report("positions", pred_xyz, ref["positions"])
report("states", s_after_transition, ref["states"])

# --- torsion/frame intermediates (match OpenFold feats.py) ---
default_frames, _, _, _ = ESMEmbed._init_residue_constants!(sm, angles)
idx = aatype .+ 1
df = permutedims(default_frames, (2, 3, 4, 1))
df_sel = NNlib.gather(df, idx)
default_4x4 = permutedims(df_sel, (4, 5, 1, 2, 3))

bb_shape = (size(angles)[1:end-2]..., 1, 2)
bb_rot = ESMEmbed.zeros_like(angles, eltype(angles), bb_shape...)
ESMEmbed._view_last2(bb_rot, 1, 2) .= 1
alpha = cat(bb_rot, angles; dims=ndims(angles) - 1)

all_rots = ESMEmbed.zeros_like(angles, eltype(angles), size(default_4x4)...)
ESMEmbed._view_last2(all_rots, 1, 1) .= 1
ESMEmbed._view_last2(all_rots, 2, 2) .= ESMEmbed._view_last1(alpha, 2)
ESMEmbed._view_last2(all_rots, 2, 3) .= -ESMEmbed._view_last1(alpha, 1)
ESMEmbed._view_last2(all_rots, 3, 2) .= ESMEmbed._view_last1(alpha, 1)
ESMEmbed._view_last2(all_rots, 3, 3) .= ESMEmbed._view_last1(alpha, 2)

default_r = ESMEmbed.rigid_from_tensor_4x4(default_4x4)
all_rots_r = ESMEmbed.rigid_from_tensor_4x4(all_rots)
all_frames = ESMEmbed.compose(default_r, all_rots_r)

chi2_frame_to_frame = ESMEmbed.rigid_index(all_frames, Colon(), Colon(), 6)
chi3_frame_to_frame = ESMEmbed.rigid_index(all_frames, Colon(), Colon(), 7)
chi4_frame_to_frame = ESMEmbed.rigid_index(all_frames, Colon(), Colon(), 8)

chi1_frame_to_bb = ESMEmbed.rigid_index(all_frames, Colon(), Colon(), 5)
chi2_frame_to_bb = ESMEmbed.compose(chi1_frame_to_bb, chi2_frame_to_frame)
chi3_frame_to_bb = ESMEmbed.compose(chi2_frame_to_bb, chi3_frame_to_frame)
chi4_frame_to_bb = ESMEmbed.compose(chi3_frame_to_bb, chi4_frame_to_frame)

rot = ESMEmbed.get_rot_mats(all_frames.rots)
trans = all_frames.trans
rot_first = rot[:, :, 1:5, :, :]
trans_first = trans[:, :, 1:5, :]
rot_chi2 = reshape(ESMEmbed.get_rot_mats(chi2_frame_to_bb.rots), size(rot, 1), size(rot, 2), 1, 3, 3)
rot_chi3 = reshape(ESMEmbed.get_rot_mats(chi3_frame_to_bb.rots), size(rot, 1), size(rot, 2), 1, 3, 3)
rot_chi4 = reshape(ESMEmbed.get_rot_mats(chi4_frame_to_bb.rots), size(rot, 1), size(rot, 2), 1, 3, 3)
trans_chi2 = reshape(chi2_frame_to_bb.trans, size(trans, 1), size(trans, 2), 1, 3)
trans_chi3 = reshape(chi3_frame_to_bb.trans, size(trans, 1), size(trans, 2), 1, 3)
trans_chi4 = reshape(chi4_frame_to_bb.trans, size(trans, 1), size(trans, 2), 1, 3)
rot_new = cat(rot_first, rot_chi2, rot_chi3, rot_chi4; dims=3)
trans_new = cat(trans_first, trans_chi2, trans_chi3, trans_chi4; dims=3)
all_frames_to_bb = ESMEmbed.Rigid(ESMEmbed.Rotation(rot_mats=rot_new), trans_new)
all_frames_to_global2 = ESMEmbed.compose(backb_to_global, all_frames_to_bb)

report("backb_rotmats", ESMEmbed.get_rot_mats(backb_to_global.rots), ref["backb_rotmats"])
report("backb_trans", backb_to_global.trans, ref["backb_trans"])
report("default_4x4", default_4x4, ref["default_4x4"])
report("all_rots_4x4", all_rots, ref["all_rots_4x4"])
report("all_frames_4x4", ESMEmbed.to_tensor_4x4(all_frames), ref["all_frames_4x4"])
report("all_frames_to_bb_4x4", ESMEmbed.to_tensor_4x4(all_frames_to_bb), ref["all_frames_to_bb_4x4"])
report("all_frames_to_global_4x4", ESMEmbed.to_tensor_4x4(all_frames_to_global2), ref["all_frames_to_global_4x4"])
