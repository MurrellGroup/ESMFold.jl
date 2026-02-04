import argparse
from pathlib import Path

import numpy as np
import torch

from export_esmfold_full_from_safetensors import (
    SafeTensorsReader,
    build_model,
    load_esm2_weights,
    load_rest_weights,
    infer,
    batch_encode_sequences,
)

from openfold.utils.rigid_utils import Rotation, Rigid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--safetensors", required=True)
    parser.add_argument("--sequence", default="ELLKKLLEELKG")
    parser.add_argument("--output", default="esmfold_structure_intermediate.npz")
    parser.add_argument("--num-recycles", type=int, default=0)
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    with SafeTensorsReader(Path(args.safetensors)) as reader:
        model = build_model(reader, use_esm_attn_map=False)
        load_esm2_weights(model.esm, reader)
        load_rest_weights(model, reader)

        model.esm.eval()
        model.esm_s_mlp.eval()
        model.embedding.eval()
        model.trunk.eval()
        model.distogram_head.eval()
        model.ptm_head.eval()
        model.lm_head.eval()
        model.lddt_head.eval()

        # Run full trunk once to get s_s / s_z
        output = infer(model, args.sequence, num_recycles=args.num_recycles)
        s_s = output["s_s"]
        s_z = output["s_z"]

        # Rebuild mask/residx consistent with infer
        aatype, mask, residx, _linker_mask, _chain_index = batch_encode_sequences(
            [args.sequence], 512, "G" * 25
        )
        aatype = aatype.to(s_s.device)
        mask = mask.to(s_s.device)
        residx = residx.to(s_s.device)

        # Inputs to structure module
        single = model.trunk.trunk2sm_s(s_s)
        pair = model.trunk.trunk2sm_z(s_z)

        sm = model.trunk.structure_module

        # --- Structure module step-by-step (block 0) ---
        s_ln_s = sm.layer_norm_s(single)
        z = sm.layer_norm_z(pair)
        s_initial = s_ln_s
        s_in = sm.linear_in(s_ln_s)

        rigids = Rigid.identity(
            s_in.shape[:-1],
            s_in.dtype,
            s_in.device,
            sm.training,
            fmt="quat",
        )

        ipa_out = sm.ipa(s_in, z, rigids, mask.float())
        s_after_ipa = s_in + ipa_out
        s_after_ipa = sm.ipa_dropout(s_after_ipa)
        s_after_ln = sm.layer_norm_ipa(s_after_ipa)
        s_after_transition = sm.transition(s_after_ln)

        bb_update = sm.bb_update(s_after_transition)
        rigids1 = rigids.compose_q_update_vec(bb_update)

        backb_to_global = Rigid(
            Rotation(rot_mats=rigids1.get_rots().get_rot_mats(), quats=None),
            rigids1.get_trans(),
        )
        backb_to_global = backb_to_global.scale_translation(
            sm.trans_scale_factor
        )

        unnormalized_angles, angles = sm.angle_resnet(s_after_transition, s_initial)
        # torsion -> frames (with intermediates)
        sm._init_residue_constants(angles.dtype, angles.device)
        default_frames = sm.default_frames

        rigid_type = type(backb_to_global)
        default_4x4 = default_frames[aatype, ...]
        default_r = rigid_type.from_tensor_4x4(default_4x4)

        bb_rot = angles.new_zeros((*((1,) * len(angles.shape[:-1])), 2))
        bb_rot[..., 1] = 1
        alpha = torch.cat([bb_rot.expand(*angles.shape[:-2], -1, -1), angles], dim=-2)

        all_rots = alpha.new_zeros(default_r.shape + (4, 4))
        all_rots[..., 0, 0] = 1
        all_rots[..., 1, 1] = alpha[..., 1]
        all_rots[..., 1, 2] = -alpha[..., 0]
        all_rots[..., 2, 1:3] = alpha
        all_rots_r = rigid_type.from_tensor_4x4(all_rots)
        all_frames = default_r.compose(all_rots_r)

        chi2_frame_to_frame = all_frames[..., 5]
        chi3_frame_to_frame = all_frames[..., 6]
        chi4_frame_to_frame = all_frames[..., 7]

        chi1_frame_to_bb = all_frames[..., 4]
        chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
        chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
        chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

        all_frames_to_bb = rigid_type.cat(
            [
                all_frames[..., :5],
                chi2_frame_to_bb.unsqueeze(-1),
                chi3_frame_to_bb.unsqueeze(-1),
                chi4_frame_to_bb.unsqueeze(-1),
            ],
            dim=-1,
        )
        all_frames_to_global = backb_to_global[..., None].compose(all_frames_to_bb)
        pred_xyz = sm.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global, aatype
        )

        scaled_rigids = rigids1.scale_translation(sm.trans_scale_factor)

        export = {
            "single": single.detach().cpu().numpy(),
            "pair": pair.detach().cpu().numpy(),
            "aatype": aatype.detach().cpu().numpy(),
            "mask": mask.detach().cpu().numpy(),
            "residx": residx.detach().cpu().numpy(),
            "s_s": s_s.detach().cpu().numpy(),
            "s_z": s_z.detach().cpu().numpy(),
            "s_ln_s": s_ln_s.detach().cpu().numpy(),
            "z_ln": z.detach().cpu().numpy(),
            "s_initial": s_initial.detach().cpu().numpy(),
            "s_in": s_in.detach().cpu().numpy(),
            "ipa_out": ipa_out.detach().cpu().numpy(),
            "s_after_ipa": s_after_ipa.detach().cpu().numpy(),
            "s_after_ln": s_after_ln.detach().cpu().numpy(),
            "s_after_transition": s_after_transition.detach().cpu().numpy(),
            "bb_update": bb_update.detach().cpu().numpy(),
            "rigids_t7": rigids1.to_tensor_7().detach().cpu().numpy(),
            "backb_t7": backb_to_global.to_tensor_7().detach().cpu().numpy(),
            "unnormalized_angles": unnormalized_angles.detach().cpu().numpy(),
            "angles": angles.detach().cpu().numpy(),
            "frames": scaled_rigids.to_tensor_7().detach().cpu().numpy(),
            "sidechain_frames": all_frames_to_global.to_tensor_4x4().detach().cpu().numpy(),
            "positions": pred_xyz.detach().cpu().numpy(),
            "states": s_after_transition.detach().cpu().numpy(),
            # torsion/frame intermediates
            "backb_rotmats": backb_to_global.get_rots().get_rot_mats().detach().cpu().numpy(),
            "backb_trans": backb_to_global.get_trans().detach().cpu().numpy(),
            "default_4x4": default_4x4.detach().cpu().numpy(),
            "all_rots_4x4": all_rots.detach().cpu().numpy(),
            "all_frames_4x4": all_frames.to_tensor_4x4().detach().cpu().numpy(),
            "all_frames_to_bb_4x4": all_frames_to_bb.to_tensor_4x4().detach().cpu().numpy(),
            "all_frames_to_global_4x4": all_frames_to_global.to_tensor_4x4().detach().cpu().numpy(),
        }

        np.savez(args.output, **export)


if __name__ == "__main__":
    main()
