import argparse
import math
from pathlib import Path
import numpy as np
import torch

from export_esmfold_full_from_safetensors import (
    SafeTensorsReader,
    build_model,
    load_esm2_weights,
    load_rest_weights,
    infer,
)

from openfold.utils.rigid_utils import Rigid
from openfold.utils.tensor_utils import permute_final_dims, flatten_final_dims

RESTYPES = [
    "A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V",
]
RESTYPES_WITH_X = RESTYPES + ["X"]
RESTYPE_ORDER_WITH_X = {aa: i for i, aa in enumerate(RESTYPES_WITH_X)}


def sequence_to_af2_indices(seq: str):
    return [RESTYPE_ORDER_WITH_X.get(ch, RESTYPE_ORDER_WITH_X["X"]) for ch in seq]


def ipa_internals(ipa, s, z, r, mask):
    # q, k, v
    q = ipa.linear_q(s)
    q = q.view(q.shape[:-1] + (ipa.no_heads, -1))

    # local point projections (before applying rigids)
    q_pts_linear = ipa.linear_q_points.linear(s)
    q_pts_local = q_pts_linear
    q_pts_local = torch.split(q_pts_local, q_pts_local.shape[-1] // 3, dim=-1)
    q_pts_local = torch.stack(q_pts_local, dim=-1).view(
        *s.shape[:-1], ipa.no_heads, ipa.no_qk_points, 3
    )

    kv_pts_linear = ipa.linear_kv_points.linear(s)
    kv_pts_local = kv_pts_linear
    kv_pts_local = torch.split(kv_pts_local, kv_pts_local.shape[-1] // 3, dim=-1)
    kv_pts_local = torch.stack(kv_pts_local, dim=-1).view(
        *s.shape[:-1], ipa.no_heads, ipa.no_qk_points + ipa.no_v_points, 3
    )
    k_pts_local = kv_pts_local[..., : ipa.no_qk_points, :]
    v_pts_local = kv_pts_local[..., ipa.no_qk_points :, :]

    q_pts = ipa.linear_q_points(s, r)

    kv = ipa.linear_kv(s)
    kv = kv.view(kv.shape[:-1] + (ipa.no_heads, -1))
    k, v = torch.split(kv, ipa.c_hidden, dim=-1)

    kv_pts = ipa.linear_kv_points(s, r)
    k_pts, v_pts = torch.split(kv_pts, [ipa.no_qk_points, ipa.no_v_points], dim=-2)

    # bias
    b = ipa.linear_b(z)

    # scalar attention
    a = torch.matmul(
        permute_final_dims(q, (1, 0, 2)),  # [*, H, L, C]
        permute_final_dims(k, (1, 2, 0)),  # [*, H, C, L]
    )
    a = a * math.sqrt(1.0 / (3.0 * ipa.c_hidden))
    a = a + (math.sqrt(1.0 / 3.0) * permute_final_dims(b, (2, 0, 1)))

    # point attention
    pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
    pt_att = pt_att ** 2
    pt_att = sum(torch.unbind(pt_att, dim=-1))

    head_weights = ipa.softplus(ipa.head_weights).view(
        *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
    )
    head_weights = head_weights * math.sqrt(
        1.0 / (3.0 * (ipa.no_qk_points * 9.0 / 2.0))
    )
    pt_att = (pt_att * head_weights).sum(dim=-1) * (-0.5)
    pt_att = permute_final_dims(pt_att, (2, 0, 1))

    square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    square_mask = ipa.inf * (square_mask - 1)
    a = a + pt_att + square_mask.unsqueeze(-3)
    attn = ipa.softmax(a)

    # output from values
    o = torch.matmul(attn, v.transpose(-2, -3)).transpose(-2, -3)
    o_flat = flatten_final_dims(o, 2)

    # point outputs
    o_pt = torch.sum(
        (
            attn[..., None, :, :, None]
            * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
        ),
        dim=-2,
    )
    o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
    o_pt = r[..., None, None].invert_apply(o_pt)

    o_pt_norm = flatten_final_dims(
        torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + ipa.eps), 2
    )

    o_pt_reshaped = o_pt.reshape(*o_pt.shape[:-3], -1, 3)
    o_pt_x, o_pt_y, o_pt_z = torch.unbind(o_pt_reshaped, dim=-1)

    # pair output
    o_pair = torch.matmul(attn.transpose(-2, -3), z)
    o_pair = flatten_final_dims(o_pair, 2)

    concat = torch.cat((o_flat, o_pt_x, o_pt_y, o_pt_z, o_pt_norm, o_pair), dim=-1)
    out = ipa.linear_out(concat)

    return {
        "q": q,
        "k": k,
        "v": v,
        "q_pts": q_pts,
        "k_pts": k_pts,
        "v_pts": v_pts,
        "q_pts_local": q_pts_local,
        "k_pts_local": k_pts_local,
        "v_pts_local": v_pts_local,
        "q_pts_linear": q_pts_linear,
        "kv_pts_linear": kv_pts_linear,
        "b": b,
        "attn_logits": a,
        "attn": attn,
        "o_flat": o_flat,
        "o_pt_norm": o_pt_norm,
        "o_pair": o_pair,
        "out": out,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--safetensors", required=True)
    parser.add_argument("--sequence", default="ELLKKLLEELKG")
    parser.add_argument("--output", default="esmfold_ipa_debug.npz")
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

        # run full model with no recycling to get trunk outputs
        structure = infer(model, args.sequence, num_recycles=0)
        s_s = model.trunk.trunk2sm_s(structure["s_s"])
        s_z = model.trunk.trunk2sm_z(structure["s_z"])

        # build mask from sequence
        aa = torch.tensor([sequence_to_af2_indices(args.sequence)], dtype=torch.long)
        mask = torch.ones_like(aa).float()

        sm = model.trunk.structure_module
        s = sm.layer_norm_s(s_s)
        z = sm.layer_norm_z(s_z)
        s = sm.linear_in(s)
        rigids = Rigid.identity(s.shape[:-1], s.dtype, s.device, False, fmt="quat")

        dbg = ipa_internals(sm.ipa, s, z, rigids, mask)
        dbg["s"] = s
        dbg["z"] = z

        export = {}
        for k, v in dbg.items():
            if torch.is_tensor(v):
                export[k] = v.cpu().numpy()

        np.savez(args.output, **export)


if __name__ == "__main__":
    main()
