import argparse
import numpy as np
import torch

from export_esmfold_full_from_safetensors import (
    SafeTensorsReader,
    build_model,
    load_esm2_weights,
    load_rest_weights,
    compute_language_model_representations,
    infer_config,
)

RESTYPES = [
    "A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V",
]
RESTYPES_WITH_X = RESTYPES + ["X"]
RESTYPE_ORDER_WITH_X = {aa: i for i, aa in enumerate(RESTYPES_WITH_X)}


def sequence_to_af2_indices(seq: str):
    return [RESTYPE_ORDER_WITH_X.get(ch, RESTYPE_ORDER_WITH_X["X"]) for ch in seq]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--safetensors", required=True)
    parser.add_argument("--sequence", default="ELLKKLLEELKG")
    parser.add_argument("--output", default="esmfold_block0_debug.npz")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    from pathlib import Path
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

        # build s_s_0, s_z_0
        seq = args.sequence
        aa = torch.tensor([sequence_to_af2_indices(seq)], dtype=torch.long)
        mask = torch.ones_like(aa)

        esmaa = model.af2_to_esm[(aa + 1).masked_fill(mask != 1, 0)]
        esm_s, _ = compute_language_model_representations(model, esmaa, use_esm_attn_map=False)
        esm_s = esm_s.to(model.esm_s_combine.dtype).detach()
        esm_s = (torch.softmax(model.esm_s_combine, 0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = model.esm_s_mlp(esm_s) + model.embedding(aa)
        B, L, _ = s_s_0.shape
        s_z_0 = s_s_0.new_zeros(B, L, L, model.cfg["c_z"])

        block = model.trunk.blocks[0]
        tri_mask = mask.unsqueeze(2) * mask.unsqueeze(1)

        bias = block.pair_to_sequence(s_z_0)
        seq_ln = block.layernorm_1(s_s_0)
        proj_out = block.seq_attention.proj(seq_ln)
        B, L, C3 = proj_out.shape
        H = block.seq_attention.num_heads
        head_width = block.seq_attention.head_width
        t = proj_out.view(B, L, H, 3 * head_width)
        q = t[..., :head_width]
        k = t[..., head_width:2 * head_width]
        v = t[..., 2 * head_width:3 * head_width]
        seq_attn_out, attn = block.seq_attention(seq_ln, mask=mask, bias=bias)
        g_proj_out = block.seq_attention.g_proj(seq_ln)
        q_py = q.permute(0, 2, 1, 3)
        k_py = k.permute(0, 2, 1, 3)
        logits = torch.einsum("...qc,...kc->...qk", q_py, k_py)
        seq_state_attn = s_s_0 + seq_attn_out
        seq_state_mlp = block.mlp_seq(seq_state_attn)

        # manual attention to capture pre-o_proj outputs
        t_attn = block.seq_attention.proj(seq_ln)
        t_attn = t_attn.reshape(B, L, H, 3 * head_width).permute(0, 2, 1, 3)
        q2 = t_attn[..., :head_width]
        k2 = t_attn[..., head_width:2 * head_width]
        v2 = t_attn[..., 2 * head_width:3 * head_width]
        q2 = q2 * block.seq_attention.rescale_factor
        a2 = torch.einsum("...qc,...kc->...qk", q2, k2)
        a2 = a2 + bias.permute(0, 3, 1, 2)
        if mask is not None:
            a2 = a2.masked_fill(mask[:, None, None, :].expand_as(a2) == 0, float("-inf"))
        a2 = torch.softmax(a2, dim=-1)
        y2 = torch.einsum("...hqk,...hkc->...qhc", a2, v2)
        y2 = y2.reshape(B, L, H * head_width)
        if block.seq_attention.gated:
            y2_gated = block.seq_attention.g_proj(seq_ln).sigmoid() * y2
        else:
            y2_gated = y2

        pair_state = s_z_0 + block.sequence_to_pair(seq_state_mlp)

        tri_mul_out = block.tri_mul_out(pair_state, mask=tri_mask)
        pair_state = pair_state + tri_mul_out

        tri_mul_in = block.tri_mul_in(pair_state, mask=tri_mask)
        pair_state = pair_state + tri_mul_in

        pair_state_before_start = pair_state
        tri_att_start = block.tri_att_start(pair_state_before_start, mask=tri_mask, chunk_size=None)
        pair_state = pair_state_before_start + tri_att_start

        pair_state_before_end = pair_state
        tri_att_end = block.tri_att_end(pair_state_before_end, mask=tri_mask, chunk_size=None)
        pair_state = pair_state_before_end + tri_att_end

        # Triangle attention internals (start/end) to pinpoint mismatches
        from openfold.utils.tensor_utils import permute_final_dims, flatten_final_dims

        def tri_attn_internals(ta, x, mask):
            if not ta.starting:
                x = x.transpose(-2, -3)
                mask = mask.transpose(-1, -2)
            x = ta.layer_norm(x)
            mask_bias = (ta.inf * (mask - 1))[..., :, None, None, :]
            triangle_bias = permute_final_dims(ta.linear(x), (2, 0, 1))
            triangle_bias = triangle_bias.unsqueeze(-4)
            q, k, v = ta.mha._prep_qkv(x, x, apply_scale=True)
            logits = torch.matmul(q, k.transpose(-1, -2))
            logits = logits + mask_bias + triangle_bias
            attn = torch.softmax(logits, dim=-1)
            o = torch.matmul(attn, v)  # [*, H, Q, C]
            o = o.transpose(-2, -3)    # [*, Q, H, C]
            if ta.mha.linear_g is not None:
                g = torch.sigmoid(ta.mha.linear_g(x))
                g = g.view(g.shape[:-1] + (ta.mha.no_heads, -1))
                o = o * g
            o_flat = flatten_final_dims(o, 2)  # [*, Q, H*C]
            return logits, attn, o_flat, mask_bias, triangle_bias, x, q, k, v

        tri_att_start_logits, tri_att_start_attn, tri_att_start_pre_o, tri_att_start_mask_bias, tri_att_start_triangle_bias, tri_att_start_ln, tri_att_start_q, tri_att_start_k, tri_att_start_v = tri_attn_internals(
            block.tri_att_start, pair_state_before_start, tri_mask
        )
        tri_att_end_logits, tri_att_end_attn, tri_att_end_pre_o, tri_att_end_mask_bias, tri_att_end_triangle_bias, tri_att_end_ln, tri_att_end_q, tri_att_end_k, tri_att_end_v = tri_attn_internals(
            block.tri_att_end, pair_state_before_end, tri_mask
        )

        pair_state_mlp = block.mlp_pair(pair_state)

        export = {
            "s_s_0": s_s_0.cpu().numpy(),
            "s_z_0": s_z_0.cpu().numpy(),
            "bias": bias.cpu().numpy(),
            "seq_ln": seq_ln.cpu().numpy(),
            "proj_out": proj_out.cpu().numpy(),
            "seq_attn_out": seq_attn_out.cpu().numpy(),
            "seq_attn_pre_o_proj": y2_gated.cpu().numpy(),
            "seq_attn_pre_gate": y2.cpu().numpy(),
            "g_proj_out": g_proj_out.cpu().numpy(),
            "logits": logits.cpu().numpy(),
            "q": q.cpu().numpy(),
            "k": k.cpu().numpy(),
            "v": v.cpu().numpy(),
            "attn": attn.cpu().numpy(),
            "seq_state_attn": seq_state_attn.cpu().numpy(),
            "seq_state_mlp": seq_state_mlp.cpu().numpy(),
            "pair_state_seq2pair": (s_z_0 + block.sequence_to_pair(seq_state_mlp)).cpu().numpy(),
            "tri_mul_out": tri_mul_out.cpu().numpy(),
            "tri_mul_in": tri_mul_in.cpu().numpy(),
            "pair_state_before_start": pair_state_before_start.cpu().numpy(),
            "pair_state_before_end": pair_state_before_end.cpu().numpy(),
            "tri_att_start": tri_att_start.cpu().numpy(),
            "tri_att_end": tri_att_end.cpu().numpy(),
            "tri_att_start_logits": tri_att_start_logits.cpu().numpy(),
            "tri_att_start_attn": tri_att_start_attn.cpu().numpy(),
            "tri_att_start_pre_o": tri_att_start_pre_o.cpu().numpy(),
            "tri_att_start_mask_bias": tri_att_start_mask_bias.cpu().numpy(),
            "tri_att_start_triangle_bias": tri_att_start_triangle_bias.cpu().numpy(),
            "tri_att_start_ln": tri_att_start_ln.cpu().numpy(),
            "tri_att_start_q": tri_att_start_q.cpu().numpy(),
            "tri_att_start_k": tri_att_start_k.cpu().numpy(),
            "tri_att_start_v": tri_att_start_v.cpu().numpy(),
            "tri_att_end_logits": tri_att_end_logits.cpu().numpy(),
            "tri_att_end_attn": tri_att_end_attn.cpu().numpy(),
            "tri_att_end_pre_o": tri_att_end_pre_o.cpu().numpy(),
            "tri_att_end_mask_bias": tri_att_end_mask_bias.cpu().numpy(),
            "tri_att_end_triangle_bias": tri_att_end_triangle_bias.cpu().numpy(),
            "tri_att_end_ln": tri_att_end_ln.cpu().numpy(),
            "tri_att_end_q": tri_att_end_q.cpu().numpy(),
            "tri_att_end_k": tri_att_end_k.cpu().numpy(),
            "tri_att_end_v": tri_att_end_v.cpu().numpy(),
            "pair_state_final": pair_state_mlp.cpu().numpy(),
        }
        np.savez(args.output, **export)


if __name__ == "__main__":
    main()
