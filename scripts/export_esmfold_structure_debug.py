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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--safetensors", required=True)
    parser.add_argument("--sequence", default="ELLKKLLEELKG")
    parser.add_argument("--output", default="esmfold_structure_debug.npz")
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

        sm_out = model.trunk.structure_module(
            {"single": single, "pair": pair},
            aatype,
            mask.float(),
        )

        export = {
            "single": single.detach().cpu().numpy(),
            "pair": pair.detach().cpu().numpy(),
            "aatype": aatype.detach().cpu().numpy(),
            "mask": mask.detach().cpu().numpy(),
            "residx": residx.detach().cpu().numpy(),
            "s_s": s_s.detach().cpu().numpy(),
            "s_z": s_z.detach().cpu().numpy(),
        }
        for key in [
            "frames",
            "sidechain_frames",
            "unnormalized_angles",
            "angles",
            "positions",
            "states",
        ]:
            export[key] = sm_out[key].detach().cpu().numpy()

        np.savez(args.output, **export)


if __name__ == "__main__":
    main()
