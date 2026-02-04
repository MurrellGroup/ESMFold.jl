import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

# ---- minimal SafeTensors reader (no external deps) ----
DTYPE_MAP = {
    "F16": torch.float16,
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
}


class SafeTensorsReader:
    def __init__(self, path: Path):
        self.path = path
        self._file = None
        self._header = None
        self._base = None

    def __enter__(self):
        self._file = self.path.open("rb")
        header_len = struct.unpack("<Q", self._file.read(8))[0]
        self._header = json.loads(self._file.read(header_len))
        self._base = 8 + header_len
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._file is not None:
            self._file.close()

    @property
    def header(self):
        return self._header

    def load(self, name: str) -> torch.Tensor:
        entry = self._header[name]
        dtype = DTYPE_MAP[entry["dtype"]]
        shape = entry["shape"]
        start, end = entry["data_offsets"]
        self._file.seek(self._base + start)
        data = self._file.read(end - start)
        tensor = torch.frombuffer(memoryview(data), dtype=dtype)
        return tensor.reshape(shape)


RESTYPES = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
RESTYPES_WITH_X = RESTYPES + ["X"]
RESTYPE_ORDER_WITH_X = {aa: i for i, aa in enumerate(RESTYPES_WITH_X)}


def sequence_to_af2_indices(seq: str):
    return [RESTYPE_ORDER_WITH_X.get(ch, RESTYPE_ORDER_WITH_X["X"]) for ch in seq]


def infer_config(reader: SafeTensorsReader):
    keys = list(reader.header.keys())
    layer_ids = []
    for key in keys:
        if not key.startswith("esm.encoder.layer."):
            continue
        parts = key.split(".")
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[3])
        except ValueError:
            continue
        layer_ids.append(idx)
    if not layer_ids:
        raise RuntimeError("Unable to infer num_layers from safetensors header")
    num_layers = max(layer_ids) + 1

    embed_entry = reader.header["esm.embeddings.word_embeddings.weight"]
    embed_dim = embed_entry["shape"][1]

    inv_entry = reader.header["esm.encoder.layer.0.attention.self.rotary_embeddings.inv_freq"]
    head_dim = inv_entry["shape"][0] * 2
    attention_heads = embed_dim // head_dim

    cs_entry = reader.header["esm_s_mlp.1.weight"]
    c_s = cs_entry["shape"][1]

    c_z = 1
    if "esm_z_mlp.1.weight" in reader.header:
        cz_entry = reader.header["esm_z_mlp.1.weight"]
        c_z = cz_entry["shape"][1]

    # full config
    block_ids = []
    for key in keys:
        if not key.startswith("trunk.blocks."):
            continue
        parts = key.split(".")
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[2])
        except ValueError:
            continue
        block_ids.append(idx)
    num_blocks = max(block_ids) + 1 if block_ids else 0

    ln_entry = reader.header["trunk.blocks.0.layernorm_1.weight"]
    sequence_state_dim = ln_entry["shape"][0]
    z_entry = reader.header["trunk.blocks.0.tri_att_start.layer_norm.weight"]
    pairwise_state_dim = z_entry["shape"][0]

    seq_heads_entry = reader.header["trunk.blocks.0.pair_to_sequence.linear.weight"]
    sequence_num_heads = seq_heads_entry["shape"][0]
    sequence_head_width = sequence_state_dim // sequence_num_heads

    pair_heads_entry = reader.header["trunk.blocks.0.tri_att_start.linear.weight"]
    pairwise_num_heads = pair_heads_entry["shape"][0]
    pairwise_head_width = pairwise_state_dim // pairwise_num_heads

    pos_entry = reader.header["trunk.pairwise_positional_embedding.embedding.weight"]
    position_bins = (pos_entry["shape"][0] - 2) // 2

    lddt_entry = reader.header["lddt_head.1.weight"]
    lddt_head_hid_dim = lddt_entry["shape"][0]

    return dict(
        num_layers=num_layers,
        embed_dim=embed_dim,
        attention_heads=attention_heads,
        c_s=sequence_state_dim,
        c_z=pairwise_state_dim,
        sequence_head_width=sequence_head_width,
        pairwise_head_width=pairwise_head_width,
        position_bins=position_bins,
        num_blocks=num_blocks,
        lddt_head_hid_dim=lddt_head_hid_dim,
    )


# ---- model wiring ----

def build_model(reader: SafeTensorsReader, use_esm_attn_map: bool = False):
    import types
    import sys

    # stub openfold.np.protein to avoid heavy deps
    prot_mod = types.ModuleType("openfold.np.protein")
    class Protein:
        pass
    prot_mod.Protein = Protein
    prot_mod.to_pdb = lambda *args, **kwargs: None
    sys.modules["openfold.np.protein"] = prot_mod

    from esm.data import Alphabet
    from esm.model.esm2 import ESM2
    from esm.esmfold.v1.trunk import FoldingTrunk, StructureModuleConfig
    from esm.esmfold.v1.categorical_mixture import categorical_lddt
    from openfold.np import residue_constants as rc

    cfg = infer_config(reader)

    alphabet = Alphabet.from_architecture("ESM-1b")
    esm = ESM2(
        num_layers=cfg["num_layers"],
        embed_dim=cfg["embed_dim"],
        attention_heads=cfg["attention_heads"],
        alphabet=alphabet,
        token_dropout=True,
    )

    # Build ESMFold-lite components
    esm_s_mlp = torch.nn.Sequential(
        torch.nn.LayerNorm(cfg["embed_dim"]),
        torch.nn.Linear(cfg["embed_dim"], cfg["c_s"]),
        torch.nn.ReLU(),
        torch.nn.Linear(cfg["c_s"], cfg["c_s"]),
    )

    if use_esm_attn_map:
        # not used in this parity check
        esm_z_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(cfg["num_layers"] * cfg["attention_heads"]),
            torch.nn.Linear(cfg["num_layers"] * cfg["attention_heads"], cfg["c_z"]),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg["c_z"], cfg["c_z"]),
        )
    else:
        esm_z_mlp = None

    n_tokens_embed = 21 + 2  # restype_num + 2 (unknown, mask) + padding
    embedding = torch.nn.Embedding(n_tokens_embed, cfg["c_s"], padding_idx=0)

    structure_module = StructureModuleConfig().__dict__
    trunk = FoldingTrunk(
        num_blocks=cfg["num_blocks"],
        sequence_state_dim=cfg["c_s"],
        pairwise_state_dim=cfg["c_z"],
        sequence_head_width=cfg["sequence_head_width"],
        pairwise_head_width=cfg["pairwise_head_width"],
        position_bins=cfg["position_bins"],
        dropout=0.0,
        layer_drop=0.0,
        cpu_grad_checkpoint=False,
        max_recycles=4,
        chunk_size=None,
        structure_module=structure_module,
    )

    distogram_head = torch.nn.Linear(cfg["c_z"], 64)
    ptm_head = torch.nn.Linear(cfg["c_z"], 64)
    lm_head = torch.nn.Linear(cfg["c_s"], n_tokens_embed)
    lddt_bins = 50
    lddt_head = torch.nn.Sequential(
        torch.nn.LayerNorm(structure_module["c_s"]),
        torch.nn.Linear(structure_module["c_s"], cfg["lddt_head_hid_dim"]),
        torch.nn.Linear(cfg["lddt_head_hid_dim"], cfg["lddt_head_hid_dim"]),
        torch.nn.Linear(cfg["lddt_head_hid_dim"], 37 * lddt_bins),
    )

    # pack into a simple namespace
    model = types.SimpleNamespace()
    model.cfg = cfg
    model.alphabet = alphabet
    model.esm = esm
    model.esm_s_mlp = esm_s_mlp
    model.esm_z_mlp = esm_z_mlp
    model.embedding = embedding
    model.trunk = trunk
    model.distogram_head = distogram_head
    model.ptm_head = ptm_head
    model.lm_head = lm_head
    model.lddt_head = lddt_head
    model.lddt_bins = lddt_bins
    model.distogram_bins = 64

    # buffers/params
    model.af2_to_esm = reader.load("af2_to_esm").long()
    model.esm_s_combine = torch.nn.Parameter(reader.load("esm_s_combine").float())

    # attach helpers
    model.categorical_lddt = categorical_lddt
    def make_atom14_masks(protein):
        restype_atom14_to_atom37 = []
        restype_atom37_to_atom14 = []
        restype_atom14_mask = []

        for rt in rc.restypes:
            atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
            restype_atom14_to_atom37.append(
                [(rc.atom_order[name] if name else 0) for name in atom_names]
            )
            atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
            restype_atom37_to_atom14.append(
                [
                    (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
                    for name in rc.atom_types
                ]
            )
            restype_atom14_mask.append([(1.0 if name else 0.0) for name in atom_names])

        restype_atom14_to_atom37.append([0] * 14)
        restype_atom37_to_atom14.append([0] * 37)
        restype_atom14_mask.append([0.0] * 14)

        restype_atom14_to_atom37 = torch.tensor(
            restype_atom14_to_atom37, dtype=torch.int32, device=protein["aatype"].device
        )
        restype_atom37_to_atom14 = torch.tensor(
            restype_atom37_to_atom14, dtype=torch.int32, device=protein["aatype"].device
        )
        restype_atom14_mask = torch.tensor(
            restype_atom14_mask, dtype=torch.float32, device=protein["aatype"].device
        )

        protein_aatype = protein["aatype"].to(torch.long)
        residx_atom14_to_atom37 = restype_atom14_to_atom37[protein_aatype]
        residx_atom14_mask = restype_atom14_mask[protein_aatype]

        protein["atom14_atom_exists"] = residx_atom14_mask
        protein["residx_atom14_to_atom37"] = residx_atom14_to_atom37.long()

        residx_atom37_to_atom14 = restype_atom37_to_atom14[protein_aatype]
        protein["residx_atom37_to_atom14"] = residx_atom37_to_atom14.long()

        restype_atom37_mask = torch.zeros([21, 37], dtype=torch.float32, device=protein["aatype"].device)
        for restype, restype_letter in enumerate(rc.restypes):
            restype_name = rc.restype_1to3[restype_letter]
            atom_names = rc.residue_atoms[restype_name]
            for atom_name in atom_names:
                atom_type = rc.atom_order[atom_name]
                restype_atom37_mask[restype, atom_type] = 1

        residx_atom37_mask = restype_atom37_mask[protein_aatype]
        protein["atom37_atom_exists"] = residx_atom37_mask

        return protein

    model.make_atom14_masks = make_atom14_masks
    def _calculate_bin_centers(boundaries: torch.Tensor) -> torch.Tensor:
        step = boundaries[1] - boundaries[0]
        bin_centers = boundaries + step / 2
        return torch.cat([bin_centers, bin_centers[-1:] + step], dim=0)

    def _calculate_expected_aligned_error(boundaries, aligned_distance_error_probs):
        bin_centers = _calculate_bin_centers(boundaries)
        # reshape bin centers for broadcasting
        view = [1] * (aligned_distance_error_probs.ndim - 1) + [bin_centers.shape[0]]
        expected = torch.sum(aligned_distance_error_probs * bin_centers.view(*view), dim=-1)
        return expected, bin_centers[-1]

    def compute_predicted_aligned_error(logits, max_bin=31, no_bins=64):
        boundaries = torch.linspace(0.0, float(max_bin), steps=no_bins - 1, device=logits.device)
        aligned_confidence_probs = torch.softmax(logits, dim=-1)
        predicted_aligned_error, max_predicted_aligned_error = _calculate_expected_aligned_error(
            boundaries, aligned_confidence_probs
        )
        return dict(
            aligned_confidence_probs=aligned_confidence_probs,
            predicted_aligned_error=predicted_aligned_error,
            max_predicted_aligned_error=max_predicted_aligned_error,
        )

    def compute_tm(logits, residue_weights=None, asym_id=None, interface=False, max_bin=31, no_bins=64, eps=1e-8, **kwargs):
        if residue_weights is None:
            residue_weights = logits.new_ones(logits.shape[-2])
        boundaries = torch.linspace(0.0, float(max_bin), steps=no_bins - 1, device=logits.device)
        bin_centers = _calculate_bin_centers(boundaries)
        clipped_n = max(torch.sum(residue_weights), 19)
        d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3.0) - 1.8

        probs = torch.softmax(logits, dim=-1)
        tm_per_bin = 1.0 / (1.0 + (bin_centers ** 2) / (d0 ** 2))
        predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

        n = residue_weights.shape[-1]
        pair_mask = residue_weights.new_ones((n, n), dtype=torch.int32)
        if interface and (asym_id is not None):
            if len(asym_id.shape) > 1:
                batch_size = asym_id.shape[0]
                pair_mask = residue_weights.new_ones((batch_size, n, n), dtype=torch.int32)
            pair_mask *= (asym_id[..., None] != asym_id[..., None, :]).to(dtype=pair_mask.dtype)

        predicted_tm_term = predicted_tm_term * pair_mask
        pair_residue_weights = pair_mask * (residue_weights[..., None, :] * residue_weights[..., :, None])
        denom = eps + torch.sum(pair_residue_weights, dim=-1, keepdims=True)
        normed_residue_mask = pair_residue_weights / denom
        per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)

        weighted = per_alignment * residue_weights
        argmax = (weighted == torch.max(weighted)).nonzero()[0]
        return per_alignment[tuple(argmax)]

    model.compute_predicted_aligned_error = compute_predicted_aligned_error
    model.compute_tm = compute_tm

    return model


def load_esm2_weights(esm, reader: SafeTensorsReader):
    state = esm.state_dict()

    state["embed_tokens.weight"].copy_(reader.load("esm.embeddings.word_embeddings.weight").float())
    state["emb_layer_norm_after.weight"].copy_(reader.load("esm.encoder.emb_layer_norm_after.weight").float())
    state["emb_layer_norm_after.bias"].copy_(reader.load("esm.encoder.emb_layer_norm_after.bias").float())

    num_layers = esm.num_layers
    for i in range(num_layers):
        prefix = f"esm.encoder.layer.{i}"
        state[f"layers.{i}.self_attn.q_proj.weight"].copy_(reader.load(f"{prefix}.attention.self.query.weight").float())
        state[f"layers.{i}.self_attn.q_proj.bias"].copy_(reader.load(f"{prefix}.attention.self.query.bias").float())
        state[f"layers.{i}.self_attn.k_proj.weight"].copy_(reader.load(f"{prefix}.attention.self.key.weight").float())
        state[f"layers.{i}.self_attn.k_proj.bias"].copy_(reader.load(f"{prefix}.attention.self.key.bias").float())
        state[f"layers.{i}.self_attn.v_proj.weight"].copy_(reader.load(f"{prefix}.attention.self.value.weight").float())
        state[f"layers.{i}.self_attn.v_proj.bias"].copy_(reader.load(f"{prefix}.attention.self.value.bias").float())
        state[f"layers.{i}.self_attn.out_proj.weight"].copy_(reader.load(f"{prefix}.attention.output.dense.weight").float())
        state[f"layers.{i}.self_attn.out_proj.bias"].copy_(reader.load(f"{prefix}.attention.output.dense.bias").float())

        state[f"layers.{i}.self_attn_layer_norm.weight"].copy_(reader.load(f"{prefix}.attention.LayerNorm.weight").float())
        state[f"layers.{i}.self_attn_layer_norm.bias"].copy_(reader.load(f"{prefix}.attention.LayerNorm.bias").float())

        state[f"layers.{i}.fc1.weight"].copy_(reader.load(f"{prefix}.intermediate.dense.weight").float())
        state[f"layers.{i}.fc1.bias"].copy_(reader.load(f"{prefix}.intermediate.dense.bias").float())
        state[f"layers.{i}.fc2.weight"].copy_(reader.load(f"{prefix}.output.dense.weight").float())
        state[f"layers.{i}.fc2.bias"].copy_(reader.load(f"{prefix}.output.dense.bias").float())

        state[f"layers.{i}.final_layer_norm.weight"].copy_(reader.load(f"{prefix}.LayerNorm.weight").float())
        state[f"layers.{i}.final_layer_norm.bias"].copy_(reader.load(f"{prefix}.LayerNorm.bias").float())

    esm.load_state_dict(state, strict=False)
    esm.eval()


def load_rest_weights(model, reader: SafeTensorsReader):
    # esm_s_mlp
    model.esm_s_mlp[0].weight.data.copy_(reader.load("esm_s_mlp.0.weight").float())
    model.esm_s_mlp[0].bias.data.copy_(reader.load("esm_s_mlp.0.bias").float())
    model.esm_s_mlp[1].weight.data.copy_(reader.load("esm_s_mlp.1.weight").float())
    model.esm_s_mlp[1].bias.data.copy_(reader.load("esm_s_mlp.1.bias").float())
    model.esm_s_mlp[3].weight.data.copy_(reader.load("esm_s_mlp.3.weight").float())
    model.esm_s_mlp[3].bias.data.copy_(reader.load("esm_s_mlp.3.bias").float())

    model.embedding.weight.data.copy_(reader.load("embedding.weight").float())

    # trunk + heads
    state = model.trunk.state_dict()
    for key in state.keys():
        full_key = f"trunk.{key}"
        if full_key not in reader.header:
            # Handle PointProjection linear keys (safetensors omit the ".linear" segment)
            if ".ipa.linear_q_points.linear." in full_key:
                full_key = full_key.replace(".ipa.linear_q_points.linear.", ".ipa.linear_q_points.")
            elif ".ipa.linear_kv_points.linear." in full_key:
                full_key = full_key.replace(".ipa.linear_kv_points.linear.", ".ipa.linear_kv_points.")
            else:
                continue
        tensor = reader.load(full_key).float()
        target = state[key]
        if tensor.shape == target.shape:
            target.copy_(tensor)
            continue
        if tensor.ndim == 2 and tensor.T.shape == target.shape:
            target.copy_(tensor.T)
            continue
        if tensor.ndim == 3 and tensor.permute(1, 0, 2).shape == target.shape:
            target.copy_(tensor.permute(1, 0, 2))
            continue
        raise RuntimeError(
            f"shape mismatch for {full_key}: safetensors {tuple(tensor.shape)} vs target {tuple(target.shape)}"
        )
    model.trunk.load_state_dict(state, strict=False)

    model.distogram_head.weight.data.copy_(reader.load("distogram_head.weight").float())
    model.distogram_head.bias.data.copy_(reader.load("distogram_head.bias").float())

    model.ptm_head.weight.data.copy_(reader.load("ptm_head.weight").float())
    model.ptm_head.bias.data.copy_(reader.load("ptm_head.bias").float())

    model.lm_head.weight.data.copy_(reader.load("lm_head.weight").float())
    model.lm_head.bias.data.copy_(reader.load("lm_head.bias").float())

    model.lddt_head[0].weight.data.copy_(reader.load("lddt_head.0.weight").float())
    model.lddt_head[0].bias.data.copy_(reader.load("lddt_head.0.bias").float())
    model.lddt_head[1].weight.data.copy_(reader.load("lddt_head.1.weight").float())
    model.lddt_head[1].bias.data.copy_(reader.load("lddt_head.1.bias").float())
    model.lddt_head[2].weight.data.copy_(reader.load("lddt_head.2.weight").float())
    model.lddt_head[2].bias.data.copy_(reader.load("lddt_head.2.bias").float())
    model.lddt_head[3].weight.data.copy_(reader.load("lddt_head.3.weight").float())
    model.lddt_head[3].bias.data.copy_(reader.load("lddt_head.3.bias").float())


# ---- inference helpers ----

def encode_sequence(seq: str, residue_index_offset: int = 512, chain_linker: str = "G" * 25):
    if chain_linker is None:
        chain_linker = ""
    if residue_index_offset is None:
        residue_index_offset = 0

    chains = seq.split(":")
    seq = chain_linker.join(chains)

    unk_idx = RESTYPE_ORDER_WITH_X["X"]
    encoded = torch.tensor([RESTYPE_ORDER_WITH_X.get(aa, unk_idx) for aa in seq], dtype=torch.long)
    residx = torch.arange(len(encoded), dtype=torch.long)

    if residue_index_offset > 0:
        start = 0
        for i, chain in enumerate(chains):
            residx[start : start + len(chain) + len(chain_linker)] += i * residue_index_offset
            start += len(chain) + len(chain_linker)

    linker_mask = torch.ones_like(encoded, dtype=torch.float32)
    chain_index = []
    offset = 0
    for i, chain in enumerate(chains):
        if i > 0:
            chain_index.extend([i - 1] * len(chain_linker))
        chain_index.extend([i] * len(chain))
        offset += len(chain)
        linker_mask[offset : offset + len(chain_linker)] = 0
        offset += len(chain_linker)

    chain_index = torch.tensor(chain_index, dtype=torch.long)
    return encoded, residx, linker_mask, chain_index


def collate_dense_tensors(samples, pad_v: float = 0):
    if len(samples) == 0:
        return torch.Tensor()
    if len(set(x.dim() for x in samples)) != 1:
        raise RuntimeError("Samples has varying dimensions")
    (device,) = tuple(set(x.device for x in samples))
    max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
    result = torch.empty(len(samples), *max_shape, dtype=samples[0].dtype, device=device)
    result.fill_(pad_v)
    for i in range(len(samples)):
        t = samples[i]
        result[i][tuple(slice(0, k) for k in t.shape)] = t
    return result


def batch_encode_sequences(sequences, residue_index_offset=512, chain_linker="G" * 25):
    aatype_list = []
    residx_list = []
    linker_mask_list = []
    chain_index_list = []
    for seq in sequences:
        aatype_seq, residx_seq, linker_mask_seq, chain_index_seq = encode_sequence(
            seq,
            residue_index_offset=residue_index_offset,
            chain_linker=chain_linker,
        )
        aatype_list.append(aatype_seq)
        residx_list.append(residx_seq)
        linker_mask_list.append(linker_mask_seq)
        chain_index_list.append(chain_index_seq)

    aatype = collate_dense_tensors(aatype_list)
    mask = collate_dense_tensors([aatype.new_ones(len(aatype_seq)) for aatype_seq in aatype_list])
    residx = collate_dense_tensors(residx_list)
    linker_mask = collate_dense_tensors(linker_mask_list)
    chain_index_list = collate_dense_tensors(chain_index_list, -1)

    return aatype, mask, residx, linker_mask, chain_index_list


# ---- core forward ----

def compute_language_model_representations(model, esmaa: torch.Tensor, use_esm_attn_map: bool):
    batch_size = esmaa.size(0)
    bosi, eosi = model.alphabet.cls_idx, model.alphabet.eos_idx
    bos = esmaa.new_full((batch_size, 1), bosi)
    eos = esmaa.new_full((batch_size, 1), model.alphabet.padding_idx)
    esmaa = torch.cat([bos, esmaa, eos], dim=1)
    esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi

    res = model.esm(
        esmaa,
        repr_layers=range(model.esm.num_layers + 1),
        need_head_weights=use_esm_attn_map,
    )
    esm_s = torch.stack([v for _, v in sorted(res["representations"].items())], dim=2)
    esm_s = esm_s[:, 1:-1]
    if use_esm_attn_map:
        esm_z = res["attentions"].permute(0, 4, 3, 1, 2).flatten(3, 4)[:, 1:-1, 1:-1, :]
    else:
        esm_z = None
    return esm_s, esm_z


def infer(model, sequences, residx=None, masking_pattern=None, num_recycles=None, residue_index_offset=512, chain_linker="G" * 25, use_esm_attn_map: bool = False):
    if isinstance(sequences, str):
        sequences = [sequences]

    aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(
        sequences, residue_index_offset, chain_linker
    )

    if residx is None:
        residx = _residx
    elif not isinstance(residx, torch.Tensor):
        residx = collate_dense_tensors(residx)

    device = next(model.esm.parameters()).device
    aatype = aatype.to(device)
    mask = mask.to(device)
    residx = residx.to(device)
    linker_mask = linker_mask.to(device)

    # ESM
    aa = aatype
    esmaa = model.af2_to_esm[(aa + 1).masked_fill(mask != 1, 0)]
    if masking_pattern is not None:
        esmaa = esmaa.clone()
        esmaa[masking_pattern == 1] = model.alphabet.mask_idx

    esm_s, esm_z = compute_language_model_representations(model, esmaa, use_esm_attn_map)
    esm_s = esm_s.to(model.esm_s_combine.dtype).detach()

    esm_s = (torch.softmax(model.esm_s_combine, 0).unsqueeze(0) @ esm_s).squeeze(2)
    s_s_0 = model.esm_s_mlp(esm_s)

    if use_esm_attn_map:
        esm_z = esm_z.to(model.esm_s_combine.dtype).detach()
        s_z_0 = model.esm_z_mlp(esm_z)
    else:
        B, L = aa.shape
        s_z_0 = s_s_0.new_zeros(B, L, L, model.cfg["c_z"])

    s_s_0 = s_s_0 + model.embedding(aa)

    structure = model.trunk(s_s_0, s_z_0, aa, residx, mask, no_recycles=num_recycles)
    structure = {k: v for k, v in structure.items() if k in ["s_z", "s_s", "frames", "sidechain_frames", "unnormalized_angles", "angles", "positions", "states"]}

    disto_logits = model.distogram_head(structure["s_z"])
    disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
    structure["distogram_logits"] = disto_logits

    structure["lm_logits"] = model.lm_head(structure["s_s"])
    structure["aatype"] = aa
    model.make_atom14_masks(structure)

    for k in ["atom14_atom_exists", "atom37_atom_exists"]:
        structure[k] = structure[k] * mask.unsqueeze(-1)
    structure["residue_index"] = residx

    lddt_head = model.lddt_head(structure["states"]).reshape(
        structure["states"].shape[0],
        aa.shape[0],
        aa.shape[1],
        -1,
        model.lddt_bins,
    )
    structure["lddt_head"] = lddt_head
    plddt = model.categorical_lddt(lddt_head[-1], bins=model.lddt_bins)
    structure["plddt"] = 100 * plddt

    ptm_logits = model.ptm_head(structure["s_z"])
    structure["ptm_logits"] = ptm_logits

    seqlen = mask.type(torch.int64).sum(1)
    structure["ptm"] = torch.stack([
        model.compute_tm(batch_ptm_logits[None, :sl, :sl], max_bins=31, no_bins=model.distogram_bins)
        for batch_ptm_logits, sl in zip(ptm_logits, seqlen)
    ])

    structure.update(model.compute_predicted_aligned_error(ptm_logits, max_bin=31, no_bins=model.distogram_bins))

    structure["atom37_atom_exists"] = structure["atom37_atom_exists"] * linker_mask.unsqueeze(2)
    structure["mean_plddt"] = (structure["plddt"] * structure["atom37_atom_exists"]).sum(dim=(1, 2)) / structure["atom37_atom_exists"].sum(dim=(1, 2))
    structure["chain_index"] = chain_index

    # include pre-trunk features for parity
    structure["s_s_0"] = s_s_0
    structure["s_z_0"] = s_z_0

    return structure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--safetensors", required=True)
    parser.add_argument("--sequence", default="ELLKKLLEELKG")
    parser.add_argument("--output", default="esmfold_full_ref.npz")
    parser.add_argument("--num-recycles", type=int, default=None)
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

        output = infer(model, args.sequence, num_recycles=args.num_recycles)

        # export key tensors
        export = {}
        for k, v in output.items():
            if torch.is_tensor(v):
                export[k] = v.cpu().numpy()
        np.savez(args.output, **export)


if __name__ == "__main__":
    main()
