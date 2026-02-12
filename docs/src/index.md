```@meta
CurrentModule = ESMFold
```

# ESMFold

A Julia port of the **full ESMFold model**: ESM2 embeddings + folding trunk + structure module.

## Quickstart

```julia
using ESMFold

model = load_ESMFold()
output = infer(model, "ACDEFGHIK")
pdb = output_to_pdb(output)[1]
```

## Outputs

`infer` returns a dictionary with structure + confidence outputs, including:

- `positions`, `frames`, `angles`, `states`
- `plddt`, `mean_plddt`, `ptm`, and `predicted_aligned_error`

Use `output_to_pdb` to export PDBs.

## Input Modes

- `AbstractMatrix{Int}` shaped `(B, L)`
- `Vector{Vector{Int}}` (auto-padded)
- `Vector{String}` or a single `String`

## Pipeline API

The inference pipeline is decomposed into composable stages. Each stage can be called
independently for research workflows (extracting embeddings, running partial inference,
feeding custom features, etc.).

```
prepare_inputs  →  run_embedding  →  run_trunk  →  run_heads  →  (post-processing)
                   ╰─ run_esm2        ╰─ run_trunk_single_pass
                                      ╰─ run_structure_module
```

### Stages

- **`prepare_inputs(model, sequences)`** — encode sequences and transfer to model device
- **`run_esm2(model, inputs)`** — raw ESM2 forward with BOS/EOS wrapping
- **`run_embedding(model, inputs)`** — ESM2 + projection to trunk dimensions → `(s_s_0, s_z_0)`
- **`run_trunk(model, s_s_0, s_z_0, inputs)`** — full trunk with recycling + structure module
- **`run_trunk_single_pass(model, s_s, s_z, inputs)`** — one pass through 48 blocks (no recycling, no structure module)
- **`run_structure_module(model, s_s, s_z, inputs)`** — structure module on arbitrary trunk outputs
- **`run_heads(model, structure, inputs)`** — all output heads (distogram, PTM, lDDT, LM)
- **`run_pipeline(model, sequences)`** — full pipeline, identical output to `infer()`

### AD-compatible ESM2

`esm2_forward_ad(esm, tokens_bt)` is a Zygote-compatible ESM2 forward that replaces
in-place ops with allocating equivalents. Use it when you need gradients through the
language model.

### Constants

`DISTOGRAM_BINS`, `LDDT_BINS`, `NUM_ATOM_TYPES`, `RECYCLE_DISTANCE_BINS` — named
constants for model dimensions, replacing magic numbers.

See the README for detailed examples.

```@index
```

```@autodocs
Modules = [ESMFold]
```
