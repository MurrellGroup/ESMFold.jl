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
- `Vector{Vector{Int}}` (auto‑padded)
- `Vector{String}` or a single `String`

See the README for more usage examples and batch folding.

```@index
```

```@autodocs
Modules = [ESMFold]
```
