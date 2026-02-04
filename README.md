# ESMFold.jl

A Julia port of the **full ESMFold model**: ESM2 embeddings + folding trunk + structure module.
This repo runs end‑to‑end folding on CPU, and will run on GPU when you move the model/tensors
to the GPU.

## Quickstart (single sequence)

```julia
using ESMFold

# Download weights from Hugging Face and build the full folding model
model = load_ESMFold()

seq = "ELLKKLLEELKG"
output = infer(model, seq)

# PDB output
pdb = output_to_pdb(output)[1]
println(pdb)
```

## Batch Folding

```julia
using ESMFold

model = load_ESMFold()
seqs = ["ELLKKLLEELKG", "ACDEFGHIKLMNPQRSTVWY"]

output = infer(model, seqs)

# PDBs for each sequence
pdbs = output_to_pdb(output)
```

You can also go directly to PDBs:

```julia
pdbs = infer_pdbs(model, seqs)
```

## Confidence Metrics

`infer` returns a dictionary with confidence outputs. You can access these directly or use
`confidence_metrics`:

```julia
metrics = confidence_metrics(output)

# Per‑residue plDDT (0‑100)
plddt = metrics.plddt

# Mean plDDT per sequence
mean_plddt = metrics.mean_plddt

# Predicted TM‑score per sequence
ptm = metrics.ptm

# Predicted aligned error (PAE)
pae = metrics.predicted_aligned_error
max_pae = metrics.max_predicted_aligned_error
```

## Weights And Caching

`load_ESMFold()` downloads the safetensors checkpoint from Hugging Face using
`HuggingFaceApi.hf_hub_download`. By default it pulls:

- `repo_id = "facebook/esmfold_v1"`
- `filename = "model.safetensors"`
- `revision = "ba837a3"`

Downloaded files are cached by HuggingFaceApi in your Julia depot (via OhMyArtifacts).
You can override the source if you want to point at a PR or a specific commit:

```julia
model = load_ESMFold(
    repo_id = "facebook/esmfold_v1",
    filename = "model.safetensors",
    revision = "refs/pr/123",
)
```

You can also skip network access and use the local cache only:

```julia
model = load_ESMFold(local_files_only=true)
```

## Testing

The regression test script in `scripts/test.jl` folds `"ELLKKLLEELKG"` and compares the
resulting PDB against `scripts/output_ELLKKLLEELKG.pdb`:

```bash
julia --project=. scripts/test.jl
```

## Notes

- CPU‑only execution is supported.
- The implementation follows the ESMFold Python model closely and is parity‑checked
  against the official model within floating‑point tolerances.

## License

This package reuses ESM/ESMFold code concepts and weight formats. Please refer to the
original ESM/ESMFold licenses and terms for model usage.
