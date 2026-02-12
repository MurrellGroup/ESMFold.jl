# ESMFold.jl

A Julia port of the **full ESMFold model**: ESM2 embeddings + folding trunk + structure module.
This repo runs end‑to‑end folding on CPU, and will run on GPU when you move the model/tensors
to the GPU.

## Installation

Some dependencies (Onion, Einops, BatchedTransformations, etc.) live in the
MurrellGroup registry. Add it **once** alongside the default General registry:

```julia
using Pkg
Pkg.Registry.add("https://github.com/MurrellGroup/MurrellGroupRegistry")
```

Then install ESMFold.jl (from a local clone):

```julia
Pkg.develop(path="path/to/ESMFold.jl")
```

Or, if the package is registered in the MurrellGroup registry:

```julia
Pkg.add("ESMFold")
```

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

## Pipeline API

In addition to the monolithic `infer()`, ESMFold.jl exports composable pipeline stages
that give you access to intermediate representations. All functions work on both CPU and
GPU — tensors follow the model device automatically.

### Pipeline overview

```
prepare_inputs  →  run_embedding  →  run_trunk  →  run_heads  →  (post‑processing)
                   ╰─ run_esm2        ╰─ run_trunk_single_pass
                                      ╰─ run_structure_module
```

`run_pipeline(model, sequences)` chains all stages and produces output identical to
`infer()`. The individual stages can be called separately for research workflows.

### Stage reference

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `prepare_inputs(model, seqs)` | sequences | NamedTuple | Encode + device transfer |
| `run_esm2(model, inputs)` | prepared inputs | `ESM2Output` | Raw ESM2 with BOS/EOS wrapping |
| `run_embedding(model, inputs)` | prepared inputs | `(s_s_0, s_z_0)` | ESM2 + projection to trunk dims |
| `run_trunk(model, s_s_0, s_z_0, inputs)` | embeddings | Dict | Full trunk: recycling + structure module |
| `run_trunk_single_pass(model, s_s, s_z, inputs)` | states | `(s_s, s_z)` | One pass through 48 blocks (no recycling) |
| `run_structure_module(model, s_s, s_z, inputs)` | trunk states | Dict | Structure module on custom states |
| `run_heads(model, structure, inputs)` | structure Dict | Dict | Distogram, PTM, lDDT, LM heads |
| `run_pipeline(model, seqs)` | sequences | Dict | Full pipeline (identical to `infer`) |

### Examples

**Get ESM2 embeddings:**

```julia
inputs = prepare_inputs(model, "MKQLLED...")
esm_out = run_esm2(model, inputs; repr_layers=collect(0:33))
esm_out.representations[33]  # (B, T, C) last-layer hidden states
```

**Get trunk output without the structure module:**

```julia
inputs = prepare_inputs(model, "MKQLLED...")
emb = run_embedding(model, inputs)
result = run_trunk_single_pass(model, emb.s_s_0, emb.s_z_0, inputs)
result.s_s  # (1024, L, B) sequence state
result.s_z  # (128, L, L, B) pairwise state
```

**Run structure module on custom features:**

```julia
structure = run_structure_module(model, custom_s_s, custom_s_z, inputs)
```

**Get distograms from one pass:**

```julia
emb = run_embedding(model, inputs)
result = run_trunk_single_pass(model, emb.s_s_0, emb.s_z_0, inputs)
structure = run_structure_module(model, result.s_s, result.s_z, inputs)
output = run_heads(model, structure, inputs)
output[:distogram_logits]  # (64, L, L, B)
```

### AD‑compatible ESM2 forward

The standard ESM2 forward uses in‑place GPU ops that Zygote cannot differentiate.
`esm2_forward_ad` provides an allocating replacement:

```julia
using Zygote

# tokens_bt: (B, T) 0-indexed token array (from ESM2's Alphabet conventions)
grads = Zygote.gradient(model.embed.esm) do esm
    x = esm2_forward_ad(esm, tokens_bt)
    sum(x)
end
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

## GPU Inference

ESMFold.jl has no direct CUDA dependency. To run on GPU, add `CUDA.jl` and
`cuDNN.jl` to your own project environment, move the model with `Flux.gpu`,
and call `infer` as usual:

```julia
using CUDA, cuDNN
using Flux
using ESMFold

model = load_ESMFold()
gpu_model = Flux.gpu(model)

output = infer(gpu_model, "ELLKKLLEELKG")
pdb = output_to_pdb(output)[1]
```

All intermediate tensors automatically follow the model to the GPU.
`output_to_pdb` handles moving results back to CPU.

## Notes

- Both CPU and GPU execution are supported.
- The implementation follows the ESMFold Python model closely and is parity‑checked
  against the official model within floating‑point tolerances.

## License

This package reuses ESM/ESMFold code concepts and weight formats. Please refer to the
original ESM/ESMFold licenses and terms for model usage.
