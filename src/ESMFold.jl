module ESMFold

using LinearAlgebra
using Statistics

using Einops
using Flux
using NNlib
using Onion
using NPZ
using JSON
using SpecialFunctions
using HuggingFaceApi

include("device_utils.jl")
include("safetensors.jl")

export Alphabet, RestypeTable, Alphabet_from_architecture
export ESM2Config, ESM2
export ESMFoldEmbedConfig, ESMFoldEmbed
export FoldingTrunkConfig, FoldingTrunk
export StructureModuleConfig
export ESMFoldConfig, ESMFoldModel
export ESMFoldModelJL
export load_esmfold_npz!, load_esm2_npz!, load_esmfold_safetensors!
export load_ESM, load_ESMFold
export sequence_to_af2_indices
export infer, infer_pdb, infer_pdbs, output_to_pdb
export confidence_metrics
export set_training!, is_training
export make_atom14_masks_jl!
export compute_tm_jl, compute_predicted_aligned_error_jl, categorical_lddt_jl

include("alphabet.jl")
include("residue_constants.jl")
include("openfold_utils.jl")
include("layernorm_last.jl")
include("rigid.jl")
include("openfold_feats.jl")
include("esmfold_misc.jl")
include("rotary.jl")
include("attention.jl")
include("esm2.jl")
include("esmfold_embed.jl")
include("triangular.jl")
include("structure_module.jl")
include("folding_trunk.jl")
include("openfold_infer_utils.jl")
include("protein.jl")
include("esmfold_full.jl")
include("weights.jl")

end
