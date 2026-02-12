module ESMFold

using LinearAlgebra

using Flux
using NNlib
using Onion
using NPZ
using JSON
using HuggingFaceApi

include("device_utils.jl")
include("safetensors.jl")
include("constants.jl")

# GPU support
using CUDA

# Import set_chunk_size! for extension on ESMFoldModel
import Onion: set_chunk_size!

# Import all protein layer types and utilities from Onion
using Onion: LayerNormFirst, LinearFirst, layernorm_inplace!,
    AbstractRotation, RotMatRotation, QuatRotation, Rigid,
    rot_from_mat, rot_from_quat, rot_matmul_first, rot_vec_mul_first,
    quat_multiply_first, quat_multiply_by_vec_first, quat_to_rot_first,
    rotation_identity, get_rot_mats, get_quats,
    compose_q_update_vec, rigid_identity, apply_rotation, apply_rigid,
    invert_apply_rigid, compose, scale_translation, to_tensor_7, to_tensor_4x4,
    rigid_index,
    permute_final_dims, flatten_final_dims, dict_multimap, stack_dicts,
    one_hot_last, collate_dense_tensors,
    rigid_from_tensor_4x4, torsion_angles_to_frames,
    frames_and_literature_positions_to_atom14_pos, atom14_to_atom37,
    RotaryEmbedding, rotate_half, apply_rotary_pos_emb,
    _update_cos_sin!,
    ESMFoldAttention, SharedDropout, SequenceToPair, PairToSequence, ResidueMLP,
    set_training!, is_training,
    encode_sequence, batch_encode_sequences,
    CategoricalMixture, categorical_lddt,
    ESMMultiheadAttention, _reshape_heads, _unshape_heads, _apply_key_padding_mask!,
    OFMultiheadAttention, TriangleAttention, TriangleMultiplicativeUpdate,
    TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming,
    TriangularSelfAttentionBlock,
    StructureModuleConfig, PointProjection, ESMFoldIPA,
    BackboneUpdate, StructureModuleTransitionLayer, StructureModuleTransition,
    AngleResnetBlock, AngleResnet, StructureModule,
    _init_residue_constants!,
    ESMFoldEmbedConfig, LayerNormMLP,
    FoldingTrunkConfig, RelativePosition, FoldingTrunk,
    cross_first, distogram, set_chunk_size!,
    layernorm_first_forward, flash_attention_forward, flash_attention_bias_forward,
    rotary_pos_emb_forward, combine_projections_forward,
    _view_first1, _view_first2, _view_last1, _view_last2,
    _reshape_first_corder

# Import residue constants from Onion
using Onion: restypes, restypes_with_x, restype_order, restype_order_with_x, restype_num,
    atom_types, atom_order, atom_type_num,
    restype_1to3, restype_3to1,
    residue_atoms, restype_name_to_atom14_names,
    restype_atom14_to_rigid_group, restype_atom14_mask,
    restype_atom14_rigid_group_positions, restype_rigid_group_default_frame

# Alias: InvariantPointAttention → ESMFoldIPA for backward compatibility
const InvariantPointAttention = ESMFoldIPA

export Alphabet, RestypeTable, Alphabet_from_architecture
export ESM2Config, ESM2
export ESMFoldEmbedConfig, ESMFoldEmbed
export FoldingTrunkConfig, FoldingTrunk
export StructureModuleConfig
export ESMFoldConfig, ESMFoldModel
export load_esmfold_npz!, load_esm2_npz!, load_esmfold_safetensors!
export load_ESM, load_ESMFold
export sequence_to_af2_indices
export infer, infer_pdb, infer_pdbs, output_to_pdb
export confidence_metrics
export set_training!, is_training
export make_atom14_masks!
export compute_tm, compute_predicted_aligned_error, categorical_lddt
export DISTOGRAM_BINS, LDDT_BINS, NUM_ATOM_TYPES, RECYCLE_DISTANCE_BINS
export esm2_forward_ad
export prepare_inputs, run_esm2, run_embedding
export run_trunk, run_trunk_single_pass
export run_structure_module, run_heads, run_pipeline

include("alphabet.jl")
include("esm2.jl")
include("esm2_ad.jl")
include("esmfold_embed.jl")
include("openfold_infer_utils.jl")
include("protein.jl")
include("esmfold_full.jl")
include("pipeline.jl")
include("weights.jl")

end
