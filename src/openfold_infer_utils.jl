# OpenFold inference utilities.
# Atom14/atom37 data tables and confidence metrics come from ProtInterop.
# This file keeps only ESMFold-specific wrappers.

using ProtInterop: OF_RESTYPE_ATOM14_TO_ATOM37, OF_RESTYPE_ATOM37_TO_ATOM14,
    OF_RESTYPE_ATOM14_MASK, OF_RESTYPE_ATOM37_MASK

# ── Aliases for backward compatibility ──────────────────────────────────────

const _restype_atom14_to_atom37 = OF_RESTYPE_ATOM14_TO_ATOM37
const _restype_atom37_to_atom14 = OF_RESTYPE_ATOM37_TO_ATOM14
const _restype_atom14_mask = OF_RESTYPE_ATOM14_MASK
const _restype_atom37_mask = OF_RESTYPE_ATOM37_MASK

# ── make_atom14_masks! (ESMFold-specific: mutating API, modifies Dict) ──────

function make_atom14_masks!(protein::AbstractDict)
    protein_aatype = protein[:aatype] .+ 1
    mask_type = haskey(protein, :s_s) ? eltype(protein[:s_s]) : Float32

    restype_atom14_to_atom37 = to_device(_restype_atom14_to_atom37, protein_aatype, Int)
    restype_atom37_to_atom14 = to_device(_restype_atom37_to_atom14, protein_aatype, Int)
    restype_atom14_mask = to_device(_restype_atom14_mask, protein_aatype, mask_type)
    restype_atom37_mask = to_device(_restype_atom37_mask, protein_aatype, mask_type)

    residx_atom14_to_atom37 = restype_atom14_to_atom37[protein_aatype, :]
    residx_atom14_to_atom37 = permutedims(residx_atom14_to_atom37, (3, 1, 2))
    protein[:atom14_atom_exists] = permutedims(restype_atom14_mask[protein_aatype, :], (3, 1, 2))
    protein[:residx_atom14_to_atom37] = residx_atom14_to_atom37 .- 1

    residx_atom37_to_atom14 = restype_atom37_to_atom14[protein_aatype, :]
    residx_atom37_to_atom14 = permutedims(residx_atom37_to_atom14, (3, 1, 2))
    protein[:residx_atom37_to_atom14] = residx_atom37_to_atom14 .- 1
    protein[:atom37_atom_exists] = permutedims(restype_atom37_mask[protein_aatype, :], (3, 1, 2))

    return protein
end

# ── _calculate_expected_aligned_error (ESMFold-specific helper) ─────────────

function _calculate_expected_aligned_error(alignment_confidence_breaks, aligned_distance_error_probs)
    bin_centers = ProtInterop._calculate_bin_centers(alignment_confidence_breaks)
    bview = reshape(bin_centers, ntuple(_ -> 1, ndims(aligned_distance_error_probs) - 1)..., length(bin_centers))
    expected = sum(aligned_distance_error_probs .* bview; dims=ndims(aligned_distance_error_probs))
    expected = dropdims(expected; dims=ndims(expected))
    return expected, bin_centers[end]
end

# Confidence metrics (compute_plddt, compute_predicted_aligned_error, compute_tm)
# are now imported from ProtInterop via `using ProtInterop` in the module definition.
# ESMFold re-exports them.
