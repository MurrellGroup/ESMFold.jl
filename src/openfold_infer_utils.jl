using NNlib

const _restype_atom14_to_atom37 = let
    rows = Vector{Vector{Int}}()
    for rt in restypes
        atom_names = restype_name_to_atom14_names[restype_1to3[rt]]
        push!(rows, [name == "" ? 1 : (atom_order[name] + 1) for name in atom_names])
    end
    push!(rows, fill(1, 14)) # UNK
    reduce(vcat, (reshape(r, 1, :) for r in rows))
end

const _restype_atom37_to_atom14 = let
    rows = Vector{Vector{Int}}()
    for rt in restypes
        atom_names = restype_name_to_atom14_names[restype_1to3[rt]]
        atom_name_to_idx14 = Dict(name => i for (i, name) in enumerate(atom_names) if name != "")
        push!(rows, [get(atom_name_to_idx14, name, 1) for name in atom_types])
    end
    push!(rows, fill(1, length(atom_types))) # UNK
    reduce(vcat, (reshape(r, 1, :) for r in rows))
end

const _restype_atom14_mask = let
    rows = Vector{Vector{Float32}}()
    for rt in restypes
        atom_names = restype_name_to_atom14_names[restype_1to3[rt]]
        push!(rows, [name == "" ? 0f0 : 1f0 for name in atom_names])
    end
    push!(rows, fill(0f0, 14)) # UNK
    reduce(vcat, (reshape(r, 1, :) for r in rows))
end

const _restype_atom37_mask = let
    mask = zeros(Float32, length(restypes), length(atom_types))
    for (restype, restype_letter) in enumerate(restypes)
        restype_name = restype_1to3[restype_letter]
        atom_names = residue_atoms[restype_name]
        for atom_name in atom_names
            atom_type = atom_order[atom_name] + 1
            mask[restype, atom_type] = 1f0
        end
    end
    mask = vcat(mask, zeros(Float32, 1, length(atom_types))) # UNK
    mask
end

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

function _calculate_bin_centers(boundaries::AbstractArray)
    step = boundaries[2] - boundaries[1]
    bin_centers = boundaries .+ step / 2
    return vcat(bin_centers, bin_centers[end] + step)
end

function _calculate_expected_aligned_error(alignment_confidence_breaks, aligned_distance_error_probs)
    bin_centers = _calculate_bin_centers(alignment_confidence_breaks)
    bview = reshape(bin_centers, ntuple(_ -> 1, ndims(aligned_distance_error_probs) - 1)..., length(bin_centers))
    expected = sum(aligned_distance_error_probs .* bview; dims=ndims(aligned_distance_error_probs))
    expected = dropdims(expected; dims=ndims(expected))
    return expected, bin_centers[end]
end

function compute_predicted_aligned_error(
    logits;
    max_bin::Int = 31,
    no_bins::Int = 64,
)
    boundaries = range(0f0, Float32(max_bin); length=no_bins - 1)
    boundaries = to_device(collect(boundaries), logits, Float32)

    aligned_confidence_probs = NNlib.softmax(logits; dims=1)
    bin_centers = _calculate_bin_centers(boundaries)
    bview = reshape(bin_centers, length(bin_centers), ntuple(_ -> 1, ndims(aligned_confidence_probs) - 1)...)
    expected = sum(aligned_confidence_probs .* bview; dims=1)
    expected = dropdims(expected; dims=1)

    return Dict(
        :aligned_confidence_probs => aligned_confidence_probs,
        :predicted_aligned_error => expected,
        :max_predicted_aligned_error => bin_centers[end],
    )
end

function compute_tm(
    logits;
    residue_weights = nothing,
    asym_id = nothing,
    interface::Bool = false,
    max_bin::Int = 31,
    no_bins::Int = 64,
    eps::Real = 1e-8,
)
    if residue_weights === nothing
        residue_weights = ones_like(logits, size(logits, 2))
    end

    boundaries = range(0f0, Float32(max_bin); length=no_bins - 1)
    boundaries = to_device(collect(boundaries), logits, Float32)
    bin_centers = _calculate_bin_centers(boundaries)

    clipped_n = max(sum(residue_weights), 19)
    d0 = 1.24f0 * (clipped_n - 15)^(1f0 / 3f0) - 1.8f0

    probs = NNlib.softmax(logits; dims=1)
    tm_per_bin = 1f0 ./ (1f0 .+ (bin_centers .^ 2) ./ (d0 ^ 2))
    tm_view = reshape(tm_per_bin, length(tm_per_bin), ntuple(_ -> 1, ndims(probs) - 1)...)
    predicted_tm_term = sum(probs .* tm_view; dims=1)
    predicted_tm_term = dropdims(predicted_tm_term; dims=1)

    n = size(predicted_tm_term, 1)
    pair_mask = ones_like(predicted_tm_term, Int, n, n)
    if interface && asym_id !== nothing
        pair_mask .= (reshape(asym_id, :, 1) .!= reshape(asym_id, 1, :))
    end

    predicted_tm_term = predicted_tm_term .* pair_mask

    pair_residue_weights = pair_mask .* (reshape(residue_weights, :, 1) .* reshape(residue_weights, 1, :))
    denom = eps .+ sum(pair_residue_weights; dims=2)
    normed_residue_mask = pair_residue_weights ./ denom
    per_alignment = sum(predicted_tm_term .* normed_residue_mask; dims=2)
    per_alignment = dropdims(per_alignment; dims=2)

    weighted = per_alignment .* residue_weights
    max_idx = argmax(weighted)
    return per_alignment[max_idx]
end
