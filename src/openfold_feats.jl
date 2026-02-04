using NNlib

function torsion_angles_to_frames(r::Rigid, alpha::AbstractArray, aatype::AbstractArray, default_frames::AbstractArray)
    # default_frames: (21, 8, 4, 4)
    # gather default frames for each residue
    idx = aatype .+ 1
    df = permutedims(default_frames, (2, 3, 4, 1))
    df_sel = NNlib.gather(df, idx)
    default_4x4 = permutedims(df_sel, (4, 5, 1, 2, 3))

    default_r = rigid_from_tensor_4x4(default_4x4)

    # prepend backbone rotation
    bb_shape = (size(alpha)[1:end-2]..., 1, 2)
    bb_rot = zeros_like(alpha, eltype(alpha), bb_shape...)
    _view_last2(bb_rot, 1, 2) .= 1
    alpha = cat(bb_rot, alpha; dims=ndims(alpha) - 1)

    # build rotation matrices from angles
    all_rots = zeros_like(alpha, eltype(alpha), size(default_4x4)...)
    _view_last2(all_rots, 1, 1) .= 1
    _view_last2(all_rots, 2, 2) .= _view_last1(alpha, 2)
    _view_last2(all_rots, 2, 3) .= -_view_last1(alpha, 1)
    _view_last2(all_rots, 3, 2) .= _view_last1(alpha, 1)
    _view_last2(all_rots, 3, 3) .= _view_last1(alpha, 2)

    all_rots_r = rigid_from_tensor_4x4(all_rots)
    all_frames = compose(default_r, all_rots_r)

    # chi frames
    chi2_frame_to_frame = rigid_index(all_frames, Colon(), Colon(), 6)
    chi3_frame_to_frame = rigid_index(all_frames, Colon(), Colon(), 7)
    chi4_frame_to_frame = rigid_index(all_frames, Colon(), Colon(), 8)

    chi1_frame_to_bb = rigid_index(all_frames, Colon(), Colon(), 5)
    chi2_frame_to_bb = compose(chi1_frame_to_bb, chi2_frame_to_frame)
    chi3_frame_to_bb = compose(chi2_frame_to_bb, chi3_frame_to_frame)
    chi4_frame_to_bb = compose(chi3_frame_to_bb, chi4_frame_to_frame)

    rot = get_rot_mats(all_frames.rots)
    trans = all_frames.trans
    rot_first = rot[:, :, 1:5, :, :]
    trans_first = trans[:, :, 1:5, :]
    rot_chi2 = reshape(get_rot_mats(chi2_frame_to_bb.rots), size(rot, 1), size(rot, 2), 1, 3, 3)
    rot_chi3 = reshape(get_rot_mats(chi3_frame_to_bb.rots), size(rot, 1), size(rot, 2), 1, 3, 3)
    rot_chi4 = reshape(get_rot_mats(chi4_frame_to_bb.rots), size(rot, 1), size(rot, 2), 1, 3, 3)
    trans_chi2 = reshape(chi2_frame_to_bb.trans, size(trans, 1), size(trans, 2), 1, 3)
    trans_chi3 = reshape(chi3_frame_to_bb.trans, size(trans, 1), size(trans, 2), 1, 3)
    trans_chi4 = reshape(chi4_frame_to_bb.trans, size(trans, 1), size(trans, 2), 1, 3)
    rot_new = cat(rot_first, rot_chi2, rot_chi3, rot_chi4; dims=3)
    trans_new = cat(trans_first, trans_chi2, trans_chi3, trans_chi4; dims=3)
    all_frames_to_bb = Rigid(Rotation(rot_mats=rot_new), trans_new)

    # compose with backbone
    all_frames_to_global = compose(r, all_frames_to_bb)
    return all_frames_to_global
end

function frames_and_literature_positions_to_atom14_pos(
    r::Rigid,
    aatype::AbstractArray,
    default_frames::AbstractArray,
    group_idx::AbstractArray,
    atom_mask::AbstractArray,
    lit_positions::AbstractArray,
)
    idx = aatype .+ 1
    # gather group indices and masks
    g = permutedims(group_idx, (2, 1))
    group_sel = NNlib.gather(g, idx)
    group_idx_sel = permutedims(group_sel, (2, 3, 1))

    am = permutedims(atom_mask, (2, 1))
    mask_sel = NNlib.gather(am, idx)
    atom_mask_sel = permutedims(mask_sel, (2, 3, 1))
    atom_mask_sel = reshape(atom_mask_sel, size(atom_mask_sel)..., 1)

    lp = permutedims(lit_positions, (2, 3, 1))
    lp_sel = NNlib.gather(lp, idx)
    lit_pos_sel = permutedims(lp_sel, (3, 4, 1, 2))

    group_mask = one_hot_last(group_idx_sel, size(default_frames, 2))
    group_mask = convert.(eltype(r.trans), group_mask)

    rot = get_rot_mats(r.rots)          # (B, L, 8, 3, 3)
    trans = r.trans                    # (B, L, 8, 3)

    rot_exp = reshape(rot, size(rot, 1), size(rot, 2), 1, size(rot, 3), 3, 3)
    mask_exp = reshape(group_mask, size(group_mask, 1), size(group_mask, 2), size(group_mask, 3), size(group_mask, 4), 1, 1)
    rot_atom = sum(rot_exp .* mask_exp; dims=4)
    rot_atom = dropdims(rot_atom; dims=4)

    trans_exp = reshape(trans, size(trans, 1), size(trans, 2), 1, size(trans, 3), 3)
    mask_t = reshape(group_mask, size(group_mask, 1), size(group_mask, 2), size(group_mask, 3), size(group_mask, 4), 1)
    trans_atom = sum(trans_exp .* mask_t; dims=4)
    trans_atom = dropdims(trans_atom; dims=4)

    pred = rot_vec_mul(rot_atom, lit_pos_sel) .+ trans_atom
    pred = pred .* atom_mask_sel
    return pred
end

function atom14_to_atom37(atom14::AbstractArray, batch::AbstractDict)
    idx = batch[:residx_atom37_to_atom14]
    # atom14: (B, L, 14, 3) -> (3, B, L, 14)
    src = permutedims(atom14, (4, 1, 2, 3))

    # Ensure idx has batch dimension: (B, L, 37)
    if ndims(idx) == 2
        B = size(atom14, 1)
        L = size(atom14, 2)
        idx = reshape(idx, 1, L, size(idx, 2))
        idx = repeat(idx, B, 1, 1)
    end

    # Convert 0-based mapping to 1-based for Julia gather; masked positions will be zeroed later.
    idx1 = idx .+ 1

    # Build tuple indices into (B, L, 14) so gather doesn't duplicate batch dims.
    B, L, A = size(idx1, 1), size(idx1, 2), size(idx1, 3)
    idx_cart = Array{CartesianIndex{3}}(undef, B, L, A)
    for b in 1:B, l in 1:L, a in 1:A
        idx_cart[b, l, a] = CartesianIndex(b, l, idx1[b, l, a])
    end
    idx_cart = to_device(idx_cart, atom14, CartesianIndex{3})

    gathered = NNlib.gather(src, idx_cart)  # (3, B, L, 37)
    out = permutedims(gathered, (2, 3, 4, 1))  # (B, L, 37, 3)

    atom37_exists = batch[:atom37_atom_exists]
    if ndims(atom37_exists) == 2
        atom37_exists = reshape(atom37_exists, 1, size(atom37_exists, 1), size(atom37_exists, 2))
        atom37_exists = repeat(atom37_exists, size(atom14, 1), 1, 1)
    end
    out .= out .* reshape(atom37_exists, size(atom37_exists)..., 1)
    return out
end
