using LinearAlgebra
import BatchedTransformations as BT

# Quaternion and rigid utilities (device-agnostic)

@inline function _view_last2(x, I, J)
    return view(x, ntuple(_ -> Colon(), ndims(x) - 2)..., I, J)
end

@inline function _view_last1(x, I)
    return view(x, ntuple(_ -> Colon(), ndims(x) - 1)..., I)
end

@inline function _view_first2(x, I, J)
    return view(x, I, J, ntuple(_ -> Colon(), ndims(x) - 2)...)
end

@inline function _view_first1(x, I)
    return view(x, I, ntuple(_ -> Colon(), ndims(x) - 1)...)
end

function rot_matmul(a::AbstractArray, b::AbstractArray)
    # a, b: (..., 3, 3)
    a11 = _view_last2(a, 1, 1); a12 = _view_last2(a, 1, 2); a13 = _view_last2(a, 1, 3)
    a21 = _view_last2(a, 2, 1); a22 = _view_last2(a, 2, 2); a23 = _view_last2(a, 2, 3)
    a31 = _view_last2(a, 3, 1); a32 = _view_last2(a, 3, 2); a33 = _view_last2(a, 3, 3)

    b11 = _view_last2(b, 1, 1); b12 = _view_last2(b, 1, 2); b13 = _view_last2(b, 1, 3)
    b21 = _view_last2(b, 2, 1); b22 = _view_last2(b, 2, 2); b23 = _view_last2(b, 2, 3)
    b31 = _view_last2(b, 3, 1); b32 = _view_last2(b, 3, 2); b33 = _view_last2(b, 3, 3)

    c11 = a11 .* b11 .+ a12 .* b21 .+ a13 .* b31
    c12 = a11 .* b12 .+ a12 .* b22 .+ a13 .* b32
    c13 = a11 .* b13 .+ a12 .* b23 .+ a13 .* b33

    c21 = a21 .* b11 .+ a22 .* b21 .+ a23 .* b31
    c22 = a21 .* b12 .+ a22 .* b22 .+ a23 .* b32
    c23 = a21 .* b13 .+ a22 .* b23 .+ a23 .* b33

    c31 = a31 .* b11 .+ a32 .* b21 .+ a33 .* b31
    c32 = a31 .* b12 .+ a32 .* b22 .+ a33 .* b32
    c33 = a31 .* b13 .+ a32 .* b23 .+ a33 .* b33

    mat = cat(
        cat(c11, c12, c13; dims=ndims(c11)+1),
        cat(c21, c22, c23; dims=ndims(c11)+1),
        cat(c31, c32, c33; dims=ndims(c11)+1);
        dims=ndims(c11)+2,
    )
    # Ensure last two dims are (row, col)
    return permutedims(mat, (1:ndims(mat)-2..., ndims(mat), ndims(mat)-1))
end

function rot_vec_mul(r::AbstractArray, t::AbstractArray)
    # r: (..., 3, 3), t: (..., 3)
    x = _view_last1(t, 1)
    y = _view_last1(t, 2)
    z = _view_last1(t, 3)
    o1 = _view_last2(r, 1, 1) .* x .+ _view_last2(r, 1, 2) .* y .+ _view_last2(r, 1, 3) .* z
    o2 = _view_last2(r, 2, 1) .* x .+ _view_last2(r, 2, 2) .* y .+ _view_last2(r, 2, 3) .* z
    o3 = _view_last2(r, 3, 1) .* x .+ _view_last2(r, 3, 2) .* y .+ _view_last2(r, 3, 3) .* z
    return cat(o1, o2, o3; dims=ndims(o1)+1)
end

function quat_multiply(q1::AbstractArray, q2::AbstractArray)
    a1 = _view_last1(q1, 1); b1 = _view_last1(q1, 2); c1 = _view_last1(q1, 3); d1 = _view_last1(q1, 4)
    a2 = _view_last1(q2, 1); b2 = _view_last1(q2, 2); c2 = _view_last1(q2, 3); d2 = _view_last1(q2, 4)
    w = a1 .* a2 .- b1 .* b2 .- c1 .* c2 .- d1 .* d2
    x = a1 .* b2 .+ b1 .* a2 .+ c1 .* d2 .- d1 .* c2
    y = a1 .* c2 .- b1 .* d2 .+ c1 .* a2 .+ d1 .* b2
    z = a1 .* d2 .+ b1 .* c2 .- c1 .* b2 .+ d1 .* a2
    return cat(w, x, y, z; dims=ndims(w)+1)
end

function quat_multiply_by_vec(q::AbstractArray, v::AbstractArray)
    a = _view_last1(q, 1); b = _view_last1(q, 2); c = _view_last1(q, 3); d = _view_last1(q, 4)
    vx = _view_last1(v, 1); vy = _view_last1(v, 2); vz = _view_last1(v, 3)
    w = -b .* vx .- c .* vy .- d .* vz
    x = a .* vx .+ c .* vz .- d .* vy
    y = a .* vy .- b .* vz .+ d .* vx
    z = a .* vz .+ b .* vy .- c .* vx
    return cat(w, x, y, z; dims=ndims(w)+1)
end

function quat_to_rot(quat::AbstractArray)
    q = quat ./ sqrt.(sum(quat .^ 2; dims=ndims(quat)))
    w = _view_last1(q, 1); x = _view_last1(q, 2); y = _view_last1(q, 3); z = _view_last1(q, 4)

    ww = w .* w; xx = x .* x; yy = y .* y; zz = z .* z
    wx = w .* x; wy = w .* y; wz = w .* z
    xy = x .* y; xz = x .* z; yz = y .* z

    r11 = 1 .- 2 .* (yy .+ zz)
    r12 = 2 .* (xy .- wz)
    r13 = 2 .* (xz .+ wy)

    r21 = 2 .* (xy .+ wz)
    r22 = 1 .- 2 .* (xx .+ zz)
    r23 = 2 .* (yz .- wx)

    r31 = 2 .* (xz .- wy)
    r32 = 2 .* (yz .+ wx)
    r33 = 1 .- 2 .* (xx .+ yy)

    rot = cat(
        cat(r11, r12, r13; dims=ndims(r11)+1),
        cat(r21, r22, r23; dims=ndims(r11)+1),
        cat(r31, r32, r33; dims=ndims(r11)+1);
        dims=ndims(r11)+2,
    )
    # Ensure last two dims are (row, col)
    return permutedims(rot, (1:ndims(rot)-2..., ndims(rot), ndims(rot)-1))
end

function rot_to_quat(rot::AbstractArray)
    r11 = _view_last2(rot, 1, 1); r22 = _view_last2(rot, 2, 2); r33 = _view_last2(rot, 3, 3)
    r12 = _view_last2(rot, 1, 2); r21 = _view_last2(rot, 2, 1)
    r13 = _view_last2(rot, 1, 3); r31 = _view_last2(rot, 3, 1)
    r23 = _view_last2(rot, 2, 3); r32 = _view_last2(rot, 3, 2)

    trace = r11 .+ r22 .+ r33

    mask0 = trace .> 0
    mask1 = .!mask0 .& (r11 .> r22) .& (r11 .> r33)
    mask2 = .!mask0 .& .!mask1 .& (r22 .> r33)
    mask3 = .!mask0 .& .!mask1 .& .!mask2

    s0 = sqrt.(trace .+ 1) .* 2
    w0 = 0.25 .* s0
    x0 = (r32 .- r23) ./ s0
    y0 = (r13 .- r31) ./ s0
    z0 = (r21 .- r12) ./ s0

    s1 = sqrt.(1 .+ r11 .- r22 .- r33) .* 2
    w1 = (r32 .- r23) ./ s1
    x1 = 0.25 .* s1
    y1 = (r12 .+ r21) ./ s1
    z1 = (r13 .+ r31) ./ s1

    s2 = sqrt.(1 .+ r22 .- r11 .- r33) .* 2
    w2 = (r13 .- r31) ./ s2
    x2 = (r12 .+ r21) ./ s2
    y2 = 0.25 .* s2
    z2 = (r23 .+ r32) ./ s2

    s3 = sqrt.(1 .+ r33 .- r11 .- r22) .* 2
    w3 = (r21 .- r12) ./ s3
    x3 = (r13 .+ r31) ./ s3
    y3 = (r23 .+ r32) ./ s3
    z3 = 0.25 .* s3

    w = ifelse.(mask0, w0, ifelse.(mask1, w1, ifelse.(mask2, w2, w3)))
    x = ifelse.(mask0, x0, ifelse.(mask1, x1, ifelse.(mask2, x2, x3)))
    y = ifelse.(mask0, y0, ifelse.(mask1, y1, ifelse.(mask2, y2, y3)))
    z = ifelse.(mask0, z0, ifelse.(mask1, z1, ifelse.(mask2, z2, z3)))

    quat = cat(w, x, y, z; dims=ndims(w)+1)
    quat = quat ./ sqrt.(sum(quat .^ 2; dims=ndims(quat)))
    return quat
end

struct Rotation
    rot_mats::Union{AbstractArray,Nothing}
    quats::Union{AbstractArray,Nothing}
end

function Rotation(; rot_mats=nothing, quats=nothing, normalize_quats::Bool=true)
    if (rot_mats === nothing) == (quats === nothing)
        error("Exactly one of rot_mats or quats must be provided")
    end
    if quats !== nothing && normalize_quats
        quats = quats ./ sqrt.(sum(quats .^ 2; dims=ndims(quats)))
    end
    return Rotation(rot_mats, quats)
end

function rotation_identity(shape::Tuple, like::AbstractArray; fmt::Symbol=:quat)
    if fmt == :rot_mat
        eye = Matrix{eltype(like)}(I, 3, 3)
        eye = reshape(eye, ntuple(_ -> 1, length(shape))..., 3, 3)
        rot = repeat(eye, shape..., 1, 1)
        rot = to_device(rot, like, eltype(like))
        return Rotation(rot_mats=rot)
    elseif fmt == :quat
        zero = zeros_like(like, eltype(like), shape..., 1)
        one = ones_like(like, eltype(like), shape..., 1)
        q = cat(one, zero, zero, zero; dims=ndims(zero))
        return Rotation(quats=q, normalize_quats=false)
    else
        error("Unknown rotation format: $fmt")
    end
end

function get_rot_mats(r::Rotation)
    if r.rot_mats !== nothing
        return r.rot_mats
    end
    @assert r.quats !== nothing
    return quat_to_rot(r.quats)
end

function get_quats(r::Rotation)
    if r.quats !== nothing
        return r.quats
    end
    @assert r.rot_mats !== nothing
    return rot_to_quat(r.rot_mats)
end

function compose_q_update_vec(r::Rotation, q_update_vec::AbstractArray; normalize_quats::Bool=true)
    q = get_quats(r)
    new_quats = q .+ quat_multiply_by_vec(q, q_update_vec)
    if normalize_quats
        new_quats = new_quats ./ sqrt.(sum(new_quats .^ 2; dims=ndims(new_quats)))
    end
    return Rotation(quats=new_quats, normalize_quats=false)
end

function compose_r(r1::Rotation, r2::Rotation)
    rot1 = get_rot_mats(r1)
    rot2 = get_rot_mats(r2)
    return Rotation(rot_mats=rot_matmul(rot1, rot2))
end

function apply_rotation(r::Rotation, pts::AbstractArray)
    rot = get_rot_mats(r)
    return rot_vec_mul(rot, pts)
end

function invert_rotation(r::Rotation)
    rot = get_rot_mats(r)
    return Rotation(rot_mats=permutedims(rot, (1:ndims(rot)-2..., ndims(rot), ndims(rot)-1)))
end

struct Rigid
    rots::Rotation
    trans::AbstractArray
end

function rigid_identity(shape::Tuple, like::AbstractArray; fmt::Symbol=:quat)
    rots = rotation_identity(shape, like; fmt=fmt)
    trans = zeros_like(like, eltype(like), shape..., 3)
    return Rigid(rots, trans)
end

function get_rots(r::Rigid)
    return r.rots
end

function get_trans(r::Rigid)
    return r.trans
end

function _align_batch(a::AbstractArray, target_batch_dims::Int)
    batch_dims = ndims(a) - 2
    if batch_dims < target_batch_dims
        extra = target_batch_dims - batch_dims
        return reshape(a, size(a)[1:batch_dims]..., ntuple(_ -> 1, extra)..., 3, 3)
    end
    return a
end

function _align_trans(a::AbstractArray, target_batch_dims::Int)
    batch_dims = ndims(a) - 1
    if batch_dims < target_batch_dims
        extra = target_batch_dims - batch_dims
        return reshape(a, size(a)[1:batch_dims]..., ntuple(_ -> 1, extra)..., 3)
    end
    return a
end

function apply_rigid(r::Rigid, pts::AbstractArray)
    rot = get_rot_mats(r.rots)
    trans = r.trans
    batch_dims = ndims(trans) - 1
    extra = ndims(pts) - (batch_dims + 1)
    rot = reshape(rot, size(rot)[1:batch_dims]..., ntuple(_ -> 1, extra)..., 3, 3)
    trans = reshape(trans, size(trans)[1:batch_dims]..., ntuple(_ -> 1, extra)..., 3)
    return rot_vec_mul(rot, pts) .+ trans
end

function invert_apply_rigid(r::Rigid, pts::AbstractArray)
    rot = get_rot_mats(r.rots)
    trans = r.trans
    batch_dims = ndims(trans) - 1
    extra = ndims(pts) - (batch_dims + 1)
    rot = reshape(rot, size(rot)[1:batch_dims]..., ntuple(_ -> 1, extra)..., 3, 3)
    trans = reshape(trans, size(trans)[1:batch_dims]..., ntuple(_ -> 1, extra)..., 3)
    return rot_vec_mul(permutedims(rot, (1:ndims(rot)-2..., ndims(rot), ndims(rot)-1)), pts .- trans)
end

function compose_q_update_vec(r::Rigid, q_update_vec::AbstractArray)
    q_vec = _view_last1(q_update_vec, 1:3)
    t_vec = _view_last1(q_update_vec, 4:6)
    new_rots = compose_q_update_vec(r.rots, q_vec)
    trans_update = apply_rotation(r.rots, t_vec)
    new_trans = r.trans .+ trans_update
    return Rigid(new_rots, new_trans)
end

function compose(r1::Rigid, r2::Rigid)
    rot1 = get_rot_mats(r1.rots)
    rot2 = get_rot_mats(r2.rots)
    batch_dims = max(ndims(rot1) - 2, ndims(rot2) - 2)
    rot1 = _align_batch(rot1, batch_dims)
    rot2 = _align_batch(rot2, batch_dims)
    trans1 = _align_trans(r1.trans, batch_dims)
    trans2 = _align_trans(r2.trans, batch_dims)
    new_rot = rot_matmul(rot1, rot2)
    new_trans = rot_vec_mul(rot1, trans2) .+ trans1
    return Rigid(Rotation(rot_mats=new_rot), new_trans)
end

function scale_translation(r::Rigid, factor::Real)
    return Rigid(r.rots, r.trans .* factor)
end

function to_tensor_4x4(r::Rigid)
    rot = get_rot_mats(r.rots)
    trans = r.trans
    trans_col = reshape(trans, size(trans)..., 1)
    top = cat(rot, trans_col; dims=ndims(rot))
    bottom_zeros = zeros_like(trans, eltype(trans), size(trans)[1:end-1]..., 1, 3)
    bottom_ones = ones_like(trans, eltype(trans), size(trans)[1:end-1]..., 1, 1)
    bottom = cat(bottom_zeros, bottom_ones; dims=ndims(bottom_zeros))
    return cat(top, bottom; dims=ndims(top) - 1)
end

function to_tensor_7(r::Rigid)
    q = get_quats(r.rots)
    return cat(q, r.trans; dims=ndims(q))
end

function rigid_from_tensor_4x4(t::AbstractArray)
    rot = _view_last2(t, 1:3, 1:3)
    trans = _view_last2(t, 1:3, 4)
    return Rigid(Rotation(rot_mats=rot), trans)
end

function rigid_index(r::Rigid, inds...)
    rot = get_rot_mats(r.rots)[inds..., :, :]
    trans = r.trans[inds..., :]
    return Rigid(Rotation(rot_mats=rot), trans)
end

# Julia-convention (feature-first, batch-last) rigid utilities.

struct RotationJL
    rot_mats::Union{AbstractArray,Nothing}
    quats::Union{AbstractArray,Nothing}
end

function RotationJL(; rot_mats=nothing, quats=nothing, normalize_quats::Bool=true)
    if (rot_mats === nothing) == (quats === nothing)
        error("Exactly one of rot_mats or quats must be provided")
    end
    if quats !== nothing && normalize_quats
        quats = quats ./ sqrt.(sum(quats .^ 2; dims=1))
    end
    return RotationJL(rot_mats, quats)
end

function rot_matmul_first(a::AbstractArray, b::AbstractArray)
    a11 = _view_first2(a, 1, 1); a12 = _view_first2(a, 1, 2); a13 = _view_first2(a, 1, 3)
    a21 = _view_first2(a, 2, 1); a22 = _view_first2(a, 2, 2); a23 = _view_first2(a, 2, 3)
    a31 = _view_first2(a, 3, 1); a32 = _view_first2(a, 3, 2); a33 = _view_first2(a, 3, 3)

    b11 = _view_first2(b, 1, 1); b12 = _view_first2(b, 1, 2); b13 = _view_first2(b, 1, 3)
    b21 = _view_first2(b, 2, 1); b22 = _view_first2(b, 2, 2); b23 = _view_first2(b, 2, 3)
    b31 = _view_first2(b, 3, 1); b32 = _view_first2(b, 3, 2); b33 = _view_first2(b, 3, 3)

    c11 = a11 .* b11 .+ a12 .* b21 .+ a13 .* b31
    c12 = a11 .* b12 .+ a12 .* b22 .+ a13 .* b32
    c13 = a11 .* b13 .+ a12 .* b23 .+ a13 .* b33

    c21 = a21 .* b11 .+ a22 .* b21 .+ a23 .* b31
    c22 = a21 .* b12 .+ a22 .* b22 .+ a23 .* b32
    c23 = a21 .* b13 .+ a22 .* b23 .+ a23 .* b33

    c31 = a31 .* b11 .+ a32 .* b21 .+ a33 .* b31
    c32 = a31 .* b12 .+ a32 .* b22 .+ a33 .* b32
    c33 = a31 .* b13 .+ a32 .* b23 .+ a33 .* b33

    r1 = cat(
        reshape(c11, 1, 1, size(c11)...),
        reshape(c12, 1, 1, size(c12)...),
        reshape(c13, 1, 1, size(c13)...);
        dims=2,
    )
    r2 = cat(
        reshape(c21, 1, 1, size(c21)...),
        reshape(c22, 1, 1, size(c22)...),
        reshape(c23, 1, 1, size(c23)...);
        dims=2,
    )
    r3 = cat(
        reshape(c31, 1, 1, size(c31)...),
        reshape(c32, 1, 1, size(c32)...),
        reshape(c33, 1, 1, size(c33)...);
        dims=2,
    )
    return cat(r1, r2, r3; dims=1)
end

function rot_vec_mul_first(r::AbstractArray, t::AbstractArray)
    x = _view_first1(t, 1)
    y = _view_first1(t, 2)
    z = _view_first1(t, 3)
    o1 = _view_first2(r, 1, 1) .* x .+ _view_first2(r, 1, 2) .* y .+ _view_first2(r, 1, 3) .* z
    o2 = _view_first2(r, 2, 1) .* x .+ _view_first2(r, 2, 2) .* y .+ _view_first2(r, 2, 3) .* z
    o3 = _view_first2(r, 3, 1) .* x .+ _view_first2(r, 3, 2) .* y .+ _view_first2(r, 3, 3) .* z
    o1 = reshape(o1, 1, size(o1)...)
    o2 = reshape(o2, 1, size(o2)...)
    o3 = reshape(o3, 1, size(o3)...)
    return cat(o1, o2, o3; dims=1)
end

function quat_multiply_first(q1::AbstractArray, q2::AbstractArray)
    a1 = _view_first1(q1, 1); b1 = _view_first1(q1, 2); c1 = _view_first1(q1, 3); d1 = _view_first1(q1, 4)
    a2 = _view_first1(q2, 1); b2 = _view_first1(q2, 2); c2 = _view_first1(q2, 3); d2 = _view_first1(q2, 4)
    w = a1 .* a2 .- b1 .* b2 .- c1 .* c2 .- d1 .* d2
    x = a1 .* b2 .+ b1 .* a2 .+ c1 .* d2 .- d1 .* c2
    y = a1 .* c2 .- b1 .* d2 .+ c1 .* a2 .+ d1 .* b2
    z = a1 .* d2 .+ b1 .* c2 .- c1 .* b2 .+ d1 .* a2
    w = reshape(w, 1, size(w)...)
    x = reshape(x, 1, size(x)...)
    y = reshape(y, 1, size(y)...)
    z = reshape(z, 1, size(z)...)
    return cat(w, x, y, z; dims=1)
end

function quat_multiply_by_vec_first(q::AbstractArray, v::AbstractArray)
    a = _view_first1(q, 1); b = _view_first1(q, 2); c = _view_first1(q, 3); d = _view_first1(q, 4)
    vx = _view_first1(v, 1); vy = _view_first1(v, 2); vz = _view_first1(v, 3)
    w = -b .* vx .- c .* vy .- d .* vz
    x = a .* vx .+ c .* vz .- d .* vy
    y = a .* vy .- b .* vz .+ d .* vx
    z = a .* vz .+ b .* vy .- c .* vx
    w = reshape(w, 1, size(w)...)
    x = reshape(x, 1, size(x)...)
    y = reshape(y, 1, size(y)...)
    z = reshape(z, 1, size(z)...)
    return cat(w, x, y, z; dims=1)
end

function quat_to_rot_first(quat::AbstractArray)
    q = quat ./ sqrt.(sum(quat .^ 2; dims=1))
    w = _view_first1(q, 1); x = _view_first1(q, 2); y = _view_first1(q, 3); z = _view_first1(q, 4)

    ww = w .* w; xx = x .* x; yy = y .* y; zz = z .* z
    wx = w .* x; wy = w .* y; wz = w .* z
    xy = x .* y; xz = x .* z; yz = y .* z

    r11 = 1 .- 2 .* (yy .+ zz)
    r12 = 2 .* (xy .- wz)
    r13 = 2 .* (xz .+ wy)

    r21 = 2 .* (xy .+ wz)
    r22 = 1 .- 2 .* (xx .+ zz)
    r23 = 2 .* (yz .- wx)

    r31 = 2 .* (xz .- wy)
    r32 = 2 .* (yz .+ wx)
    r33 = 1 .- 2 .* (xx .+ yy)

    r1 = cat(
        reshape(r11, 1, 1, size(r11)...),
        reshape(r12, 1, 1, size(r12)...),
        reshape(r13, 1, 1, size(r13)...);
        dims=2,
    )
    r2 = cat(
        reshape(r21, 1, 1, size(r21)...),
        reshape(r22, 1, 1, size(r22)...),
        reshape(r23, 1, 1, size(r23)...);
        dims=2,
    )
    r3 = cat(
        reshape(r31, 1, 1, size(r31)...),
        reshape(r32, 1, 1, size(r32)...),
        reshape(r33, 1, 1, size(r33)...);
        dims=2,
    )
    return cat(r1, r2, r3; dims=1)
end

function rotation_identity_jl(shape::Tuple, like::AbstractArray; fmt::Symbol=:quat)
    if fmt == :rot_mat
        eye = Matrix{eltype(like)}(I, 3, 3)
        eye = reshape(eye, 3, 3, ntuple(_ -> 1, length(shape))...)
        rot = repeat(eye, 1, 1, shape...)
        rot = to_device(rot, like, eltype(like))
        return RotationJL(rot_mats=rot)
    elseif fmt == :quat
        zero = zeros_like(like, eltype(like), 1, shape...)
        one = ones_like(like, eltype(like), 1, shape...)
        q = cat(one, zero, zero, zero; dims=1)
        return RotationJL(quats=q, normalize_quats=false)
    else
        error("Unknown rotation format: $fmt")
    end
end

function get_rot_mats_jl(r::RotationJL)
    if r.rot_mats !== nothing
        return r.rot_mats
    end
    @assert r.quats !== nothing
    return quat_to_rot_first(r.quats)
end

function get_quats_jl(r::RotationJL)
    if r.quats !== nothing
        return r.quats
    end
    @assert r.rot_mats !== nothing
    error("rot_to_quat_first not implemented")
end

function compose_q_update_vec_jl(r::RotationJL, q_update_vec::AbstractArray; normalize_quats::Bool=true)
    q = get_quats_jl(r)
    new_quats = q .+ quat_multiply_by_vec_first(q, q_update_vec)
    if normalize_quats
        new_quats = new_quats ./ sqrt.(sum(new_quats .^ 2; dims=1))
    end
    return RotationJL(quats=new_quats, normalize_quats=false)
end

struct RigidJL
    rots::RotationJL
    trans::AbstractArray
end

function rigid_identity_jl(shape::Tuple, like::AbstractArray; fmt::Symbol=:quat)
    rots = rotation_identity_jl(shape, like; fmt=fmt)
    trans = zeros_like(like, eltype(like), 3, shape...)
    return RigidJL(rots, trans)
end

function apply_rotation_jl(r::RotationJL, pts::AbstractArray)
    rot = get_rot_mats_jl(r)
    return rot_vec_mul_first(rot, pts)
end

function apply_rigid_jl(r::RigidJL, pts::AbstractArray)
    rot = get_rot_mats_jl(r.rots)
    trans = r.trans
    batch_dims = ndims(trans) - 1
    extra = ndims(pts) - (batch_dims + 1)
    extra < 0 && error("apply_rigid_jl: point rank $(ndims(pts)) incompatible with trans rank $(ndims(trans))")
    dims = size(pts)
    m = extra == 0 ? 1 : prod(dims[2:1 + extra])
    x = reshape(pts, 3, m, size(trans)[2:end]...)
    trans_bt = reshape(trans, 3, 1, size(trans)[2:end]...)
    rigid_bt = BT.Translation(trans_bt) ∘ BT.Rotation(rot)
    y = BT.transform(rigid_bt, x)
    return reshape(y, dims)
end

function invert_apply_rigid_jl(r::RigidJL, pts::AbstractArray)
    rot = get_rot_mats_jl(r.rots)
    trans = r.trans
    batch_dims = ndims(trans) - 1
    extra = ndims(pts) - (batch_dims + 1)
    extra < 0 && error("invert_apply_rigid_jl: point rank $(ndims(pts)) incompatible with trans rank $(ndims(trans))")
    dims = size(pts)
    m = extra == 0 ? 1 : prod(dims[2:1 + extra])
    x = reshape(pts, 3, m, size(trans)[2:end]...)
    trans_bt = reshape(trans, 3, 1, size(trans)[2:end]...)
    rigid_bt = BT.Translation(trans_bt) ∘ BT.Rotation(rot)
    y = BT.inverse_transform(rigid_bt, x)
    return reshape(y, dims)
end

function compose_q_update_vec_jl(r::RigidJL, q_update_vec::AbstractArray)
    q_vec = _view_first1(q_update_vec, 1:3)
    t_vec = _view_first1(q_update_vec, 4:6)
    new_rots = compose_q_update_vec_jl(r.rots, q_vec)
    trans_update = apply_rotation_jl(r.rots, t_vec)
    new_trans = r.trans .+ trans_update
    return RigidJL(new_rots, new_trans)
end

function scale_translation_jl(r::RigidJL, factor::Real)
    return RigidJL(r.rots, r.trans .* factor)
end

function to_tensor_7_jl(r::RigidJL)
    q = get_quats_jl(r.rots)
    return cat(q, r.trans; dims=1)
end

function to_tensor_4x4_jl(r::RigidJL)
    rot = get_rot_mats_jl(r.rots)  # (3, 3, batch...)
    trans = r.trans               # (3, batch...)
    trans_row = reshape(trans, 3, 1, size(trans)[2:end]...)
    top = cat(rot, trans_row; dims=2) # (3, 4, batch...)
    bottom_zeros = zeros_like(trans, eltype(trans), 1, 3, size(trans)[2:end]...)
    bottom_ones = ones_like(trans, eltype(trans), 1, 1, size(trans)[2:end]...)
    bottom = cat(bottom_zeros, bottom_ones; dims=2)
    return cat(top, bottom; dims=1)
end

function _align_batch_first(a::AbstractArray, target_batch_dims::Int)
    batch_dims = ndims(a) - 2
    if batch_dims < target_batch_dims
        extra = target_batch_dims - batch_dims
        return reshape(a, size(a, 1), size(a, 2), ntuple(_ -> 1, extra)..., size(a)[3:end]...)
    end
    return a
end

function _align_trans_first(a::AbstractArray, target_batch_dims::Int)
    batch_dims = ndims(a) - 1
    if batch_dims < target_batch_dims
        extra = target_batch_dims - batch_dims
        return reshape(a, size(a, 1), ntuple(_ -> 1, extra)..., size(a)[2:end]...)
    end
    return a
end

function compose_jl(r1::RigidJL, r2::RigidJL)
    rot1 = get_rot_mats_jl(r1.rots)
    rot2 = get_rot_mats_jl(r2.rots)
    batch_dims = max(ndims(rot1) - 2, ndims(rot2) - 2)
    rot1 = _align_batch_first(rot1, batch_dims)
    rot2 = _align_batch_first(rot2, batch_dims)
    trans1 = _align_trans_first(r1.trans, batch_dims)
    trans2 = _align_trans_first(r2.trans, batch_dims)
    new_rot = rot_matmul_first(rot1, rot2)
    new_trans = rot_vec_mul_first(rot1, trans2) .+ trans1
    return RigidJL(RotationJL(rot_mats=new_rot), new_trans)
end

function rigid_index_jl(r::RigidJL, inds...)
    rot = get_rot_mats_jl(r.rots)
    rot = view(rot, :, :, inds...)
    trans = view(r.trans, :, inds...)
    return RigidJL(RotationJL(rot_mats=rot), trans)
end
