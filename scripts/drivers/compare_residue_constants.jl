using Pkg
Pkg.activate("/Users/benmurrell/JuliaM3/juliaESM"; io=devnull)

using NPZ
using Statistics
using ESMEmbed

ref = NPZ.npzread("/Users/benmurrell/JuliaM3/juliaESM/residue_constants_ref.npz")

function diff_stats(a, b)
    max_abs = maximum(abs.(a .- b))
    mean_abs = mean(abs.(a .- b))
    return max_abs, mean_abs
end

function report(name, a, b)
    max_abs, mean_abs = diff_stats(Float32.(a), Float32.(b))
    println(name, " max_abs=", max_abs, " mean_abs=", mean_abs)
end

report(
    "restype_rigid_group_default_frame",
    ESMEmbed.restype_rigid_group_default_frame,
    ref["restype_rigid_group_default_frame"],
)
report(
    "restype_atom14_to_rigid_group",
    ESMEmbed.restype_atom14_to_rigid_group,
    ref["restype_atom14_to_rigid_group"],
)
report(
    "restype_atom14_mask",
    ESMEmbed.restype_atom14_mask,
    ref["restype_atom14_mask"],
)
report(
    "restype_atom14_rigid_group_positions",
    ESMEmbed.restype_atom14_rigid_group_positions,
    ref["restype_atom14_rigid_group_positions"],
)
