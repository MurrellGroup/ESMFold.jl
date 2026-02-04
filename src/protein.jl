using Printf

const PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
const PDB_MAX_CHAINS = length(PDB_CHAIN_IDS)

struct Protein
    atom_positions::AbstractArray
    aatype::AbstractVector{Int}
    atom_mask::AbstractArray
    residue_index::AbstractVector{Int}
    b_factors::AbstractArray
    chain_index::Union{Nothing,AbstractVector{Int}}
    remark::Union{Nothing,String}
    parents::Union{Nothing,Vector{String}}
    parents_chain_index::Union{Nothing,Vector{Int}}
end

function _chain_end(atom_index::Int, end_resname::String, chain_name::Char, residue_index::Int)
    @sprintf("TER   %5d      %3s %1c%4d", atom_index, end_resname, chain_name, residue_index)
end

function get_pdb_headers(prot::Protein, chain_id::Int=0)
    lines = String[]
    prot.remark !== nothing && push!(lines, "REMARK $(prot.remark)")

    parents = prot.parents
    if prot.parents_chain_index !== nothing
        parents = [p for (p, i) in zip(parents, prot.parents_chain_index) if i == chain_id]
    end
    if parents === nothing || isempty(parents)
        parents = ["N/A"]
    end
    push!(lines, "PARENT " * join(parents, " "))
    return lines
end

function to_pdb(prot::Protein)
    restypes_local = vcat(restypes, ["X"])
    res_1to3 = r -> get(restype_1to3, restypes_local[r + 1], "UNK")
    atom_types_local = atom_types

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index
    b_factors = prot.b_factors
    chain_index = prot.chain_index === nothing ? fill(0, length(aatype)) : prot.chain_index

    if any(aatype .> restype_num)
        error("Invalid aatypes.")
    end

    chain_ids = Dict{Int,Char}()
    for i in unique(chain_index)
        if i < 0
            chain_ids[i] = PDB_CHAIN_IDS[end]
            continue
        end
        i >= PDB_MAX_CHAINS && error("The PDB format supports at most $PDB_MAX_CHAINS chains.")
        chain_ids[i] = PDB_CHAIN_IDS[i + 1]
    end

    pdb_lines = String[]
    append!(pdb_lines, get_pdb_headers(prot))
    push!(pdb_lines, "MODEL     1")

    atom_index = 1
    last_chain_index = chain_index[1]
    prev_chain_index = chain_index[1]

    for i in 1:length(aatype)
        if last_chain_index != chain_index[i]
            push!(pdb_lines, _chain_end(atom_index, res_1to3(aatype[i - 1]), chain_ids[chain_index[i - 1]], residue_index[i - 1]))
            last_chain_index = chain_index[i]
            atom_index += 1
        end

        res_name_3 = res_1to3(aatype[i])
        for (atom_name, pos, mask, b_factor) in zip(atom_types_local, eachrow(atom_positions[i, :, :]), atom_mask[i, :], b_factors[i, :])
            mask < 0.5f0 && continue
            record_type = "ATOM"
            name = length(atom_name) == 4 ? atom_name : " " * atom_name
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[1]
            charge = ""
            chain_tag = chain_ids[chain_index[i]]
            atom_line = @sprintf(
                "%-6s%5d %-4s%1s%3s %1c%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2c%2s",
                record_type,
                atom_index,
                name,
                alt_loc,
                res_name_3,
                chain_tag,
                residue_index[i],
                insertion_code,
                pos[1],
                pos[2],
                pos[3],
                occupancy,
                b_factor,
                element,
                charge,
            )
            push!(pdb_lines, atom_line)
            atom_index += 1
        end

        should_terminate = (i == length(aatype))
        if i != length(aatype) && chain_index[i + 1] != prev_chain_index
            should_terminate = true
            prev_chain_index = chain_index[i + 1]
        end

        if should_terminate
            chain_tag = chain_ids[chain_index[i]]
            push!(
                pdb_lines,
                @sprintf("TER   %5d      %3s %1c%4d", atom_index, res_1to3(aatype[i]), chain_tag, residue_index[i]),
            )
            atom_index += 1
            if i != length(aatype)
                append!(pdb_lines, get_pdb_headers(prot, prev_chain_index))
            end
        end
    end

    push!(pdb_lines, "ENDMDL")
    push!(pdb_lines, "END")
    pdb_lines = map(line -> rpad(line, 80), pdb_lines)
    return join(pdb_lines, "\n") * "\n"
end

function output_to_pdb(output::AbstractDict)
    final_atom_positions = atom14_to_atom37(output[:positions][end, :, :, :, :], output)
    cpu_out = Dict{Symbol,Any}()
    for (k, v) in output
        if v isa AbstractArray
            cpu_out[k] = Array(v)
        else
            cpu_out[k] = v
        end
    end
    final_atom_positions = Array(final_atom_positions)
    final_atom_mask = cpu_out[:atom37_atom_exists]

    pdbs = String[]
    for i in 1:size(cpu_out[:aatype], 1)
        aa = cpu_out[:aatype][i, :]
        pred_pos = final_atom_positions[i, :, :, :]
        mask = final_atom_mask[i, :, :]
        resid = cpu_out[:residue_index][i, :] .+ 1
        b_factors = cpu_out[:plddt][i, :, :]
        chain_index = haskey(cpu_out, :chain_index) ? cpu_out[:chain_index][i, :] : nothing
        prot = Protein(
            pred_pos,
            vec(aa),
            mask,
            vec(resid),
            b_factors,
            chain_index === nothing ? nothing : vec(chain_index),
            nothing,
            nothing,
            nothing,
        )
        push!(pdbs, to_pdb(prot))
    end
    return pdbs
end
