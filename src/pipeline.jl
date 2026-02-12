"""
    prepare_inputs(model::ESMFoldModel, sequences; kw...) → NamedTuple

Encode protein sequences and transfer to model device.

Returns a NamedTuple with fields:
- `aa`: amino acid indices, `(L, B)` layout, on model device
- `mask`: padding mask, `(L, B)` layout, on model device
- `residx`: residue indices, `(L, B)` layout, on model device
- `linker_mask`: chain linker mask, `(B, L)` layout, on model device
- `chain_index`: chain indices, `(B, L)` layout, on CPU
- `masking_pattern`: optional masking pattern, `(L, B)` layout or `nothing`

# Example
```julia
inputs = prepare_inputs(model, "MKQLLED...")
inputs = prepare_inputs(model, ["SEQ1", "SEQ2"])
```
"""
function prepare_inputs(
    model::ESMFoldModel,
    sequences::Union{AbstractString,AbstractVector{<:AbstractString}};
    residx = nothing,
    masking_pattern = nothing,
    residue_index_offset::Int = 512,
    chain_linker::Union{AbstractString,Int} = "G"^25,
)
    seqs = isa(sequences, AbstractString) ? [sequences] : sequences

    aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(
        seqs;
        residue_index_offset = residue_index_offset,
        chain_linker = chain_linker,
    )

    if residx === nothing
        residx = _residx
    elseif !isa(residx, AbstractArray)
        residx = collate_dense_tensors(residx)
    end

    like = model.embed.esm.embed_tokens.weight
    aatype = to_device(aatype, like, Int)
    mask = to_device(mask, like, eltype(like))
    residx = to_device(residx, like, Int)
    linker_mask = to_device(linker_mask, like, eltype(like))
    if masking_pattern !== nothing
        masking_pattern = to_device(masking_pattern, like, eltype(aatype))
    end

    return (
        aa = permutedims(aatype, (2, 1)),
        mask = permutedims(mask, (2, 1)),
        residx = permutedims(residx, (2, 1)),
        linker_mask = linker_mask,
        chain_index = chain_index,
        masking_pattern = masking_pattern === nothing ? nothing : permutedims(masking_pattern, (2, 1)),
    )
end

"""
    run_esm2(model::ESMFoldModel, inputs; repr_layers, need_head_weights) → ESM2Output

Run the ESM2 language model on prepared inputs.

Performs AF2→ESM index conversion, BOS/EOS wrapping, and calls the ESM2 model directly.
Uses the fast in-place forward path. For AD-compatible forward, use [`esm2_forward_ad`](@ref) instead.

# Arguments
- `model::ESMFoldModel`: The ESMFold model.
- `inputs::NamedTuple`: From [`prepare_inputs`](@ref).
- `repr_layers`: Layer indices for hidden representations (default: `[model.embed.esm.num_layers]`).
- `need_head_weights`: Return attention weights (default: `false`).

# Returns
`ESM2Output` with `logits`, `representations` Dict, `attentions`. All representations include
BOS/EOS tokens (strip with `[:, 2:end-1, :]` if needed).
"""
function run_esm2(
    model::ESMFoldModel,
    inputs::NamedTuple;
    repr_layers::AbstractVector{Int} = [model.embed.esm.num_layers],
    need_head_weights::Bool = false,
)
    embed = model.embed

    # inputs.aa is (L, B); embed helpers expect (B, L)
    aa_bl = permutedims(inputs.aa, (2, 1))
    mask_bl = permutedims(inputs.mask, (2, 1))

    # Convert AF2 indices → ESM indices
    esmaa = _af2_idx_to_esm_idx(embed, aa_bl, mask_bl)

    # Apply masking pattern if present
    if inputs.masking_pattern !== nothing
        masking_pattern_bl = permutedims(inputs.masking_pattern, (2, 1))
        esmaa = _mask_inputs_to_esm(embed, esmaa, masking_pattern_bl)
    end

    # BOS/EOS wrapping (replicates _compute_language_model_representations logic)
    bosi = embed.esm_dict.cls_idx
    eosi = embed.esm_dict.eos_idx
    pad = embed.esm_dict.padding_idx
    batch_size = size(esmaa, 1)

    bos = to_device(fill(bosi, batch_size, 1), esmaa, eltype(esmaa))
    eos = to_device(fill(pad, batch_size, 1), esmaa, eltype(esmaa))
    esmaa2 = hcat(bos, esmaa, eos)

    lengths = sum(esmaa2 .!= pad, dims=2)
    positions = to_device(reshape(1:size(esmaa2, 2), 1, :), esmaa2, Int)
    eos_mask = positions .== (lengths .+ 1)
    esmaa2 = ifelse.(eos_mask, eosi, esmaa2)

    return embed.esm(
        esmaa2;
        repr_layers = repr_layers,
        need_head_weights = need_head_weights,
    )
end

"""
    run_embedding(model::ESMFoldModel, inputs) → NamedTuple

Run ESM2 + projection to get trunk-dimension embeddings.

Returns a NamedTuple with:
- `s_s_0`: sequence state, `(c_s, L, B)`
- `s_z_0`: pairwise state, `(c_z, L, L, B)`

# Example
```julia
inputs = prepare_inputs(model, "MKQLLED...")
emb = run_embedding(model, inputs)
emb.s_s_0  # (1024, L, B) sequence embedding
emb.s_z_0  # (128, L, L, B) pairwise embedding
```
"""
function run_embedding(model::ESMFoldModel, inputs::NamedTuple)
    # embed expects (B, L) layout
    aa_bl = permutedims(inputs.aa, (2, 1))
    mask_bl = permutedims(inputs.mask, (2, 1))
    masking_pattern_bl = inputs.masking_pattern === nothing ? nothing : permutedims(inputs.masking_pattern, (2, 1))

    embed_out = model.embed(
        aa_bl;
        mask = mask_bl,
        masking_pattern = masking_pattern_bl,
        return_pair = model.cfg.use_esm_attn_map,
    )

    if model.cfg.use_esm_attn_map
        s_s_0 = embed_out.sequence
        s_z_0 = embed_out.pair
    else
        s_s_0 = embed_out
        s_z_0 = nothing
    end

    s_z_0 = if s_z_0 === nothing
        zeros_like(
            s_s_0,
            model.cfg.trunk.pairwise_state_dim,
            size(s_s_0, 2),
            size(s_s_0, 2),
            size(s_s_0, 3),
        )
    else
        s_z_0
    end

    return (s_s_0 = s_s_0, s_z_0 = s_z_0)
end

"""
    run_trunk(model::ESMFoldModel, s_s_0, s_z_0, inputs; num_recycles) → Dict

Run the full folding trunk: all recycle iterations + structure module.

# Arguments
- `s_s_0`: initial sequence state `(c_s, L, B)` from [`run_embedding`](@ref)
- `s_z_0`: initial pairwise state `(c_z, L, L, B)` from [`run_embedding`](@ref)
- `inputs`: from [`prepare_inputs`](@ref)
- `num_recycles`: number of recycling iterations (default: `model.cfg.trunk.max_recycles`)

# Returns
Dict with `:s_s`, `:s_z`, `:positions`, `:states`, `:single`, `:frames`, etc.
"""
function run_trunk(
    model::ESMFoldModel,
    s_s_0::AbstractArray,
    s_z_0::AbstractArray,
    inputs::NamedTuple;
    num_recycles = nothing,
)
    return model.trunk(
        s_s_0,
        s_z_0,
        inputs.aa,
        inputs.residx,
        inputs.mask;
        no_recycles = num_recycles,
    )
end

"""
    run_trunk_single_pass(model::ESMFoldModel, s_s, s_z, inputs) → NamedTuple

Run one trunk iteration: pairwise positional embedding + all blocks. No structure module,
no recycling.

Returns `(s_s, s_z)` — updated sequence and pairwise states.

# Example
```julia
emb = run_embedding(model, inputs)
result = run_trunk_single_pass(model, emb.s_s_0, emb.s_z_0, inputs)
result.s_s  # updated sequence state
result.s_z  # updated pairwise state
```
"""
function run_trunk_single_pass(
    model::ESMFoldModel,
    s_s::AbstractArray,
    s_z::AbstractArray,
    inputs::NamedTuple,
)
    trunk = model.trunk
    mask = inputs.mask
    residx = inputs.residx

    # Add pairwise positional embedding
    z = s_z .+ trunk.pairwise_positional_embedding(residx, mask=mask)

    # Optimize mask: if fully unmasked, pass nothing to blocks
    block_mask = mask
    if mask !== nothing
        mask_total = sum(mask)
        mask_sum = mask_total isa AbstractArray ? Int(Array(mask_total)[]) : Int(mask_total)
        if mask_sum == length(mask)
            block_mask = nothing
        end
    end

    s = s_s
    for block in trunk.blocks
        s, z = block(s, z; mask=block_mask, residue_index=residx, chunk_size=trunk.chunk_size)
    end

    return (s_s = s, s_z = z)
end

"""
    run_structure_module(model::ESMFoldModel, s_s, s_z, inputs) → Dict

Run the structure module on trunk outputs.

Applies trunk→SM projections (`trunk2sm_s`, `trunk2sm_z`) then calls the structure module.
Also stores `s_s` and `s_z` in the returned Dict for downstream use by [`run_heads`](@ref).

# Arguments
- `s_s`: sequence state `(c_s, L, B)`, e.g. from [`run_trunk_single_pass`](@ref)
- `s_z`: pairwise state `(c_z, L, L, B)`
- `inputs`: from [`prepare_inputs`](@ref)

# Returns
Dict with `:positions`, `:states`, `:single`, `:frames`, `:s_s`, `:s_z`, etc.
"""
function run_structure_module(
    model::ESMFoldModel,
    s_s::AbstractArray,
    s_z::AbstractArray,
    inputs::NamedTuple,
)
    trunk = model.trunk
    mask_f = inputs.mask === nothing ? nothing : convert.(eltype(s_s), inputs.mask)

    sm_input = Dict(:single => trunk.trunk2sm_s(s_s), :pair => trunk.trunk2sm_z(s_z))
    structure = trunk.structure_module(sm_input, inputs.aa, mask_f)
    structure[:s_s] = s_s
    structure[:s_z] = s_z
    return structure
end

"""
    run_heads(model::ESMFoldModel, structure, inputs) → Dict

Run all output heads: distogram, PTM, lDDT, LM.

Takes a structure Dict (from [`run_trunk`](@ref) or [`run_structure_module`](@ref)) and returns
a new Dict augmented with head outputs and confidence metrics. The input Dict is not mutated.

# Returns
Dict with all structure fields plus:
`:distogram_logits`, `:lm_logits`, `:aatype`, `:residue_index`,
`:lddt_head`, `:plddt`, `:ptm_logits`, `:ptm`,
`:aligned_confidence_probs`, `:predicted_aligned_error`, `:max_predicted_aligned_error`
"""
function run_heads(model::ESMFoldModel, structure::AbstractDict, inputs::NamedTuple)
    output = copy(structure)
    mask = inputs.mask

    # Distogram head
    disto_logits = model.distogram_head(output[:s_z])
    disto_logits = (disto_logits .+ permutedims(disto_logits, (1, 3, 2, 4))) ./ 2
    output[:distogram_logits] = disto_logits

    # LM head
    output[:lm_logits] = model.lm_head(output[:s_s])
    output[:aatype] = inputs.aa

    # Atom masks
    make_atom14_masks!(output)
    for k in (:atom14_atom_exists, :atom37_atom_exists)
        output[k] .*= reshape(mask, 1, size(mask, 1), size(mask, 2))
    end
    output[:residue_index] = inputs.residx

    # lDDT head
    states = output[:states]
    states_cfirst = permutedims(states, (2, 1, 3, 4))
    lddt_logits = model.lddt_head(states_cfirst)
    lddt_tmp = _reshape_first_corder(lddt_logits, NUM_ATOM_TYPES, model.lddt_bins)
    lddt_head = permutedims(lddt_tmp, (3, 4, 5, 2, 1))
    output[:lddt_head] = lddt_head

    plddt = categorical_lddt(lddt_head[end, :, :, :, :], bins=model.lddt_bins)
    output[:plddt] = 100f0 .* plddt

    # PTM head
    ptm_logits = model.ptm_head(output[:s_z])
    output[:ptm_logits] = ptm_logits

    seqlen_cpu = Array(sum(mask .== 1; dims=1))
    ptm_logits_cpu = Array(ptm_logits)
    ptm_vals = Vector{eltype(ptm_logits_cpu)}(undef, size(ptm_logits_cpu, 4))
    for b in 1:size(ptm_logits_cpu, 4)
        sl = Int(seqlen_cpu[1, b])
        ptm_vals[b] = compute_tm(ptm_logits_cpu[:, 1:sl, 1:sl, b]; max_bin=31, no_bins=model.distogram_bins)
    end
    output[:ptm] = to_device(reshape(collect(ptm_vals), size(ptm_logits, 4)), ptm_logits, eltype(ptm_logits))

    # Predicted aligned error
    structure_update = compute_predicted_aligned_error(ptm_logits; max_bin=31, no_bins=model.distogram_bins)
    for (k, v) in structure_update
        output[k] = v
    end

    return output
end

"""
    run_pipeline(model::ESMFoldModel, sequences; kw...) → Dict

Full inference pipeline via composable stages. Produces output identical to `infer(model, sequences; kw...)`.

# Stages
1. [`prepare_inputs`](@ref) — encode sequences, transfer to device
2. [`run_embedding`](@ref) — ESM2 + projection to trunk dimensions
3. [`run_trunk`](@ref) — full trunk with recycling + structure module
4. [`run_heads`](@ref) — distogram, PTM, lDDT, LM heads
5. Post-processing — linker mask, mean_plddt, chain_index

# Example
```julia
# Equivalent to infer(model, "ELLKKLLEELKG")
output = run_pipeline(model, "ELLKKLLEELKG")
output[:plddt]        # per-residue confidence
output[:mean_plddt]   # mean confidence
```
"""
function run_pipeline(
    model::ESMFoldModel,
    sequences::Union{AbstractString,AbstractVector{<:AbstractString}};
    residx = nothing,
    masking_pattern = nothing,
    num_recycles = nothing,
    residue_index_offset::Int = 512,
    chain_linker::Union{AbstractString,Int} = "G"^25,
)
    inputs = prepare_inputs(model, sequences;
        residx = residx,
        masking_pattern = masking_pattern,
        residue_index_offset = residue_index_offset,
        chain_linker = chain_linker,
    )

    emb = run_embedding(model, inputs)

    structure = run_trunk(model, emb.s_s_0, emb.s_z_0, inputs; num_recycles = num_recycles)

    output = run_heads(model, structure, inputs)

    # Post-processing (identical to infer())
    linker_mask = inputs.linker_mask
    output[:atom37_atom_exists] .*= reshape(permutedims(linker_mask, (2, 1)), 1, size(linker_mask, 2), size(linker_mask, 1))

    atom37 = permutedims(output[:atom37_atom_exists], (2, 3, 1))
    weighted_plddt = output[:plddt] .* atom37
    numerator = sum(weighted_plddt; dims=(1, 3))
    denom = sum(atom37; dims=(1, 3))
    output[:mean_plddt] = numerator ./ denom
    output[:chain_index] = permutedims(inputs.chain_index, (2, 1))

    return output
end
