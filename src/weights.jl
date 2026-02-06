function _npz_state(path::AbstractString)
    data = NPZ.npzread(path)
    meta_path = path * ".names.json"
    if isfile(meta_path)
        meta = JSON.parsefile(meta_path)
        names = meta["names"]
        state = Dict{String,Any}()
        for (i, name) in enumerate(names)
            key = "arr_$(i)"
            state[name] = data[key]
        end
        return state
    end
    return Dict{String,Any}(data)
end

function _assign!(dest, src)
    dest .= src
    return dest
end

function _assign_transpose!(dest, src)
    dest .= permutedims(src, (2, 1))
    return dest
end

function load_esm2_npz!(model::ESM2, path::AbstractString)
    state = _npz_state(path)

    _assign_transpose!(model.embed_tokens.weight, state["esm.embed_tokens.weight"])

    for i in 0:(model.num_layers - 1)
        layer = model.layers[i + 1]
        prefix = "esm.layers.$i"

        _assign!(layer.self_attn.q_proj.weight, state["$prefix.self_attn.q_proj.weight"])
        _assign!(layer.self_attn.k_proj.weight, state["$prefix.self_attn.k_proj.weight"])
        _assign!(layer.self_attn.v_proj.weight, state["$prefix.self_attn.v_proj.weight"])
        _assign!(layer.self_attn.out_proj.weight, state["$prefix.self_attn.out_proj.weight"])

        _assign!(layer.self_attn.q_proj.bias, state["$prefix.self_attn.q_proj.bias"])
        _assign!(layer.self_attn.k_proj.bias, state["$prefix.self_attn.k_proj.bias"])
        _assign!(layer.self_attn.v_proj.bias, state["$prefix.self_attn.v_proj.bias"])
        _assign!(layer.self_attn.out_proj.bias, state["$prefix.self_attn.out_proj.bias"])

        _assign!(layer.self_attn_layer_norm.w, state["$prefix.self_attn_layer_norm.weight"])
        _assign!(layer.self_attn_layer_norm.b, state["$prefix.self_attn_layer_norm.bias"])

        _assign!(layer.fc1.weight, state["$prefix.fc1.weight"])
        _assign!(layer.fc1.bias, state["$prefix.fc1.bias"])
        _assign!(layer.fc2.weight, state["$prefix.fc2.weight"])
        _assign!(layer.fc2.bias, state["$prefix.fc2.bias"])

        _assign!(layer.final_layer_norm.w, state["$prefix.final_layer_norm.weight"])
        _assign!(layer.final_layer_norm.b, state["$prefix.final_layer_norm.bias"])
    end

    _assign!(model.emb_layer_norm_after.w, state["esm.emb_layer_norm_after.weight"])
    _assign!(model.emb_layer_norm_after.b, state["esm.emb_layer_norm_after.bias"])

    if haskey(state, "esm.lm_head.dense.weight")
        _assign!(model.lm_head.dense.weight, state["esm.lm_head.dense.weight"])
        _assign!(model.lm_head.dense.bias, state["esm.lm_head.dense.bias"])
        _assign!(model.lm_head.layer_norm.w, state["esm.lm_head.layer_norm.weight"])
        _assign!(model.lm_head.layer_norm.b, state["esm.lm_head.layer_norm.bias"])
        _assign!(model.lm_head.bias, state["esm.lm_head.bias"])
        # tie weights
        model.lm_head.weight .= model.embed_tokens.weight
    end

    return model
end

function load_esmfold_safetensors!(model::ESMFoldEmbed, reader::SafeTensors.Reader)
    SafeTensors.read_into!(reader, "af2_to_esm", model.af2_to_esm)
    SafeTensors.read_into!(reader, "esm_s_combine", model.esm_s_combine)

    SafeTensors.read_into!(reader, "esm_s_mlp.0.weight", model.esm_s_mlp.norm.w)
    SafeTensors.read_into!(reader, "esm_s_mlp.0.bias", model.esm_s_mlp.norm.b)
    SafeTensors.read_into!(reader, "esm_s_mlp.1.weight", model.esm_s_mlp.fc1.weight)
    SafeTensors.read_into!(reader, "esm_s_mlp.1.bias", model.esm_s_mlp.fc1.bias)
    SafeTensors.read_into!(reader, "esm_s_mlp.3.weight", model.esm_s_mlp.fc2.weight)
    SafeTensors.read_into!(reader, "esm_s_mlp.3.bias", model.esm_s_mlp.fc2.bias)

    if model.esm_z_mlp !== nothing
        SafeTensors.read_into!(reader, "esm_z_mlp.0.weight", model.esm_z_mlp.norm.w)
        SafeTensors.read_into!(reader, "esm_z_mlp.0.bias", model.esm_z_mlp.norm.b)
        SafeTensors.read_into!(reader, "esm_z_mlp.1.weight", model.esm_z_mlp.fc1.weight)
        SafeTensors.read_into!(reader, "esm_z_mlp.1.bias", model.esm_z_mlp.fc1.bias)
        SafeTensors.read_into!(reader, "esm_z_mlp.3.weight", model.esm_z_mlp.fc2.weight)
        SafeTensors.read_into!(reader, "esm_z_mlp.3.bias", model.esm_z_mlp.fc2.bias)
    end

    # embedding.weight in checkpoint is (n_tokens, c_s); Flux expects (c_s, n_tokens)
    permutedims!(
        model.embedding.weight,
        SafeTensors.read_tensor(reader, "embedding.weight"),
        (2, 1),
    )

    # ESM2 weights
    # word_embeddings in checkpoint is (vocab, dim); Flux expects (dim, vocab)
    permutedims!(
        model.esm.embed_tokens.weight,
        SafeTensors.read_tensor(reader, "esm.embeddings.word_embeddings.weight"),
        (2, 1),
    )
    SafeTensors.read_into!(reader, "esm.encoder.emb_layer_norm_after.weight", model.esm.emb_layer_norm_after.w)
    SafeTensors.read_into!(reader, "esm.encoder.emb_layer_norm_after.bias", model.esm.emb_layer_norm_after.b)

    for i in 0:(model.esm.num_layers - 1)
        layer = model.esm.layers[i + 1]
        prefix = "esm.encoder.layer.$i"

        SafeTensors.read_into!(reader, "$prefix.attention.self.query.weight", layer.self_attn.q_proj.weight)
        SafeTensors.read_into!(reader, "$prefix.attention.self.query.bias", layer.self_attn.q_proj.bias)
        SafeTensors.read_into!(reader, "$prefix.attention.self.key.weight", layer.self_attn.k_proj.weight)
        SafeTensors.read_into!(reader, "$prefix.attention.self.key.bias", layer.self_attn.k_proj.bias)
        SafeTensors.read_into!(reader, "$prefix.attention.self.value.weight", layer.self_attn.v_proj.weight)
        SafeTensors.read_into!(reader, "$prefix.attention.self.value.bias", layer.self_attn.v_proj.bias)
        SafeTensors.read_into!(reader, "$prefix.attention.output.dense.weight", layer.self_attn.out_proj.weight)
        SafeTensors.read_into!(reader, "$prefix.attention.output.dense.bias", layer.self_attn.out_proj.bias)

        SafeTensors.read_into!(reader, "$prefix.attention.LayerNorm.weight", layer.self_attn_layer_norm.w)
        SafeTensors.read_into!(reader, "$prefix.attention.LayerNorm.bias", layer.self_attn_layer_norm.b)

        SafeTensors.read_into!(reader, "$prefix.intermediate.dense.weight", layer.fc1.weight)
        SafeTensors.read_into!(reader, "$prefix.intermediate.dense.bias", layer.fc1.bias)
        SafeTensors.read_into!(reader, "$prefix.output.dense.weight", layer.fc2.weight)
        SafeTensors.read_into!(reader, "$prefix.output.dense.bias", layer.fc2.bias)

        SafeTensors.read_into!(reader, "$prefix.LayerNorm.weight", layer.final_layer_norm.w)
        SafeTensors.read_into!(reader, "$prefix.LayerNorm.bias", layer.final_layer_norm.b)
    end

    return model
end

function load_esmfold_safetensors!(model::ESMFoldEmbed, path::AbstractString)
    reader = SafeTensors.Reader(path)
    return load_esmfold_safetensors!(model, reader)
end

function _load_layernorm!(reader::SafeTensors.Reader, prefix::String, ln::LayerNormFirst)
    SafeTensors.read_into!(reader, "$prefix.weight", ln.w)
    SafeTensors.read_into!(reader, "$prefix.bias", ln.b)
end

function _load_linear!(reader::SafeTensors.Reader, prefix::String, lin::LinearFirst)
    SafeTensors.read_into!(reader, "$prefix.weight", lin.weight)
    if lin.use_bias
        SafeTensors.read_into!(reader, "$prefix.bias", lin.bias)
    end
end

function _load_embedding_weight!(reader::SafeTensors.Reader, name::String, emb)
    permutedims!(emb.weight, SafeTensors.read_tensor(reader, name), (2, 1))
end

function _load_residue_mlp!(reader::SafeTensors.Reader, prefix::String, mlp::ResidueMLP)
    _load_layernorm!(reader, "$prefix.mlp.0", mlp.norm)
    _load_linear!(reader, "$prefix.mlp.1", mlp.fc1)
    _load_linear!(reader, "$prefix.mlp.3", mlp.fc2)
end

function _load_triangle_mul!(reader::SafeTensors.Reader, prefix::String, mul::TriangleMultiplicativeUpdate)
    _load_layernorm!(reader, "$prefix.layer_norm_in", mul.layer_norm_in)
    _load_layernorm!(reader, "$prefix.layer_norm_out", mul.layer_norm_out)
    _load_linear!(reader, "$prefix.linear_a_p", mul.linear_a_p)
    _load_linear!(reader, "$prefix.linear_a_g", mul.linear_a_g)
    _load_linear!(reader, "$prefix.linear_b_p", mul.linear_b_p)
    _load_linear!(reader, "$prefix.linear_b_g", mul.linear_b_g)
    _load_linear!(reader, "$prefix.linear_g", mul.linear_g)
    _load_linear!(reader, "$prefix.linear_z", mul.linear_z)
end

function _load_of_mha!(reader::SafeTensors.Reader, prefix::String, mha::OFMultiheadAttention)
    _load_linear!(reader, "$prefix.linear_q", mha.linear_q)
    _load_linear!(reader, "$prefix.linear_k", mha.linear_k)
    _load_linear!(reader, "$prefix.linear_v", mha.linear_v)
    _load_linear!(reader, "$prefix.linear_o", mha.linear_o)
    mha.linear_g !== nothing && _load_linear!(reader, "$prefix.linear_g", mha.linear_g)
end

function _load_triangle_attention!(reader::SafeTensors.Reader, prefix::String, attn::TriangleAttention)
    _load_layernorm!(reader, "$prefix.layer_norm", attn.layer_norm)
    _load_linear!(reader, "$prefix.linear", attn.linear)
    _load_of_mha!(reader, "$prefix.mha", attn.mha)
end

function _load_structure_module!(reader::SafeTensors.Reader, prefix::String, sm::StructureModule)
    _load_layernorm!(reader, "$prefix.layer_norm_s", sm.layer_norm_s)
    _load_layernorm!(reader, "$prefix.layer_norm_z", sm.layer_norm_z)
    _load_linear!(reader, "$prefix.linear_in", sm.linear_in)

    ipa = sm.ipa
    _load_linear!(reader, "$prefix.ipa.linear_q", ipa.linear_q)
    _load_linear!(reader, "$prefix.ipa.linear_q_points", ipa.linear_q_points.linear)
    _load_linear!(reader, "$prefix.ipa.linear_kv", ipa.linear_kv)
    _load_linear!(reader, "$prefix.ipa.linear_kv_points", ipa.linear_kv_points.linear)
    _load_linear!(reader, "$prefix.ipa.linear_b", ipa.linear_b)
    _load_linear!(reader, "$prefix.ipa.linear_out", ipa.linear_out)
    SafeTensors.read_into!(reader, "$prefix.ipa.head_weights", ipa.head_weights)

    _load_layernorm!(reader, "$prefix.layer_norm_ipa", sm.layer_norm_ipa)

    _load_layernorm!(reader, "$prefix.transition.layer_norm", sm.transition.layer_norm)
    for (i, layer) in enumerate(sm.transition.layers)
        lp = "$prefix.transition.layers.$(i - 1)"
        _load_linear!(reader, "$lp.linear_1", layer.linear_1)
        _load_linear!(reader, "$lp.linear_2", layer.linear_2)
        _load_linear!(reader, "$lp.linear_3", layer.linear_3)
    end

    _load_linear!(reader, "$prefix.bb_update.linear", sm.bb_update.linear)

    ar = sm.angle_resnet
    _load_linear!(reader, "$prefix.angle_resnet.linear_in", ar.linear_in)
    _load_linear!(reader, "$prefix.angle_resnet.linear_initial", ar.linear_initial)
    _load_linear!(reader, "$prefix.angle_resnet.linear_out", ar.linear_out)
    for (i, layer) in enumerate(ar.layers)
        lp = "$prefix.angle_resnet.layers.$(i - 1)"
        _load_linear!(reader, "$lp.linear_1", layer.linear_1)
        _load_linear!(reader, "$lp.linear_2", layer.linear_2)
    end
end

function load_esmfold_safetensors!(model::ESMFoldModel, reader::SafeTensors.Reader)
    load_esmfold_safetensors!(model.embed, reader)

    _load_embedding_weight!(reader, "trunk.pairwise_positional_embedding.embedding.weight", model.trunk.pairwise_positional_embedding.embedding)

    _load_layernorm!(reader, "trunk.recycle_s_norm", model.trunk.recycle_s_norm)
    _load_layernorm!(reader, "trunk.recycle_z_norm", model.trunk.recycle_z_norm)
    permutedims!(model.trunk.recycle_disto.weight, SafeTensors.read_tensor(reader, "trunk.recycle_disto.weight"), (2, 1))

    _load_linear!(reader, "trunk.trunk2sm_s", model.trunk.trunk2sm_s)
    _load_linear!(reader, "trunk.trunk2sm_z", model.trunk.trunk2sm_z)

    for (i, block) in enumerate(model.trunk.blocks)
        prefix = "trunk.blocks.$(i - 1)"
        _load_layernorm!(reader, "$prefix.layernorm_1", block.layernorm_1)

        _load_layernorm!(reader, "$prefix.sequence_to_pair.layernorm", block.sequence_to_pair.layernorm)
        _load_linear!(reader, "$prefix.sequence_to_pair.proj", block.sequence_to_pair.proj)
        _load_linear!(reader, "$prefix.sequence_to_pair.o_proj", block.sequence_to_pair.o_proj)

        _load_layernorm!(reader, "$prefix.pair_to_sequence.layernorm", block.pair_to_sequence.layernorm)
        _load_linear!(reader, "$prefix.pair_to_sequence.linear", block.pair_to_sequence.linear)

        _load_linear!(reader, "$prefix.seq_attention.proj", block.seq_attention.proj)
        _load_linear!(reader, "$prefix.seq_attention.o_proj", block.seq_attention.o_proj)
        block.seq_attention.g_proj !== nothing && _load_linear!(reader, "$prefix.seq_attention.g_proj", block.seq_attention.g_proj)

        _load_triangle_mul!(reader, "$prefix.tri_mul_out", block.tri_mul_out.inner)
        _load_triangle_mul!(reader, "$prefix.tri_mul_in", block.tri_mul_in.inner)

        _load_triangle_attention!(reader, "$prefix.tri_att_start", block.tri_att_start)
        _load_triangle_attention!(reader, "$prefix.tri_att_end", block.tri_att_end)

        _load_residue_mlp!(reader, "$prefix.mlp_seq", block.mlp_seq)
        _load_residue_mlp!(reader, "$prefix.mlp_pair", block.mlp_pair)
    end

    _load_structure_module!(reader, "trunk.structure_module", model.trunk.structure_module)

    _load_linear!(reader, "distogram_head", model.distogram_head)
    _load_linear!(reader, "ptm_head", model.ptm_head)
    _load_linear!(reader, "lm_head", model.lm_head)

    _load_layernorm!(reader, "lddt_head.0", model.lddt_head.norm)
    _load_linear!(reader, "lddt_head.1", model.lddt_head.linear_1)
    _load_linear!(reader, "lddt_head.2", model.lddt_head.linear_2)
    _load_linear!(reader, "lddt_head.3", model.lddt_head.linear_3)

    return model
end

function load_esmfold_safetensors!(model::ESMFoldModel, path::AbstractString)
    reader = SafeTensors.Reader(path)
    return load_esmfold_safetensors!(model, reader)
end

function _infer_esmfold_config(reader::SafeTensors.Reader)
    header_keys = collect(keys(reader.header))

    layer_ids = Int[]
    for key in header_keys
        startswith(key, "esm.encoder.layer.") || continue
        parts = split(key, '.')
        length(parts) < 4 && continue
        idx = tryparse(Int, parts[4])
        idx === nothing || push!(layer_ids, idx)
    end
    isempty(layer_ids) && error("Unable to infer num_layers from safetensors header.")
    num_layers = maximum(layer_ids) + 1

    embed_entry = reader.header["esm.embeddings.word_embeddings.weight"]
    embed_dim = embed_entry.shape[2]

    inv_entry = reader.header["esm.encoder.layer.0.attention.self.rotary_embeddings.inv_freq"]
    head_dim = inv_entry.shape[1] * 2
    attention_heads = embed_dim ÷ head_dim

    cs_entry = reader.header["esm_s_mlp.1.weight"]
    c_s = cs_entry.shape[1]

    c_z = 1
    if haskey(reader.header, "esm_z_mlp.1.weight")
        cz_entry = reader.header["esm_z_mlp.1.weight"]
        c_z = cz_entry.shape[1]
    end

    return num_layers, embed_dim, attention_heads, c_s, c_z
end

function _infer_esmfold_full_config(reader::SafeTensors.Reader)
    num_layers, embed_dim, attention_heads, c_s, c_z = _infer_esmfold_config(reader)

    block_ids = Int[]
    for key in keys(reader.header)
        startswith(key, "trunk.blocks.") || continue
        parts = split(key, '.')
        length(parts) < 3 && continue
        idx = tryparse(Int, parts[3])
        idx === nothing || push!(block_ids, idx)
    end
    num_blocks = isempty(block_ids) ? 0 : maximum(block_ids) + 1

    ln_entry = reader.header["trunk.blocks.0.layernorm_1.weight"]
    sequence_state_dim = ln_entry.shape[1]
    z_entry = reader.header["trunk.blocks.0.tri_att_start.layer_norm.weight"]
    pairwise_state_dim = z_entry.shape[1]

    seq_heads_entry = reader.header["trunk.blocks.0.pair_to_sequence.linear.weight"]
    sequence_num_heads = seq_heads_entry.shape[1]
    sequence_head_width = sequence_state_dim ÷ sequence_num_heads

    pair_heads_entry = reader.header["trunk.blocks.0.tri_att_start.linear.weight"]
    pairwise_num_heads = pair_heads_entry.shape[1]
    pairwise_head_width = pairwise_state_dim ÷ pairwise_num_heads

    pos_entry = reader.header["trunk.pairwise_positional_embedding.embedding.weight"]
    position_bins = (pos_entry.shape[1] - 2) ÷ 2

    lddt_entry = reader.header["lddt_head.1.weight"]
    lddt_head_hid_dim = lddt_entry.shape[1]

    return (
        num_layers,
        embed_dim,
        attention_heads,
        sequence_state_dim,
        pairwise_state_dim,
        sequence_head_width,
        pairwise_head_width,
        position_bins,
        num_blocks,
        lddt_head_hid_dim,
    )
end

function load_ESM(;
    repo_id::AbstractString = "facebook/esmfold_v1",
    filename::AbstractString = "model.safetensors",
    revision::AbstractString = "ba837a3",
    cache::Bool = true,
    local_files_only::Bool = false,
    use_esm_attn_map::Bool = false,
)
    path = hf_hub_download(
        repo_id,
        filename;
        revision = revision,
        cache = cache,
        local_files_only = local_files_only,
    )

    reader = SafeTensors.Reader(path)
    num_layers, embed_dim, attention_heads, c_s, c_z = _infer_esmfold_config(reader)

    use_esm_attn_map && !haskey(reader.header, "esm_z_mlp.1.weight") &&
        error("use_esm_attn_map=true but esm_z_mlp weights are not present in the checkpoint.")

    alphabet = Alphabet_from_architecture("ESM-1b")
    esm = ESM2(
        num_layers,
        embed_dim,
        attention_heads;
        alphabet = alphabet,
        token_dropout = true,
    )
    model = ESMFoldEmbed(
        esm;
        c_s = c_s,
        c_z = c_z,
        use_esm_attn_map = use_esm_attn_map,
    )
    load_esmfold_safetensors!(model, reader)
    return model
end

function load_ESMFold(;
    repo_id::AbstractString = "facebook/esmfold_v1",
    filename::AbstractString = "model.safetensors",
    revision::AbstractString = "ba837a3",
    cache::Bool = true,
    local_files_only::Bool = false,
    use_esm_attn_map::Bool = false,
)
    path = hf_hub_download(
        repo_id,
        filename;
        revision = revision,
        cache = cache,
        local_files_only = local_files_only,
    )

    reader = SafeTensors.Reader(path)
    (
        num_layers,
        embed_dim,
        attention_heads,
        c_s,
        c_z,
        sequence_head_width,
        pairwise_head_width,
        position_bins,
        num_blocks,
        lddt_head_hid_dim,
    ) = _infer_esmfold_full_config(reader)

    use_esm_attn_map && !haskey(reader.header, "esm_z_mlp.1.weight") &&
        error("use_esm_attn_map=true but esm_z_mlp weights are not present in the checkpoint.")

    alphabet = Alphabet_from_architecture("ESM-1b")
    esm = ESM2(
        num_layers,
        embed_dim,
        attention_heads;
        alphabet = alphabet,
        token_dropout = true,
    )

    trunk_cfg = FoldingTrunkConfig(
        num_blocks,
        c_s,
        c_z,
        sequence_head_width,
        pairwise_head_width,
        position_bins,
        0f0,
        0f0,
        false,
        4,
        nothing,
        StructureModuleConfig(),
    )
    cfg = ESMFoldConfig(; trunk=trunk_cfg, lddt_head_hid_dim=lddt_head_hid_dim, use_esm_attn_map=use_esm_attn_map)

    model = ESMFoldModel(esm; cfg=cfg)
    load_esmfold_safetensors!(model, reader)
    return model
end
