def count_gpt_parameters(cfg):
    V = cfg["vocab_size"]
    C = cfg["context_length"]
    d = cfg["emb_dim"]
    L = cfg["n_layers"]
    qkv_bias = cfg["qkv_bias"]

    token_emb = V * d
    pos_emb = C * d

    qkv = 3 * d * d
    if qkv_bias:
        qkv += 3 * d

    attn_out = d * d + d

    layer_norms = 4 * d

    ffn = (
        d * (4 * d) + 4 * d
        + (4 * d) * d + d
    )

    block_params = qkv + attn_out + layer_norms + ffn

    final_ln = 2 * d
    lm_head = d * V

    total_params = (
        token_emb
        + pos_emb
        + L * block_params
        + final_ln
        + lm_head
    )

    tied_params = total_params - lm_head

    return [total_params, tied_params]