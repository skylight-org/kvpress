# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def search_hyperplane(X, max_iter: int = 1000):
    """
    Given a tensor X of shape (bsz, seq_len, head_dim), search for a hyperplane Y (bsz, head_dim)
    such that for every i, <X[:, i], Y> <= 0. Returns - 1e5 * Y / ||Y|| ** 2 to ensure exp(<X, Y>) = 0
    Raises a ValueError if no such hyperplane is found

    Parameters
    ----------
    X : torch.Tensor
        Query tensor with shape (batch_size, seq_len, head_dim) representing
        the query vectors for which we want to find a nullifying hyperplane.
    max_iter : int, default=1000
        Maximum number of iterations to search for the hyperplane. If no valid
        hyperplane is found within this limit, a ValueError is raised.

    Returns
    -------
    torch.Tensor
        Hyperplane tensor with shape (batch_size, head_dim) scaled by -1e5 / ||Y||²
        to ensure that exp(<X, Y>) ≈ 0 for all queries in X.

    Raises
    ------
    ValueError
        If no valid hyperplane is found within max_iter iterations.
    """
    Y = X.mean(1)  # this initialization is enough for most cases
    for _ in range(max_iter):
        mask = torch.bmm(X, Y.unsqueeze(-1)) <= 0
        if not mask.any():
            return -1e5 * Y / Y.norm(dim=-1, keepdim=True) ** 2
        Y += (X * mask).sum(1) / mask.sum(1).clamp(min=1)
    raise ValueError("Could not find fake keys such that for every query q, exp(<q, k>) = 0")


def add_attention_bias(bias, query, key, attention_mask):
    """
    Add a cache-owned attention bias to the attention mask of the first context positions.

    attention_bias has shape (bsz, num_key_value_heads, ctx_len): an additive logit bias per KV pair, with
    -inf for dropped pairs (see BernoulliPress). It is repeated across the query heads of each KV group.
    Keys past ctx_len (question and generated tokens) get no bias.

    Parameters
    ----------
    bias : torch.Tensor
        Additive bias with shape (bsz, num_key_value_heads, ctx_len).
    query : torch.Tensor
        Query tensor with shape (bsz, num_heads, q_len, head_dim).
    key : torch.Tensor
        Key tensor with shape (bsz, num_key_value_heads, k_len, head_dim).
    attention_mask : torch.Tensor or None
        Mask passed to the attention function: None, boolean (True means attend) or additive.

    Returns
    -------
    torch.Tensor
        Additive attention mask with shape broadcastable to (bsz, num_heads, q_len, k_len).
    """
    q_len, k_len = query.shape[2], key.shape[2]
    bsz, num_key_value_heads, ctx_len = bias.shape
    if ctx_len > k_len:
        raise ValueError(f"attention_bias covers {ctx_len} keys but only {k_len} are present (stale bias?)")

    num_groups = query.shape[1] // num_key_value_heads
    full = torch.zeros((bsz, query.shape[1], q_len, k_len), dtype=query.dtype, device=query.device)
    full[..., :ctx_len] = bias.repeat_interleave(num_groups, dim=1)[:, :, None, :].to(query.dtype)

    if attention_mask is None:
        # SDPA only applies causal masking when no mask is given, so once a mask is passed the causal structure
        # must be written into it. The queries are the last q_len positions of the k_len keys.
        if q_len > 1:
            positions = torch.arange(k_len, device=query.device)
            future = positions[None, :] > (torch.arange(q_len, device=query.device)[:, None] + (k_len - q_len))
            full = full.masked_fill(future[None, None], -float("inf"))
        return full
    if attention_mask.dtype == torch.bool:
        # Boolean masks (True means attend) must be converted to additive form first: adding a float tensor to a
        # boolean one would turn True and False into 1.0 and 0.0 and silently remove the masking.
        attention_mask = torch.zeros_like(attention_mask, dtype=query.dtype).masked_fill_(
            ~attention_mask, -float("inf")
        )
    return attention_mask + full


def attention_patch(func):
    """
    Decorator to update the keys before the attention computation at the indices provided in module.masked_key_indices
    The keys are updated with a fake key k such that exp(<q, k>) = 0 to fake head-wise compression
    This solution is not optimal as it does not reduce peak memory and slightly increases runtime

    It also adds a cache-owned attention bias supplied through ``kvpress_metadata`` (see add_attention_bias).

    When micro-metric logging is enabled, logs attention sparsity and (when the full KV is still
    present) relative attention-output error against a dense forward.

    Parameters
    ----------
    func : callable
        The original attention function to be patched. Should accept parameters
        (module, query, key, value, attention_mask, dropout, **kwargs).

    Returns
    -------
    callable
        The wrapped attention function that supports head-wise key masking.
    """

    def wrapper(module, query, key, value, attention_mask, dropout, **kwargs):
        if query.shape[2] == key.shape[2]:
            module.masked_key_indices = None
        metadata = kwargs.pop("kvpress_metadata", None)
        layer_idx = int(module.layer_idx)
        vkvwr_state = None if metadata is None else metadata.get("vkvwr", {}).get(layer_idx)
        attention_bias = None if metadata is None else metadata.get("attention_bias", {}).get(layer_idx)
        original_attention_mask = attention_mask
        dense_key = None
        sparsify_source = None
        apply_vkvwr = vkvwr_state is not None and query.shape[2] != key.shape[2]

        # Prefill keeps q_len == k_len: do not sparsify. Decode / cached forwards use the real query.
        if apply_vkvwr:
            if module.config._attn_implementation != "sdpa":
                raise ValueError("vkvwr decode bias is only supported with attn_implementation='sdpa'")
            from kvpress.metric_logging import bias_sparsity, maybe_log_sparsity
            from kvpress.presses.vkvwr_press import decode_attention_bias

            bias = decode_attention_bias(query, key, vkvwr_state)
            maybe_log_sparsity(
                bias_sparsity(bias),
                layer_idx=layer_idx,
                q_len=query.shape[2],
                k_len=key.shape[2],
                source="vkvwr_bias",
            )
            attention_mask = add_attention_bias(bias, query, key, attention_mask)
            sparsify_source = "vkvwr_bias"
        elif attention_bias is not None:
            if module.config._attn_implementation != "sdpa":
                raise ValueError("attention_bias is only supported with attn_implementation='sdpa'")
            from kvpress.metric_logging import bias_sparsity, maybe_log_sparsity

            maybe_log_sparsity(
                bias_sparsity(attention_bias),
                layer_idx=layer_idx,
                q_len=query.shape[2],
                k_len=key.shape[2],
                source="attention_bias",
            )
            attention_mask = add_attention_bias(attention_bias, query, key, attention_mask)
            sparsify_source = "attention_bias"
        elif getattr(module, "masked_key_indices", None) is not None:
            # Decoding: build fake keys k s.t. exp(<q, k>) = 0
            from kvpress.metric_logging import (
                masked_key_sparsity,
                maybe_log_sparsity,
                want_attention_output_error,
            )

            bsz, num_heads, seq_len, head_dim = query.shape
            num_key_value_heads = key.shape[1]
            num_groups = num_heads // num_key_value_heads
            batch_indices, head_indices, seq_indices = module.masked_key_indices

            maybe_log_sparsity(
                masked_key_sparsity(
                    module.masked_key_indices,
                    batch_size=bsz,
                    num_kv_heads=num_key_value_heads,
                    seq_len=key.shape[2],
                ),
                layer_idx=layer_idx,
                q_len=query.shape[2],
                k_len=key.shape[2],
                source="masked_keys",
            )
            if want_attention_output_error():
                dense_key = key.clone()

            # Build a fake key k per key group such that for every query q, exp(<q, k>) = 0
            q = query.view(bsz, num_key_value_heads, num_groups, seq_len, head_dim)
            q = q.reshape(bsz * num_key_value_heads, num_groups * seq_len, head_dim)
            k = search_hyperplane(q)
            k = k.view(bsz, num_key_value_heads, head_dim)

            # At indices, update the keys to the fake keys
            key[batch_indices, head_indices, seq_indices] = k[batch_indices, head_indices]
            sparsify_source = "masked_keys"

        # see https://github.com/NVIDIA/kvpress/pull/115#issuecomment-3183785597
        # cu_seq_lens_k are only in kwargs if model.generate is used.
        if "cu_seq_lens_k" in kwargs:
            kwargs["cu_seq_lens_k"][-1] = key.shape[-2]

        if sparsify_source is not None:
            from kvpress.metric_logging import maybe_log_attention_output_error, want_attention_output_error

            if want_attention_output_error():
                sparse_result = func(module, query, key, value, attention_mask, dropout, **kwargs)
                dense_result = func(
                    module,
                    query,
                    dense_key if dense_key is not None else key,
                    value,
                    original_attention_mask,
                    dropout,
                    **kwargs,
                )
                maybe_log_attention_output_error(
                    sparse_result,
                    dense_result,
                    layer_idx=layer_idx,
                    q_len=query.shape[2],
                    k_len=key.shape[2],
                    source=sparsify_source,
                )
                return sparse_result

        return func(module, query, key, value, attention_mask, dropout, **kwargs)

    return wrapper


def patch_attention_functions():
    """
    Apply attention patching to all transformer attention functions.

    This function automatically patches all attention functions registered in
    transformers' ALL_ATTENTION_FUNCTIONS to support head-wise key masking.
    It enables KVPress compression methods that require head-specific masking
    (like AdaKV) to work correctly during text generation.

    The patching is applied globally and affects all transformer models loaded
    after this function is called. It's automatically called when importing
    kvpress to ensure compatibility with head-wise compression methods.

    Notes
    -----
    This function modifies the global attention functions in the transformers
    library. The modifications do not affect models that don't use head-wise compression (i.e. don't have
    module.masked_key_indices).
    """
    for name, func in ALL_ATTENTION_FUNCTIONS.items():
        ALL_ATTENTION_FUNCTIONS[name] = attention_patch(func)
