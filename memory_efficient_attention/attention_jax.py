import functools
import jax
import math

from jax import numpy as jnp


def _query_chunk_attention(query, key, value, mask, bias, precision, key_chunk_size=4096, weights_out=None):
    num_kv, num_heads, k_features = key.shape[-3:]
    v_features = value.shape[-1]
    num_q = query.shape[-3]
    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / jnp.sqrt(k_features)

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(query, key, value, mask, bias, weights_out):
        attn_weights = jnp.einsum('...qhd,...khd->...qhk', query, key, precision=precision)
        if bias is not None:
            bias = jnp.einsum('...hqk->...qhk', bias)
            attn_weights = attn_weights + bias
        if mask is not None:
            big_neg = jnp.finfo(attn_weights.dtype).min
            mask = jnp.einsum('...hqk->...qhk', mask)
            attn_weights = jnp.where(mask, attn_weights, big_neg)
        if weights_out is not None:
            weights_out[:] = jax.lax.stop_gradient(attn_weights)
        max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        max_score = jax.lax.stop_gradient(max_score)
        exp_weights = jnp.exp(attn_weights - max_score)
        exp_values = jnp.einsum('...vhf,...qhv->...qhf', value, exp_weights, precision=precision)
        max_score = jnp.einsum('...qhk->...qh', max_score)
        return exp_values, exp_weights.sum(axis=-1), max_score

    def chunk_scanner(chunk_idx):
        key_chunk = jax.lax.dynamic_slice(
            key, tuple([0] * (key.ndim - 3)) + (chunk_idx, 0, 0),
            slice_sizes=tuple(key.shape[:-3]) + (key_chunk_size, num_heads, k_features))
        value_chunk = jax.lax.dynamic_slice(
            value, tuple([0] * (value.ndim - 3)) + (chunk_idx, 0, 0),
            slice_sizes=tuple(value.shape[:-3]) + (key_chunk_size, num_heads, v_features))
        if bias is not None:
            bias_chunk = jax.lax.dynamic_slice(
                bias, tuple([0] * (bias.ndim - 3)) + (0, 0, chunk_idx),
                slice_sizes=tuple(bias.shape[:-3]) + (num_heads, num_q, key_chunk_size))
        else:
            bias_chunk = None
        if mask is not None:
            mask_chunk = jax.lax.dynamic_slice(
                mask, tuple([0] * (mask.ndim - 3)) + (0, 0, chunk_idx),
                slice_sizes=tuple(mask.shape[:-3]) + (num_heads, num_q, key_chunk_size))
        else:
            mask_chunk = None
        if weights_out is not None:
            weights_chunk = jax.lax.dynamic_slice(
                weights_out, tuple([0] * (weights_out.ndim - 3)) + (0, 0, chunk_idx),
                slice_sizes=tuple(weights_out.shape[:-3]) + (num_q, num_heads, key_chunk_size))
        else:
            weights_chunk = None
        return summarize_chunk(query, key_chunk, value_chunk, mask_chunk, bias_chunk, weights_chunk)

    chunk_values, chunk_weights, chunk_max = jax.lax.map(
        chunk_scanner, xs=jnp.arange(0, num_kv, key_chunk_size))

    global_max = jnp.max(chunk_max, axis=0, keepdims=True)
    max_diffs = jnp.exp(chunk_max - global_max)
    chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(axis=0)
    all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)
    return all_values / all_weights


def efficient_dot_product_attention(query, key, value,
                                    mask=None, bias=None,
                                    precision=jax.lax.Precision.HIGHEST,
                                    query_chunk_size=1024,
                                    key_chunk_size=4096,
                                    return_weights=False):
    """Computes efficient dot-product attention given query, key, and value.
      This is efficient version of attention presented in
      https://arxiv.org/abs/2112.05682v2 which comes with O(sqrt(n)) memory requirements.
      Note: query, key, value needn't have any batch dimensions.
      Args:
        query: queries for calculating attention with shape of
          `[batch..., q_length, num_heads, qk_depth_per_head]`.
        key: keys for calculating attention with shape of
          `[batch..., kv_length, num_heads, qk_depth_per_head]`.
        value: values to be used in attention with shape of
          `[batch..., kv_length, num_heads, v_depth_per_head]`.
        bias: bias for the attention weights. This should be broadcastable to the
          shape `[batch..., num_heads, q_length, kv_length]`.
          This can be used for incorporating causal masks, padding masks,
          proximity bias, etc.
        mask: mask for the attention weights. This should be broadcastable to the
          shape `[batch..., num_heads, q_length, kv_length]`.
          This can be used for incorporating causal masks.
          Attention weights are masked out if their corresponding mask value
          is `False`.
        query_chunk_size: int: query chunks size
        key_chunk_size: int: key chunks size
        precision: numerical precision of the computation see `jax.lax.Precision`
                for details.
        return_weights: If specified, a tuple of (output, attentionts) will be returned.
      Returns:
        Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
      """
    num_q, num_heads, q_features = query.shape[-3:]
    num_kv = key.shape[-3]

    if return_weights:
        weights = jnp.empty((*value.shape[:-3], num_q, num_heads, num_kv))
    else:
        weights = None

    def chunk_scanner(chunk_idx, _):
        query_chunk = jax.lax.dynamic_slice(
            query, tuple([0] * (query.ndim - 3)) + (chunk_idx, 0, 0),
            slice_sizes=tuple(query.shape[:-3]) + (min(query_chunk_size, num_q), num_heads, q_features))
        if mask is not None:
            mask_chunk = jax.lax.dynamic_slice(
                mask, tuple([0] * (mask.ndim - 3)) + (0, chunk_idx, 0),
                slice_sizes=tuple(mask.shape[:-3]) + (num_heads, min(query_chunk_size, num_q), num_kv))
        else:
            mask_chunk = None
        if bias is not None:
            bias_chunk = jax.lax.dynamic_slice(
                bias, tuple([0] * (bias.ndim - 3)) + (0, chunk_idx, 0),
                slice_sizes=tuple(bias.shape[:-3]) + (num_heads, min(query_chunk_size, num_q), num_kv))
        else:
            bias_chunk = None
        if weights is not None:
            weights_chunk = jax.lax.dynamic_slice(
                weights, tuple([0] * (weights.ndim - 3)) + (chunk_idx, 0, 0),
                slice_sizes=tuple(weights.shape[:-3]) + (min(query_chunk_size, num_q), num_heads, num_kv))
        else:
            weights_chunk = None
        return (chunk_idx + query_chunk_size,
                _query_chunk_attention(query_chunk, key, value, mask_chunk, bias_chunk,
                                       precision=precision, key_chunk_size=key_chunk_size, weights_out=weights_chunk))

    _, res = jax.lax.scan(
        chunk_scanner, init=0, xs=None, length=math.ceil(num_q / query_chunk_size))
    if return_weights:
        return jnp.concatenate(res, axis=-3), weights
    else:
        return jnp.concatenate(res, axis=-3)
