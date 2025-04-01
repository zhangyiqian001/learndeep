# import math
# import warnings
# from abc import abstractmethod
# from typing import Optional
#
# import torch
# import torch.nn as nn
# from einops import rearrange
#
#
# def _reset_causal(
#         num_query_tokens: int, num_key_tokens: int, original_causal: bool
# ):
#     # disable causal when it is not needed
#     # necessary for flash & triton for generation with kv_cache
#     if original_causal and num_query_tokens != num_key_tokens:
#         if num_query_tokens != 1:
#             raise NotImplementedError(
#                 "MPT does not support query and key with different number of"
#                 " tokens, unless number of query tokens is 1."
#             )
#         else:
#             return False
#     return original_causal
#
#
# def check_valid_inputs(*tensors, valid_dtypes=[torch.float16, torch.bfloat16]):
#     for tensor in tensors:
#         if tensor.dtype not in valid_dtypes:
#             raise TypeError(f"{tensor.dtype=} must be in {valid_dtypes=}.")
#         if not tensor.is_cuda:
#             raise TypeError(f"Inputs must be cuda tensors ({tensor.is_cuda=}).")
#
#
# def flash_attn_fn(
#         query,
#         key,
#         value,
#         heads,
#         past_key_value=None,
#         softmax_scale=None,
#         bias=None,
#         key_padding_mask=None,
#         causal=False,
#         dropout=0.0,
#         training=False,
#         needs_weights=False,
#         multiquery=False,
# ):
#     try:
#         # type: ignore # yapf: disable # isort: skip
#         from flash_attn import bert_padding, flash_attn_interface
#     except BaseException:
#         raise RuntimeError("Please install flash-attn==1.0.3.post0")
#
#     check_valid_inputs(query, key, value)
#
#     if past_key_value is not None:
#         if len(past_key_value) != 0:
#             key = torch.cat([past_key_value[0], key], dim=1)
#             value = torch.cat([past_key_value[1], value], dim=1)
#
#         past_key_value = (key, value)
#
#     if bias is not None:
#         # clamp to 0 necessary for torch 2.0 compile()
#         _s_q = max(0, bias.size(2) - query.size(1))
#         _s_k = max(0, bias.size(3) - key.size(1))
#         bias = bias[:, :, _s_q:, _s_k:]
#
#     if bias is not None:
#         raise NotImplementedError("bias not implemented for flash attn.")
#
#     batch_size, seqlen = query.shape[:2]
#
#     if key_padding_mask is None:
#         key_padding_mask = torch.ones_like(key[:, :, 0], dtype=torch.bool)
#     query_padding_mask = key_padding_mask[:, -query.size(1):]
#
#     (
#         query_unpad,
#         indices_q,
#         cu_seqlens_q,
#         max_seqlen_q,
#     ) = bert_padding.unpad_input(query, query_padding_mask)
#     query_unpad = rearrange(query_unpad, "nnz (h d) -> nnz h d", h=heads)
#
#     key_unpad, _, cu_seqlens_k, max_seqlen_k = bert_padding.unpad_input(
#         key, key_padding_mask
#     )
#     key_unpad = rearrange(
#         key_unpad, "nnz (h d) -> nnz h d", h=1 if multiquery else heads
#     )
#
#     value_unpad, _, _, _ = bert_padding.unpad_input(value, key_padding_mask)
#     value_unpad = rearrange(
#         value_unpad, "nnz (h d) -> nnz h d", h=1 if multiquery else heads
#     )
#
#     if multiquery:
#         key_unpad = key_unpad.expand(
#             key_unpad.size(0), heads, key_unpad.size(-1)
#         )
#         value_unpad = value_unpad.expand(
#             value_unpad.size(0), heads, value_unpad.size(-1)
#         )
#
#     dropout = dropout if training else 0.0
#
#     reset_causal = _reset_causal(query.size(1), key.size(1), causal)
#
#     output_unpad = flash_attn_interface.flash_attn_unpadded_func(
#         query_unpad,
#         key_unpad,
#         value_unpad,
#         cu_seqlens_q,
#         cu_seqlens_k,
#         max_seqlen_q,
#         max_seqlen_k,
#         dropout,
#         softmax_scale=softmax_scale,
#         causal=reset_causal,
#         return_attn_probs=needs_weights,
#     )
#
#     output = bert_padding.pad_input(
#         rearrange(output_unpad, "nnz h d -> nnz (h d)"),
#         indices_q,
#         batch_size,
#         seqlen,
#     )
#     return output, None, past_key_value
#
# def triton_flash_attn_fn(
#         query,
#         key,
#         value,
#         heads,
#         past_key_value=None,
#         softmax_scale=None,
#         bias=None,
#         key_padding_mask=None,
#         causal=False,
#         dropout=0.0,
#         training=False,
#         needs_weights=False,
#         multiquery=False,
# ):
#     try:
#         from llmfoundry.models.layers.flash_attn_triton import flash_attn_func
#     except BaseException:
#         _installed = False
#         if version.parse(torch.__version__) < version.parse("2.0.0"):
#             _installed = True
#             # if torch1.13.1 revert to using triton flash attn from HazyResearch
#             # with flash-attn==1.0.3.post0 and triton==2.0.0.dev20221202
#             try:
#                 from flash_attn.flash_attn_triton import flash_attn_func
#             except BaseException:
#                 _installed = False
#         if not _installed:
#             # installing triton-pre-mlir works for both torch1.13.1 and torch2.0+
#             # default recommendation is to install this variant
#             raise RuntimeError(
#                 "Requirements for `attn_impl: triton` not installed. Either (1)"
#                 " have a CUDA-compatible GPU and `pip install .[gpu]` if"
#                 " installing from source or `pip install"
#                 " triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python`"
#                 " if installing from pypi, or (2) use torch attn"
#                 " model.attn_config.attn_impl=torch (torch attn_impl will be"
#                 " slow). Note: (1) requires you have CMake and PyTorch already"
#                 " installed."
#             )
#
#     check_valid_inputs(query, key, value)
#
#     if past_key_value is not None:
#         if len(past_key_value) != 0:
#             key = torch.cat([past_key_value[0], key], dim=1)
#             value = torch.cat([past_key_value[1], value], dim=1)
#
#         past_key_value = (key, value)
#
#     if bias is not None:
#         # clamp to 0 necessary for torch 2.0 compile()
#         _s_q = max(0, bias.size(2) - query.size(1))
#         _s_k = max(0, bias.size(3) - key.size(1))
#         bias = bias[:, :, _s_q:, _s_k:]
#
#     if dropout:
#         raise NotImplementedError(
#             "Dropout not implemented for attn_impl: triton."
#         )
#
#     if needs_weights:
#         raise NotImplementedError(
#             "attn_impl: triton cannot return attn weights."
#         )
#
#     if key_padding_mask is not None:
#         warnings.warn(
#             "Propagating key_padding_mask to the attention module "
#             + "and applying it within the attention module can cause "
#             + "unnecessary computation/memory usage. Consider integrating "
#             + "into bias once and passing that to each attention "
#             + "module instead."
#         )
#         b_size, s_k = key_padding_mask.shape[:2]
#
#         if bias is None:
#             bias = query.new_zeros(b_size, 1, 1, s_k)
#
#         bias = bias.masked_fill(
#             ~key_padding_mask.view((b_size, 1, 1, s_k)),
#             torch.finfo(query.dtype).min,
#         )
#
#     query = rearrange(query, "b s (h d) -> b s h d", h=heads)
#     key = rearrange(key, "b s (h d) -> b s h d", h=1 if multiquery else heads)
#     value = rearrange(
#         value, "b s (h d) -> b s h d", h=1 if multiquery else heads
#     )
#
#     if multiquery:
#         # necessary to repeat instead of expand tensor because
#         # output contains NaN in edge cases such as with head dimension = 8
#         key = key.repeat(1, 1, heads, 1)
#         value = value.repeat(1, 1, heads, 1)
#
#     reset_causal = _reset_causal(query.size(1), key.size(1), causal)
#     attn_output = flash_attn_func(
#         query, key, value, bias, reset_causal, softmax_scale
#     )
#
#     output = attn_output.view(*attn_output.shape[:2], -1)
#
#     return output, None, past_key_value
# class BaseAttention(nn.Module):
#     @abstractmethod
#     def __init__(self):
#         super().__init__()
#
#     @abstractmethod
#     def forward(self, x, context=None, mask=None):
#         pass
#
#
# class MultiQueryAttention(BaseAttention):
#     """Multi-Query self attention.
#
#     Using torch or triton attention implemetation enables user to also use
#     additive bias.
#
#     Look for documentation
#     """
#
#     def __init__(
#             self,
#             d_model: int,
#             heads: int,
#             attn_impl: str = "torch",
#             clip_qkv: Optional[float] = None,
#             qk_ln: bool = False,
#             softmax_scale: Optional[float] = None,
#             attn_pdrop: float = 0.0,
#             norm_type: str = "low_precision_layernorm",
#             fc_type: str = "torch",
#             verbose: int = 0,
#             device: Optional[str] = None,
#     ):
#         super().__init__()
#
#         self.attn_impl = attn_impl
#         self.clip_qkv = clip_qkv
#         self.qk_ln = qk_ln
#
#         self.d_model = d_model
#         self.heads = heads
#         self.head_dim = d_model // heads
#         self.softmax_scale = softmax_scale
#         if self.softmax_scale is None:
#             self.softmax_scale = 1 / math.sqrt(self.head_dim)
#         self.attn_dropout = attn_pdrop
#
#         fc_kwargs = {}
#         if fc_type != "te":
#             fc_kwargs["device"] = device
#         # - vchiley
#         self.Wqkv = nn.Linear(
#             d_model,
#             d_model + 2 * self.head_dim,
#             **fc_kwargs,
#         )
#         # for param init fn; enables shape based init of fused layers
#         fuse_splits = (d_model, d_model + self.head_dim)
#         self.Wqkv._fused = (0, fuse_splits)  # type: ignore
#
#         if self.qk_ln:
#             norm_class = nn.LayerNorm
#             self.q_ln = norm_class(d_model, device=device)
#             self.k_ln = norm_class(self.head_dim, device=device)
#
#         if self.attn_impl == "flash":
#             self.attn_fn = flash_attn_fn
#         elif self.attn_impl == "triton":
#             self.attn_fn = triton_flash_attn_fn
#             if verbose:
#                 warnings.warn(
#                     "While `attn_impl: triton` can be faster than `attn_impl:"
#                     " flash` "
#                     + "it uses more memory. When training larger models"
#                       " this can"
#                       " trigger "
#                     + "alloc retries which hurts performance. If"
#                       " encountered, we"
#                       " recommend "
#                     + "using `attn_impl: flash` if your model does not use"
#                       " `alibi` or `prefix_lm`."
#                 )
#         elif self.attn_impl == "torch":
#             self.attn_fn = scaled_multihead_dot_product_attention
#             if torch.cuda.is_available() and verbose:
#                 warnings.warn(
#                     "Using `attn_impl: torch`. If your model does not use"
#                     " `alibi` or "
#                     + "`prefix_lm` we recommend using `attn_impl: flash`"
#                       " otherwise "
#                     + "we recommend using `attn_impl: triton`."
#                 )
#         else:
#             raise ValueError(f"{attn_impl=} is an invalid setting.")
#
#         self.out_proj = FC_CLASS_REGISTRY[fc_type](
#             self.d_model,
#             self.d_model,
#             **fc_kwargs,
#         )
#         self.out_proj._is_residual = True  # type: ignore
#
#     def forward(
#             self,
#             x,
#             past_key_value=None,
#             bias=None,
#             mask=None,
#             causal=True,
#             needs_weights=False,
#     ):
#         qkv = self.Wqkv(x)
#
#         if self.clip_qkv:
#             qkv = qkv.clamp(min=-self.clip_qkv, max=self.clip_qkv)
#
#         query, key, value = qkv.split(
#             [self.d_model, self.head_dim, self.head_dim], dim=2
#         )
#
#         key_padding_mask = mask
#
#         if self.qk_ln:
#             # Applying layernorm to qk
#             dtype = query.dtype
#             query = self.q_ln(query).to(dtype)
#             key = self.k_ln(key).to(dtype)
#
#         context, attn_weights, past_key_value = self.attn_fn(
#             query,
#             key,
#             value,
#             self.heads,
#             past_key_value=past_key_value,
#             softmax_scale=self.softmax_scale,
#             bias=bias,
#             key_padding_mask=key_padding_mask,
#             causal=causal,
#             dropout=self.attn_dropout,
#             training=self.training,
#             needs_weights=needs_weights,
#             multiquery=True,
#         )
#
#         return self.out_proj(context), attn_weights, past_key_value
