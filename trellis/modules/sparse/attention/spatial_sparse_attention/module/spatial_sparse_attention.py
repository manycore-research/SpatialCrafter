import torch
import torch.nn as nn
from einops import rearrange
from flash_attn import flash_attn_varlen_func
from ..ops import (
    spatial_selection_attention,
    get_block_score,
    sparse_window_attention,
)
from .compression_block import SparseDownBlock3d_v1, SparseDownBlock3d_v2
import trellis.modules.sparse as sp
from trellis.modules.sparse.basic import SparseTensor


class SpatialSparseAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        compression_block_size: int = 4,
        selection_block_size: int = 8,
        topk: int = 8,
        window_size: int = 8,
        shift_window: int = 4,
        resolution: int = 64,
        compression_version: str = "v2",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.compression_block_size = compression_block_size
        self.selection_block_size = selection_block_size
        self.topk = topk
        self.window_size = window_size
        self.shift_window = shift_window
        self.resolution = resolution

        # qkv proj and o proj
        self.proj_q = sp.SparseLinear(hidden_size, num_q_heads * head_dim, bias=False)
        self.proj_k = sp.SparseLinear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.proj_v = sp.SparseLinear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.proj_o = torch.nn.Linear(num_q_heads * head_dim, hidden_size, bias=False)

        # ssa parameteres
        if compression_version == "v1":
            compression_block = SparseDownBlock3d_v1
        elif compression_version == "v2":
            compression_block = SparseDownBlock3d_v2
        else:
            raise NotImplementedError("only support v1 or v2 compression block")
        # downsample key and value features by 4x
        self.compression_key = compression_block(num_kv_heads * head_dim, num_kv_heads * head_dim, factor=compression_block_size)
        self.compression_value = compression_block(num_kv_heads * head_dim, num_kv_heads * head_dim, factor=compression_block_size)
        self.intra_block_pe = torch.nn.Parameter(
            torch.zeros(
                compression_block_size,
                compression_block_size,
                compression_block_size,
                num_kv_heads * head_dim,
            )
        )

        # gate function
        self.gate = torch.nn.Sequential(
            sp.SparseLinear(hidden_size, 3, bias=False),
            sp.SparseSigmoid(),
        )

    def sparse3d_compression(self, x: sp.SparseTensor, key=True):
        _, num_heads, num_dim = x.feats.shape
        x = x.replace(x.feats.view(-1, num_heads * num_dim))
        if key:
            coords = x.coords
            intra_block_coords = coords[..., 1:] % self.compression_block_size
            intra_block_pos = self.intra_block_pe[intra_block_coords[:, 0], intra_block_coords[:, 1], intra_block_coords[:, 2]].to(x.dtype)
            x = x.replace(x.feats + intra_block_pos)
            y = self.compression_key(x)
        else:
            y = self.compression_value(x)
        y = y.replace(y.feats.view(-1, num_heads, num_dim))
        return y

    def forward(self, x: sp.SparseTensor):
        # dtype and shape check
        assert x.shape[-1] == self.hidden_size
        assert self.selection_block_size % self.compression_block_size == 0
        # # Reset peak memory stats at the beginning
        # if torch.cuda.is_available():
        #     torch.cuda.reset_peak_memory_stats()
        #     torch.cuda.synchronize()

        # # Memory tracking state
        # prev_allocated = 0
        # prev_reserved = 0

        # # Helper function for memory profiling
        # def log_memory(tag: str):
        #     nonlocal prev_allocated, prev_reserved
        #     if torch.cuda.is_available():
        #         torch.cuda.synchronize()
        #         allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        #         reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        #         peak = torch.cuda.max_memory_allocated() / 1024**3  # GB

        #         # Calculate delta
        #         delta_allocated = allocated - prev_allocated
        #         delta_reserved = reserved - prev_reserved

        #         # Format output with colors (optional, works in most terminals)
        #         delta_alloc_str = f"+{delta_allocated:.3f}" if delta_allocated >= 0 else f"{delta_allocated:.3f}"
        #         delta_resv_str = f"+{delta_reserved:.3f}" if delta_reserved >= 0 else f"{delta_reserved:.3f}"

        #         print(
        #             f"[{tag:30s}] Alloc: {allocated:6.3f} GB ({delta_alloc_str:8s} GB) | "
        #             f"Resv: {reserved:6.3f} GB ({delta_resv_str:8s} GB) | Peak: {peak:6.3f} GB"
        #         )

        #         prev_allocated = allocated
        #         prev_reserved = reserved

        #         return allocated, reserved, peak
        #     return 0, 0, 0

        # # Initial memory state
        # print("=" * 110)
        # print("CUDA Memory Profile - Spatial Sparse Attention")
        # print("=" * 110)
        # log_memory("Start")

        # qkv proj
        q = x.replace(self.proj_q(x).feats.view(-1, self.num_q_heads, self.head_dim))
        k = x.replace(self.proj_k(x).feats.view(-1, self.num_kv_heads, self.head_dim))
        v = x.replace(self.proj_v(x).feats.view(-1, self.num_kv_heads, self.head_dim))
        # log_memory("After QKV projection")
        # print(f"[SpatialSparseAttention] q shape: {q.feats.shape}, k shape: {k.feats.shape}, v shape: {v.feats.shape}")

        # compression attention
        compressed_k = self.sparse3d_compression(k, key=True)
        compressed_v = self.sparse3d_compression(v, key=False)
        # log_memory("After QKV compression")
        # print(f"[SpatialSparseAttention] compressed_k shape: {compressed_k.feats.shape}, compressed_v shape: {compressed_v.feats.shape}")

        compressed_cu_seqlens = (
            torch.tensor([s.start for s in compressed_v.layout] + [s.stop for s in compressed_v.layout if s.stop not in [s.start for s in compressed_v.layout]])
            .to(compressed_v.device)
            .to(torch.int32)
        )
        compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]

        cu_seqlens = (
            torch.tensor([s.start for s in x.layout] + [s.stop for s in x.layout if s.stop not in [s.start for s in x.layout]]).to(x.device).to(torch.int32)
        )
        # print(f"[SpatialSparseAttention] cu_seqlens: {cu_seqlens}")
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

        # compressed_attn_output: [num_q_tokens, num_q_head, head_dim] 每个token的weighted sum of compressed_v;
        # lse: [num_q_tokens, num_q_heads] 每个token每个头的log-sum-exp 值，logsumexp[i, h] = log(Σ_j exp(S[i,h,j])), i表示第i个query token, h表示第h个query head, j表示compressed_k中的token index, S[i,h,j]未归一化
        compressed_attn_output, lse, _ = flash_attn_varlen_func(
            q.feats,
            compressed_k.feats,
            compressed_v.feats,
            cu_seqlens,
            compressed_cu_seqlens,
            seqlens.max().item(),
            compressed_seqlens.max().item(),
            causal=False,
            return_attn_probs=True,
        )
        # log_memory("After compressed attention")
        # print(f"[SpatialSparseAttention] compressed_attn_output shape: {compressed_attn_output.shape}, softmax_lse shape: {lse.shape}")

        with torch.no_grad():
            block_topk, cu_seqblocks, cu_block_include_tokens = get_block_score(
                q=q,
                compressed_k=compressed_k,
                lse=lse,
                resolution=self.resolution,
                kernel_stride=self.compression_block_size,
                block_size=self.selection_block_size,
                topk=self.topk,
                cu_seqlens=cu_seqlens,
                compressed_cu_seqlens=compressed_cu_seqlens,
                seqlens=seqlens,
                compressed_seqlens=compressed_seqlens,
                sm_scale=None,
            )
        # log_memory("After block score computation")
        # print(
        #     f"[SpatialSparseAttention] block_topk shape: {block_topk.shape}, cu_seqblocks shape: {cu_seqblocks.shape}, cu_block_include_tokens shape: {cu_block_include_tokens.shape}"
        # )
        # spatial selection attention
        selection_attn_output = spatial_selection_attention(
            q=q.feats,
            k=k.feats,
            v=v.feats,
            block_topk=block_topk,
            cu_seqblocks=cu_seqblocks,
            cu_block_include_tokens=cu_block_include_tokens,
            block_size=self.selection_block_size,
            cu_seqlens=cu_seqlens,
            softmax_scale=None,
        )
        # log_memory("After selection attention")
        # print(f"[SpatialSparseAttention] selection_attn_output shape: {selection_attn_output.shape}")
        # window attention
        window_attn_output = sparse_window_attention(
            q=q,
            k=k,
            v=v,
            window_size=self.window_size,
            shift_window=self.shift_window,
        ).feats
        # log_memory("After window attention")
        # gate average
        gate = self.gate(x).feats
        attn_output = gate[:, 0:1, None] * compressed_attn_output + gate[:, 1:2, None] * selection_attn_output + gate[:, 2:3, None] * window_attn_output
        # log_memory("After gating_mlp")
        # rearrange and output proj
        attn_output = rearrange(attn_output, "n h d -> n (h d)")
        attn_output = self.proj_o(attn_output)
        # log_memory("After output projection")
        return x.replace(attn_output)
