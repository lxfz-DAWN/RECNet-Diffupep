```python
StripedHyena(
  (embedding_layer): ESMCembeding(
    (model): ESMC(
      (embed): Embedding(64, 1152)
      (transformer): TransformerStack(
        (blocks): ModuleList(
          (0-35): 36 x UnifiedTransformerBlock(
            (attn): FlashMultiHeadAttention(
              (layernorm_qkv): Sequential(
                (0): LayerNorm((1152,), eps=1e-05, elementwise_affine=True)
                (1): Linear(in_features=1152, out_features=3456, bias=False)
              )
              (out_proj): Linear(in_features=1152, out_features=1152, bias=False)
              (q_ln): LayerNorm((1152,), eps=1e-05, elementwise_affine=True)
              (k_ln): LayerNorm((1152,), eps=1e-05, elementwise_affine=True)
              (rotary): TritonRotaryEmbedding()
            )
            (ffn): Sequential(
              (0): LayerNorm((1152,), eps=1e-05, elementwise_affine=True)
              (1): Linear(in_features=1152, out_features=6144, bias=False)
              (2): SwiGLU()
              (3): Linear(in_features=3072, out_features=1152, bias=False)
            )
          )
        )
        (norm): LayerNorm((1152,), eps=1e-05, elementwise_affine=True)
      )
      (sequence_head): Sequential(
        (0): Linear(in_features=1152, out_features=1152, bias=True)
        (1): GELU(approximate='none')
        (2): LayerNorm((1152,), eps=1e-05, elementwise_affine=True)
        (3): Linear(in_features=1152, out_features=64, bias=True)
      )
    )
  )
  (norm): RMSNorm()
  (unembed): ESMCembeding(
    (model): ESMC(
      (embed): Embedding(64, 1152)
      (transformer): TransformerStack(
        (blocks): ModuleList(
          (0-35): 36 x UnifiedTransformerBlock(
            (attn): FlashMultiHeadAttention(
              (layernorm_qkv): Sequential(
                (0): LayerNorm((1152,), eps=1e-05, elementwise_affine=True)
                (1): Linear(in_features=1152, out_features=3456, bias=False)
              )
              (out_proj): Linear(in_features=1152, out_features=1152, bias=False)
              (q_ln): LayerNorm((1152,), eps=1e-05, elementwise_affine=True)
              (k_ln): LayerNorm((1152,), eps=1e-05, elementwise_affine=True)
              (rotary): TritonRotaryEmbedding()
            )
            (ffn): Sequential(
              (0): LayerNorm((1152,), eps=1e-05, elementwise_affine=True)
              (1): Linear(in_features=1152, out_features=6144, bias=False)
              (2): SwiGLU()
              (3): Linear(in_features=3072, out_features=1152, bias=False)
            )
          )
        )
        (norm): LayerNorm((1152,), eps=1e-05, elementwise_affine=True)
      )
      (sequence_head): Sequential(
        (0): Linear(in_features=1152, out_features=1152, bias=True)
        (1): GELU(approximate='none')
        (2): LayerNorm((1152,), eps=1e-05, elementwise_affine=True)
        (3): Linear(in_features=1152, out_features=64, bias=True)
      )
    )
  )
  (blocks): ModuleList(
    (0-7): 8 x ParallelGatedConvBlock(
      (pre_norm): RMSNorm()
      (post_norm): RMSNorm()
      (filter): ParallelHyenaFilter()
      (projections): Linear(in_features=4096, out_features=12288, bias=True)
      (out_filter_dense): Linear(in_features=4096, out_features=4096, bias=True)
      (mlp): ParallelGatedMLP(
        (l1): Linear(in_features=4096, out_features=10928, bias=False)
        (l2): Linear(in_features=4096, out_features=10928, bias=False)
        (l3): Linear(in_features=10928, out_features=4096, bias=False)
      )
    )
    (8): AttentionBlock(
      (pre_norm): RMSNorm()
      (post_norm): RMSNorm()
      (inner_mha_cls): MHA(
        (rotary_emb): RotaryEmbedding()
        (Wqkv): Linear(in_features=4096, out_features=12288, bias=True)
        (inner_attn): FlashSelfAttention(
          (drop): Dropout(p=0.0, inplace=False)
        )
        (inner_cross_attn): FlashCrossAttention(
          (drop): Dropout(p=0.0, inplace=False)
        )
        (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
      )
      (mlp): ParallelGatedMLP(
        (l1): Linear(in_features=4096, out_features=10928, bias=False)
        (l2): Linear(in_features=4096, out_features=10928, bias=False)
        (l3): Linear(in_features=10928, out_features=4096, bias=False)
      )
    )
    (9-15): 7 x ParallelGatedConvBlock(
      (pre_norm): RMSNorm()
      (post_norm): RMSNorm()
      (filter): ParallelHyenaFilter()
      (projections): Linear(in_features=4096, out_features=12288, bias=True)
      (out_filter_dense): Linear(in_features=4096, out_features=4096, bias=True)
      (mlp): ParallelGatedMLP(
        (l1): Linear(in_features=4096, out_features=10928, bias=False)
        (l2): Linear(in_features=4096, out_features=10928, bias=False)
        (l3): Linear(in_features=10928, out_features=4096, bias=False)
      )
    )
    (16): AttentionBlock(
      (pre_norm): RMSNorm()
      (post_norm): RMSNorm()
      (inner_mha_cls): MHA(
        (rotary_emb): RotaryEmbedding()
        (Wqkv): Linear(in_features=4096, out_features=12288, bias=True)
        (inner_attn): FlashSelfAttention(
          (drop): Dropout(p=0.0, inplace=False)
        )
        (inner_cross_attn): FlashCrossAttention(
          (drop): Dropout(p=0.0, inplace=False)
        )
        (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
      )
      (mlp): ParallelGatedMLP(
        (l1): Linear(in_features=4096, out_features=10928, bias=False)
        (l2): Linear(in_features=4096, out_features=10928, bias=False)
        (l3): Linear(in_features=10928, out_features=4096, bias=False)
      )
    )
    (17-23): 7 x ParallelGatedConvBlock(
      (pre_norm): RMSNorm()
      (post_norm): RMSNorm()
      (filter): ParallelHyenaFilter()
      (projections): Linear(in_features=4096, out_features=12288, bias=True)
      (out_filter_dense): Linear(in_features=4096, out_features=4096, bias=True)
      (mlp): ParallelGatedMLP(
        (l1): Linear(in_features=4096, out_features=10928, bias=False)
        (l2): Linear(in_features=4096, out_features=10928, bias=False)
        (l3): Linear(in_features=10928, out_features=4096, bias=False)
      )
    )
    (24): AttentionBlock(
      (pre_norm): RMSNorm()
      (post_norm): RMSNorm()
      (inner_mha_cls): MHA(
        (rotary_emb): RotaryEmbedding()
        (Wqkv): Linear(in_features=4096, out_features=12288, bias=True)
        (inner_attn): FlashSelfAttention(
          (drop): Dropout(p=0.0, inplace=False)
        )
        (inner_cross_attn): FlashCrossAttention(
          (drop): Dropout(p=0.0, inplace=False)
        )
        (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
      )
      (mlp): ParallelGatedMLP(
        (l1): Linear(in_features=4096, out_features=10928, bias=False)
        (l2): Linear(in_features=4096, out_features=10928, bias=False)
        (l3): Linear(in_features=10928, out_features=4096, bias=False)
      )
    )
    (25-31): 7 x ParallelGatedConvBlock(
      (pre_norm): RMSNorm()
      (post_norm): RMSNorm()
      (filter): ParallelHyenaFilter()
      (projections): Linear(in_features=4096, out_features=12288, bias=True)
      (out_filter_dense): Linear(in_features=4096, out_features=4096, bias=True)
      (mlp): ParallelGatedMLP(
        (l1): Linear(in_features=4096, out_features=10928, bias=False)
        (l2): Linear(in_features=4096, out_features=10928, bias=False)
        (l3): Linear(in_features=10928, out_features=4096, bias=False)
      )
    )
    (32): AttentionBlock(
      (pre_norm): RMSNorm()
      (post_norm): RMSNorm()
      (inner_mha_cls): MHA(
        (rotary_emb): RotaryEmbedding()
        (Wqkv): Linear(in_features=4096, out_features=12288, bias=True)
        (inner_attn): FlashSelfAttention(
          (drop): Dropout(p=0.0, inplace=False)
        )
        (inner_cross_attn): FlashCrossAttention(
          (drop): Dropout(p=0.0, inplace=False)
        )
        (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
      )
      (mlp): ParallelGatedMLP(
        (l1): Linear(in_features=4096, out_features=10928, bias=False)
        (l2): Linear(in_features=4096, out_features=10928, bias=False)
        (l3): Linear(in_features=10928, out_features=4096, bias=False)
      )
    )
    (33-34): 2 x ParallelGatedConvBlock(
      (pre_norm): RMSNorm()
      (post_norm): RMSNorm()
      (filter): ParallelHyenaFilter()
      (projections): Linear(in_features=4096, out_features=12288, bias=True)
      (out_filter_dense): Linear(in_features=4096, out_features=4096, bias=True)
      (mlp): ParallelGatedMLP(
        (l1): Linear(in_features=4096, out_features=10928, bias=False)
        (l2): Linear(in_features=4096, out_features=10928, bias=False)
        (l3): Linear(in_features=10928, out_features=4096, bias=False)
      )
    )
  )
)
```