# Upstream PR Draft: Fix crash with quantized K cache on MLA models without FA

**Target repo:** `ggml-org/llama.cpp`
**Scope:** One 3-line change in `src/llama-graph.cpp`
**Impact:** Unblocks `--cache-type-k iq4_nl|q8_0|q4_0|...` on MLA-arch models
(Mistral Large 3, Kimi K2.5, some DeepSeek variants) when Flash Attention is
unavailable (Pascal/SM61 and below, AMD without ROCm FA, CPU-only builds,
some embedded inference paths).

> **AGENTS.md compliance reminder:** this file is a draft / investigation
> writeup. The PR itself must be authored by a human contributor who has
> reviewed the change, can defend it, and can discuss it with maintainers.
> Draft language here is a starting point, not submission copy.

## Problem

On MLA architectures (`deepseek2`), the V tensor at attention time is a
*view* of the K cache — V shares K's latent storage. When `--cache-type-k`
is any quantized type, V inherits that quant type.

On the non-FlashAttention path in `build_attn_mha`, the V view is
materialized via:

```cpp
if (!v_trans) {
    v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
}
```

`ggml_transpose` on a quantized view yields a block-misaligned strided
layout; `ggml_cont` then needs a kernel that materializes a contiguous
quantized tensor from block-misaligned strides. **No backend implements
this**:

- CUDA `ggml_cuda_cpy` falls through to `GGML_ABORT("unsupported type
  combination (iq4_nl to iq4_nl)")` at `cpy.cu:547`
- CPU `ggml_compute_forward_dup_bytes` at `ops.cpp:397` iterates
  `i00 < ne00` (element count) with `i00*nb00` (block stride),
  overflowing for any `ne00 > block_size`

The crash surfaces during server startup via
`common_speculative_is_compat()`'s probe decode — before any real user
input.

## Root cause

The block-size granularity of quantized tensors means *no* legitimate
byte-copy kernel can transpose-and-materialize a quantized tensor
whose dim-0 length isn't naturally aligned to the target layout. The
operation itself is ill-defined; the right answer is to avoid
constructing it.

## Reproducer

- Model: any `deepseek2`-arch GGUF with MLA enabled (Mistral Large 3
  675B Q4_K_M confirmed; Kimi K2.5 expected identical).
- Build: mainline b8180+ CUDA (SM61 / Pascal or any non-FA environment)
- Command:
  ```
  llama-server -m <model> --cache-type-k iq4_nl --cache-type-v f16 \
      --ctx-size 16384 -ngl 99
  ```
- Expected: abort at load time with
  ```
  ggml/src/ggml-cuda/cpy.cu:547: ggml_cuda_cpy: unsupported type combination (iq4_nl to iq4_nl)
  ```

Note: DS R1 does NOT hit this with the same binary + same config. The
graph it builds happens not to schedule the problematic
`transpose(quant_view) → cont` op. Other deepseek2 variants (ML3, Kimi
K2.5) do schedule it. Root cause of the graph-level divergence not
pinned down, but the fix below sidesteps the issue for all variants.

## Fix

In `llama-graph.cpp` in `build_attn_mha`, insert a dequantize cast
before the transpose+cont when V is a quantized view:

```cpp
if (!v_trans) {
    if (ggml_is_quantized(v->type)) {
        v = ggml_cast(ctx0, v, GGML_TYPE_F32);
    }
    v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
    cb(v, "v_cont", il);
}
```

F32 chosen over F16: Pascal has rate-limited fp16 compute (measured:
F32 3.32 tok/s vs F16 3.06 tok/s on Mistral Large 3), and CPU backend
only implements `is_quantized → F32` (`ops.cpp:567`), not `→ F16`.
F32 is the universal lowest-common-denominator.

Cost: V-cast working memory inflates from `quant_bpw * n_kv_cells *
kv_lora_rank` to `4 * n_kv_cells * kv_lora_rank` during attention.
For ML3 at 16K ctx: ~32 MB/layer × 61 layers = ~2 GB transient peak.
Persistent KV cache unchanged.

## Measured results (4x Tesla P40, Pascal SM61, non-FA)

### Mistral Large 3 675B, 16K ctx

| `--cache-type-k` | tok/s | K cache | Baseline |
|---|---|---|---|
| q8_0 (baseline) | 3.69 | 8.5 bpw | mainline |
| **iq4_nl (with fix)** | **3.32** | **4.5 bpw** | this PR |

47% K cache savings, ~10% throughput cost. Coherent multi-prompt output.

### Kimi K2.5 1T, 32K ctx

| `--cache-type-k` | tok/s | K cache (32K) | Baseline |
|---|---|---|---|
| q8_0 (prior, 131K ctx) | 4.72 | ~1166 MiB proj | mainline |
| **iq4_nl (with fix, 32K)** | **3.73–4.19** | **617 MiB** | this PR |

K cache halved; same fix, same arch (deepseek2 MLA). Not a
like-for-like comparison (q8_0 baseline was at full 131K ctx), but
confirms the fix generalizes across MLA models.

At 131K ctx the K savings scale linearly — significant VRAM freed
on 24GB Pascal cards.

## Tests

- [ ] Existing CI tests pass (MHA models, DS R1, small models)
- [ ] Manual: Mistral Large 3 load + multi-prompt inference @ 16K ctx
- [ ] Manual: verify DS R1 unchanged (V not quantized in its graph path)
- [ ] Manual: `test-backend-ops cpy` for iq4_nl → F32 coverage

## Related observations (not part of this PR)

The following gaps exist but are out of scope:

- `ggml-cpu/ops.cpp:397` (non-contig dst, quant src) has a latent bug
  iterating elements with block strides. Never hit in practice because
  this PR avoids the op, but worth a follow-up.
- CUDA does not implement `is_quantized → is_quantized` cpy at all.
  Our fork added a kernel for this as investigation residue
  (`cuda cpy: add non-contiguous same-type quant copy kernel`) — not
  needed upstream once this graph fix lands.
- On CUDA, adding `IQ4_NL → F32` and `IQ4_NL → F16` cpy dispatches
  allows `ggml_cast` to stay on the CUDA backend. We measured the
  CPU-scheduled route is marginally faster than keeping on CUDA due
  to overlap with MoE expert compute — worth investigating upstream
  but not required by this fix.

## Author checklist before submitting

- [ ] Rebase cleanly on current `master`
- [ ] Strip AI-assistance markers from commit message; author in
      human-authored form per `AGENTS.md`
- [ ] Add a `test-backend-ops` case for the specific MLA path if
      existing coverage is insufficient
- [ ] Verify no behavioral change on DS R1 / any already-working
      MHA+quant config
- [ ] Discuss with a maintainer whether F32 vs F16 cast is the right
      default (maintainers may want it hardware-gated)
