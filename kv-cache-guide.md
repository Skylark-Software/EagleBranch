# KV cache preset guide

The `--cache-type-k` / `--cache-type-v` flags control the storage format
of the attention KV cache. Trade-off: smaller formats save VRAM/RAM but
cost some throughput. Pick based on your bottleneck.

> **Availability:** the `tbq*` types are **TurboQuant**, Skylark Software's
> proprietary KV-cache quantization family. They are included in the
> [release binaries](https://github.com/Skylark-Software/EagleBranch/releases)
> but are **not present in the public source** — a build from this repo has
> only the mainline types. Licensing/source access: info@skylarksoftware.me.

## All supported types (widest to tightest compression)

| Type | bpw | Compression vs f16 | Architecture | Notes |
|---|---|---|---|---|
| `f16` | 16.0 | 1x | all | No compression; fastest |
| `bf16` | 16.0 | 1x | all | Same size as f16; SM80+ hardware only |
| `q8_0` | 8.5 | 1.88x | MHA; MLA requires FA (no Pascal MLA) | Minimal quality loss |
| `q5_0` / `q5_1` | ~5.5 | ~2.9x | MHA | Mainline |
| `q4_0` / `q4_1` | ~4.5 | ~3.5x | MHA | Mainline |
| `iq4_nl` | 4.5 | 3.56x | MHA + MLA | Production default for MLA on Pascal |
| **`tbq4_1`** | 4.125 | 3.88x | MHA only (block=128) | Quality-leaning TurboQuant; smaller than iq4_nl (measured numbers TBD) |
| **`tbq3_2`** | 3.5 | 4.57x | MHA + MLA (block=32) | 22% smaller K than IQ4_NL |
| **`tbq3_1`** | 3.125 | 5.12x | MHA only (block=128) | Smallest KV; MHA only |

## Architecture detection (important)

- **MHA** (multi-head attention, `head_dim=128`): Mistral Small/Large,
  Llama family, Mixtral, Pixtral, Qwen2 family, most models
- **MLA** (multi-query latent attention, `head_dim=576`): DeepSeek V2
  / V3 / R1, Mistral Large 3, Kimi K2 / K2.5 — anything using the
  `deepseek2` architecture tag

For MLA on Pascal (P40, P100, etc.), Flash Attention is auto-disabled
by llama.cpp (head_dim > 256 ceiling). This means:
- V must stay `f16` (quantized V would require FA)
- K can be any non-FA-required type (f16, q8_0, iq4_nl, tbq3_2)
- TBQ3_1's block=128 doesn't divide 576, so TBQ3_1 isn't usable on MLA

## Picking by workload

### Throughput critical, memory plentiful

```
--cache-type-k f16 --cache-type-v f16
```

### Production default (balanced)

```
--cache-type-k q8_0 --cache-type-v f16      # MLA
--cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on   # MHA
```

### RAM-pressured, moderate compression

```
--cache-type-k iq4_nl --cache-type-v f16    # MLA (default in many configs)
--cache-type-k iq4_nl --cache-type-v iq4_nl --flash-attn on   # MHA
```

### Maximum compression (long context, tight RAM)

MHA:
```
--cache-type-k tbq3_1 --cache-type-v tbq3_1 --flash-attn on
```

MLA:
```
--cache-type-k tbq3_2 --cache-type-v f16
```

TBQ3_2 on MLA uses the proprietary Lane B fused matvec kernel (included
in this binary) — closes most of the throughput gap vs IQ4_NL.

## Measured numbers (P40, reference hardware)

### MHA — Mistral Small 24B, 32K ctx

| Preset | Gen tok/s | Δ vs f16 | Compression |
|---|---|---|---|
| `f16` | 17.5 | 0 | 1x |
| `iq4_nl` | 15.4 | -12% | 3.56x |
| `tbq3_2` | 14.6 | -17% | 4.57x |
| `tbq3_1` GPU KV | 12.9 | -26% | 5.12x |
| `tbq3_1` CPU KV (fastest TBQ3_1) | 15.9 | -9% | 5.12x |

### MLA — Mistral Large 3, 16K ctx, 4x P40

| Preset | Gen tok/s | Δ vs iq4_nl | K cache @16K |
|---|---|---|---|
| `q8_0` K + f16 V | 3.69 | +11% | 540 MiB |
| `iq4_nl` K + f16 V | 3.32 | 0 | 308 MiB |
| `tbq3_2` K + f16 V (Lane B) | 2.85 | -14% | 240 MiB |

### MLA — DeepSeek V2 Lite, 8K ctx, single P40

| Preset | Gen tok/s | Δ vs iq4_nl |
|---|---|---|
| `iq4_nl` K + f16 V | 72.3 | 0 |
| `tbq3_2` K + f16 V (Lane B) | 63-65 | -11% |

## Quality (perplexity on Mistral Small 24B, wiki.test.raw, 30 chunks, ctx=512)

| Preset | PPL | Δ vs f16 |
|---|---|---|
| `f16` | 5.3663 ± 0.137 | — |
| `iq4_nl` | 5.3944 ± 0.138 | +0.52% |
| `tbq3_2` | 5.4971 ± 0.139 | +2.44% |
| `tbq3_1` | 5.4881 ± 0.139 | +2.27% |

All within acceptable range; error bars overlap between f16 and IQ4_NL.

## Troubleshooting

- **"quantized V cache was requested, but this requires Flash Attention"**
  → your model is MLA on Pascal (no FA for head_dim=576). Use `-cache-type-v f16`.

- **Gibberish output on Qwen2.5-VL with quantized KV + FA on Pascal**
  → known upstream issue (mrope + FA interaction on SM61). Use `-cache-type-k f16`.

- **Speculative decoding refuses to load with `--mmproj`**
  → known upstream llama.cpp limit. Use one or the other.
