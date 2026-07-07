# EAGLE3 speculative decoding — user guide

`--eagle3` enables EAGLE-family speculative decoding: a small **draft head**
predicts several tokens ahead, and the full target model verifies them in a
single batched pass. Verified tokens are accepted, mispredictions are
discarded — **output is bit-identical to normal decoding**; only the speed
changes. Whether it's a speed*up* depends on your hardware (see
[When it helps](#when-it-helps--honest-numbers) below).

Available both in the public `eagle3` source branch and in the
[Skylark release binaries](https://github.com/Skylark-Software/EagleBranch/releases).

## Supported models and modes

The fork adds an `eagle3_ds` architecture (MLA + MoE, DeepSeek-V2-family)
supporting both EAGLE v1/v2 and Eagle-3/MTP styles. The method is
auto-detected from GGUF metadata (`eagle_method`).

| Target model | Mode | Draft head |
|---|---|---|
| Mistral Large 3 (675B) | EAGLE v1/v2 | [Mistral Eagle head](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle) |
| DeepSeek R1 (671B) | Eagle-3 / MTP | [DeepSeek-R1-NextN](https://huggingface.co/lmsys/DeepSeek-R1-NextN) |
| DeepSeek V3 (671B) | Eagle-3 / MTP | built-in NextN/MTP module |
| DeepSeek V2 (236B) | EAGLE | any compatible EAGLE draft head |

## Getting a draft head

Draft heads are distributed as HF checkpoints (links above) and converted
to GGUF with the fork's converter, which auto-detects the EAGLE method:

```bash
python convert_hf_to_gguf.py /path/to/Mistral-Large-3-675B-Instruct-2512-Eagle \
  --outtype q8_0 --outfile mistral-eagle-v1-q8_0.gguf
```

## Switches

| Flag | Meaning |
|---|---|
| `--eagle3` | enable EAGLE3 speculative decoding with the draft model |
| `-md`, `--model-draft` | path to the draft head GGUF |
| `--draft`, `--draft-max` | max tokens to draft per step (typical: 4–8) |
| `--draft-min`, `--draft-n-min` | min tokens to draft per step |
| `--draft-p-min` | min draft probability to keep speculating (greedy cutoff) |
| `-devd` | device(s) for the draft model (e.g. `CUDA3` to pin it to one GPU) |
| `-ngld` | layers of the draft model to offload (`99` = all) |
| `--cache-type-k-draft` / `--cache-type-v-draft` | KV cache type for the draft model (see [kv-cache-guide.md](kv-cache-guide.md)) |

## Example — Mistral Large 3 on 4× P40

```bash
./llama-speculative-simple \
  -m /path/to/Mistral-Large-3-Q4_K_M.gguf \
  -md /path/to/mistral-eagle-v1-q8_0.gguf \
  --eagle3 --no-mmap -devd CUDA3 -ngld 99 \
  -ngl 4 -c 2048 --draft 8 -n 128 \
  -p "Write a short essay about the future of artificial intelligence."
```

`llama-server` accepts the same `-md`/`--eagle3`/draft flags for serving.

## When it helps — honest numbers

Speculation pays off when **target-model verification is cheap relative to
sequential decoding** — i.e. when the target has healthy GPU headroom.
Measured reference point (Mistral Large 3 675B Q4_K_M, 4× Tesla P40 +
503 GB RAM):

| Metric | Value |
|---|---|
| Draft acceptance rate | 64.5% (71/110 tokens) |
| With speculation | 2.86 tok/s |
| Baseline (no speculation) | 4.72 tok/s |

On this configuration speculation is a net **slowdown**: the massive MoE
target spans 4 GPUs + CPU (1000+ graph splits per verification pass), so
verification cost dominates and negates the ~2.3× tokens-per-step gain.
The 64.5% acceptance confirms the implementation; the economics need a
target that verifies faster (more/faster GPU offload, or a smaller target).

**Rule of thumb:** benchmark your own config with and without `--eagle3`
(`llama-bench` or a fixed prompt + `-n 128`) before adopting it. If your
target model fits fully in GPU memory, expect a win; if it spills heavily
to CPU, expect a loss.

## Interactions and limits

- Speculative decoding will not load together with `--mmproj`
  (multimodal projector) — upstream llama.cpp limitation. Use one or the other.
- The draft model needs its own KV cache — budget a little extra memory
  (`--cache-type-k-draft`/`-v-draft` can compress it; the draft is small,
  so `f16` is usually fine).
