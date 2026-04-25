# Setup guide — Skylark llama-server

Step-by-step install + first-run for Linux x86_64 with NVIDIA GPUs.

## 1. Verify your system

```bash
# OS — Ubuntu 22.04+ / RHEL 9+ / Debian 12+ / Fedora 38+ all known to work
cat /etc/os-release | grep -E "PRETTY_NAME|VERSION_ID"

# glibc — need 2.35 or newer
ldd --version | head -1

# NVIDIA driver — need a CUDA 12+ compatible driver
nvidia-smi
```

If `nvidia-smi` reports a driver version of **535.x or newer** (CUDA 12.2+),
you're good. Older drivers won't load this binary because it's compiled for
CUDA 12+ runtime ABIs.

## 2. Verify supported GPU

This binary contains compiled SASS for compute capabilities **6.1, 7.0, 7.5,
8.0, 8.6, 8.9, 9.0** — Pascal (P40, P100, GTX 1080/Ti) through Hopper (H100,
H200). No JIT fallback. Check yours:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

Examples:

| Output | Compatible? |
|---|---|
| `Tesla P40, 6.1` | ✅ Pascal |
| `NVIDIA A100 40GB, 8.0` | ✅ Ampere |
| `NVIDIA H100 80GB, 9.0` | ✅ Hopper |
| `NVIDIA L40, 8.9` | ✅ Ada |
| `Tesla P4, 6.1` | ✅ Pascal |
| `Tesla M40, 5.2` | ❌ Maxwell — not supported |
| `Blackwell SM100+` | ❌ would need a future build |

## 3. Verify the download

```bash
sha256sum skylark-llama-server-v1.0-linux-x86_64-cuda.tar.gz
# expect: c9b4948f7cefe3e5eb5f8a69baae25e2be8e4868362b6efb9647f11f3ea1d438
```

If it doesn't match, redownload — the file is corrupted or tampered with.

## 4. Extract

```bash
tar -xzf skylark-llama-server-v1.0-linux-x86_64-cuda.tar.gz
cd skylark-llama-server-v1.0-linux-x86_64-cuda
```

You should see:
- `llama-server` (the binary)
- `lib/` (libggml*, libllama, libmtmd shared libraries)
- `NOTICE`, `README.md`, `SETUP.md` (this file), `kv-cache-guide.md`

## 5. Smoke test — does it run?

```bash
./llama-server --version
./llama-server --help | head -20
```

Expected output: GPU detection lines for each card you have, then llama-server
help text. The binary uses `RPATH=$ORIGIN/lib` so libraries resolve from the
same directory — no `LD_LIBRARY_PATH` setup needed.

## 6. Get a model

llama-server uses GGUF files. Common sources:

```bash
# Mistral Small 24B Q4_K_M — about 14 GB, works on a single 24GB GPU
huggingface-cli download bartowski/Mistral-Small-24B-Instruct-2501-GGUF \
  Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf --local-dir ./models

# DeepSeek V2 Lite Q4_K_M — about 10 GB, good for testing MLA on 1 GPU
huggingface-cli download mradermacher/DeepSeek-V2-Lite-GGUF \
  DeepSeek-V2-Lite.Q4_K_M.gguf --local-dir ./models
```

(Install `huggingface-cli` via `pip install -U "huggingface_hub[cli]"`.)

## 7. First inference run

### Mistral Small 24B (MHA — head_dim=128)

```bash
./llama-server \
  -m models/Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 32768 -ngl 99 --no-mmap \
  --cache-type-k tbq3_1 --cache-type-v tbq3_1 --flash-attn on
```

This uses TBQ3_1 KV cache (5.12× compression vs f16, +2.27% PPL on this model).
Open `http://localhost:8080` in a browser, or hit the OpenAI-compatible API
at `http://localhost:8080/v1/chat/completions`.

### DeepSeek V2 Lite (MLA — head_dim=576)

```bash
./llama-server \
  -m models/DeepSeek-V2-Lite.Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 8192 -ngl 99 --no-mmap \
  --cache-type-k tbq3_2 --cache-type-v f16
```

MLA models need `V=f16` because Pascal lacks Flash Attention for `head_dim=576`,
and quantized V requires FA. K can still be TBQ3_2 for memory savings.

## 8. Pick the right KV preset

See `kv-cache-guide.md` for the full decision tree. Quick rules:

- **MHA models** (most LLMs — Mistral, Llama, Qwen, etc.): `--cache-type-k tbq3_1 --cache-type-v tbq3_1 --flash-attn on` for max compression
- **MLA models** (DeepSeek V2/V3/R1, Mistral Large 3, Kimi K2.5): `--cache-type-k tbq3_2 --cache-type-v f16` (no `--flash-attn`)
- **Don't trust quant KV?** Start with `--cache-type-k iq4_nl` to compare quality
- **Plenty of memory?** `--cache-type-k f16 --cache-type-v f16` for max throughput
- **Qwen2-VL specifically**: stick with `f16` everywhere — known mainline issue with mrope+FA on Pascal

## 9. Multi-GPU + large MoE setup

For models that need multiple GPUs (Mistral Large 3 675B, Kimi K2.5 1T,
DeepSeek R1 671B), use `--tensor-split` or `-ot` to control layer placement.

Example for Mistral Large 3 on 4× P40:

```bash
numactl --interleave=all ./llama-server \
  -m models/mistral-large-3-q4_k_m-00001-of-00011.gguf \
  --host 0.0.0.0 --port 8082 \
  -c 16384 -ngl 99 --no-mmap \
  -ot "blk\.[34]\.ffn_.*_exps=CUDA1,blk\.[56]\.ffn_.*_exps=CUDA2,blk\.[78]\.ffn_.*_exps=CUDA3,.*_exps=CPU" \
  --cache-type-k tbq3_2 --cache-type-v f16 \
  --numa distribute -np 1 --prio 2 --poll 50
```

Specifics depend on your GPU memory and CPU RAM. Start with the upstream
[llama.cpp docs on tensor-split + MoE expert pinning](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

## Troubleshooting

### "CUDA error: forward compatibility was attempted"

Your driver is older than required. Update to **535.x or newer** for CUDA 12+.

### "ggml_cuda_init: failed to initialize CUDA: ..."

`nvidia-smi` works but llama-server can't see CUDA. Usually a driver/runtime
version mismatch — verify with `nvidia-smi` and check your driver matches.

### "quantized V cache was requested, but this requires Flash Attention"

You're running an MLA model on Pascal with `--cache-type-v` set to a quant
type. Pascal can't run Flash Attention on MLA's `head_dim=576`. Use
`--cache-type-v f16` instead.

### Output is gibberish on Qwen2-VL with quant KV

Known mainline issue with mrope + FA on Pascal. Use `--cache-type-k f16
--cache-type-v f16` for Qwen2-VL specifically.

### "out of memory" with multi-GPU MoE

Reduce context (`-c`), reduce GPU expert count (less aggressive `-ot` patterns
that route more experts to CPU), or use a smaller quant of the model.

### Server starts but generation is very slow on MLA

Make sure you're using the binary's RPATH (don't set `LD_LIBRARY_PATH`
manually) — overriding the bundled libs with system ggml/llama can revert
you to the unpatched MLA path that crashes or stalls.

## Support

- Licensing, commercial use, source access: **info@skylarksoftware.me**
- Upstream llama.cpp issues (features, base-library bugs): https://github.com/ggml-org/llama.cpp/issues

This binary doesn't have its own issue tracker.
