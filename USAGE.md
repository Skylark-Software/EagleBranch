# Usage guide — Skylark llama-server

Practical examples of how to actually talk to the server once it's
running. Assumes you've followed [SETUP.md](./SETUP.md) and have
`llama-server` listening on `http://localhost:8080`.

> Looking for the **launch switches** instead? KV-cache compression types
> (`--cache-type-*`, TurboQuant) are in [kv-cache-guide.md](./kv-cache-guide.md);
> speculative decoding (`--eagle3`, draft models) is in [EAGLE3.md](./EAGLE3.md).

## Web UI

The simplest path: open `http://localhost:8080` in any browser. Built-in
chat UI, slot status panel, and config inspector. No additional setup
needed.

## OpenAI-compatible API

The server speaks the OpenAI Chat Completions API on `/v1/chat/completions`,
which means **any OpenAI client library works** by pointing `base_url` at
your llama-server.

### curl

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user",   "content": "Say hello in one short sentence."}
    ],
    "temperature": 0.7,
    "max_tokens": 60
  }'
```

`"model"` is ignored by llama-server (you set the model when you launched
the server). Pass any string.

### Python (openai client)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="local",  # required by the client lib but not validated
)

response = client.chat.completions.create(
    model="local",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Say hello in one short sentence."},
    ],
    temperature=0.7,
    max_tokens=60,
)
print(response.choices[0].message.content)
```

### Streaming

Add `"stream": true` to the request body. The server emits Server-Sent
Events with the same shape as OpenAI's streaming API.

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Count from 1 to 5 slowly."}],
    "stream": true,
    "max_tokens": 50
  }'
```

```python
stream = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Count from 1 to 5 slowly."}],
    stream=True,
    max_tokens=50,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
print()
```

## Native llama.cpp endpoints (lower-level)

Useful when you want fine control or when interacting from shell scripts.

### `/completion` — raw completion (no chat formatting)

```bash
curl -s http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "n_predict": 20,
    "temperature": 0
  }' | jq -r .content
```

### `/health` — server liveness

```bash
curl -s http://localhost:8080/health
# {"status":"ok"}
```

### `/tokenize` — see how text becomes tokens

```bash
curl -s http://localhost:8080/tokenize \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello world!"}' | jq
```

### `/slots` — what generation slots are doing right now

```bash
curl -s http://localhost:8080/slots | jq '.[].state'
```

## Generation parameters worth knowing

| Parameter | Default | Notes |
|---|---|---|
| `max_tokens` / `n_predict` | -1 (unlimited) | Cap per-request output length |
| `temperature` | 0.8 | 0 = deterministic, higher = more random |
| `top_p` | 0.95 | Nucleus sampling cutoff |
| `top_k` | 40 | Limit vocabulary at each step |
| `min_p` | 0.05 | Cut tokens below this fraction of top probability |
| `repeat_penalty` | 1.1 | Discourage repetition (>1.0 penalizes) |
| `seed` | -1 | Set for reproducible outputs |
| `stop` | `[]` | Array of strings; generation halts on any match |
| `cache_prompt` | true | Reuse KV across requests with shared prefix |

Full list: `curl http://localhost:8080/props` after starting the server.

## Speculative decoding (binary distribution)

To get a draft model accelerating generation, pass `-md` (draft model)
when launching the server. The proprietary binary supports EAGLE v1/v2/v3
and MTP/NextN draft heads; mainline llama.cpp supports vanilla speculative.

### DeepSeek R1 with MTP/NextN draft

```bash
./llama-server \
  -m models/DeepSeek-R1-Q4_K_M.gguf \
  -md models/deepseek-r1-nextn-q8_0.gguf \
  --eagle3 \
  --host 0.0.0.0 --port 8083 \
  -c 6144 -ngl 99 --no-mmap \
  --cache-type-k iq4_nl --cache-type-v f16 \
  --draft-max 8 --draft-min 0
```

### Mistral Large 3 with EAGLE v1 draft

```bash
./llama-server \
  -m models/Mistral-Large-3-Q4_K_M.gguf \
  -md models/mistral-eagle-v1-q8_0.gguf \
  --eagle3 \
  --host 0.0.0.0 --port 8082 \
  -c 16384 -ngl 99 --no-mmap \
  --cache-type-k tbq3_2 --cache-type-v f16 \
  --draft-max 8 --draft-min 0
```

The server reports speculation acceptance rate in the per-request timing
output — look for `n_drafted` / `n_accept` in the `/slots` response or
server log.

**Limitation**: speculative decoding is incompatible with `--mmproj`
(vision encoders) in upstream llama.cpp. If you need vision, run
without speculation.

## Multimodal (vision)

Pass `--mmproj` pointing at a vision encoder GGUF. The model must have
been trained with vision (Pixtral, Qwen2.5-VL, Kimi K2.5 multimodal,
Mistral Large 3 multimodal, etc.).

```bash
./llama-server \
  -m models/Pixtral-12B-Q4_K_M.gguf \
  --mmproj models/Pixtral-12B-mmproj-F16.gguf \
  --host 0.0.0.0 --port 8094 \
  -c 8192 -ngl 99 --no-mmap \
  --cache-type-k iq4_nl --cache-type-v iq4_nl --flash-attn on
```

Send images via the OpenAI vision API format:

```bash
curl http://localhost:8094/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
      ]
    }],
    "max_tokens": 200
  }'
```

Local files via `data:image/jpeg;base64,...` URI also work.

## Multi-instance (run several models concurrently)

Launch each on its own port + GPU:

```bash
# First instance — Mistral Small on GPU 0
CUDA_VISIBLE_DEVICES=0 ./llama-server \
  -m models/Mistral-Small-24B-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8090 \
  -c 32768 -ngl 99 --no-mmap \
  --cache-type-k tbq3_1 --cache-type-v tbq3_1 --flash-attn on &

# Second instance — Pixtral on GPU 1
CUDA_VISIBLE_DEVICES=1 ./llama-server \
  -m models/Pixtral-12B-Q4_K_M.gguf \
  --mmproj models/Pixtral-12B-mmproj-F16.gguf \
  --host 0.0.0.0 --port 8094 \
  -c 8192 -ngl 99 --no-mmap \
  --cache-type-k iq4_nl --cache-type-v iq4_nl --flash-attn on &
```

Each gets its own `/v1/chat/completions` endpoint at its respective port.

## Stopping the server

```bash
# foreground: Ctrl-C
# background: find and kill
pkill -f 'llama-server.*--port 8080'
```

The server flushes per-request state cleanly on SIGINT/SIGTERM.

## Where to next

- [SETUP.md](./SETUP.md) — install + first run
- [kv-cache-guide.md (in the bundle)](https://github.com/Skylark-Software/EagleBranch/releases/tag/v1.0) — pick the right `--cache-type-*` for your model
- Upstream llama.cpp [server docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) — full endpoint reference
- [OpenAI API reference](https://platform.openai.com/docs/api-reference/chat) — for client-side parameter docs

## Support

Licensing, commercial use, source access for the proprietary extensions:
**info@skylarksoftware.me**
