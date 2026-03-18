# Qwen3-ASR in llama.cpp

Native Qwen3-ASR-1.7B speech recognition running entirely in llama.cpp, integrated with AlignAtt simultaneous MT.

## Quick Start

```bash
# SSH to A40
ssh -p 3622 fuxa@quest.ms.mff.cuni.cz
cd /home/fuxa/llama.cpp

# ASR only
build/bin/llama-mtmd-cli \
    -m ~/qwen3-asr-1.7b-decoder-v2.gguf \
    --mmproj ~/qwen3-asr-1.7b-mmproj-v4.gguf \
    --chat-template chatml \
    -p "<__media__>Detect the language and recognize the speech." \
    --audio test.wav -n 500 --temp 0

# Full simultaneous cascade (ASR + AlignAtt MT)
python3 tools/simultaneous_cascade.py test.wav output.jsonl

# Batch cascade (ASR + full MT, higher quality)
python3 tools/cascade_test.py test.wav
```

## Architecture

```
Audio WAV (16kHz mono)
  |
  v
[Whisper mel preprocessing] 128 bins, 3000 frames (30s)
  |
  v
[Chunked Conv2D encoder] 30 chunks x 100 frames -> 390 tokens x 1024 dim
  |  - 3 layers Conv2D(stride=2, pad=1) + GELU
  |  - Per-chunk positional embeddings (sinusoidal)
  |  - conv_out linear projection (7680 -> 1024)
  |
  v
[Windowed Transformer] 24 layers, 104-token attention windows
  |
  v
[Projector MLP] 1024 -> GELU -> 2048
  |
  v
[Qwen3 Decoder] 28 layers, generates text
  |
  v
"language English<asr_text>transcription here"
```

## GGUF Models

| File | Size | Description |
|------|------|-------------|
| `qwen3-asr-1.7b-decoder-v2.gguf` | 3.8 GB | Decoder with fixed BOS/EOS tokens |
| `qwen3-asr-1.7b-mmproj-v4.gguf` | 641 MB | Encoder with reordered conv_out weight |
| `Qwen3.5-4B-Q8_0.gguf` | 4.2 GB | MT model (AlignAtt compatible) |

### Converting from HuggingFace

```bash
# Decoder
python3 convert_hf_to_gguf.py Qwen/Qwen3-ASR-1.7B --outfile decoder.gguf

# Encoder (mmproj)
python3 convert_hf_to_gguf.py Qwen/Qwen3-ASR-1.7B --mmproj --outfile mmproj.gguf
```

The conversion includes:
- BOS/EOS token fix (151643/151645 instead of default comma)
- conv_out weight reordering (F-fastest to C-fastest for ggml permute)
- conv2d bias unsqueeze for broadcast

## Key Technical Fixes

### 1. Chunked Conv2D (not global)
The encoder MUST split mel into 100-frame chunks. Global processing gives cosine similarity 0.40 with reference. Chunked gives 1.0.

### 2. Windowed Attention (104-token windows)
The transformer uses block-diagonal attention masks with 104-token windows (13 tokens/chunk x 8 chunks). Global attention degrades quality.

### 3. conv_out Weight Reorder
ggml's `ggml_cont` for 4D permutation produces C-fastest memory order. PyTorch's flatten produces F-fastest. The conv_out weight's 7680 dimension is reordered during GGUF conversion: `weight.reshape(1024, 480, 16).permute(0, 2, 1).reshape(1024, 7680)`.

### 4. BOS/EOS Token Fix
Qwen3-ASR config has `bos_token_id: None`, causing GGUF to default to token 11 (comma) as EOS. Fixed to 151643 (endoftext) and 151645 (im_end).

## Performance (A40)

| Stage | Time | Notes |
|-------|------|-------|
| Encoder (30s audio) | 242 ms | Conv2D + transformer |
| Prompt eval (390 tokens) | 291 ms | 1400 tok/s |
| Generation | 7.15 ms/tok | 140 tok/s |
| Total ASR (3.5s audio) | 3.8s | Including model load |
| Total ASR (40s audio) | 4.3s | Encoder-bound |

## Branch

`qwen3-asr-integration` on A40 at `/home/fuxa/llama.cpp/`

Built on top of the attention extraction API (PR #20086) from `QuentinFuxa/llama.cpp`.

## Files Changed

- `convert_hf_to_gguf.py` -- Qwen3ASR model/mmproj conversion
- `tools/mtmd/models/qwen3a.cpp` -- Audio encoder graph
- `tools/mtmd/clip.cpp` -- Chunked data feeding, hybrid mode, debug
- `tools/mtmd/clip-graph.h` -- build_vit with optional attention mask
- `gguf-py/gguf/constants.py` -- QWEN3A projector type
- `gguf-py/gguf/tensor_mapping.py` -- Audio tower tensor names
- `tools/simultaneous_cascade.py` -- End-to-end cascade script
- `tools/cascade_test.py` -- Batch cascade test
