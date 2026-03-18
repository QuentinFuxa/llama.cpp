#!/bin/bash
# Demo: Full llama.cpp Simultaneous Speech Translation
# Run on A40 VM at /home/fuxa/llama.cpp/
set -e

LLAMA=/home/fuxa/llama.cpp
ASR_DEC=$LLAMA/../../qwen3-asr-1.7b-decoder-v2.gguf
ASR_ENC=$LLAMA/../../qwen3-asr-1.7b-mmproj-v4.gguf

echo "=============================================="
echo " llama.cpp Simultaneous Speech Translation"
echo " Qwen3-ASR-1.7B + Qwen3.5-4B AlignAtt"
echo "=============================================="
echo

# Test 1: ASR only
echo "=== Test 1: ASR (Qwen3-ASR-1.7B) ==="
$LLAMA/build/bin/llama-mtmd-cli \
    -m /home/fuxa/qwen3-asr-1.7b-decoder-v2.gguf \
    --mmproj /home/fuxa/qwen3-asr-1.7b-mmproj-v4.gguf \
    --chat-template chatml \
    -p "<__media__>Detect the language and recognize the speech." \
    --audio /tmp/test_speech_real.wav \
    -n 200 --temp 0 2>&1 | grep "language"
echo

# Test 2: Full cascade (batch MT)
echo "=== Test 2: Batch Cascade (ASR + MT) ==="
python3 $LLAMA/tools/cascade_test.py /tmp/test_speech_real.wav
echo

# Test 3: Simultaneous cascade (AlignAtt streaming MT)
echo "=== Test 3: Simultaneous Cascade (ASR + AlignAtt MT) ==="
python3 $LLAMA/tools/simultaneous_cascade.py /tmp/test_speech_real.wav /tmp/demo_output.jsonl
echo

echo "=============================================="
echo " All tests complete!"
echo "=============================================="
