#!/usr/bin/env python3
"""Full llama.cpp cascade: Qwen3-ASR -> Qwen3-4B MT (EN->ZH)"""
import subprocess, sys, time, re

LLAMA = "/home/fuxa/llama.cpp/build/bin"
ASR_DEC = "/home/fuxa/qwen3-asr-1.7b-decoder-v2.gguf"
ASR_ENC = "/home/fuxa/qwen3-asr-1.7b-mmproj-v4.gguf"
MT_MODEL = "/home/fuxa/Qwen3-4B-Q8_0.gguf"

audio = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_speech_real.wav"
print(f"=== llama.cpp Full Cascade: EN->ZH ===\nAudio: {audio}\n")

# Stage 1: ASR
t0 = time.time()
r = subprocess.run([
    f"{LLAMA}/llama-mtmd-cli", "-m", ASR_DEC, "--mmproj", ASR_ENC,
    "--chat-template", "chatml",
    "-p", "<__media__>Detect the language and recognize the speech.",
    "--audio", audio, "-n", "500", "--temp", "0"
], capture_output=True, text=True, timeout=120)
asr_time = time.time() - t0

asr_text = ""
for line in (r.stdout + r.stderr).split("\n"):
    if "<asr_text>" in line:
        asr_text = line.split("<asr_text>")[-1]
        asr_text = re.sub(r"<\|[^>]+\|>", "", asr_text).strip()
        break

print(f"[ASR] ({asr_time:.1f}s) {asr_text}")
if not asr_text:
    print("ASR FAILED!"); sys.exit(1)

# Stage 2: MT
t0 = time.time()
prompt = (
    f"<|im_start|>user\n"
    f"Translate to Chinese: {asr_text}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)
r = subprocess.run([
    f"{LLAMA}/llama-completion", "-m", MT_MODEL,
    "-ngl", "99", "-n", "500", "--temp", "0", "-e",
    "--no-display-prompt", "-p", prompt
], capture_output=True, text=True, timeout=300)
mt_time = time.time() - t0

# Extract translation (after </think>)
mt_out = r.stdout
if "</think>" in mt_out:
    mt_text = mt_out.split("</think>")[-1].strip()
else:
    mt_text = mt_out.strip()
mt_text = re.sub(r"<\|[^>]+\|>", "", mt_text).strip()

print(f"[MT]  ({mt_time:.1f}s) {mt_text}")
print(f"\n=== RESULT ===")
print(f"EN: {asr_text}")
print(f"ZH: {mt_text}")
print(f"Total: {asr_time + mt_time:.1f}s")
