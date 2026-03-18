#!/usr/bin/env python3
"""
Bridge: Qwen3-ASR (llama.cpp) -> AlignAtt Streaming MT (llama.cpp)

Runs ASR via llama-mtmd-cli, converts output to word-level JSONL,
then feeds to alignatt_simulstreaming_llamacpp.py for streaming MT.
"""
import subprocess, sys, os, json, time, re, tempfile

LLAMA_DIR = "/home/fuxa/llama.cpp"
IWSLT_DIR = "/home/fuxa/iwslt26-sst/SimulMT_tests"
ASR_DEC = "/home/fuxa/qwen3-asr-1.7b-decoder-v2.gguf"
ASR_ENC = "/home/fuxa/qwen3-asr-1.7b-mmproj-v4.gguf"
MT_MODEL = "/home/fuxa/Qwen3.5-4B-Q8_0.gguf"
MT_HEADS = f"{IWSLT_DIR}/heads/translation_heads_qwen3.5_4b_en_zh.json"

def run_asr(audio_path):
    """Run ASR and return transcribed text."""
    r = subprocess.run([
        f"{LLAMA_DIR}/build/bin/llama-mtmd-cli",
        "-m", ASR_DEC, "--mmproj", ASR_ENC,
        "--chat-template", "chatml",
        "-p", "<__media__>Detect the language and recognize the speech.",
        "--audio", audio_path, "-n", "500", "--temp", "0"
    ], capture_output=True, text=True, timeout=120)

    for line in (r.stdout + r.stderr).split("\n"):
        if "<asr_text>" in line:
            text = line.split("<asr_text>")[-1]
            return re.sub(r"<\|[^>]+\|>", "", text).strip()
    return ""

def text_to_word_jsonl(text, audio_duration=30.0):
    """Convert text to word-level JSONL (simulating streaming ASR output)."""
    words = text.split()
    if not words:
        return []

    # Distribute emission times evenly across audio duration
    time_per_word = audio_duration / len(words)

    entries = []
    for i, word in enumerate(words):
        t = (i + 1) * time_per_word
        entry = {
            "text": word,
            "emission_time": round(t, 3),
            "is_final": word.endswith(".") or word.endswith("!") or word.endswith("?") or i == len(words) - 1
        }
        entries.append(entry)
    return entries

def run_streaming_mt(jsonl_path, output_path):
    """Run AlignAtt streaming MT on word-level JSONL."""
    cmd = [
        "python3", f"{IWSLT_DIR}/alignatt/alignatt_simulstreaming_llamacpp.py",
        "--model", MT_MODEL,
        "--heads", MT_HEADS,
        "--border-distance", "3",
        "--context-sentences", "5",
        "--prompt-format", "qwen3.5-ctx-v5",
        "--kvcache",
        "--word-batch", "2",
        "--input", jsonl_path,
        "--output", output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return r

if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_speech_real.wav"
    output = sys.argv[2] if len(sys.argv) > 2 else "/tmp/cascade_output.jsonl"

    print(f"=== llama.cpp Full Simultaneous Cascade ===")
    print(f"Audio: {audio}")
    print(f"Output: {output}\n")

    # Stage 1: ASR
    t0 = time.time()
    asr_text = run_asr(audio)
    asr_time = time.time() - t0
    print(f"[ASR] ({asr_time:.1f}s) {asr_text}")

    if not asr_text:
        print("ASR FAILED!"); sys.exit(1)

    # Convert to word JSONL
    entries = text_to_word_jsonl(asr_text)
    jsonl_path = "/tmp/asr_words.jsonl"
    with open(jsonl_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print(f"[JSONL] {len(entries)} words written to {jsonl_path}")

    # Stage 2: Streaming MT with AlignAtt
    print(f"[MT] Starting AlignAtt streaming translation...")
    t0 = time.time()
    r = run_streaming_mt(jsonl_path, output)
    mt_time = time.time() - t0

    # Read output
    if os.path.exists(output):
        with open(output) as f:
            mt_lines = [json.loads(l) for l in f if l.strip()]
        finals = [l for l in mt_lines if l.get("is_final")]
        mt_text = " ".join(l.get("text", "") for l in finals)
        print(f"[MT] ({mt_time:.1f}s) {mt_text}")
    else:
        print(f"[MT] ({mt_time:.1f}s) Output file not created")
        if r.stderr:
            print(f"  stderr: {r.stderr[-200:]}")

    print(f"\n=== RESULT ===")
    print(f"EN: {asr_text}")
    print(f"ZH: {mt_text if 'mt_text' in dir() else 'N/A'}")
    print(f"Total: {asr_time + mt_time:.1f}s")
