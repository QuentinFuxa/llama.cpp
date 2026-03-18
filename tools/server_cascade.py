#!/usr/bin/env python3
"""
Server-mode cascade: both ASR and MT run as persistent llama-server instances.
ASR: port 8080, MT: port 8081.

Start servers:
  llama-server -m qwen3-asr-1.7b-decoder-v2.gguf --mmproj qwen3-asr-1.7b-mmproj-v4.gguf --chat-template chatml -ngl 99 --port 8080
  llama-server -m Qwen3-4B-Q8_0.gguf --chat-template chatml -ngl 99 --port 8081
"""
import requests, base64, time, re, sys

ASR_URL = "http://localhost:8080/v1/chat/completions"
MT_URL = "http://localhost:8081/v1/chat/completions"

def cascade(audio_path, target_lang="Chinese"):
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    # ASR
    t0 = time.time()
    r1 = requests.post(ASR_URL, json={
        "messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
            {"type": "text", "text": "Detect the language and recognize the speech."}
        ]}],
        "max_tokens": 500, "temperature": 0,
    }, timeout=60)
    asr_time = time.time() - t0
    asr_raw = r1.json()["choices"][0]["message"]["content"]
    en_text = re.sub(r"<\|[^>]+\|>", "", asr_raw.split("<asr_text>")[-1]).strip()

    # MT
    t0 = time.time()
    r2 = requests.post(MT_URL, json={
        "messages": [{"role": "user", "content": f"Translate to {target_lang}: {en_text}"}],
        "max_tokens": 1000, "temperature": 0,
    }, timeout=300)
    mt_time = time.time() - t0
    mt_raw = r2.json()["choices"][0]["message"]["content"]
    mt_text = mt_raw.split("</think>")[-1].strip() if "</think>" in mt_raw else mt_raw
    mt_text = re.sub(r"<\|[^>]+\|>", "", mt_text).strip()

    return {"en": en_text, "zh": mt_text, "asr_time": asr_time, "mt_time": mt_time}

if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_speech_real.wav"
    r = cascade(audio)
    print(f"EN: {r['en']}")
    print(f"ZH: {r['zh']}")
    print(f"Time: {r['asr_time']:.2f}s ASR + {r['mt_time']:.2f}s MT = {r['asr_time']+r['mt_time']:.2f}s total")
