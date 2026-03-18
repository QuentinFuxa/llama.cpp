#!/usr/bin/env python3
"""
Python wrapper for Qwen3-ASR running in llama.cpp.

Provides a simple API that matches the WhisperLiveKit interface:
  asr = LlamaCppASR(decoder_path, mmproj_path)
  text = asr.transcribe(audio_path)
  text = asr.transcribe_array(audio_np, sr=16000)

Can be used as a drop-in replacement for the PyTorch Qwen3-ASR backend.
"""
import subprocess
import tempfile
import wave
import struct
import re
import os
import time
import numpy as np
from typing import Optional


class LlamaCppASR:
    """Qwen3-ASR-1.7B via llama-mtmd-cli."""

    def __init__(
        self,
        decoder_path: str = "/home/fuxa/qwen3-asr-1.7b-decoder-v2.gguf",
        mmproj_path: str = "/home/fuxa/qwen3-asr-1.7b-mmproj-v4.gguf",
        llama_bin: str = "/home/fuxa/llama.cpp/build/bin/llama-mtmd-cli",
        n_gpu_layers: int = 99,
        max_tokens: int = 500,
    ):
        self.decoder_path = decoder_path
        self.mmproj_path = mmproj_path
        self.llama_bin = llama_bin
        self.n_gpu_layers = n_gpu_layers
        self.max_tokens = max_tokens

        # Verify files exist
        for f in [decoder_path, mmproj_path, llama_bin]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Not found: {f}")

    def transcribe(self, audio_path: str, timeout: int = 120) -> dict:
        """Transcribe a WAV file. Returns dict with 'text', 'language', 'time'."""
        t0 = time.time()
        cmd = [
            self.llama_bin,
            "-m", self.decoder_path,
            "--mmproj", self.mmproj_path,
            "--chat-template", "chatml",
            "-p", "<__media__>Detect the language and recognize the speech.",
            "--audio", audio_path,
            "-n", str(self.max_tokens),
            "--temp", "0",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - t0

        # Parse output
        text = ""
        language = ""
        for line in (result.stdout + result.stderr).split("\n"):
            if "<asr_text>" in line:
                parts = line.split("<asr_text>")
                if len(parts) >= 2:
                    # Extract language
                    lang_part = parts[0]
                    if "language " in lang_part:
                        language = lang_part.split("language ")[-1].strip()
                    # Extract text
                    text = re.sub(r"<\|[^>]+\|>", "", parts[-1]).strip()
                break

        return {
            "text": text,
            "language": language,
            "time": elapsed,
        }

    def transcribe_array(self, audio: np.ndarray, sr: int = 16000, timeout: int = 120) -> dict:
        """Transcribe a numpy audio array. Saves to temp WAV first."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            # Write WAV
            samples = (audio * 32767).astype(np.int16)
            with wave.open(tmp_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(samples.tobytes())

            return self.transcribe(tmp_path, timeout=timeout)
        finally:
            os.unlink(tmp_path)


class LlamaCppCascade:
    """Full cascade: ASR + MT, both via llama.cpp."""

    def __init__(
        self,
        asr: Optional[LlamaCppASR] = None,
        mt_model: str = "/home/fuxa/Qwen3-4B-Q8_0.gguf",
        llama_completion: str = "/home/fuxa/llama.cpp/build/bin/llama-completion",
        target_lang: str = "Chinese",
        n_gpu_layers: int = 99,
        max_mt_tokens: int = 500,
    ):
        self.asr = asr or LlamaCppASR()
        self.mt_model = mt_model
        self.llama_completion = llama_completion
        self.target_lang = target_lang
        self.n_gpu_layers = n_gpu_layers
        self.max_mt_tokens = max_mt_tokens

    def translate(self, audio_path: str) -> dict:
        """Transcribe + translate an audio file."""
        # ASR
        asr_result = self.asr.transcribe(audio_path)
        if not asr_result["text"]:
            return {"asr": asr_result, "translation": "", "total_time": asr_result["time"]}

        # MT
        t0 = time.time()
        prompt = (
            f"<|im_start|>user\n"
            f"Translate to {self.target_lang}: {asr_result['text']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        result = subprocess.run([
            self.llama_completion, "-m", self.mt_model,
            "-ngl", str(self.n_gpu_layers),
            "-n", str(self.max_mt_tokens),
            "--temp", "0", "-e", "--no-display-prompt",
            "-p", prompt,
        ], capture_output=True, text=True, timeout=300)
        mt_time = time.time() - t0

        # Extract translation (after </think>)
        mt_out = result.stdout
        if "</think>" in mt_out:
            translation = mt_out.split("</think>")[-1].strip()
        else:
            translation = mt_out.strip()
        translation = re.sub(r"<\|[^>]+\|>", "", translation).strip()

        return {
            "asr": asr_result,
            "translation": translation,
            "mt_time": mt_time,
            "total_time": asr_result["time"] + mt_time,
        }


if __name__ == "__main__":
    import sys

    audio = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_speech_real.wav"

    # Test ASR
    print("=== ASR Test ===")
    asr = LlamaCppASR()
    result = asr.transcribe(audio)
    print(f"Language: {result['language']}")
    print(f"Text: {result['text']}")
    print(f"Time: {result['time']:.1f}s")

    # Test Cascade
    print("\n=== Cascade Test ===")
    cascade = LlamaCppCascade(asr=asr)
    result = cascade.translate(audio)
    print(f"EN: {result['asr']['text']}")
    print(f"ZH: {result['translation']}")
    print(f"Total: {result['total_time']:.1f}s")
