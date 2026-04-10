import os
import wave
import subprocess
import threading
import queue
import tempfile
import pygame
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
pygame.init()

# ─── Configuration ────────────────────────────────────────────────────────────

PIPER_MODEL = os.getenv(
    "PIPER_MODEL",
    str(Path(__file__).parent.parent / "tts_models" / "en_US-lessac-medium.onnx")
)

# ─── TTS Engine ───────────────────────────────────────────────────────────────

class TTSEngine:
    """
    Converts text to speech using Piper TTS.
    Supports both simple one-shot synthesis and a streaming queue
    that lets you feed sentences in while audio is still playing.
    """

    def __init__(self):
        self._verify_model()
        self._playback_queue = queue.Queue()
        self._playback_thread = threading.Thread(
            target=self._playback_worker,
            daemon=True
        )
        self._playback_thread.start()
        print(f"[TTS] Engine ready. Model: {PIPER_MODEL}")

    def _verify_model(self):
        model_path = Path(PIPER_MODEL)
        if not model_path.exists():
            raise FileNotFoundError(
                f"[TTS] Model not found at {PIPER_MODEL}\n"
                f"      Make sure you downloaded en_US-lessac-medium.onnx "
                f"into the tts_models folder."
            )

    def _synthesize(self, text: str) -> str:
        """
        Run Piper on a string of text and return the path to a wav file.
        Uses a temp file so we don't litter the project with audio files.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()

        try:
            proc = subprocess.run(
                [
                    "python", "-m", "piper",
                    "--model", PIPER_MODEL,
                    "--output_file", tmp.name,
                ],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=15
            )
            if proc.returncode != 0:
                print(f"[TTS] Piper error: {proc.stderr.decode()}")
                return None
        except subprocess.TimeoutExpired:
            print("[TTS] Piper timed out.")
            return None

        return tmp.name

    def _play_wav(self, filepath: str):
        """Play a wav file using pygame and delete it afterward."""
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            print(f"[TTS] Playback error: {e}")
        finally:
            try:
                pygame.mixer.music.unload()
                os.unlink(filepath)
            except Exception:
                pass

    def _playback_worker(self):
        """
        Runs in a background thread. Pulls wav file paths off the queue
        and plays them one at a time in order. This means you can push
        multiple sentences to the queue and they play sequentially without
        blocking the main thread.
        """
        while True:
            filepath = self._playback_queue.get()
            if filepath is None:
                break
            self._play_wav(filepath)
            self._playback_queue.task_done()

    # ── Public API ─────────────────────────────────────────────────────────

    def speak(self, text: str):
        cleaned = text.strip()
        if not cleaned:
            return
        if len(cleaned) < 4:
            return
        if all(c in '.,!?;:()-–—"\'' for c in cleaned):
            return
        # Uncomment to debug TTS: print(f"[TTS] Speaking: {cleaned[:60]}")
        wav_path = self._synthesize(cleaned)
        if wav_path:
            self._playback_queue.put(wav_path)

    def speak_and_wait(self, text: str):
        """
        Speak a string and block until audio finishes.
        Useful for one-off lines like startup sounds or error messages.
        """
        self.speak(text)
        self._playback_queue.join()

    def wait_until_done(self):
        """Block until the playback queue is fully empty."""
        self._playback_queue.join()

    def stop(self):
        """Shut down the playback worker cleanly."""
        self._playback_queue.put(None)
        self._playback_thread.join()


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("TTS Module Test")
    print("=" * 50)

    tts = TTSEngine()

    # Test 1: basic speech
    print("\nTest 1 — single sentence")
    tts.speak_and_wait("Hello! I am Pip, your robot companion.")

    # Test 2: queued sentences play back to back
    print("\nTest 2 — queued sentences")
    tts.speak("This is the first sentence.")
    tts.speak("This is the second sentence, queued immediately.")
    tts.speak("And this is the third.")
    tts.wait_until_done()

    # Test 3: interactive
    print("\nTest 3 — type something and hear it spoken. Type quit to exit.")
    while True:
        try:
            text = input("Type: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if text.lower() == "quit":
            break
        if text:
            tts.speak_and_wait(text)

    tts.stop()
    print("\n[TTS] Done.")