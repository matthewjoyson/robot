import sys
import os
import threading
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add project root to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from speech.llm import LLMClient, stream_to_sentences
from speech.tts import TTSEngine

# ─── Conversation Loop ────────────────────────────────────────────────────────

class PipConversation:
    """
    Wires the LLM and TTS together into a full conversation loop.
    LLM streams tokens → buffered into sentences → TTS speaks each
    sentence as it arrives, while the LLM is still generating the rest.
    """

    def __init__(self):
        print("[Pip] Starting up...")
        self.tts = TTSEngine()
        self.llm = LLMClient()
        self._speaking = False
        print("[Pip] Ready.")

    def _speak_stream(self, user_input: str):
        """
        Gets a streaming response from the LLM and feeds each complete
        sentence to TTS immediately. The user starts hearing the response
        before the LLM has finished generating it.
        """
        self._speaking = True
        stream = self.llm.chat_stream(user_input)

        for sentence in stream_to_sentences(stream):
            self.tts.speak(sentence)

        # Wait for all queued audio to finish before accepting next input
        self.tts.wait_until_done()
        self._speaking = False

    def say(self, text: str):
        """Speak a line directly without going through the LLM."""
        self.tts.speak_and_wait(text)

    def chat(self, user_input: str):
        """Send a message and speak the response."""
        self._speak_stream(user_input)

    def run(self):
        """
        Main interactive loop. Type messages, hear Pip respond.
        This is the text-based version — we'll add voice input next.
        """
        self.say("Hello! I'm Pip. How can I help you today?")

        print("\n" + "=" * 50)
        print("Pip is ready. Type a message and press Enter.")
        print("Commands: 'quit' to exit, 'clear' to reset memory")
        print("=" * 50 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n[Exiting]")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                self.say("Goodbye!")
                break

            if user_input.lower() == "clear":
                self.llm.history.clear()
                self.say("Memory cleared. Fresh start!")
                continue

            # Show a thinking indicator while waiting for first token
            print("Pip: ", end="", flush=True)
            full_response = []

            stream = self.llm.chat_stream(user_input)
            for sentence in stream_to_sentences(stream):
                full_response.append(sentence)
                print(sentence, end=" ", flush=True)
                self.tts.speak(sentence)

            print()
            self.tts.wait_until_done()

        self.tts.stop()


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pip = PipConversation()
    pip.run()