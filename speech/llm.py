import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL      = os.getenv("LLM_MODEL", "phi3:mini")

# This is your robot's personality. Change this freely — it has a big impact.
SYSTEM_PROMPT = """
You are a small, friendly robot companion named Pip. You are curious, warm,
and a little playful. You live in the same room as your human and enjoy
conversation. Keep your responses conversational and fairly brief — you are
speaking out loud, not writing an essay. Avoid bullet points and lists.
Respond in 1 to 3 sentences unless the human specifically asks for more detail.
"""

# How many conversation turns to remember (each turn = 1 human + 1 assistant message)
MAX_HISTORY_TURNS = 10


# ─── Conversation History ─────────────────────────────────────────────────────

class ConversationHistory:
    """
    Stores the back-and-forth between the human and the robot.
    Automatically trims old turns so we don't overflow the model's context window.
    """

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS):
        self.max_turns = max_turns
        self.messages: list[dict] = []

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str):
        self.messages.append({"role": "assistant", "content": text})
        self._trim()

    def _trim(self):
        # Each "turn" is a user + assistant pair = 2 messages.
        # We keep at most max_turns * 2 messages.
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_messages(self) -> list[dict]:
        return self.messages.copy()

    def clear(self):
        self.messages = []
        print("[LLM] Conversation history cleared.")

# ─── LLM Client ───────────────────────────────────────────────────────────────

class LLMClient:
    """
    Sends messages to Ollama and gets responses back.
    Supports both streaming (word by word) and non-streaming (all at once).
    """

    def __init__(self):
        self.history = ConversationHistory()
        self._verify_connection()

    def _verify_connection(self):
        """Check Ollama is running before we try to use it."""
        try:
            r = requests.get(f"{OLLAMA_URL}", timeout=3)
            if r.ok:
                print(f"[LLM] Connected to Ollama at {OLLAMA_URL}")
                print(f"[LLM] Using model: {MODEL}")
            else:
                print(f"[LLM] Warning: Ollama returned status {r.status_code}")
        except requests.exceptions.ConnectionError:
            print("[LLM] ERROR: Cannot reach Ollama.")
            print("       Make sure the Ollama app is running on Windows.")
            print(f"       Expected it at: {OLLAMA_URL}")

    def _build_payload(self, stream: bool) -> dict:
        """Assemble the full request body for the Ollama API."""
        return {
            "model": MODEL,
            "system": SYSTEM_PROMPT.strip(),
            "messages": self.history.get_messages(),
            "stream": stream,
            "options": {
                # Lower temperature = more focused/predictable responses
                # Higher = more creative but less consistent
                "temperature": 0.7,
                # Limits response length. Good for a voice robot — keeps it brief.
                "num_predict": 256,
            }
        }

    def chat(self, user_input: str) -> str:
        """
        Send a message and get the full response at once.
        Simpler, but the robot is silent until the entire response is ready.
        Use this for testing. Use chat_stream() in the real voice pipeline.
        """
        self.history.add_user(user_input)

        try:
            payload = self._build_payload(stream=False)
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            reply = data["message"]["content"].strip()

        except requests.exceptions.Timeout:
            reply = "Sorry, I took too long to think. Could you ask me again?"
        except requests.exceptions.ConnectionError:
            reply = "I lost connection to my brain. Is Ollama still running?"
        except (KeyError, json.JSONDecodeError) as e:
            reply = "I got a strange response I couldn't understand."
            print(f"[LLM] Parse error: {e}")

        self.history.add_assistant(reply)
        return reply

    def chat_stream(self, user_input: str):
        """
        Send a message and get the response as a generator, word by word.
        This is what the voice pipeline will use — it lets TTS start speaking
        the first sentence while the model is still generating the rest.

        Usage:
            for chunk in client.chat_stream("Hello!"):
                print(chunk, end="", flush=True)
        """
        self.history.add_user(user_input)
        full_reply = []

        try:
            payload = self._build_payload(stream=True)
            with requests.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
                stream=True,
                timeout=60
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        chunk = data.get("message", {}).get("content", "")
                        if chunk:
                            full_reply.append(chunk)
                            yield chunk
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.Timeout:
            fallback = "Sorry, I took too long to think."
            yield fallback
            full_reply.append(fallback)
        except requests.exceptions.ConnectionError:
            fallback = "I lost connection to my brain."
            yield fallback
            full_reply.append(fallback)

        # Save the complete response to history once streaming is done
        self.history.add_assistant("".join(full_reply))


    # ─── Sentence Buffer ──────────────────────────────────────────────────────────

def stream_to_sentences(stream):
    """
    Takes a word-by-word token stream and yields complete sentences.
    This lets TTS speak sentence 1 while the LLM is still writing sentence 2.

    Usage:
        for sentence in stream_to_sentences(client.chat_stream("Hi")):
            tts.speak(sentence)  # speak each sentence as it becomes ready
    """
    buffer = ""
    sentence_endings = {'.', '!', '?'}

    for chunk in stream:
        buffer += chunk
        # Check if we've accumulated a complete sentence
        while any(marker in buffer for marker in sentence_endings):
            earliest = len(buffer)
            split_at = -1
            for marker in sentence_endings:
                idx = buffer.find(marker)
                if idx != -1 and idx < earliest:
                    earliest = idx
                    split_at = idx

            if split_at == -1:
                break

            sentence = buffer[:split_at + 1].strip()
            buffer = buffer[split_at + 1:].lstrip()

            if sentence:
                yield sentence

    # Yield anything left over (response that didn't end with punctuation)
    if buffer.strip():
        yield buffer.strip()  


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("LLM Module Test")
    print("Type a message, press Enter. Type 'quit' to exit.")
    print("Type 'clear' to reset conversation history.")
    print("=" * 50)

    client = LLMClient()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[Exiting]")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            client.history.clear()
            continue

        print("Pip (streaming): ", end="", flush=True)
        for chunk in client.chat_stream(user_input):
            print(chunk, end="", flush=True)
        print()  # newline after response

        print(f"\n[History contains {len(client.history.messages)} messages]")