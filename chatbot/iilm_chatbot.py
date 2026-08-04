"""
IILM University Terminal Chatbot
Powered by Google Gemini API

Usage:
  1. Copy .env.example to .env and add your Gemini API key
  2. Run: python iilm_chatbot.py
  3. Type your questions about IILM University
  4. Type 'quit', 'exit', or 'bye' to stop
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print("Missing package. Install it with:")
    print("  pip install python-dotenv")
    sys.exit(1)

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Missing package. Install it with:")
    print("  pip install google-genai")
    sys.exit(1)


BASE_DIR = Path(__file__).parent
KNOWLEDGE_FILE = BASE_DIR / "iilm_knowledge.txt"
ENV_FILE = BASE_DIR / ".env"
MODEL_NAME = "gemini-2.0-flash"

SYSTEM_PROMPT = """You are a helpful assistant for IILM University, Greater Noida.
Your job is to answer questions about IILM University using ONLY the knowledge provided below.
Be friendly, concise, and accurate. If the answer is not in the knowledge base, say you don't have that information and suggest contacting admissions@iilm.edu or calling +91-8065905224.

KNOWLEDGE BASE:
{knowledge}
"""


def load_knowledge() -> str:
    if not KNOWLEDGE_FILE.exists():
        print(f"Warning: {KNOWLEDGE_FILE} not found. Chatbot will have limited info.")
        return ""
    return KNOWLEDGE_FILE.read_text(encoding="utf-8")


def is_valid_gemini_key(key: str) -> bool:
    if key.startswith("AIza") and len(key) >= 35:
        return True
    if key.startswith("AQ.") and len(key) >= 20:
        return True
    return False


def get_api_key() -> str:
    # override=True ensures .env wins over a bad shell variable
    load_dotenv(ENV_FILE, override=True)

    key = os.environ.get("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    if key:
        if key in ("your_gemini_api_key_here", "your_api_key_here"):
            print("\nReplace the placeholder in .env with your real Gemini API key.")
            sys.exit(1)
        if key.startswith("gen-lang-client-"):
            print("\nYou pasted a project/client ID, not the API key.")
            print("In Google AI Studio, click your key and copy the full secret.")
            print("It should start with AQ. or AIza — not 'gen-lang-client-'.")
            print("Also run: Remove-Item Env:GEMINI_API_KEY")
            sys.exit(1)
        if not is_valid_gemini_key(key):
            print("\nYour GEMINI_API_KEY does not look valid.")
            print("Get a key from Google AI Studio: https://aistudio.google.com/apikey")
            print("Valid keys start with 'AIza' (older) or 'AQ.' (newer auth keys).")
            print("Make sure .env has: GEMINI_API_KEY=AQ.your_full_key")
            sys.exit(1)
        os.environ["GEMINI_API_KEY"] = key
        return key

    if not ENV_FILE.exists():
        print(f"\nNo .env file found. Create one at: {ENV_FILE}")
        print("Copy .env.example to .env and add your API key.")
    else:
        print("\nGEMINI_API_KEY is missing or empty in your .env file.")

    key = input("Enter your Gemini API key: ").strip()
    if not key:
        print("API key is required. Get one at: https://aistudio.google.com/apikey")
        sys.exit(1)
    return key


def create_chat():
    client = genai.Client()
    knowledge = load_knowledge()
    system_instruction = SYSTEM_PROMPT.format(knowledge=knowledge)

    chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
        ),
    )
    return client, chat


def main():
    print("=" * 55)
    print("   IILM University Chatbot")
    print("   Ask anything about IILM University, Greater Noida")
    print("   Type 'quit' to exit")
    print("=" * 55)

    get_api_key()
    client, chat = create_chat()

    print("\nChatbot ready! Ask your question.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye", "q"):
            print("Goodbye! Visit https://iilm.edu for more info.")
            break

        try:
            response = chat.send_message(user_input)
            print(f"\nBot: {response.text}\n")
        except Exception as e:
            error_text = str(e)
            if "API_KEY_INVALID" in error_text or "API key not valid" in error_text:
                print("\nError: Gemini rejected the API key.")
                print("Your key format looks fine. Try these steps:")
                print("1. Open https://aistudio.google.com/apikey")
                print("2. Create a NEW key (new keys start with AQ.)")
                print("3. Make sure Generative Language API is enabled")
                print("4. Paste the full key in .env with no quotes or spaces")
            elif "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                print("\nError: API quota exceeded (free tier limit reached).")
                print("Wait a few minutes and try again, or check usage at:")
                print("https://aistudio.google.com/")
            elif "404" in error_text and "model" in error_text.lower():
                print(f"\nError: Model '{MODEL_NAME}' is not available on your account.")
                print("Try again later or create a new API key in Google AI Studio.")
            else:
                print(f"\nError: {e}")
                print("Check your API key and internet connection.")
            print()


if __name__ == "__main__":
    main()
