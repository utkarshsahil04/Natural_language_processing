# IILM University Chatbot

A simple terminal chatbot that answers questions about **IILM University, Greater Noida** using the Google Gemini API.

Ask about admissions, programs, contact details, placements, campus life, and more.

## Features

- Terminal-based chat (no UI required)
- Powered by Google Gemini (`gemini-2.0-flash`)
- Custom knowledge base for IILM University
- API key stored securely in a `.env` file
- Multi-turn conversation (remembers context within a session)

## Project Structure

```
NLP/
├── iilm_chatbot.py      # Main chatbot script
├── iilm_knowledge.txt   # IILM University information (knowledge base)
├── requirements.txt     # Python dependencies
├── .env.example         # Template for API key setup
├── .env                 # Your API key (not committed to git)
└── README.md
```

## Prerequisites

- Python 3.10+
- A free [Google AI Studio](https://aistudio.google.com/apikey) API key

## Setup

### 1. Clone or open the project

```powershell
cd "D:\COllege\3rd Year\NLP"
```

### 2. Create a virtual environment (optional but recommended)

```powershell
python -m venv env
.\env\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure your API key

Copy the example file and add your Gemini API key:

```powershell
copy .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=AQ.your_full_api_key_here
```

Get your key from [Google AI Studio](https://aistudio.google.com/apikey) — **not** Android Studio.

| Key format | Description |
|------------|-------------|
| `AQ....`   | New auth keys (current default) |
| `AIza....` | Older standard keys |

Do **not** use a project/client ID like `gen-lang-client-0038169940` — that is not the API key.

## Usage

Run the chatbot:

```powershell
.\env\Scripts\python.exe iilm_chatbot.py
```

Or, if the virtual environment is activated:

```powershell
python iilm_chatbot.py
```

### Example questions

- What programs does IILM offer?
- How do I apply for B.Tech?
- What is the admission helpline for MBA?
- Tell me about placements at IILM
- What is the campus address?

Type `quit`, `exit`, or `bye` to stop the chatbot.

## Customizing the Knowledge Base

Edit `iilm_knowledge.txt` to add or update IILM University information. The chatbot uses this file as its source of truth and will only answer based on the content provided.

After editing, restart the chatbot to load the updated knowledge.

## Troubleshooting

### "API key not valid"

- Copy the full key from [Google AI Studio](https://aistudio.google.com/apikey)
- Ensure there are no quotes or spaces in `.env`
- Use the secret key (`AQ.` or `AIza`), not the client/project ID

### "Your GEMINI_API_KEY does not look valid"

If you previously set a bad key in PowerShell, clear it:

```powershell
Remove-Item Env:GEMINI_API_KEY
```

The chatbot reads from `.env` first, but clearing the shell variable avoids confusion.

### "API quota exceeded"

The free Gemini tier has rate limits. Wait a few minutes and try again, or check usage at [Google AI Studio](https://aistudio.google.com/).

### "Missing package"

Install dependencies inside your virtual environment:

```powershell
pip install -r requirements.txt
```

## Tech Stack

- **Python 3**
- **Google Gen AI SDK** (`google-genai`)
- **python-dotenv** — environment variable management
- **Gemini 2.0 Flash** — language model

## License

This project is for educational purposes (NLP course).
