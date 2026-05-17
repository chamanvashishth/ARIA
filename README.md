# ARIA

ARIA is a lightweight Python command-line assistant scaffold. It includes separate files for the application entry point, response logic, configuration, persistent memory, and chat history.

## Project Structure

```text
ARIA/
│── main.py
│── responses.py
│── memory.json
│── chat_history.txt
│── config.json
│── README.md
│── .gitignore
```

## Files

- `main.py` starts the command-line assistant, loads configuration and memory, and records chat history.
- `responses.py` contains response-generation helpers.
- `memory.json` stores basic assistant memory such as interaction counts and notes.
- `chat_history.txt` stores conversation history written by the application.
- `config.json` stores assistant configuration values.
- `.gitignore` excludes common generated Python files and local environment artifacts.

## Getting Started

Run ARIA with Python 3.10 or newer:

```bash
python main.py
```

Type `exit` or `quit` to end the session.
