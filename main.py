"""Entry point for ARIA, a small command-line assistant scaffold."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from responses import get_response

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
MEMORY_PATH = BASE_DIR / "memory.json"
HISTORY_PATH = BASE_DIR / "chat_history.txt"


def load_json(path: Path, default: dict) -> dict:
    """Load JSON data from *path*, returning *default* when the file is empty."""
    if not path.exists() or path.stat().st_size == 0:
        return default.copy()

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: dict) -> None:
    """Write JSON data to *path* with consistent formatting."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def append_history(user_message: str, assistant_response: str) -> None:
    """Append one conversation turn to the chat history file."""
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with HISTORY_PATH.open("a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] User: {user_message}\n")
        file.write(f"[{timestamp}] ARIA: {assistant_response}\n")


def main() -> None:
    """Run ARIA in an interactive terminal session."""
    config = load_json(CONFIG_PATH, {"assistant_name": "ARIA"})
    memory = load_json(MEMORY_PATH, {"interactions": 0, "notes": []})
    assistant_name = config.get("assistant_name", "ARIA")

    print(f"{assistant_name} is ready. Type 'exit' or 'quit' to end the session.")

    while True:
        user_message = input("You: ").strip()
        if user_message.lower() in {"exit", "quit"}:
            print(f"{assistant_name}: Goodbye!")
            break

        assistant_response = get_response(user_message, memory)
        print(f"{assistant_name}: {assistant_response}")

        memory["interactions"] = int(memory.get("interactions", 0)) + 1
        save_json(MEMORY_PATH, memory)
        append_history(user_message, assistant_response)


if __name__ == "__main__":
    main()
