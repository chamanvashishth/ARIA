"""Interactive AI Chatbot Model.

Built with Python, Scikit-Learn (Neural Network / MLP), and TF-IDF NLP.
Classifies user queries against intent patterns and returns contextual responses.
"""

from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any, TypedDict

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

BASE_DIR = Path(__file__).resolve().parent
INTENTS_FILE = BASE_DIR / "intents.json"
MODEL_FILE = BASE_DIR / "chatbot_model.pkl"

__all__ = ["ChatbotModel", "ChatbotResponse", "INTENTS_FILE", "MODEL_FILE"]


class ChatbotResponse(TypedDict):
    """Structured response payload returned by ChatbotModel."""

    response: str
    intent: str
    confidence: float
    is_fallback: bool


class ChatbotModel:
    """Neural Network Intent Classification Model using TF-IDF and MLPClassifier."""

    def __init__(self, confidence_threshold: float = 0.35) -> None:
        self.confidence_threshold: float = float(confidence_threshold)
        self.vectorizer: TfidfVectorizer | None = None
        self.classifier: MLPClassifier | None = None
        self.intents: dict[str, dict[str, Any]] = {}
        self.responses: dict[str, list[str]] = {}

    def load_intents(self, filepath: Path | str = INTENTS_FILE) -> tuple[list[str], list[str]]:
        """Load intent patterns and responses from JSON dataset.

        Args:
            filepath: Path to the intents JSON file.

        Returns:
            Tuple of (training patterns list, corresponding intent tags list).

        Raises:
            FileNotFoundError: If the intents file does not exist.
            ValueError: If the intents file is empty or corrupted.
        """
        path = Path(filepath).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Intents file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in intents file: {exc}") from exc

        patterns: list[str] = []
        tags: list[str] = []
        self.intents = {}
        self.responses = {}

        raw_intents = data.get("intents", [])
        if not raw_intents:
            raise ValueError("Intents file contains no intents data.")

        for intent in raw_intents:
            tag = intent.get("tag", "").strip()
            if not tag:
                continue

            self.intents[tag] = intent
            self.responses[tag] = intent.get("responses", [])

            for pattern in intent.get("patterns", []):
                cleaned_pattern = pattern.strip()
                if cleaned_pattern:
                    patterns.append(cleaned_pattern)
                    tags.append(tag)

        return patterns, tags

    def train(self, filepath: Path | str = INTENTS_FILE) -> float:
        """Train the Neural Network intent classification model.

        Args:
            filepath: Path to the training dataset.

        Returns:
            Training accuracy score between 0.0 and 1.0.
        """
        print("[*] Training AI Chatbot Model...")
        patterns, tags = self.load_intents(filepath)

        if not patterns:
            raise ValueError("Cannot train model: No patterns found in dataset.")

        # 1. Feature extraction using TF-IDF (Text to numerical vectors)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            sublinear_tf=True,
        )
        x_train = self.vectorizer.fit_transform(patterns)
        y_train = tags

        # 2. Neural Network Classifier (Multi-Layer Perceptron)
        self.classifier = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42,
        )
        self.classifier.fit(x_train, y_train)

        accuracy = float(self.classifier.score(x_train, y_train))
        print(f"[+] Model trained successfully! Training Accuracy: {accuracy * 100:.1f}%")
        self.save()
        return accuracy

    def save(self, filepath: Path | str = MODEL_FILE) -> bool:
        """Save the trained model and vectorizer to disk."""
        if self.vectorizer is None or self.classifier is None:
            raise RuntimeError("Cannot save untrained model.")

        path = Path(filepath).resolve()
        payload = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "responses": self.responses,
            "intents": self.intents,
            "threshold": self.confidence_threshold,
        }

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(payload, f)
            print(f"[+] Model saved to: {path.name}")
            return True
        except OSError as exc:
            print(f"[-] Error saving model: {exc}")
            return False

    def load(self, filepath: Path | str = MODEL_FILE) -> bool:
        """Load trained model from disk cache."""
        path = Path(filepath).resolve()
        if not path.exists():
            return False

        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)

            self.vectorizer = payload["vectorizer"]
            self.classifier = payload["classifier"]
            self.responses = payload.get("responses", {})
            self.intents = payload.get("intents", {})
            self.confidence_threshold = float(payload.get("threshold", 0.35))
            return True
        except Exception as exc:
            print(f"[-] Error loading model cache: {exc}")
            return False

    def predict_intent(self, text: str) -> tuple[str, float]:
        """Predict the intent tag and confidence probability for a given user input.

        Args:
            text: Raw user query string.

        Returns:
            Tuple of (predicted intent tag, confidence probability float).
        """
        if self.classifier is None or self.vectorizer is None:
            raise RuntimeError("Model is not loaded or trained yet.")

        vec = self.vectorizer.transform([text])
        probabilities = self.classifier.predict_proba(vec)[0]
        max_idx = int(probabilities.argmax())
        confidence = float(probabilities[max_idx])
        tag = str(self.classifier.classes_[max_idx])

        return tag, confidence

    def get_detailed_response(self, text: str) -> ChatbotResponse:
        """Return structured response with intent, confidence score, and fallback status.

        Args:
            text: Raw user query string.

        Returns:
            Structured ChatbotResponse dict.
        """
        cleaned = text.strip() if text else ""
        if not cleaned:
            return {
                "response": "Please enter a valid command, query, or expression.",
                "intent": "empty_input",
                "confidence": 0.0,
                "is_fallback": True,
            }

        tag, confidence = self.predict_intent(cleaned)
        is_fallback = confidence < self.confidence_threshold

        if is_fallback:
            response_text = (
                "Query not recognized in current knowledge base.\n\n"
                "*Available topics: System Architecture, Python, Neural Networks, "
                "Math calculations (`calc <expr>`), or Diagnostics.*"
            )
        else:
            possible_responses = self.responses.get(tag, [])
            if possible_responses:
                response_text = random.choice(possible_responses)
            else:
                response_text = "Specified intent has no registered response template."

        return {
            "response": response_text,
            "intent": tag,
            "confidence": round(confidence, 4),
            "is_fallback": is_fallback,
        }

    def get_response(self, text: str, debug: bool = False) -> str:
        """Generate response based on classified intent and confidence score.

        Args:
            text: User input.
            debug: If True, prints classification debug information to stdout.

        Returns:
            Response string.
        """
        detailed = self.get_detailed_response(text)
        tag = detailed["intent"]
        confidence = detailed["confidence"]

        if debug:
            print(f"   [Debug: Intent = '{tag}' | Confidence = {confidence * 100:.1f}%]")

        return detailed["response"]

    def get_status(self) -> dict[str, Any]:
        """Return diagnostic status information for the model."""
        is_ready = self.classifier is not None and self.vectorizer is not None
        classes = list(self.classifier.classes_) if self.classifier is not None else []
        return {
            "ready": is_ready,
            "classes_count": len(classes),
            "classes": classes,
            "responses_count": len(self.responses),
            "confidence_threshold": self.confidence_threshold,
        }


def main() -> None:
    """Interactive command-line interface for the chatbot model."""
    print("=" * 60)
    print("  AI Chatbot Model (Python + Neural Network)")
    print("=" * 60)
    print("Commands:")
    print("  'quit' or 'exit' : Close the chat")
    print("  '/debug'         : Toggle confidence & intent inspect mode")
    print("  '/retrain'       : Retrain model on intents.json")
    print("  '/status'        : Print model status and tag list")
    print("  '/help'          : Show this commands overview")
    print("=" * 60)

    bot = ChatbotModel()

    # Load existing model or train if first run
    if not bot.load():
        bot.train()
    else:
        print("[+] Loaded existing trained model from cache.")

    debug_mode = False
    print("\nChatbot: Hello! I'm ready. How can I help you today?")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nChatbot: Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ["quit", "exit", "bye", "q"]:
            print("Chatbot: Goodbye! Keep exploring AI!")
            break

        if cmd == "/debug":
            debug_mode = not debug_mode
            print(f"[Debug mode is now {'ENABLED' if debug_mode else 'DISABLED'}]")
            continue

        if cmd == "/retrain":
            bot.train()
            print("Chatbot: Retraining complete!")
            continue

        if cmd == "/status":
            status = bot.get_status()
            print(f"[Model Status: Ready={status['ready']} | Classes={status['classes_count']} | Threshold={status['confidence_threshold']}]")
            print(f" Tags: {', '.join(status['classes'])}")
            continue

        if cmd == "/help":
            print("Available commands: /debug, /retrain, /status, /help, quit, exit")
            continue

        response = bot.get_response(user_input, debug=debug_mode)
        print(f"Chatbot: {response}")


if __name__ == "__main__":
    main()
