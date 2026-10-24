import pytest

from chatbot import ChatbotModel, ChatbotResponse


def test_chatbot_model_init_and_threshold() -> None:
    bot = ChatbotModel(confidence_threshold=0.42)
    assert bot.confidence_threshold == 0.42
    assert bot.vectorizer is None
    assert bot.classifier is None


def test_chatbot_load_intents() -> None:
    bot = ChatbotModel()
    patterns, tags = bot.load_intents()
    assert len(patterns) > 0
    assert len(tags) == len(patterns)
    assert "greeting" in tags
    assert "greeting" in bot.responses
    assert len(bot.responses["greeting"]) > 0


def test_chatbot_train_and_predict() -> None:
    bot = ChatbotModel(confidence_threshold=0.30)
    accuracy = bot.train()
    assert accuracy > 0.90

    # Status check
    status = bot.get_status()
    assert status["ready"] is True
    assert status["classes_count"] >= 10

    # Predictions
    tag, conf = bot.predict_intent("hello")
    assert tag == "greeting"
    assert conf > 0.30

    detailed: ChatbotResponse = bot.get_detailed_response("hello")
    assert detailed["intent"] == "greeting"
    assert not detailed["is_fallback"]
    assert len(detailed["response"]) > 0


def test_chatbot_fallback_for_unrecognized_query() -> None:
    bot = ChatbotModel(confidence_threshold=0.75)
    bot.load()

    # Empty query
    empty_res = bot.get_detailed_response("   ")
    assert empty_res["is_fallback"] is True
    assert empty_res["intent"] == "empty_input"

    # Garbage gibberish query
    gibberish = bot.get_detailed_response("xyz123abc987qwerty")
    assert gibberish["is_fallback"] is True

