"""Train ARIA's experimental Transformer on permitted social text using VGPU only.

This is a small research training workflow, not a production chatbot fine-tune.
No Twitter/X scraping or restricted tweet ingestion is performed.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aria.brain.optim import SGD
from aria.brain.transformer import TransformerLanguageModel
from aria.compute.backends import VirtualGPUBackend
from aria.tokenizer.core import ByteTokenizer
from aria.training.checkpoint import TrainingConfig, save_training_checkpoint
from aria.training.dataset import TokenWindowDataset
from aria.training.trainer import LanguageModelTrainer

HF_DIALOGUE_DATASET = "3nesdeniz/english-daily-dialogues-10k"


def normalize_text(text: str) -> str:
    """Remove obvious identifiers before local training, preserving conversational cues."""
    text = re.sub(r"https?://\\S+", "[link]", text)
    text = re.sub(r"\\b[\\w.+-]+@[\\w.-]+\\.[A-Za-z]{2,}\\b", "[email]", text)
    text = re.sub(r"(?<!\\w)@[A-Za-z0-9_]{2,}", "[user]", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text[:4000]


def seed_records(path: Path) -> list[str]:
    """Each nonempty line is one original, short conversational example."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [clean for line in lines if (clean := normalize_text(line)) and
            not clean.startswith("#")]


def split_records(records: list[str], *, seed: int) -> tuple[list[str], list[str]]:
    """Split by complete examples, before converting to overlapping token windows."""
    unique = list(dict.fromkeys(records))
    if len(unique) < 6:
        raise ValueError("at least six distinct examples are required")
    random.Random(seed).shuffle(unique)
    count = max(2, len(unique) // 5)
    return unique[count:], unique[:count]


def load_hf_dialogues(limit: int) -> tuple[list[str], list[str]]:
    """Use only the publisher's separate train and validation splits."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install optional datasets support: pip install datasets") from exc

    if limit < 1:
        raise ValueError("limit must be positive")
    train_split = load_dataset(HF_DIALOGUE_DATASET, split="train")
    validation_split = load_dataset(HF_DIALOGUE_DATASET, split="validation")
    train = [normalize_text(row["text"]) for row in train_split.select(
        range(min(limit, len(train_split)))
    )]
    validation = [normalize_text(row["text"]) for row in validation_split.select(
        range(min(max(2, limit // 5), len(validation_split)))
    )]
    return [x for x in train if x], [x for x in validation if x]


def make_datasets(train: list[str], validation: list[str], sequence_length: int):
    tokenizer = ByteTokenizer()
    training_tokens = tokenizer.encode("\n\n".join(train), add_eos=True)
    validation_tokens = tokenizer.encode("\n\n".join(validation), add_eos=True)
    return (
        TokenWindowDataset(training_tokens, sequence_length, stride=sequence_length),
        TokenWindowDataset(validation_tokens, sequence_length, stride=sequence_length),
    )


def validation_loss(model: TransformerLanguageModel, dataset: TokenWindowDataset,
                    max_windows: int) -> float:
    if max_windows < 1:
        raise ValueError("max_windows must be positive")
    losses = []
    for index in range(min(max_windows, len(dataset))):
        inputs, targets = dataset[index]
        losses.append(model.loss_batch([inputs], [targets]).item())
    if not losses:
        raise ValueError("validation dataset has no windows")
    result = math.fsum(losses) / len(losses)
    if not math.isfinite(result):
        raise ValueError("non-finite validation loss")
    return result


def train(args: argparse.Namespace) -> dict:
    if args.steps < 1 or args.sequence_length < 2 or args.batch_size < 1:
        raise ValueError("steps, batch size and sequence length must be positive")
    source = ROOT / "data" / "social_seed.txt"
    training, validation = split_records(seed_records(source), seed=args.seed)
    sources = ["ARIA original social and gesture examples"]
    if args.source == "hf":
        extra_train, extra_val = load_hf_dialogues(args.max_dialogues)
        training.extend(extra_train)
        validation.extend(extra_val)
        sources.append(HF_DIALOGUE_DATASET + " (CC BY 4.0)")
    train_dataset, validation_dataset = make_datasets(
        training, validation, args.sequence_length
    )

    # This explicitly forbids both the default VCPU and the inference-only VNPU.
    backend = VirtualGPUBackend(tile_size=args.tile_size)
    if backend.info.device != "vgpu" or not backend.info.supports_autodiff:
        raise RuntimeError("VGPU autodiff backend required for training")

    model = TransformerLanguageModel(
        vocab_size=ByteTokenizer.VOCAB_SIZE,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_layers=1,
        max_sequence_length=args.sequence_length,
        seed=args.seed,
        backend=backend,
    )
    optimizer = SGD(model.parameters(), learning_rate=args.learning_rate)
    trainer = LanguageModelTrainer(model, optimizer, train_dataset,
                                   batch_size=args.batch_size)
    before = validation_loss(model, validation_dataset, args.validation_windows)
    history = trainer.train(args.steps)
    after = validation_loss(model, validation_dataset, args.validation_windows)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "social_transformer_vgpu.json"
    config = TrainingConfig(
        learning_rate=args.learning_rate,
        sequence_length=args.sequence_length,
        steps=args.steps,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    save_training_checkpoint(
        model, optimizer, trainer, config=config,
        checkpoint_path=checkpoint_path, losses=[step.loss for step in history]
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend.info.device,
        "execution": backend.info.execution,
        "hardware_gpu_acceleration": False,
        "model": "experimental TransformerLanguageModel",
        "sources": sources,
        "training_records": len(training),
        "validation_records": len(validation),
        "training_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "steps": len(history),
        "initial_validation_loss": before,
        "final_validation_loss": after,
        "initial_training_loss": history[0].loss,
        "final_training_loss": history[-1].loss,
        "checkpoint": str(checkpoint_path),
        "note": "No tweet/X content used. Weights are not automatically loaded by chatbot.py.",
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="ARIA local virtual GPU social text training")
    parser.add_argument("--source", choices=["seed", "hf"], default="seed",
                        help="hf additionally loads licensed 2026 synthetic dialogues")
    parser.add_argument("--max-dialogues", type=int, default=40)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--intermediate-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-windows", type=int, default=3)
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "training_runs" / "social-vgpu"))
    train(parser.parse_args())


if __name__ == "__main__":
    main()
