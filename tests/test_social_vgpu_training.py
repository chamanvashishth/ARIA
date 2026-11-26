"""Focused verification for the opt-in VGPU social training path."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from aria.compute.backends import VirtualGPUBackend
from scripts.train_social_vgpu import (
    make_datasets,
    normalize_text,
    seed_records,
    split_records,
    train,
)


def test_personal_identifiers_removed_from_social_text():
    value = normalize_text("Hi @alice email alice@example.com https://example.com 👋")
    assert "@alice" not in value
    assert "alice@example.com" not in value
    assert "https://example.com" not in value
    assert "👋" in value


def test_no_token_window_leakage_between_seed_splits():
    records = seed_records(Path("data/social_seed.txt"))
    training, validation = split_records(records, seed=42)
    assert len(records) >= 30
    assert set(training).isdisjoint(validation)
    tr, va = make_datasets(training, validation, sequence_length=8)
    assert len(tr) > 0 and len(va) > 0


def test_vgpu_only_train_and_checkpoint(tmp_path):
    assert VirtualGPUBackend(tile_size=4).info.device == "vgpu"
    args = argparse.Namespace(
        source="seed", max_dialogues=1, steps=1, sequence_length=6,
        hidden_size=4, intermediate_size=8, batch_size=1,
        learning_rate=0.005, tile_size=4, seed=7,
        validation_windows=1, output_dir=str(tmp_path),
    )
    report = train(args)
    assert report["backend"] == "vgpu"
    assert report["hardware_gpu_acceleration"] is False
    assert math.isfinite(report["final_validation_loss"])
    assert (tmp_path / "social_transformer_vgpu.json").exists()
    assert (tmp_path / "training_report.json").exists()
