# ARIA social conversation training (virtual GPU only)

This is a reproducible **small language-model research experiment**, not a
production conversational model. The trainer **forces ARIA's
`VirtualGPUBackend`**, a tiled software matrix-multiplication emulator running
on the host CPU. It does not use a physical GPU or train the separate
scikit-learn intent classifier. The `chatbot.py` model is unchanged.

## Data sources

1. `data/social_seed.txt`: 64 original synthetic examples authored for ARIA.
   They include greetings, Hindi greetings, emoji, feelings and textual gesture
   markers such as `*waves*`, `[nods]`, and `[shrugs]`. This is not visual
   body-language or camera gesture recognition.
2. **Optional 2026 conversational corpus**:
   [Enes Deniz, English Daily Dialogues 10K](https://huggingface.co/datasets/3nesdeniz/english-daily-dialogues-10k).
   10,000 synthetic dialogues, CC BY 4.0. The script samples the publisher's
   train and validation splits separately; see the dataset card for limitations.
   Attribution: Enes Deniz (2026), English Daily Dialogues 10K,
   Hugging Face, CC BY 4.0.
3. **X/Twitter posts are not fetched or trained on.** X's current developer
   terms restrict scraping, redistribution and foundation-model training from
   X content. See
   [X restricted uses](https://docs.x.com/developer-terms/restricted-use-cases)
   and [X developer policy](https://docs.x.com/developer-terms/policy).
   Public availability of a third-party tweet dump is not proof of training
   rights. Obtain explicit authorization before pursuing that data.

## Run the offline smoke training

From the repository root:

```bash
python scripts/train_social_vgpu.py --source seed --steps 6
```

This uses only the original, committed ARIA social examples and never fetches
external content.

## Train with the 2026 dialogue dataset

```bash
pip install datasets
python scripts/train_social_vgpu.py --source hf --max-dialogues 40 --steps 6
```

Downloads happen **only** when `--source hf` is selected. The larger corpus
is not committed to Git. The generated checkpoint and metrics are stored in
`data/training_runs/social-vgpu/` and should not be committed without review.

## Model and outputs

The experiment trains a small 1-layer `TransformerLanguageModel` with
`ByteTokenizer`, `LanguageModelTrainer`, and `SGD`. Defaults: hidden size 8,
intermediate size 16, context length 12 bytes, batch size 1, six updates.

Training is next-byte prediction. It is **not** instruction fine-tuning and
does not teach robust conversational skills by itself. Outputs:
- `social_transformer_vgpu.json`: ARIA-native model checkpoint
- `training_report.json`: device, sources, counts and held-out loss

Validation uses held-out *records* before tokenization. A few windows are
evaluated to keep VGPU CPU-emulation costs bounded. These metrics are diagnostics,
not a benchmark of assistant helpfulness. No checkpoint is automatically loaded
by the primary chatbot; inference integration requires a separate change.

## Research cautions

The 2026 dialogue corpus is synthetic and may encourage template-like replies.
Keep source attribution, respect training permissions, avoid private messages,
and evaluate on unseen conversational prompts before claiming quality gains.
Do not present the VGPU emulator as CUDA or hardware acceleration.
