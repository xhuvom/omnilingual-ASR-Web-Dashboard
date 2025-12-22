# I have downloaded the meta omnilingual asr . Now I
_Exported on 12/12/2025 at 07:10:01 GMT+6 from Cursor (2.0.34)_

---

**User**

I have downloaded the meta omnilingual asr . Now I want to install and run it in my local pc. Follow the @README.md and install the requiremnts in a new conda env and then download the model and run it for both bangla and english language audio transcription.

Ive installed dependenciesin the omnilingual-asr conda env @bash (6-285) Now what?

Oh, but downlodign the model is taking so much space in my pc. delete the partiallydownloaded model and show me the links to donwload model

---

**Cursor**

Deleted the partial download cache at `~/.cache/fairseq2/assets`, so you’ve reclaimed the space that the interrupted model pull was using.

Here are the official direct download URLs from the README:

- `omniASR_W2V_300M`: https://dl.fbaipublicfiles.com/mms/omniASR-W2V-300M.pt  
- `omniASR_W2V_1B`: https://dl.fbaipublicfiles.com/mms/omniASR-W2V-1B.pt  
- `omniASR_W2V_3B`: https://dl.fbaipublicfiles.com/mms/omniASR-W2V-3B.pt  
- `omniASR_W2V_7B`: https://dl.fbaipublicfiles.com/mms/omniASR-W2V-7B.pt  

- `omniASR_CTC_300M`: https://dl.fbaipublicfiles.com/mms/omniASR-CTC-300M.pt  
- `omniASR_CTC_1B`: https://dl.fbaipublicfiles.com/mms/omniASR-CTC-1B.pt  
- `omniASR_CTC_3B`: https://dl.fbaipublicfiles.com/mms/omniASR-CTC-3B.pt  
- `omniASR_CTC_7B`: https://dl.fbaipublicfiles.com/mms/omniASR-CTC-7B.pt  

- `omniASR_LLM_300M`: https://dl.fbaipublicfiles.com/mms/omniASR-LLM-300M.pt  
- `omniASR_LLM_1B`: https://dl.fbaipublicfiles.com/mms/omniASR-LLM-1B.pt  
- `omniASR_LLM_3B`: https://dl.fbaipublicfiles.com/mms/omniASR-LLM-3B.pt  
- `omniASR_LLM_7B`: https://dl.fbaipublicfiles.com/mms/omniASR-LLM-7B.pt  
- `omniASR_LLM_7B_ZS`: https://dl.fbaipublicfiles.com/mms/omniASR-LLM-7B-ZS.pt  

Tokenizers:

- `omniASR_tokenizer`: https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model  
- `omniASR_tokenizer_v7`: https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer_v7.model  

Let me know when you’re ready to attempt another download (perhaps a smaller checkpoint) or if you’d like help managing disk space first.

---

**User**

Ive downloaded the model and tokenizer on /media/sigmind/Gaming1/asr folder, now tweak the run script to load the model and run inference on a sample audio file,. Generate a test script for bangla and english language 

---

**Cursor**

```python
#!/usr/bin/env python3
"""
Utility script to run Omnilingual ASR inference against locally downloaded
model/tokenizer assets.

The script expects that you have created custom fairseq2 asset cards pointing
to the local checkpoints (see README instructions added by the assistant) and
placed them under ~/.config/fairseq2/assets/cards/models. With that in place,
running this script will not trigger any remote downloads.

It performs two quick sanity checks by streaming short samples from the public
`facebook/omnilingual-asr-corpus` dataset (one Bangla, one English), runs them
through the local omniASR_LLM_7B checkpoint, and prints the reference vs.
predicted text to stdout.
"""

from __future__ import annotations

import argparse
from typing import Iterable, List, Tuple

from datasets import load_dataset

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


LANG_EXAMPLES: Tuple[Tuple[str, str], ...] = (
    ("ben_Beng", "Bangla"),
    ("eng_Latn", "English"),
)


def fetch_dataset_sample(
    lang_code: str, *, max_duration_s: float = 38.0
) -> Tuple[dict, str, float]:
    """
    Stream samples from the Hugging Face dataset until we find one shorter than
    ``max_duration_s`` seconds. The returned audio dictionary is already in the
    format accepted by :class:`ASRInferencePipeline`.
    """
    dataset = load_dataset(
        "facebook/omnilingual-asr-corpus",
        lang_code,
        split="train",
        streaming=True,
    )

    for sample in dataset:
        audio = sample["audio"]
        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]
        duration = len(waveform) / sample_rate

        if duration <= max_duration_s:
            return (
                {"waveform": waveform, "sample_rate": sample_rate},
                sample["raw_text"],
                duration,
            )

    raise RuntimeError(
        f"Could not find an example shorter than {max_duration_s}s for {lang_code}."
    )


def run_inference(
    pipeline: ASRInferencePipeline,
    lang_pairs: Iterable[Tuple[str, str]],
) -> None:
    """
    Fetches one short sample per language, runs inference, and prints the output.
    """
    batch: List[dict] = []
    lang_codes: List[str] = []
    references: List[str] = []
    summaries: List[Tuple[str, str, float, str]] = []

    for lang_code, lang_label in lang_pairs:
        audio_dict, reference_text, duration = fetch_dataset_sample(lang_code)
        batch.append(audio_dict)
        lang_codes.append(lang_code)
        references.append(reference_text)
        summaries.append((lang_label, lang_code, duration, reference_text))

    print("Loaded samples:")
    for lang_label, lang_code, duration, reference in summaries:
        snippet = reference if len(reference) <= 80 else reference[:77] + "..."
        print(f"- {lang_label} ({lang_code}): {duration:.2f}s | Reference: {snippet}")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(batch, lang=lang_codes, batch_size=len(batch))

    for (lang_label, lang_code, _, reference), prediction in zip(
        summaries, outputs, strict=True
    ):
        print("\n==>", lang_label, f"({lang_code})")
        print("Reference :", reference)
        print("Prediction:", prediction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run omnilingual ASR inference using locally cached assets."
    )
    parser.add_argument(
        "--model-card",
        default="omniASR_LLM_7B_local",
        help=(
            "Name of the fairseq2 model card that points to the local checkpoint. "
            "Defaults to 'omniASR_LLM_7B_local'."
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=38.0,
        help="Maximum clip duration (in seconds) when sampling from the dataset.",
    )

    args = parser.parse_args()

    print(f"Loading pipeline with model card '{args.model_card}'...")
    pipeline = ASRInferencePipeline(model_card=args.model_card)

    run_inference(pipeline, LANG_EXAMPLES)


if __name__ == "__main__":
    main()

```

- Added custom fairseq2 asset cards at `~/.config/fairseq2/assets/cards/models/omniasr_local.yaml` so the runtime can load your locally downloaded `omniASR-LLM-7B.pt` and `omniASR_tokenizer_v7.model` without re-downloading.
- Created `scripts/run_local_inference.py`, which spins up `ASRInferencePipeline(model_card="omniASR_LLM_7B_local")`, streams one Bangla and one English clip from the public corpus, and prints reference vs. predicted transcripts.

```46:135:scripts/run_local_inference.py
print(f"Loading pipeline with model card '{args.model_card}'...")
pipeline = ASRInferencePipeline(model_card=args.model_card)

run_inference(pipeline, LANG_EXAMPLES)
```

Run it after activating the environment:

```
conda activate omnilingual-asr
python scripts/run_local_inference.py
```

The script samples short clips (≤38 s) so it can run on CPU, but the 7B checkpoint still needs significant RAM/VRAM; expect a long load time. If your assets live elsewhere or you named the files differently, edit `omniasr_local.yaml` or pass `--model-card` pointing at another card.

```python
print(f"Loading pipeline with model card '{args.model_card}'...")
pipeline = ASRInferencePipeline(model_card=args.model_card)

run_inference(pipeline, LANG_EXAMPLES)
```

```plaintext
conda activate omnilingual-asr
python scripts/run_local_inference.py
```

---

**User**

I m trying to run 3B model @run_local_inference.py (101-118)  but I see erooro @bash (427-443)  I have the model in the dir

---

**Cursor**

```python
#!/usr/bin/env python3
"""
Utility script to run Omnilingual ASR inference against locally downloaded
model/tokenizer assets.

The script expects that you have created custom fairseq2 asset cards pointing
to the local checkpoints (see README instructions added by the assistant) and
placed them under ~/.config/fairseq2/assets/cards/models. With that in place,
running this script will not trigger any remote downloads.

It performs two quick sanity checks by streaming short samples from the public
`facebook/omnilingual-asr-corpus` dataset (one Bangla, one English), runs them
through the local omniASR_LLM_7B checkpoint, and prints the reference vs.
predicted text to stdout.
"""

from __future__ import annotations

import argparse
from typing import Iterable, List, Tuple

from datasets import load_dataset

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


LANG_EXAMPLES: Tuple[Tuple[str, str], ...] = (
    ("ben_Beng", "Bangla"),
    ("eng_Latn", "English"),
)


def fetch_dataset_sample(
    lang_code: str, *, max_duration_s: float = 38.0
) -> Tuple[dict, str, float]:
    """
    Stream samples from the Hugging Face dataset until we find one shorter than
    ``max_duration_s`` seconds. The returned audio dictionary is already in the
    format accepted by :class:`ASRInferencePipeline`.
    """
    dataset = load_dataset(
        "facebook/omnilingual-asr-corpus",
        lang_code,
        split="train",
        streaming=True,
    )

    for sample in dataset:
        audio = sample["audio"]
        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]
        duration = len(waveform) / sample_rate

        if duration <= max_duration_s:
            return (
                {"waveform": waveform, "sample_rate": sample_rate},
                sample["raw_text"],
                duration,
            )

    raise RuntimeError(
        f"Could not find an example shorter than {max_duration_s}s for {lang_code}."
    )


def run_inference(
    pipeline: ASRInferencePipeline,
    lang_pairs: Iterable[Tuple[str, str]],
) -> None:
    """
    Fetches one short sample per language, runs inference, and prints the output.
    """
    batch: List[dict] = []
    lang_codes: List[str] = []
    references: List[str] = []
    summaries: List[Tuple[str, str, float, str]] = []

    for lang_code, lang_label in lang_pairs:
        audio_dict, reference_text, duration = fetch_dataset_sample(lang_code)
        batch.append(audio_dict)
        lang_codes.append(lang_code)
        references.append(reference_text)
        summaries.append((lang_label, lang_code, duration, reference_text))

    print("Loaded samples:")
    for lang_label, lang_code, duration, reference in summaries:
        snippet = reference if len(reference) <= 80 else reference[:77] + "..."
        print(f"- {lang_label} ({lang_code}): {duration:.2f}s | Reference: {snippet}")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(batch, lang=lang_codes, batch_size=len(batch))

    for (lang_label, lang_code, _, reference), prediction in zip(
        summaries, outputs, strict=True
    ):
        print("\n==>", lang_label, f"({lang_code})")
        print("Reference :", reference)
        print("Prediction:", prediction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run omnilingual ASR inference using locally cached assets."
    )
    parser.add_argument(
        "--model-card",
        default="omniASR_LLM_3B_local",
        help=(
            "Name of the fairseq2 model card that points to the local checkpoint. "
            "Defaults to 'omniASR_LLM_3B_local'."
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=38.0,
        help="Maximum clip duration (in seconds) when sampling from the dataset.",
    )

    args = parser.parse_args()

    print(f"Loading pipeline with model card '{args.model_card}'...")
    pipeline = ASRInferencePipeline(model_card=args.model_card)

    run_inference(pipeline, LANG_EXAMPLES)


if __name__ == "__main__":
    main()


```

`ModelNotKnownError` happened because fairseq2 couldn’t find a model card named `omniASR_LLM_3B_local`. I added one (plus the matching tokenizer entry) in `~/.config/fairseq2/assets/cards/models/omniasr_local.yaml`, pointing to your files under `/media/sigmind/Gaming1/asr/`. I also tweaked `scripts/run_local_inference.py` so the help text matches the new default card.

Try again after activating the env:

```
conda activate omnilingual-asr
python scripts/run_local_inference.py
```

If your filenames differ, edit the paths in `omniasr_local.yaml` accordingly.

```plaintext
conda activate omnilingual-asr
python scripts/run_local_inference.py
```

---

**User**

Nope, still out of memory, Im tryiing now 1B model, ive downloaded it, but getting error @bash (788-894) 

---

**Cursor**

```yaml
name: omniASR_tokenizer_local
tokenizer_family: char_tokenizer
tokenizer: file:///media/sigmind/Gaming1/asr/omniASR_tokenizer.model

---

name: omniASR_tokenizer_v7_local
tokenizer_family: char_tokenizer
tokenizer: file:///media/sigmind/Gaming1/asr/omniASR_tokenizer_v7.model

---

name: omniASR_LLM_3B_local
model_family: wav2vec2_llama
model_arch: 3b
tokenizer_ref: omniASR_tokenizer_local
checkpoint: file:///media/sigmind/Gaming1/asr/omniASR-LLM-3B.pt

---

name: omniASR_LLM_7B_local
model_family: wav2vec2_llama
model_arch: 7b
tokenizer_ref: omniASR_tokenizer_v7_local
checkpoint: file:///media/sigmind/Gaming1/asr/omniASR-LLM-7B.pt


---

name: omniASR_LLM_1B_local
model_family: wav2vec2_llama
model_arch: 1b
tokenizer_ref: omniASR_tokenizer_local
checkpoint: file:///media/sigmind/Gaming1/asr/omniASR-LLM-1B.pt




```

The failure was coming from the local asset card file: the 1B entry sat directly beneath the 7B section with no `---` separator, so ruamel saw two `name:` keys in the same mapping and threw `DuplicateKeyError`. While fixing it, I also pointed the 1B card at the normal tokenizer (`omniASR_tokenizer_local`); only the 7B LLM model needs the `_v7` tokenizer.

`~/.config/fairseq2/assets/cards/models/omniasr_local.yaml` now ends with:

```21:32:/home/sigmind/.config/fairseq2/assets/cards/models/omniasr_local.yaml
name: omniASR_LLM_7B_local
model_family: wav2vec2_llama
model_arch: 7b
tokenizer_ref: omniASR_tokenizer_v7_local
checkpoint: file:///media/sigmind/Gaming1/asr/omniASR-LLM-7B.pt

---

name: omniASR_LLM_1B_local
model_family: wav2vec2_llama
model_arch: 1b
tokenizer_ref: omniASR_tokenizer_local
checkpoint: file:///media/sigmind/Gaming1/asr/omniASR-LLM-1B.pt
```

Activate the environment and re-run:

```
conda activate omnilingual-asr
python scripts/run_local_inference.py --model-card omniASR_LLM_1B_local
```

If the 1B checkpoint or tokenizer live under different filenames, tweak those paths in the YAML before retrying.

```yaml
name: omniASR_LLM_7B_local
model_family: wav2vec2_llama
model_arch: 7b
tokenizer_ref: omniASR_tokenizer_v7_local
checkpoint: file:///media/sigmind/Gaming1/asr/omniASR-LLM-7B.pt

---

name: omniASR_LLM_1B_local
model_family: wav2vec2_llama
model_arch: 1b
tokenizer_ref: omniASR_tokenizer_local
checkpoint: file:///media/sigmind/Gaming1/asr/omniASR-LLM-1B.pt
```

```plaintext
conda activate omnilingual-asr
python scripts/run_local_inference.py --model-card omniASR_LLM_1B_local
```

---

**User**

Its not getting bengali language @bash (941-1014)  But I ee o-asr suports it 

Bengali

ISO Code: ben



✕

Status

Family

Script

Safe

Indo-European

Bengali (Bangla)

Download the model or transcription tool to test a sample in this language.

1 of 1600+ covered by Omni-ASR

---

**Cursor**

```python
#!/usr/bin/env python3
"""
Utility script to run Omnilingual ASR inference against locally downloaded
model/tokenizer assets.

The script expects that you have created custom fairseq2 asset cards pointing
to the local checkpoints (see README instructions added by the assistant) and
placed them under ~/.config/fairseq2/assets/cards/models. With that in place,
running this script will not trigger any remote downloads.

It performs two quick sanity checks by streaming short samples from the public
`facebook/omnilingual-asr-corpus` dataset (one Bangla, one English), runs them
through the local omniASR_LLM_7B checkpoint, and prints the reference vs.
predicted text to stdout.
"""

from __future__ import annotations

import argparse
from typing import Iterable, List, Tuple

from datasets import load_dataset

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


LANG_EXAMPLES: Tuple[dict, ...] = (
    {
        "label": "Bengali",
        "lang_code": "ben_Beng",
        "iso_639_3": "ben",
        "iso_15924": "Beng",
    },
    {
        "label": "English",
        "lang_code": "eng_Latn",
        "iso_639_3": "eng",
        "iso_15924": "Latn",
    },
)


def fetch_dataset_sample(
    iso_639_3: str,
    iso_15924: str | None = None,
    *,
    max_duration_s: float = 38.0,
) -> Tuple[dict, str, float]:
    """
    Stream samples from the Hugging Face dataset until we find one shorter than
    ``max_duration_s`` seconds. The returned audio dictionary is already in the
    format accepted by :class:`ASRInferencePipeline`.
    """
    dataset = load_dataset("facebook/omnilingual-asr-corpus", split="train", streaming=True)

    for sample in dataset:
        if sample["iso_639_3"] != iso_639_3:
            continue
        if iso_15924 is not None and sample.get("iso_15924") != iso_15924:
            continue

        audio = sample["audio"]
        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]
        duration = len(waveform) / sample_rate

        if duration <= max_duration_s:
            return (
                {"waveform": waveform, "sample_rate": sample_rate},
                sample["raw_text"],
                duration,
            )

    raise RuntimeError(
        f"Could not find an example shorter than {max_duration_s}s for ISO={iso_639_3} script={iso_15924}."
    )


def run_inference(
    pipeline: ASRInferencePipeline,
    lang_examples: Iterable[dict],
) -> None:
    """
    Fetches one short sample per language, runs inference, and prints the output.
    """
    batch: List[dict] = []
    lang_codes: List[str] = []
    references: List[str] = []
    summaries: List[Tuple[str, str, float, str]] = []

    for example in lang_examples:
        audio_dict, reference_text, duration = fetch_dataset_sample(
            example["iso_639_3"], example.get("iso_15924")
        )
        batch.append(audio_dict)
        lang_codes.append(example["lang_code"])
        references.append(reference_text)
        summaries.append((example["label"], example["lang_code"], duration, reference_text))

    print("Loaded samples:")
    for lang_label, lang_code, duration, reference in summaries:
        snippet = reference if len(reference) <= 80 else reference[:77] + "..."
        print(f"- {lang_label} ({lang_code}): {duration:.2f}s | Reference: {snippet}")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(batch, lang=lang_codes, batch_size=len(batch))

    for (lang_label, lang_code, _, reference), prediction in zip(
        summaries, outputs, strict=True
    ):
        print("\n==>", lang_label, f"({lang_code})")
        print("Reference :", reference)
        print("Prediction:", prediction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run omnilingual ASR inference using locally cached assets."
    )
    parser.add_argument(
        "--model-card",
        default="omniASR_LLM_1B_local",
        help=(
            "Name of the fairseq2 model card that points to the local checkpoint. "
            "Defaults to 'omniASR_LLM_1B_local'."
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=38.0,
        help="Maximum clip duration (in seconds) when sampling from the dataset.",
    )

    args = parser.parse_args()

    print(f"Loading pipeline with model card '{args.model_card}'...")
    pipeline = ASRInferencePipeline(model_card=args.model_card)

    run_inference(pipeline, LANG_EXAMPLES)


if __name__ == "__main__":
    main()


```

`load_dataset()` was failing because the Hugging Face corpus doesn’t expose a `ben_Beng` config—the Bangla clips live in the default split with `iso_639_3 == "ben"` and script `iso_15924 == "Beng"`. I updated `scripts/run_local_inference.py` so each language entry now records both the model’s lang code and the dataset’s ISO/script identifiers; the script streams the default split and filters until it finds a short clip matching those IDs.

```36:64:scripts/run_local_inference.py
LANG_EXAMPLES: Tuple[dict, ...] = (
    {"label": "Bengali", "lang_code": "ben_Beng", "iso_639_3": "ben", "iso_15924": "Beng"},
    {"label": "English", "lang_code": "eng_Latn", "iso_639_3": "eng", "iso_15924": "Latn"},
)
```

Run it again (still using your local 1B checkpoint):

```
conda activate omnilingual-asr
python scripts/run_local_inference.py --model-card omniASR_LLM_1B_local
```

The script will walk the default dataset until it finds Bangla + English samples under those ISO codes, then print the reference vs. predicted text. If you want to point at different languages, add another dict with the appropriate `lang_code`, `iso_639_3`, and (optionally) `iso_15924`.

```python
LANG_EXAMPLES: Tuple[dict, ...] = (
    {"label": "Bengali", "lang_code": "ben_Beng", "iso_639_3": "ben", "iso_15924": "Beng"},
    {"label": "English", "lang_code": "eng_Latn", "iso_639_3": "eng", "iso_15924": "Latn"},
)
```

```plaintext
conda activate omnilingual-asr
python scripts/run_local_inference.py --model-card omniASR_LLM_1B_local
```

---

**User**

I see its stuck here @python (1013-1024) 

---

**Cursor**

```python
#!/usr/bin/env python3
"""
Utility script to run Omnilingual ASR inference against locally downloaded
model/tokenizer assets.

The script expects that you have created custom fairseq2 asset cards pointing
to the local checkpoints (see README instructions added by the assistant) and
placed them under ~/.config/fairseq2/assets/cards/models. With that in place,
running this script will not trigger any remote downloads.

It performs two quick sanity checks by streaming short samples from the public
`facebook/omnilingual-asr-corpus` dataset (one Bangla, one English), runs them
through the local omniASR_LLM_7B checkpoint, and prints the reference vs.
predicted text to stdout.
"""

from __future__ import annotations

import argparse
from typing import Iterable, List, Tuple

from datasets import load_dataset

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


LANG_EXAMPLES: Tuple[dict, ...] = (
    {
        "label": "Bengali",
        "lang_code": "ben_Beng",
        "iso_639_3": "ben",
        "iso_15924": "Beng",
    },
    {
        "label": "English",
        "lang_code": "eng_Latn",
        "iso_639_3": "eng",
        "iso_15924": "Latn",
    },
)


def fetch_dataset_sample(
    iso_639_3: str,
    iso_15924: str | None = None,
    *,
    max_duration_s: float = 38.0,
    max_samples_to_scan: int = 50_000,
) -> Tuple[dict, str, float]:
    """
    Stream samples from the Hugging Face dataset until we find one shorter than
    ``max_duration_s`` seconds. The returned audio dictionary is already in the
    format accepted by :class:`ASRInferencePipeline`.
    """
    dataset = load_dataset("facebook/omnilingual-asr-corpus", split="train", streaming=True)

    for idx, sample in enumerate(dataset, start=1):
        if sample["iso_639_3"] != iso_639_3:
            continue
        if iso_15924 is not None and sample.get("iso_15924") != iso_15924:
            continue

        audio = sample["audio"]
        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]
        duration = len(waveform) / sample_rate

        if duration <= max_duration_s:
            return (
                {"waveform": waveform, "sample_rate": sample_rate},
                sample["raw_text"],
                duration,
            )

        if idx % 5000 == 0:
            print(
                f"  scanned {idx} samples for ISO={iso_639_3} script={iso_15924} "
                f"but none shorter than {max_duration_s}s yet..."
            )

        if idx >= max_samples_to_scan:
            break

    raise RuntimeError(
        f"Could not find an example shorter than {max_duration_s}s for ISO={iso_639_3} "
        f"script={iso_15924} within {max_samples_to_scan} samples."
    )


def run_inference(
    pipeline: ASRInferencePipeline,
    lang_examples: Iterable[dict],
) -> None:
    """
    Fetches one short sample per language, runs inference, and prints the output.
    """
    batch: List[dict] = []
    lang_codes: List[str] = []
    references: List[str] = []
    summaries: List[Tuple[str, str, float, str]] = []

    for example in lang_examples:
        print(
            f"Searching for {example['label']} sample "
            f"(ISO={example['iso_639_3']}, script={example.get('iso_15924')})..."
        )

        audio_dict, reference_text, duration = fetch_dataset_sample(
            example["iso_639_3"], example.get("iso_15924")
        )
        batch.append(audio_dict)
        lang_codes.append(example["lang_code"])
        references.append(reference_text)
        summaries.append((example["label"], example["lang_code"], duration, reference_text))

    print("Loaded samples:")
    for lang_label, lang_code, duration, reference in summaries:
        snippet = reference if len(reference) <= 80 else reference[:77] + "..."
        print(f"- {lang_label} ({lang_code}): {duration:.2f}s | Reference: {snippet}")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(batch, lang=lang_codes, batch_size=len(batch))

    for (lang_label, lang_code, _, reference), prediction in zip(
        summaries, outputs, strict=True
    ):
        print("\n==>", lang_label, f"({lang_code})")
        print("Reference :", reference)
        print("Prediction:", prediction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run omnilingual ASR inference using locally cached assets."
    )
    parser.add_argument(
        "--model-card",
        default="omniASR_LLM_1B_local",
        help=(
            "Name of the fairseq2 model card that points to the local checkpoint. "
            "Defaults to 'omniASR_LLM_1B_local'."
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=38.0,
        help="Maximum clip duration (in seconds) when sampling from the dataset.",
    )

    args = parser.parse_args()

    print(f"Loading pipeline with model card '{args.model_card}'...")
    pipeline = ASRInferencePipeline(model_card=args.model_card)

    run_inference(pipeline, LANG_EXAMPLES)


if __name__ == "__main__":
    main()


```

It wasn’t actually hung—`fetch_dataset_sample()` was streaming the entire default split until it hit a Bangla clip shorter than 38 s, which can take a long time because the dataset isn’t sorted by language or duration. I updated `scripts/run_local_inference.py` so it:

- Logs when it starts scanning for each language.
- Prints a heartbeat every 5 000 samples while searching.
- Stops after 50 000 samples and raises a clear error if nothing short enough was found.

The Bangla entry is now tried first, so you’ll see progress right away. Re-run:

```
python scripts/run_local_inference.py --model-card omniASR_LLM_1B_local
```

If it still doesn’t find a short Bangla clip, consider lowering `--max-duration` or selecting a known local audio file instead of streaming.

```plaintext
python scripts/run_local_inference.py --model-card omniASR_LLM_1B_local
```

---

**User**

@python (1009-1027) Its stuck here for 10 mins. I want to use a local audio file bangla.m4a for the test to transcribe

---

**Cursor**

```python
#!/usr/bin/env python3
"""
Utility script to run Omnilingual ASR inference against locally downloaded
model/tokenizer assets.

The script expects that you have created custom fairseq2 asset cards pointing
to the local checkpoints (see README instructions added by the assistant) and
placed them under ~/.config/fairseq2/assets/cards/models. With that in place,
running this script will not trigger any remote downloads.

It performs two quick sanity checks by streaming short samples from the public
`facebook/omnilingual-asr-corpus` dataset (one Bangla, one English), runs them
through the local omniASR_LLM_7B checkpoint, and prints the reference vs.
predicted text to stdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

from datasets import load_dataset

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


LANG_EXAMPLES: Tuple[dict, ...] = (
    {
        "label": "Bengali",
        "lang_code": "ben_Beng",
        "iso_639_3": "ben",
        "iso_15924": "Beng",
    },
    {
        "label": "English",
        "lang_code": "eng_Latn",
        "iso_639_3": "eng",
        "iso_15924": "Latn",
    },
)


def fetch_dataset_sample(
    iso_639_3: str,
    iso_15924: str | None = None,
    *,
    max_duration_s: float = 38.0,
    max_samples_to_scan: int = 50_000,
) -> Tuple[dict, str, float]:
    """
    Stream samples from the Hugging Face dataset until we find one shorter than
    ``max_duration_s`` seconds. The returned audio dictionary is already in the
    format accepted by :class:`ASRInferencePipeline`.
    """
    dataset = load_dataset("facebook/omnilingual-asr-corpus", split="train", streaming=True)

    for idx, sample in enumerate(dataset, start=1):
        if sample["iso_639_3"] != iso_639_3:
            continue
        if iso_15924 is not None and sample.get("iso_15924") != iso_15924:
            continue

        audio = sample["audio"]
        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]
        duration = len(waveform) / sample_rate

        if duration <= max_duration_s:
            return (
                {"waveform": waveform, "sample_rate": sample_rate},
                sample["raw_text"],
                duration,
            )

        if idx % 5000 == 0:
            print(
                f"  scanned {idx} samples for ISO={iso_639_3} script={iso_15924} "
                f"but none shorter than {max_duration_s}s yet..."
            )

        if idx >= max_samples_to_scan:
            break

    raise RuntimeError(
        f"Could not find an example shorter than {max_duration_s}s for ISO={iso_639_3} "
        f"script={iso_15924} within {max_samples_to_scan} samples."
    )


def run_inference(
    pipeline: ASRInferencePipeline,
    lang_examples: Iterable[dict],
) -> None:
    """
    Fetches one short sample per language, runs inference, and prints the output.
    """
    batch: List[dict] = []
    lang_codes: List[str] = []
    references: List[str] = []
    summaries: List[Tuple[str, str, float, str]] = []

    for example in lang_examples:
        print(
            f"Searching for {example['label']} sample "
            f"(ISO={example['iso_639_3']}, script={example.get('iso_15924')})..."
        )

        audio_dict, reference_text, duration = fetch_dataset_sample(
            example["iso_639_3"], example.get("iso_15924")
        )
        batch.append(audio_dict)
        lang_codes.append(example["lang_code"])
        references.append(reference_text)
        summaries.append((example["label"], example["lang_code"], duration, reference_text))

    print("Loaded samples:")
    for lang_label, lang_code, duration, reference in summaries:
        snippet = reference if len(reference) <= 80 else reference[:77] + "..."
        print(f"- {lang_label} ({lang_code}): {duration:.2f}s | Reference: {snippet}")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(batch, lang=lang_codes, batch_size=len(batch))

    for (lang_label, lang_code, _, reference), prediction in zip(
        summaries, outputs, strict=True
    ):
        print("\n==>", lang_label, f"({lang_code})")
        print("Reference :", reference)
        print("Prediction:", prediction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run omnilingual ASR inference using locally cached assets."
    )
    parser.add_argument(
        "--model-card",
        default="omniASR_LLM_1B_local",
        help=(
            "Name of the fairseq2 model card that points to the local checkpoint. "
            "Defaults to 'omniASR_LLM_1B_local'."
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=38.0,
        help="Maximum clip duration (in seconds) when sampling from the dataset.",
    )

    args = parser.parse_args()

    print(f"Loading pipeline with model card '{args.model_card}'...")
    pipeline = ASRInferencePipeline(model_card=args.model_card)

    run_inference(pipeline, LANG_EXAMPLES)


if __name__ == "__main__":
    main()


```

```python
#!/usr/bin/env python3
"""
Utility script to run Omnilingual ASR inference against locally downloaded
model/tokenizer assets.

The script expects that you have created custom fairseq2 asset cards pointing
to the local checkpoints (see README instructions added by the assistant) and
placed them under ~/.config/fairseq2/assets/cards/models. With that in place,
running this script will not trigger any remote downloads.

It performs two quick sanity checks by streaming short samples from the public
`facebook/omnilingual-asr-corpus` dataset (one Bangla, one English), runs them
through the local omniASR_LLM_7B checkpoint, and prints the reference vs.
predicted text to stdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

from datasets import load_dataset

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


LANG_EXAMPLES: Tuple[dict, ...] = (
    {
        "label": "Bengali",
        "lang_code": "ben_Beng",
        "iso_639_3": "ben",
        "iso_15924": "Beng",
    },
    {
        "label": "English",
        "lang_code": "eng_Latn",
        "iso_639_3": "eng",
        "iso_15924": "Latn",
    },
)


def fetch_dataset_sample(
    iso_639_3: str,
    iso_15924: str | None = None,
    *,
    max_duration_s: float = 38.0,
    max_samples_to_scan: int = 50_000,
) -> Tuple[dict, str, float]:
    """
    Stream samples from the Hugging Face dataset until we find one shorter than
    ``max_duration_s`` seconds. The returned audio dictionary is already in the
    format accepted by :class:`ASRInferencePipeline`.
    """
    dataset = load_dataset("facebook/omnilingual-asr-corpus", split="train", streaming=True)

    for idx, sample in enumerate(dataset, start=1):
        if sample["iso_639_3"] != iso_639_3:
            continue
        if iso_15924 is not None and sample.get("iso_15924") != iso_15924:
            continue

        audio = sample["audio"]
        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]
        duration = len(waveform) / sample_rate

        if duration <= max_duration_s:
            return (
                {"waveform": waveform, "sample_rate": sample_rate},
                sample["raw_text"],
                duration,
            )

        if idx % 5000 == 0:
            print(
                f"  scanned {idx} samples for ISO={iso_639_3} script={iso_15924} "
                f"but none shorter than {max_duration_s}s yet..."
            )

        if idx >= max_samples_to_scan:
            break

    raise RuntimeError(
        f"Could not find an example shorter than {max_duration_s}s for ISO={iso_639_3} "
        f"script={iso_15924} within {max_samples_to_scan} samples."
    )


def run_inference_from_files(
    pipeline: ASRInferencePipeline,
    audio_files: List[Path],
    lang_codes: List[str],
) -> None:
    """
    Runs inference on local audio files and prints the output.
    """
    print(f"Transcribing {len(audio_files)} audio file(s)...")
    for audio_file, lang_code in zip(audio_files, lang_codes):
        print(f"  - {audio_file.name} ({lang_code})")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(
        [str(f) for f in audio_files],
        lang=lang_codes,
        batch_size=len(audio_files),
    )

    for audio_file, lang_code, prediction in zip(audio_files, lang_codes, outputs):
        print(f"\n==> {audio_file.name} ({lang_code})")
        print("Prediction:", prediction)


def run_inference(
    pipeline: ASRInferencePipeline,
    lang_examples: Iterable[dict],
) -> None:
    """
    Fetches one short sample per language, runs inference, and prints the output.
    """
    batch: List[dict] = []
    lang_codes: List[str] = []
    references: List[str] = []
    summaries: List[Tuple[str, str, float, str]] = []

    for example in lang_examples:
        print(
            f"Searching for {example['label']} sample "
            f"(ISO={example['iso_639_3']}, script={example.get('iso_15924')})..."
        )

        audio_dict, reference_text, duration = fetch_dataset_sample(
            example["iso_639_3"], example.get("iso_15924")
        )
        batch.append(audio_dict)
        lang_codes.append(example["lang_code"])
        references.append(reference_text)
        summaries.append((example["label"], example["lang_code"], duration, reference_text))

    print("Loaded samples:")
    for lang_label, lang_code, duration, reference in summaries:
        snippet = reference if len(reference) <= 80 else reference[:77] + "..."
        print(f"- {lang_label} ({lang_code}): {duration:.2f}s | Reference: {snippet}")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(batch, lang=lang_codes, batch_size=len(batch))

    for (lang_label, lang_code, _, reference), prediction in zip(
        summaries, outputs, strict=True
    ):
        print("\n==>", lang_label, f"({lang_code})")
        print("Reference :", reference)
        print("Prediction:", prediction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run omnilingual ASR inference using locally cached assets."
    )
    parser.add_argument(
        "--model-card",
        default="omniASR_LLM_1B_local",
        help=(
            "Name of the fairseq2 model card that points to the local checkpoint. "
            "Defaults to 'omniASR_LLM_1B_local'."
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=38.0,
        help="Maximum clip duration (in seconds) when sampling from the dataset.",
    )

    args = parser.parse_args()

    print(f"Loading pipeline with model card '{args.model_card}'...")
    pipeline = ASRInferencePipeline(model_card=args.model_card)

    run_inference(pipeline, LANG_EXAMPLES)


if __name__ == "__main__":
    main()


```

```python
#!/usr/bin/env python3
"""
Utility script to run Omnilingual ASR inference against locally downloaded
model/tokenizer assets.

The script expects that you have created custom fairseq2 asset cards pointing
to the local checkpoints (see README instructions added by the assistant) and
placed them under ~/.config/fairseq2/assets/cards/models. With that in place,
running this script will not trigger any remote downloads.

It performs two quick sanity checks by streaming short samples from the public
`facebook/omnilingual-asr-corpus` dataset (one Bangla, one English), runs them
through the local omniASR_LLM_7B checkpoint, and prints the reference vs.
predicted text to stdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

from datasets import load_dataset

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


LANG_EXAMPLES: Tuple[dict, ...] = (
    {
        "label": "Bengali",
        "lang_code": "ben_Beng",
        "iso_639_3": "ben",
        "iso_15924": "Beng",
    },
    {
        "label": "English",
        "lang_code": "eng_Latn",
        "iso_639_3": "eng",
        "iso_15924": "Latn",
    },
)


def fetch_dataset_sample(
    iso_639_3: str,
    iso_15924: str | None = None,
    *,
    max_duration_s: float = 38.0,
    max_samples_to_scan: int = 50_000,
) -> Tuple[dict, str, float]:
    """
    Stream samples from the Hugging Face dataset until we find one shorter than
    ``max_duration_s`` seconds. The returned audio dictionary is already in the
    format accepted by :class:`ASRInferencePipeline`.
    """
    dataset = load_dataset("facebook/omnilingual-asr-corpus", split="train", streaming=True)

    for idx, sample in enumerate(dataset, start=1):
        if sample["iso_639_3"] != iso_639_3:
            continue
        if iso_15924 is not None and sample.get("iso_15924") != iso_15924:
            continue

        audio = sample["audio"]
        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]
        duration = len(waveform) / sample_rate

        if duration <= max_duration_s:
            return (
                {"waveform": waveform, "sample_rate": sample_rate},
                sample["raw_text"],
                duration,
            )

        if idx % 5000 == 0:
            print(
                f"  scanned {idx} samples for ISO={iso_639_3} script={iso_15924} "
                f"but none shorter than {max_duration_s}s yet..."
            )

        if idx >= max_samples_to_scan:
            break

    raise RuntimeError(
        f"Could not find an example shorter than {max_duration_s}s for ISO={iso_639_3} "
        f"script={iso_15924} within {max_samples_to_scan} samples."
    )


def run_inference_from_files(
    pipeline: ASRInferencePipeline,
    audio_files: List[Path],
    lang_codes: List[str],
) -> None:
    """
    Runs inference on local audio files and prints the output.
    """
    print(f"Transcribing {len(audio_files)} audio file(s)...")
    for audio_file, lang_code in zip(audio_files, lang_codes):
        print(f"  - {audio_file.name} ({lang_code})")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(
        [str(f) for f in audio_files],
        lang=lang_codes,
        batch_size=len(audio_files),
    )

    for audio_file, lang_code, prediction in zip(audio_files, lang_codes, outputs):
        print(f"\n==> {audio_file.name} ({lang_code})")
        print("Prediction:", prediction)


def run_inference(
    pipeline: ASRInferencePipeline,
    lang_examples: Iterable[dict],
) -> None:
    """
    Fetches one short sample per language, runs inference, and prints the output.
    """
    batch: List[dict] = []
    lang_codes: List[str] = []
    references: List[str] = []
    summaries: List[Tuple[str, str, float, str]] = []

    for example in lang_examples:
        print(
            f"Searching for {example['label']} sample "
            f"(ISO={example['iso_639_3']}, script={example.get('iso_15924')})..."
        )

        audio_dict, reference_text, duration = fetch_dataset_sample(
            example["iso_639_3"], example.get("iso_15924")
        )
        batch.append(audio_dict)
        lang_codes.append(example["lang_code"])
        references.append(reference_text)
        summaries.append((example["label"], example["lang_code"], duration, reference_text))

    print("Loaded samples:")
    for lang_label, lang_code, duration, reference in summaries:
        snippet = reference if len(reference) <= 80 else reference[:77] + "..."
        print(f"- {lang_label} ({lang_code}): {duration:.2f}s | Reference: {snippet}")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(batch, lang=lang_codes, batch_size=len(batch))

    for (lang_label, lang_code, _, reference), prediction in zip(
        summaries, outputs, strict=True
    ):
        print("\n==>", lang_label, f"({lang_code})")
        print("Reference :", reference)
        print("Prediction:", prediction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run omnilingual ASR inference using locally cached assets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use local audio files
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng
  %(prog)s --audio-file bangla.m4a ben_Beng --audio-file english.wav eng_Latn

  # Use dataset samples (default)
  %(prog)s --model-card omniASR_LLM_1B_local
        """,
    )
    parser.add_argument(
        "--model-card",
        default="omniASR_LLM_1B_local",
        help=(
            "Name of the fairseq2 model card that points to the local checkpoint. "
            "Defaults to 'omniASR_LLM_1B_local'."
        ),
    )
    parser.add_argument(
        "--audio-file",
        action="append",
        dest="audio_files",
        type=Path,
        help="Path to an audio file to transcribe. Can be specified multiple times.",
    )
    parser.add_argument(
        "--lang-code",
        action="append",
        dest="lang_codes",
        help=(
            "Language code for the corresponding audio file (e.g., 'ben_Beng', 'eng_Latn'). "
            "Must be specified once per --audio-file. Can be specified multiple times."
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=38.0,
        help="Maximum clip duration (in seconds) when sampling from the dataset.",
    )

    args = parser.parse_args()

    print(f"Loading pipeline with model card '{args.model_card}'...")
    pipeline = ASRInferencePipeline(model_card=args.model_card)

    # If audio files are provided, use them instead of dataset
    if args.audio_files:
        if not args.lang_codes:
            parser.error("--lang-code must be specified when using --audio-file")
        if len(args.audio_files) != len(args.lang_codes):
            parser.error(
                f"Number of --audio-file ({len(args.audio_files)}) must match "
                f"number of --lang-code ({len(args.lang_codes)})"
            )
        # Verify files exist
        for audio_file in args.audio_files:
            if not audio_file.exists():
                parser.error(f"Audio file not found: {audio_file}")
        run_inference_from_files(pipeline, args.audio_files, args.lang_codes)
    else:
        # Default: use dataset samples
        run_inference(pipeline, LANG_EXAMPLES)


if __name__ == "__main__":
    main()


```

```python
#!/usr/bin/env python3
"""
Utility script to run Omnilingual ASR inference against locally downloaded
model/tokenizer assets.

The script expects that you have created custom fairseq2 asset cards pointing
to the local checkpoints (see README instructions added by the assistant) and
placed them under ~/.config/fairseq2/assets/cards/models. With that in place,
running this script will not trigger any remote downloads.

It performs two quick sanity checks by streaming short samples from the public
`facebook/omnilingual-asr-corpus` dataset (one Bangla, one English), runs them
through the local omniASR_LLM_7B checkpoint, and prints the reference vs.
predicted text to stdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

from datasets import load_dataset

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


LANG_EXAMPLES: Tuple[dict, ...] = (
    {
        "label": "Bengali",
        "lang_code": "ben_Beng",
        "iso_639_3": "ben",
        "iso_15924": "Beng",
    },
    {
        "label": "English",
        "lang_code": "eng_Latn",
        "iso_639_3": "eng",
        "iso_15924": "Latn",
    },
)


def fetch_dataset_sample(
    iso_639_3: str,
    iso_15924: str | None = None,
    *,
    max_duration_s: float = 38.0,
    max_samples_to_scan: int = 50_000,
) -> Tuple[dict, str, float]:
    """
    Stream samples from the Hugging Face dataset until we find one shorter than
    ``max_duration_s`` seconds. The returned audio dictionary is already in the
    format accepted by :class:`ASRInferencePipeline`.
    """
    dataset = load_dataset("facebook/omnilingual-asr-corpus", split="train", streaming=True)

    for idx, sample in enumerate(dataset, start=1):
        if sample["iso_639_3"] != iso_639_3:
            continue
        if iso_15924 is not None and sample.get("iso_15924") != iso_15924:
            continue

        audio = sample["audio"]
        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]
        duration = len(waveform) / sample_rate

        if duration <= max_duration_s:
            return (
                {"waveform": waveform, "sample_rate": sample_rate},
                sample["raw_text"],
                duration,
            )

        if idx % 5000 == 0:
            print(
                f"  scanned {idx} samples for ISO={iso_639_3} script={iso_15924} "
                f"but none shorter than {max_duration_s}s yet..."
            )

        if idx >= max_samples_to_scan:
            break

    raise RuntimeError(
        f"Could not find an example shorter than {max_duration_s}s for ISO={iso_639_3} "
        f"script={iso_15924} within {max_samples_to_scan} samples."
    )


def run_inference_from_files(
    pipeline: ASRInferencePipeline,
    audio_files: List[Path],
    lang_codes: List[str],
) -> None:
    """
    Runs inference on local audio files and prints the output.
    """
    print(f"Transcribing {len(audio_files)} audio file(s)...")
    for audio_file, lang_code in zip(audio_files, lang_codes):
        print(f"  - {audio_file.name} ({lang_code})")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(
        [str(f) for f in audio_files],
        lang=lang_codes,
        batch_size=len(audio_files),
    )

    for audio_file, lang_code, prediction in zip(audio_files, lang_codes, outputs):
        print(f"\n==> {audio_file.name} ({lang_code})")
        print("Prediction:", prediction)


def run_inference(
    pipeline: ASRInferencePipeline,
    lang_examples: Iterable[dict],
) -> None:
    """
    Fetches one short sample per language, runs inference, and prints the output.
    """
    batch: List[dict] = []
    lang_codes: List[str] = []
    references: List[str] = []
    summaries: List[Tuple[str, str, float, str]] = []

    for example in lang_examples:
        print(
            f"Searching for {example['label']} sample "
            f"(ISO={example['iso_639_3']}, script={example.get('iso_15924')})..."
        )

        audio_dict, reference_text, duration = fetch_dataset_sample(
            example["iso_639_3"], example.get("iso_15924")
        )
        batch.append(audio_dict)
        lang_codes.append(example["lang_code"])
        references.append(reference_text)
        summaries.append((example["label"], example["lang_code"], duration, reference_text))

    print("Loaded samples:")
    for lang_label, lang_code, duration, reference in summaries:
        snippet = reference if len(reference) <= 80 else reference[:77] + "..."
        print(f"- {lang_label} ({lang_code}): {duration:.2f}s | Reference: {snippet}")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(batch, lang=lang_codes, batch_size=len(batch))

    for (lang_label, lang_code, _, reference), prediction in zip(
        summaries, outputs, strict=True
    ):
        print("\n==>", lang_label, f"({lang_code})")
        print("Reference :", reference)
        print("Prediction:", prediction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run omnilingual ASR inference using locally cached assets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use local audio files
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng --audio-file english.wav --lang-code eng_Latn

  # Use dataset samples (default)
  %(prog)s --model-card omniASR_LLM_1B_local
        """,
    )
    parser.add_argument(
        "--model-card",
        default="omniASR_LLM_1B_local",
        help=(
            "Name of the fairseq2 model card that points to the local checkpoint. "
            "Defaults to 'omniASR_LLM_1B_local'."
        ),
    )
    parser.add_argument(
        "--audio-file",
        action="append",
        dest="audio_files",
        type=Path,
        help="Path to an audio file to transcribe. Can be specified multiple times.",
    )
    parser.add_argument(
        "--lang-code",
        action="append",
        dest="lang_codes",
        help=(
            "Language code for the corresponding audio file (e.g., 'ben_Beng', 'eng_Latn'). "
            "Must be specified once per --audio-file. Can be specified multiple times."
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=38.0,
        help="Maximum clip duration (in seconds) when sampling from the dataset.",
    )

    args = parser.parse_args()

    print(f"Loading pipeline with model card '{args.model_card}'...")
    pipeline = ASRInferencePipeline(model_card=args.model_card)

    # If audio files are provided, use them instead of dataset
    if args.audio_files:
        if not args.lang_codes:
            parser.error("--lang-code must be specified when using --audio-file")
        if len(args.audio_files) != len(args.lang_codes):
            parser.error(
                f"Number of --audio-file ({len(args.audio_files)}) must match "
                f"number of --lang-code ({len(args.lang_codes)})"
            )
        # Verify files exist
        for audio_file in args.audio_files:
            if not audio_file.exists():
                parser.error(f"Audio file not found: {audio_file}")
        run_inference_from_files(pipeline, args.audio_files, args.lang_codes)
    else:
        # Default: use dataset samples
        run_inference(pipeline, LANG_EXAMPLES)


if __name__ == "__main__":
    main()


```

Updated the script to support local audio files. You can now transcribe your `bangla.m4a` file directly without streaming from the dataset.

Run:

```bash
python scripts/run_local_inference.py --audio-file bangla.m4a --lang-code ben_Beng
```

If you have multiple files (e.g., Bangla and English):

```bash
python scripts/run_local_inference.py \
  --audio-file bangla.m4a --lang-code ben_Beng \
  --audio-file english.wav --lang-code eng_Latn
```

The script will:
1. Load your local model checkpoint
2. Transcribe the audio file(s) you provide
3. Print the transcription results

**Note:** Make sure your `bangla.m4a` file is in the current directory, or provide the full path. The audio file must be shorter than 40 seconds (as per the model's limitation mentioned in the README).

If the file is in a different location, use the full path:
```bash
python scripts/run_local_inference.py --audio-file /path/to/bangla.m4a --lang-code ben_Beng
```

```shellscript
python scripts/run_local_inference.py --audio-file bangla.m4a --lang-code ben_Beng
```

```shellscript
python scripts/run_local_inference.py \
  --audio-file bangla.m4a --lang-code ben_Beng \
  --audio-file english.wav --lang-code eng_Latn
```

```shellscript
python scripts/run_local_inference.py --audio-file /path/to/bangla.m4a --lang-code ben_Beng
```

---

**User**

@bash (989-1027) Its getting error

---

**Cursor**

The `.m4a` format isn't supported by the audio decoder (libsndfile). Converting the file to WAV or FLAC, or adding automatic conversion in the script. Checking if ffmpeg is available:


Adding automatic conversion for unsupported formats using ffmpeg:


```python
#!/usr/bin/env python3
"""
Utility script to run Omnilingual ASR inference against locally downloaded
model/tokenizer assets.

The script expects that you have created custom fairseq2 asset cards pointing
to the local checkpoints (see README instructions added by the assistant) and
placed them under ~/.config/fairseq2/assets/cards/models. With that in place,
running this script will not trigger any remote downloads.

It performs two quick sanity checks by streaming short samples from the public
`facebook/omnilingual-asr-corpus` dataset (one Bangla, one English), runs them
through the local omniASR_LLM_7B checkpoint, and prints the reference vs.
predicted text to stdout.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

from datasets import load_dataset

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


LANG_EXAMPLES: Tuple[dict, ...] = (
    {
        "label": "Bengali",
        "lang_code": "ben_Beng",
        "iso_639_3": "ben",
        "iso_15924": "Beng",
    },
    {
        "label": "English",
        "lang_code": "eng_Latn",
        "iso_639_3": "eng",
        "iso_15924": "Latn",
    },
)


def fetch_dataset_sample(
    iso_639_3: str,
    iso_15924: str | None = None,
    *,
    max_duration_s: float = 38.0,
    max_samples_to_scan: int = 50_000,
) -> Tuple[dict, str, float]:
    """
    Stream samples from the Hugging Face dataset until we find one shorter than
    ``max_duration_s`` seconds. The returned audio dictionary is already in the
    format accepted by :class:`ASRInferencePipeline`.
    """
    dataset = load_dataset("facebook/omnilingual-asr-corpus", split="train", streaming=True)

    for idx, sample in enumerate(dataset, start=1):
        if sample["iso_639_3"] != iso_639_3:
            continue
        if iso_15924 is not None and sample.get("iso_15924") != iso_15924:
            continue

        audio = sample["audio"]
        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]
        duration = len(waveform) / sample_rate

        if duration <= max_duration_s:
            return (
                {"waveform": waveform, "sample_rate": sample_rate},
                sample["raw_text"],
                duration,
            )

        if idx % 5000 == 0:
            print(
                f"  scanned {idx} samples for ISO={iso_639_3} script={iso_15924} "
                f"but none shorter than {max_duration_s}s yet..."
            )

        if idx >= max_samples_to_scan:
            break

    raise RuntimeError(
        f"Could not find an example shorter than {max_duration_s}s for ISO={iso_639_3} "
        f"script={iso_15924} within {max_samples_to_scan} samples."
    )


def run_inference_from_files(
    pipeline: ASRInferencePipeline,
    audio_files: List[Path],
    lang_codes: List[str],
) -> None:
    """
    Runs inference on local audio files and prints the output.
    """
    print(f"Transcribing {len(audio_files)} audio file(s)...")
    for audio_file, lang_code in zip(audio_files, lang_codes):
        print(f"  - {audio_file.name} ({lang_code})")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(
        [str(f) for f in audio_files],
        lang=lang_codes,
        batch_size=len(audio_files),
    )

    for audio_file, lang_code, prediction in zip(audio_files, lang_codes, outputs):
        print(f"\n==> {audio_file.name} ({lang_code})")
        print("Prediction:", prediction)


def run_inference(
    pipeline: ASRInferencePipeline,
    lang_examples: Iterable[dict],
) -> None:
    """
    Fetches one short sample per language, runs inference, and prints the output.
    """
    batch: List[dict] = []
    lang_codes: List[str] = []
    references: List[str] = []
    summaries: List[Tuple[str, str, float, str]] = []

    for example in lang_examples:
        print(
            f"Searching for {example['label']} sample "
            f"(ISO={example['iso_639_3']}, script={example.get('iso_15924')})..."
        )

        audio_dict, reference_text, duration = fetch_dataset_sample(
            example["iso_639_3"], example.get("iso_15924")
        )
        batch.append(audio_dict)
        lang_codes.append(example["lang_code"])
        references.append(reference_text)
        summaries.append((example["label"], example["lang_code"], duration, reference_text))

    print("Loaded samples:")
    for lang_label, lang_code, duration, reference in summaries:
        snippet = reference if len(reference) <= 80 else reference[:77] + "..."
        print(f"- {lang_label} ({lang_code}): {duration:.2f}s | Reference: {snippet}")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(batch, lang=lang_codes, batch_size=len(batch))

    for (lang_label, lang_code, _, reference), prediction in zip(
        summaries, outputs, strict=True
    ):
        print("\n==>", lang_label, f"({lang_code})")
        print("Reference :", reference)
        print("Prediction:", prediction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run omnilingual ASR inference using locally cached assets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use local audio files
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng --audio-file english.wav --lang-code eng_Latn

  # Use dataset samples (default)
  %(prog)s --model-card omniASR_LLM_1B_local
        """,
    )
    parser.add_argument(
        "--model-card",
        default="omniASR_LLM_1B_local",
        help=(
            "Name of the fairseq2 model card that points to the local checkpoint. "
            "Defaults to 'omniASR_LLM_1B_local'."
        ),
    )
    parser.add_argument(
        "--audio-file",
        action="append",
        dest="audio_files",
        type=Path,
        help="Path to an audio file to transcribe. Can be specified multiple times.",
    )
    parser.add_argument(
        "--lang-code",
        action="append",
        dest="lang_codes",
        help=(
            "Language code for the corresponding audio file (e.g., 'ben_Beng', 'eng_Latn'). "
            "Must be specified once per --audio-file. Can be specified multiple times."
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=38.0,
        help="Maximum clip duration (in seconds) when sampling from the dataset.",
    )

    args = parser.parse_args()

    print(f"Loading pipeline with model card '{args.model_card}'...")
    pipeline = ASRInferencePipeline(model_card=args.model_card)

    # If audio files are provided, use them instead of dataset
    if args.audio_files:
        if not args.lang_codes:
            parser.error("--lang-code must be specified when using --audio-file")
        if len(args.audio_files) != len(args.lang_codes):
            parser.error(
                f"Number of --audio-file ({len(args.audio_files)}) must match "
                f"number of --lang-code ({len(args.lang_codes)})"
            )
        # Verify files exist
        for audio_file in args.audio_files:
            if not audio_file.exists():
                parser.error(f"Audio file not found: {audio_file}")
        run_inference_from_files(pipeline, args.audio_files, args.lang_codes)
    else:
        # Default: use dataset samples
        run_inference(pipeline, LANG_EXAMPLES)


if __name__ == "__main__":
    main()


```

```python
#!/usr/bin/env python3
"""
Utility script to run Omnilingual ASR inference against locally downloaded
model/tokenizer assets.

The script expects that you have created custom fairseq2 asset cards pointing
to the local checkpoints (see README instructions added by the assistant) and
placed them under ~/.config/fairseq2/assets/cards/models. With that in place,
running this script will not trigger any remote downloads.

It performs two quick sanity checks by streaming short samples from the public
`facebook/omnilingual-asr-corpus` dataset (one Bangla, one English), runs them
through the local omniASR_LLM_7B checkpoint, and prints the reference vs.
predicted text to stdout.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

from datasets import load_dataset

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


LANG_EXAMPLES: Tuple[dict, ...] = (
    {
        "label": "Bengali",
        "lang_code": "ben_Beng",
        "iso_639_3": "ben",
        "iso_15924": "Beng",
    },
    {
        "label": "English",
        "lang_code": "eng_Latn",
        "iso_639_3": "eng",
        "iso_15924": "Latn",
    },
)


def fetch_dataset_sample(
    iso_639_3: str,
    iso_15924: str | None = None,
    *,
    max_duration_s: float = 38.0,
    max_samples_to_scan: int = 50_000,
) -> Tuple[dict, str, float]:
    """
    Stream samples from the Hugging Face dataset until we find one shorter than
    ``max_duration_s`` seconds. The returned audio dictionary is already in the
    format accepted by :class:`ASRInferencePipeline`.
    """
    dataset = load_dataset("facebook/omnilingual-asr-corpus", split="train", streaming=True)

    for idx, sample in enumerate(dataset, start=1):
        if sample["iso_639_3"] != iso_639_3:
            continue
        if iso_15924 is not None and sample.get("iso_15924") != iso_15924:
            continue

        audio = sample["audio"]
        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]
        duration = len(waveform) / sample_rate

        if duration <= max_duration_s:
            return (
                {"waveform": waveform, "sample_rate": sample_rate},
                sample["raw_text"],
                duration,
            )

        if idx % 5000 == 0:
            print(
                f"  scanned {idx} samples for ISO={iso_639_3} script={iso_15924} "
                f"but none shorter than {max_duration_s}s yet..."
            )

        if idx >= max_samples_to_scan:
            break

    raise RuntimeError(
        f"Could not find an example shorter than {max_duration_s}s for ISO={iso_639_3} "
        f"script={iso_15924} within {max_samples_to_scan} samples."
    )


# Supported audio formats by libsndfile (used by fairseq2)
SUPPORTED_FORMATS = {".wav", ".flac", ".ogg", ".au", ".aiff", ".mp3"}


def convert_audio_to_wav(input_file: Path, output_file: Path | None = None) -> Path:
    """
    Convert an audio file to WAV format using ffmpeg.
    
    Args:
        input_file: Path to the input audio file
        output_file: Optional path for output file. If None, creates a temp file.
    
    Returns:
        Path to the converted WAV file
    """
    if output_file is None:
        # Create a temporary WAV file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        output_file = Path(temp_path)
        # Close the file descriptor so ffmpeg can write to it
        import os
        os.close(temp_fd)
    
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(input_file),
                "-ar", "16000",  # Resample to 16kHz (model requirement)
                "-ac", "1",      # Convert to mono
                "-y",            # Overwrite output file
                str(output_file),
            ],
            check=True,
            capture_output=True,
        )
        return output_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to convert {input_file} to WAV format. "
            f"ffmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg to convert audio files, "
            "or convert your audio files to WAV format manually."
        )


def prepare_audio_files(audio_files: List[Path]) -> Tuple[List[Path], List[Path]]:
    """
    Prepare audio files for transcription by converting unsupported formats to WAV.
    
    Returns:
        Tuple of (list of file paths to use, list of temp files to clean up)
    """
    prepared_files: List[Path] = []
    temp_files: List[Path] = []
    
    for audio_file in audio_files:
        suffix = audio_file.suffix.lower()
        if suffix in SUPPORTED_FORMATS:
            # File is already in a supported format
            prepared_files.append(audio_file)
        else:
            # Need to convert
            print(f"  Converting {audio_file.name} ({suffix}) to WAV format...")
            converted = convert_audio_to_wav(audio_file)
            prepared_files.append(converted)
            temp_files.append(converted)
    
    return prepared_files, temp_files


def run_inference_from_files(
    pipeline: ASRInferencePipeline,
    audio_files: List[Path],
    lang_codes: List[str],
) -> None:
    """
    Runs inference on local audio files and prints the output.
    """
    print(f"Transcribing {len(audio_files)} audio file(s)...")
    for audio_file, lang_code in zip(audio_files, lang_codes):
        print(f"  - {audio_file.name} ({lang_code})")
    
    # Convert unsupported formats to WAV
    prepared_files, temp_files = prepare_audio_files(audio_files)
    
    try:
        print("\nRunning transcription...")
        outputs = pipeline.transcribe(
            [str(f) for f in prepared_files],
            lang=lang_codes,
            batch_size=len(prepared_files),
        )

        for audio_file, lang_code, prediction in zip(audio_files, lang_codes, outputs):
            print(f"\n==> {audio_file.name} ({lang_code})")
            print("Prediction:", prediction)
    finally:
        # Clean up temporary converted files
        for temp_file in temp_files:
            try:
                temp_file.unlink()
            except Exception:
                pass  # Ignore cleanup errors


def run_inference(
    pipeline: ASRInferencePipeline,
    lang_examples: Iterable[dict],
) -> None:
    """
    Fetches one short sample per language, runs inference, and prints the output.
    """
    batch: List[dict] = []
    lang_codes: List[str] = []
    references: List[str] = []
    summaries: List[Tuple[str, str, float, str]] = []

    for example in lang_examples:
        print(
            f"Searching for {example['label']} sample "
            f"(ISO={example['iso_639_3']}, script={example.get('iso_15924')})..."
        )

        audio_dict, reference_text, duration = fetch_dataset_sample(
            example["iso_639_3"], example.get("iso_15924")
        )
        batch.append(audio_dict)
        lang_codes.append(example["lang_code"])
        references.append(reference_text)
        summaries.append((example["label"], example["lang_code"], duration, reference_text))

    print("Loaded samples:")
    for lang_label, lang_code, duration, reference in summaries:
        snippet = reference if len(reference) <= 80 else reference[:77] + "..."
        print(f"- {lang_label} ({lang_code}): {duration:.2f}s | Reference: {snippet}")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(batch, lang=lang_codes, batch_size=len(batch))

    for (lang_label, lang_code, _, reference), prediction in zip(
        summaries, outputs, strict=True
    ):
        print("\n==>", lang_label, f"({lang_code})")
        print("Reference :", reference)
        print("Prediction:", prediction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run omnilingual ASR inference using locally cached assets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use local audio files
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng --audio-file english.wav --lang-code eng_Latn

  # Use dataset samples (default)
  %(prog)s --model-card omniASR_LLM_1B_local
        """,
    )
    parser.add_argument(
        "--model-card",
        default="omniASR_LLM_1B_local",
        help=(
            "Name of the fairseq2 model card that points to the local checkpoint. "
            "Defaults to 'omniASR_LLM_1B_local'."
        ),
    )
    parser.add_argument(
        "--audio-file",
        action="append",
        dest="audio_files",
        type=Path,
        help="Path to an audio file to transcribe. Can be specified multiple times.",
    )
    parser.add_argument(
        "--lang-code",
        action="append",
        dest="lang_codes",
        help=(
            "Language code for the corresponding audio file (e.g., 'ben_Beng', 'eng_Latn'). "
            "Must be specified once per --audio-file. Can be specified multiple times."
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=38.0,
        help="Maximum clip duration (in seconds) when sampling from the dataset.",
    )

    args = parser.parse_args()

    print(f"Loading pipeline with model card '{args.model_card}'...")
    pipeline = ASRInferencePipeline(model_card=args.model_card)

    # If audio files are provided, use them instead of dataset
    if args.audio_files:
        if not args.lang_codes:
            parser.error("--lang-code must be specified when using --audio-file")
        if len(args.audio_files) != len(args.lang_codes):
            parser.error(
                f"Number of --audio-file ({len(args.audio_files)}) must match "
                f"number of --lang-code ({len(args.lang_codes)})"
            )
        # Verify files exist
        for audio_file in args.audio_files:
            if not audio_file.exists():
                parser.error(f"Audio file not found: {audio_file}")
        run_inference_from_files(pipeline, args.audio_files, args.lang_codes)
    else:
        # Default: use dataset samples
        run_inference(pipeline, LANG_EXAMPLES)


if __name__ == "__main__":
    main()


```

```python
#!/usr/bin/env python3
"""
Utility script to run Omnilingual ASR inference against locally downloaded
model/tokenizer assets.

The script expects that you have created custom fairseq2 asset cards pointing
to the local checkpoints (see README instructions added by the assistant) and
placed them under ~/.config/fairseq2/assets/cards/models. With that in place,
running this script will not trigger any remote downloads.

It performs two quick sanity checks by streaming short samples from the public
`facebook/omnilingual-asr-corpus` dataset (one Bangla, one English), runs them
through the local omniASR_LLM_7B checkpoint, and prints the reference vs.
predicted text to stdout.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

from datasets import load_dataset

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


LANG_EXAMPLES: Tuple[dict, ...] = (
    {
        "label": "Bengali",
        "lang_code": "ben_Beng",
        "iso_639_3": "ben",
        "iso_15924": "Beng",
    },
    {
        "label": "English",
        "lang_code": "eng_Latn",
        "iso_639_3": "eng",
        "iso_15924": "Latn",
    },
)


def fetch_dataset_sample(
    iso_639_3: str,
    iso_15924: str | None = None,
    *,
    max_duration_s: float = 38.0,
    max_samples_to_scan: int = 50_000,
) -> Tuple[dict, str, float]:
    """
    Stream samples from the Hugging Face dataset until we find one shorter than
    ``max_duration_s`` seconds. The returned audio dictionary is already in the
    format accepted by :class:`ASRInferencePipeline`.
    """
    dataset = load_dataset("facebook/omnilingual-asr-corpus", split="train", streaming=True)

    for idx, sample in enumerate(dataset, start=1):
        if sample["iso_639_3"] != iso_639_3:
            continue
        if iso_15924 is not None and sample.get("iso_15924") != iso_15924:
            continue

        audio = sample["audio"]
        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]
        duration = len(waveform) / sample_rate

        if duration <= max_duration_s:
            return (
                {"waveform": waveform, "sample_rate": sample_rate},
                sample["raw_text"],
                duration,
            )

        if idx % 5000 == 0:
            print(
                f"  scanned {idx} samples for ISO={iso_639_3} script={iso_15924} "
                f"but none shorter than {max_duration_s}s yet..."
            )

        if idx >= max_samples_to_scan:
            break

    raise RuntimeError(
        f"Could not find an example shorter than {max_duration_s}s for ISO={iso_639_3} "
        f"script={iso_15924} within {max_samples_to_scan} samples."
    )


# Supported audio formats by libsndfile (used by fairseq2)
SUPPORTED_FORMATS = {".wav", ".flac", ".ogg", ".au", ".aiff", ".mp3"}


def convert_audio_to_wav(input_file: Path, output_file: Path | None = None) -> Path:
    """
    Convert an audio file to WAV format using ffmpeg.
    
    Args:
        input_file: Path to the input audio file
        output_file: Optional path for output file. If None, creates a temp file.
    
    Returns:
        Path to the converted WAV file
    """
    if output_file is None:
        # Create a temporary WAV file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        output_file = Path(temp_path)
        # Close the file descriptor so ffmpeg can write to it
        import os
        os.close(temp_fd)
    
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(input_file),
                "-ar", "16000",  # Resample to 16kHz (model requirement)
                "-ac", "1",      # Convert to mono
                "-y",            # Overwrite output file
                str(output_file),
            ],
            check=True,
            capture_output=True,
        )
        return output_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to convert {input_file} to WAV format. "
            f"ffmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg to convert audio files, "
            "or convert your audio files to WAV format manually."
        )


def prepare_audio_files(audio_files: List[Path]) -> Tuple[List[Path], List[Path]]:
    """
    Prepare audio files for transcription by converting unsupported formats to WAV.
    
    Returns:
        Tuple of (list of file paths to use, list of temp files to clean up)
    """
    prepared_files: List[Path] = []
    temp_files: List[Path] = []
    
    for audio_file in audio_files:
        suffix = audio_file.suffix.lower()
        if suffix in SUPPORTED_FORMATS:
            # File is already in a supported format
            prepared_files.append(audio_file)
        else:
            # Need to convert
            print(f"  Converting {audio_file.name} ({suffix}) to WAV format...")
            converted = convert_audio_to_wav(audio_file)
            prepared_files.append(converted)
            temp_files.append(converted)
    
    return prepared_files, temp_files


def run_inference_from_files(
    pipeline: ASRInferencePipeline,
    audio_files: List[Path],
    lang_codes: List[str],
) -> None:
    """
    Runs inference on local audio files and prints the output.
    """
    print(f"Transcribing {len(audio_files)} audio file(s)...")
    for audio_file, lang_code in zip(audio_files, lang_codes):
        print(f"  - {audio_file.name} ({lang_code})")
    
    # Convert unsupported formats to WAV
    prepared_files, temp_files = prepare_audio_files(audio_files)
    
    try:
        print("\nRunning transcription...")
        outputs = pipeline.transcribe(
            [str(f) for f in prepared_files],
            lang=lang_codes,
            batch_size=len(prepared_files),
        )

        for audio_file, lang_code, prediction in zip(audio_files, lang_codes, outputs):
            print(f"\n==> {audio_file.name} ({lang_code})")
            print("Prediction:", prediction)
    finally:
        # Clean up temporary converted files
        for temp_file in temp_files:
            try:
                temp_file.unlink()
            except Exception:
                pass  # Ignore cleanup errors


def run_inference(
    pipeline: ASRInferencePipeline,
    lang_examples: Iterable[dict],
) -> None:
    """
    Fetches one short sample per language, runs inference, and prints the output.
    """
    batch: List[dict] = []
    lang_codes: List[str] = []
    references: List[str] = []
    summaries: List[Tuple[str, str, float, str]] = []

    for example in lang_examples:
        print(
            f"Searching for {example['label']} sample "
            f"(ISO={example['iso_639_3']}, script={example.get('iso_15924')})..."
        )

        audio_dict, reference_text, duration = fetch_dataset_sample(
            example["iso_639_3"], example.get("iso_15924")
        )
        batch.append(audio_dict)
        lang_codes.append(example["lang_code"])
        references.append(reference_text)
        summaries.append((example["label"], example["lang_code"], duration, reference_text))

    print("Loaded samples:")
    for lang_label, lang_code, duration, reference in summaries:
        snippet = reference if len(reference) <= 80 else reference[:77] + "..."
        print(f"- {lang_label} ({lang_code}): {duration:.2f}s | Reference: {snippet}")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(batch, lang=lang_codes, batch_size=len(batch))

    for (lang_label, lang_code, _, reference), prediction in zip(
        summaries, outputs, strict=True
    ):
        print("\n==>", lang_label, f"({lang_code})")
        print("Reference :", reference)
        print("Prediction:", prediction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run omnilingual ASR inference using locally cached assets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use local audio files
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng --audio-file english.wav --lang-code eng_Latn

  # Use dataset samples (default)
  %(prog)s --model-card omniASR_LLM_1B_local
        """,
    )
    parser.add_argument(
        "--model-card",
        default="omniASR_LLM_1B_local",
        help=(
            "Name of the fairseq2 model card that points to the local checkpoint. "
            "Defaults to 'omniASR_LLM_1B_local'."
        ),
    )
    parser.add_argument(
        "--audio-file",
        action="append",
        dest="audio_files",
        type=Path,
        help="Path to an audio file to transcribe. Can be specified multiple times.",
    )
    parser.add_argument(
        "--lang-code",
        action="append",
        dest="lang_codes",
        help=(
            "Language code for the corresponding audio file (e.g., 'ben_Beng', 'eng_Latn'). "
            "Must be specified once per --audio-file. Can be specified multiple times."
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=38.0,
        help="Maximum clip duration (in seconds) when sampling from the dataset.",
    )

    args = parser.parse_args()

    print(f"Loading pipeline with model card '{args.model_card}'...")
    pipeline = ASRInferencePipeline(model_card=args.model_card)

    # If audio files are provided, use them instead of dataset
    if args.audio_files:
        if not args.lang_codes:
            parser.error("--lang-code must be specified when using --audio-file")
        if len(args.audio_files) != len(args.lang_codes):
            parser.error(
                f"Number of --audio-file ({len(args.audio_files)}) must match "
                f"number of --lang-code ({len(args.lang_codes)})"
            )
        # Verify files exist
        for audio_file in args.audio_files:
            if not audio_file.exists():
                parser.error(f"Audio file not found: {audio_file}")
        run_inference_from_files(pipeline, args.audio_files, args.lang_codes)
    else:
        # Default: use dataset samples
        run_inference(pipeline, LANG_EXAMPLES)


if __name__ == "__main__":
    main()


```

```python
#!/usr/bin/env python3
"""
Utility script to run Omnilingual ASR inference against locally downloaded
model/tokenizer assets.

The script expects that you have created custom fairseq2 asset cards pointing
to the local checkpoints (see README instructions added by the assistant) and
placed them under ~/.config/fairseq2/assets/cards/models. With that in place,
running this script will not trigger any remote downloads.

It performs two quick sanity checks by streaming short samples from the public
`facebook/omnilingual-asr-corpus` dataset (one Bangla, one English), runs them
through the local omniASR_LLM_7B checkpoint, and prints the reference vs.
predicted text to stdout.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

from datasets import load_dataset

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


LANG_EXAMPLES: Tuple[dict, ...] = (
    {
        "label": "Bengali",
        "lang_code": "ben_Beng",
        "iso_639_3": "ben",
        "iso_15924": "Beng",
    },
    {
        "label": "English",
        "lang_code": "eng_Latn",
        "iso_639_3": "eng",
        "iso_15924": "Latn",
    },
)


def fetch_dataset_sample(
    iso_639_3: str,
    iso_15924: str | None = None,
    *,
    max_duration_s: float = 38.0,
    max_samples_to_scan: int = 50_000,
) -> Tuple[dict, str, float]:
    """
    Stream samples from the Hugging Face dataset until we find one shorter than
    ``max_duration_s`` seconds. The returned audio dictionary is already in the
    format accepted by :class:`ASRInferencePipeline`.
    """
    dataset = load_dataset("facebook/omnilingual-asr-corpus", split="train", streaming=True)

    for idx, sample in enumerate(dataset, start=1):
        if sample["iso_639_3"] != iso_639_3:
            continue
        if iso_15924 is not None and sample.get("iso_15924") != iso_15924:
            continue

        audio = sample["audio"]
        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]
        duration = len(waveform) / sample_rate

        if duration <= max_duration_s:
            return (
                {"waveform": waveform, "sample_rate": sample_rate},
                sample["raw_text"],
                duration,
            )

        if idx % 5000 == 0:
            print(
                f"  scanned {idx} samples for ISO={iso_639_3} script={iso_15924} "
                f"but none shorter than {max_duration_s}s yet..."
            )

        if idx >= max_samples_to_scan:
            break

    raise RuntimeError(
        f"Could not find an example shorter than {max_duration_s}s for ISO={iso_639_3} "
        f"script={iso_15924} within {max_samples_to_scan} samples."
    )


# Supported audio formats by libsndfile (used by fairseq2)
SUPPORTED_FORMATS = {".wav", ".flac", ".ogg", ".au", ".aiff", ".mp3"}


def convert_audio_to_wav(input_file: Path, output_file: Path | None = None) -> Path:
    """
    Convert an audio file to WAV format using ffmpeg.
    
    Args:
        input_file: Path to the input audio file
        output_file: Optional path for output file. If None, creates a temp file.
    
    Returns:
        Path to the converted WAV file
    """
    if output_file is None:
        # Create a temporary WAV file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        output_file = Path(temp_path)
        # Close the file descriptor so ffmpeg can write to it
        os.close(temp_fd)
    
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(input_file),
                "-ar", "16000",  # Resample to 16kHz (model requirement)
                "-ac", "1",      # Convert to mono
                "-y",            # Overwrite output file
                str(output_file),
            ],
            check=True,
            capture_output=True,
        )
        return output_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to convert {input_file} to WAV format. "
            f"ffmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg to convert audio files, "
            "or convert your audio files to WAV format manually."
        )


def prepare_audio_files(audio_files: List[Path]) -> Tuple[List[Path], List[Path]]:
    """
    Prepare audio files for transcription by converting unsupported formats to WAV.
    
    Returns:
        Tuple of (list of file paths to use, list of temp files to clean up)
    """
    prepared_files: List[Path] = []
    temp_files: List[Path] = []
    
    for audio_file in audio_files:
        suffix = audio_file.suffix.lower()
        if suffix in SUPPORTED_FORMATS:
            # File is already in a supported format
            prepared_files.append(audio_file)
        else:
            # Need to convert
            print(f"  Converting {audio_file.name} ({suffix}) to WAV format...")
            converted = convert_audio_to_wav(audio_file)
            prepared_files.append(converted)
            temp_files.append(converted)
    
    return prepared_files, temp_files


def run_inference_from_files(
    pipeline: ASRInferencePipeline,
    audio_files: List[Path],
    lang_codes: List[str],
) -> None:
    """
    Runs inference on local audio files and prints the output.
    """
    print(f"Transcribing {len(audio_files)} audio file(s)...")
    for audio_file, lang_code in zip(audio_files, lang_codes):
        print(f"  - {audio_file.name} ({lang_code})")
    
    # Convert unsupported formats to WAV
    prepared_files, temp_files = prepare_audio_files(audio_files)
    
    try:
        print("\nRunning transcription...")
        outputs = pipeline.transcribe(
            [str(f) for f in prepared_files],
            lang=lang_codes,
            batch_size=len(prepared_files),
        )

        for audio_file, lang_code, prediction in zip(audio_files, lang_codes, outputs):
            print(f"\n==> {audio_file.name} ({lang_code})")
            print("Prediction:", prediction)
    finally:
        # Clean up temporary converted files
        for temp_file in temp_files:
            try:
                temp_file.unlink()
            except Exception:
                pass  # Ignore cleanup errors


def run_inference(
    pipeline: ASRInferencePipeline,
    lang_examples: Iterable[dict],
) -> None:
    """
    Fetches one short sample per language, runs inference, and prints the output.
    """
    batch: List[dict] = []
    lang_codes: List[str] = []
    references: List[str] = []
    summaries: List[Tuple[str, str, float, str]] = []

    for example in lang_examples:
        print(
            f"Searching for {example['label']} sample "
            f"(ISO={example['iso_639_3']}, script={example.get('iso_15924')})..."
        )

        audio_dict, reference_text, duration = fetch_dataset_sample(
            example["iso_639_3"], example.get("iso_15924")
        )
        batch.append(audio_dict)
        lang_codes.append(example["lang_code"])
        references.append(reference_text)
        summaries.append((example["label"], example["lang_code"], duration, reference_text))

    print("Loaded samples:")
    for lang_label, lang_code, duration, reference in summaries:
        snippet = reference if len(reference) <= 80 else reference[:77] + "..."
        print(f"- {lang_label} ({lang_code}): {duration:.2f}s | Reference: {snippet}")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(batch, lang=lang_codes, batch_size=len(batch))

    for (lang_label, lang_code, _, reference), prediction in zip(
        summaries, outputs, strict=True
    ):
        print("\n==>", lang_label, f"({lang_code})")
        print("Reference :", reference)
        print("Prediction:", prediction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run omnilingual ASR inference using locally cached assets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use local audio files
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng --audio-file english.wav --lang-code eng_Latn

  # Use dataset samples (default)
  %(prog)s --model-card omniASR_LLM_1B_local
        """,
    )
    parser.add_argument(
        "--model-card",
        default="omniASR_LLM_1B_local",
        help=(
            "Name of the fairseq2 model card that points to the local checkpoint. "
            "Defaults to 'omniASR_LLM_1B_local'."
        ),
    )
    parser.add_argument(
        "--audio-file",
        action="append",
        dest="audio_files",
        type=Path,
        help="Path to an audio file to transcribe. Can be specified multiple times.",
    )
    parser.add_argument(
        "--lang-code",
        action="append",
        dest="lang_codes",
        help=(
            "Language code for the corresponding audio file (e.g., 'ben_Beng', 'eng_Latn'). "
            "Must be specified once per --audio-file. Can be specified multiple times."
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=38.0,
        help="Maximum clip duration (in seconds) when sampling from the dataset.",
    )

    args = parser.parse_args()

    print(f"Loading pipeline with model card '{args.model_card}'...")
    pipeline = ASRInferencePipeline(model_card=args.model_card)

    # If audio files are provided, use them instead of dataset
    if args.audio_files:
        if not args.lang_codes:
            parser.error("--lang-code must be specified when using --audio-file")
        if len(args.audio_files) != len(args.lang_codes):
            parser.error(
                f"Number of --audio-file ({len(args.audio_files)}) must match "
                f"number of --lang-code ({len(args.lang_codes)})"
            )
        # Verify files exist
        for audio_file in args.audio_files:
            if not audio_file.exists():
                parser.error(f"Audio file not found: {audio_file}")
        run_inference_from_files(pipeline, args.audio_files, args.lang_codes)
    else:
        # Default: use dataset samples
        run_inference(pipeline, LANG_EXAMPLES)


if __name__ == "__main__":
    main()


```

```python
#!/usr/bin/env python3
"""
Utility script to run Omnilingual ASR inference against locally downloaded
model/tokenizer assets.

The script expects that you have created custom fairseq2 asset cards pointing
to the local checkpoints (see README instructions added by the assistant) and
placed them under ~/.config/fairseq2/assets/cards/models. With that in place,
running this script will not trigger any remote downloads.

It performs two quick sanity checks by streaming short samples from the public
`facebook/omnilingual-asr-corpus` dataset (one Bangla, one English), runs them
through the local omniASR_LLM_7B checkpoint, and prints the reference vs.
predicted text to stdout.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from datasets import load_dataset

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


LANG_EXAMPLES: Tuple[dict, ...] = (
    {
        "label": "Bengali",
        "lang_code": "ben_Beng",
        "iso_639_3": "ben",
        "iso_15924": "Beng",
    },
    {
        "label": "English",
        "lang_code": "eng_Latn",
        "iso_639_3": "eng",
        "iso_15924": "Latn",
    },
)


def fetch_dataset_sample(
    iso_639_3: str,
    iso_15924: str | None = None,
    *,
    max_duration_s: float = 38.0,
    max_samples_to_scan: int = 50_000,
) -> Tuple[dict, str, float]:
    """
    Stream samples from the Hugging Face dataset until we find one shorter than
    ``max_duration_s`` seconds. The returned audio dictionary is already in the
    format accepted by :class:`ASRInferencePipeline`.
    """
    dataset = load_dataset("facebook/omnilingual-asr-corpus", split="train", streaming=True)

    for idx, sample in enumerate(dataset, start=1):
        if sample["iso_639_3"] != iso_639_3:
            continue
        if iso_15924 is not None and sample.get("iso_15924") != iso_15924:
            continue

        audio = sample["audio"]
        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]
        duration = len(waveform) / sample_rate

        if duration <= max_duration_s:
            return (
                {"waveform": waveform, "sample_rate": sample_rate},
                sample["raw_text"],
                duration,
            )

        if idx % 5000 == 0:
            print(
                f"  scanned {idx} samples for ISO={iso_639_3} script={iso_15924} "
                f"but none shorter than {max_duration_s}s yet..."
            )

        if idx >= max_samples_to_scan:
            break

    raise RuntimeError(
        f"Could not find an example shorter than {max_duration_s}s for ISO={iso_639_3} "
        f"script={iso_15924} within {max_samples_to_scan} samples."
    )


# Supported audio formats by libsndfile (used by fairseq2)
SUPPORTED_FORMATS = {".wav", ".flac", ".ogg", ".au", ".aiff", ".mp3"}


def convert_audio_to_wav(input_file: Path, output_file: Path | None = None) -> Path:
    """
    Convert an audio file to WAV format using ffmpeg.
    
    Args:
        input_file: Path to the input audio file
        output_file: Optional path for output file. If None, creates a temp file.
    
    Returns:
        Path to the converted WAV file
    """
    if output_file is None:
        # Create a temporary WAV file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        output_file = Path(temp_path)
        # Close the file descriptor so ffmpeg can write to it
        os.close(temp_fd)
    
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(input_file),
                "-ar", "16000",  # Resample to 16kHz (model requirement)
                "-ac", "1",      # Convert to mono
                "-y",            # Overwrite output file
                str(output_file),
            ],
            check=True,
            capture_output=True,
        )
        return output_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to convert {input_file} to WAV format. "
            f"ffmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg to convert audio files, "
            "or convert your audio files to WAV format manually."
        )


def prepare_audio_files(audio_files: List[Path]) -> Tuple[List[Path], List[Path]]:
    """
    Prepare audio files for transcription by converting unsupported formats to WAV.
    
    Returns:
        Tuple of (list of file paths to use, list of temp files to clean up)
    """
    prepared_files: List[Path] = []
    temp_files: List[Path] = []
    
    for audio_file in audio_files:
        suffix = audio_file.suffix.lower()
        if suffix in SUPPORTED_FORMATS:
            # File is already in a supported format
            prepared_files.append(audio_file)
        else:
            # Need to convert
            print(f"  Converting {audio_file.name} ({suffix}) to WAV format...")
            converted = convert_audio_to_wav(audio_file)
            prepared_files.append(converted)
            temp_files.append(converted)
    
    return prepared_files, temp_files


def run_inference_from_files(
    pipeline: ASRInferencePipeline,
    audio_files: List[Path],
    lang_codes: List[str],
) -> None:
    """
    Runs inference on local audio files and prints the output.
    """
    print(f"Transcribing {len(audio_files)} audio file(s)...")
    for audio_file, lang_code in zip(audio_files, lang_codes):
        print(f"  - {audio_file.name} ({lang_code})")
    
    # Convert unsupported formats to WAV
    prepared_files, temp_files = prepare_audio_files(audio_files)
    
    try:
        print("\nRunning transcription...")
        outputs = pipeline.transcribe(
            [str(f) for f in prepared_files],
            lang=lang_codes,
            batch_size=len(prepared_files),
        )

        for audio_file, lang_code, prediction in zip(audio_files, lang_codes, outputs):
            print(f"\n==> {audio_file.name} ({lang_code})")
            print("Prediction:", prediction)
    finally:
        # Clean up temporary converted files
        for temp_file in temp_files:
            try:
                temp_file.unlink()
            except Exception:
                pass  # Ignore cleanup errors


def run_inference(
    pipeline: ASRInferencePipeline,
    lang_examples: Iterable[dict],
) -> None:
    """
    Fetches one short sample per language, runs inference, and prints the output.
    """
    batch: List[dict] = []
    lang_codes: List[str] = []
    references: List[str] = []
    summaries: List[Tuple[str, str, float, str]] = []

    for example in lang_examples:
        print(
            f"Searching for {example['label']} sample "
            f"(ISO={example['iso_639_3']}, script={example.get('iso_15924')})..."
        )

        audio_dict, reference_text, duration = fetch_dataset_sample(
            example["iso_639_3"], example.get("iso_15924")
        )
        batch.append(audio_dict)
        lang_codes.append(example["lang_code"])
        references.append(reference_text)
        summaries.append((example["label"], example["lang_code"], duration, reference_text))

    print("Loaded samples:")
    for lang_label, lang_code, duration, reference in summaries:
        snippet = reference if len(reference) <= 80 else reference[:77] + "..."
        print(f"- {lang_label} ({lang_code}): {duration:.2f}s | Reference: {snippet}")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(batch, lang=lang_codes, batch_size=len(batch))

    for (lang_label, lang_code, _, reference), prediction in zip(
        summaries, outputs, strict=True
    ):
        print("\n==>", lang_label, f"({lang_code})")
        print("Reference :", reference)
        print("Prediction:", prediction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run omnilingual ASR inference using locally cached assets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use local audio files
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng --audio-file english.wav --lang-code eng_Latn

  # Use dataset samples (default)
  %(prog)s --model-card omniASR_LLM_1B_local
        """,
    )
    parser.add_argument(
        "--model-card",
        default="omniASR_LLM_1B_local",
        help=(
            "Name of the fairseq2 model card that points to the local checkpoint. "
            "Defaults to 'omniASR_LLM_1B_local'."
        ),
    )
    parser.add_argument(
        "--audio-file",
        action="append",
        dest="audio_files",
        type=Path,
        help="Path to an audio file to transcribe. Can be specified multiple times.",
    )
    parser.add_argument(
        "--lang-code",
        action="append",
        dest="lang_codes",
        help=(
            "Language code for the corresponding audio file (e.g., 'ben_Beng', 'eng_Latn'). "
            "Must be specified once per --audio-file. Can be specified multiple times."
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=38.0,
        help="Maximum clip duration (in seconds) when sampling from the dataset.",
    )

    args = parser.parse_args()

    print(f"Loading pipeline with model card '{args.model_card}'...")
    pipeline = ASRInferencePipeline(model_card=args.model_card)

    # If audio files are provided, use them instead of dataset
    if args.audio_files:
        if not args.lang_codes:
            parser.error("--lang-code must be specified when using --audio-file")
        if len(args.audio_files) != len(args.lang_codes):
            parser.error(
                f"Number of --audio-file ({len(args.audio_files)}) must match "
                f"number of --lang-code ({len(args.lang_codes)})"
            )
        # Verify files exist
        for audio_file in args.audio_files:
            if not audio_file.exists():
                parser.error(f"Audio file not found: {audio_file}")
        run_inference_from_files(pipeline, args.audio_files, args.lang_codes)
    else:
        # Default: use dataset samples
        run_inference(pipeline, LANG_EXAMPLES)


if __name__ == "__main__":
    main()


```

```python
#!/usr/bin/env python3
"""
Utility script to run Omnilingual ASR inference against locally downloaded
model/tokenizer assets.

The script expects that you have created custom fairseq2 asset cards pointing
to the local checkpoints (see README instructions added by the assistant) and
placed them under ~/.config/fairseq2/assets/cards/models. With that in place,
running this script will not trigger any remote downloads.

It performs two quick sanity checks by streaming short samples from the public
`facebook/omnilingual-asr-corpus` dataset (one Bangla, one English), runs them
through the local omniASR_LLM_7B checkpoint, and prints the reference vs.
predicted text to stdout.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from datasets import load_dataset

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


LANG_EXAMPLES: Tuple[dict, ...] = (
    {
        "label": "Bengali",
        "lang_code": "ben_Beng",
        "iso_639_3": "ben",
        "iso_15924": "Beng",
    },
    {
        "label": "English",
        "lang_code": "eng_Latn",
        "iso_639_3": "eng",
        "iso_15924": "Latn",
    },
)


def fetch_dataset_sample(
    iso_639_3: str,
    iso_15924: str | None = None,
    *,
    max_duration_s: float = 38.0,
    max_samples_to_scan: int = 50_000,
) -> Tuple[dict, str, float]:
    """
    Stream samples from the Hugging Face dataset until we find one shorter than
    ``max_duration_s`` seconds. The returned audio dictionary is already in the
    format accepted by :class:`ASRInferencePipeline`.
    """
    dataset = load_dataset("facebook/omnilingual-asr-corpus", split="train", streaming=True)

    for idx, sample in enumerate(dataset, start=1):
        if sample["iso_639_3"] != iso_639_3:
            continue
        if iso_15924 is not None and sample.get("iso_15924") != iso_15924:
            continue

        audio = sample["audio"]
        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]
        duration = len(waveform) / sample_rate

        if duration <= max_duration_s:
            return (
                {"waveform": waveform, "sample_rate": sample_rate},
                sample["raw_text"],
                duration,
            )

        if idx % 5000 == 0:
            print(
                f"  scanned {idx} samples for ISO={iso_639_3} script={iso_15924} "
                f"but none shorter than {max_duration_s}s yet..."
            )

        if idx >= max_samples_to_scan:
            break

    raise RuntimeError(
        f"Could not find an example shorter than {max_duration_s}s for ISO={iso_639_3} "
        f"script={iso_15924} within {max_samples_to_scan} samples."
    )


# Supported audio formats by libsndfile (used by fairseq2)
SUPPORTED_FORMATS = {".wav", ".flac", ".ogg", ".au", ".aiff", ".mp3"}


def convert_audio_to_wav(input_file: Path, output_file: Optional[Path] = None) -> Path:
    """
    Convert an audio file to WAV format using ffmpeg.
    
    Args:
        input_file: Path to the input audio file
        output_file: Optional path for output file. If None, creates a temp file.
    
    Returns:
        Path to the converted WAV file
    """
    if output_file is None:
        # Create a temporary WAV file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        output_file = Path(temp_path)
        # Close the file descriptor so ffmpeg can write to it
        os.close(temp_fd)
    
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(input_file),
                "-ar", "16000",  # Resample to 16kHz (model requirement)
                "-ac", "1",      # Convert to mono
                "-y",            # Overwrite output file
                str(output_file),
            ],
            check=True,
            capture_output=True,
        )
        return output_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to convert {input_file} to WAV format. "
            f"ffmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg to convert audio files, "
            "or convert your audio files to WAV format manually."
        )


def prepare_audio_files(audio_files: List[Path]) -> Tuple[List[Path], List[Path]]:
    """
    Prepare audio files for transcription by converting unsupported formats to WAV.
    
    Returns:
        Tuple of (list of file paths to use, list of temp files to clean up)
    """
    prepared_files: List[Path] = []
    temp_files: List[Path] = []
    
    for audio_file in audio_files:
        suffix = audio_file.suffix.lower()
        if suffix in SUPPORTED_FORMATS:
            # File is already in a supported format
            prepared_files.append(audio_file)
        else:
            # Need to convert
            print(f"  Converting {audio_file.name} ({suffix}) to WAV format...")
            converted = convert_audio_to_wav(audio_file)
            prepared_files.append(converted)
            temp_files.append(converted)
    
    return prepared_files, temp_files


def run_inference_from_files(
    pipeline: ASRInferencePipeline,
    audio_files: List[Path],
    lang_codes: List[str],
) -> None:
    """
    Runs inference on local audio files and prints the output.
    """
    print(f"Transcribing {len(audio_files)} audio file(s)...")
    for audio_file, lang_code in zip(audio_files, lang_codes):
        print(f"  - {audio_file.name} ({lang_code})")
    
    # Convert unsupported formats to WAV
    prepared_files, temp_files = prepare_audio_files(audio_files)
    
    try:
        print("\nRunning transcription...")
        outputs = pipeline.transcribe(
            [str(f) for f in prepared_files],
            lang=lang_codes,
            batch_size=len(prepared_files),
        )

        for audio_file, lang_code, prediction in zip(audio_files, lang_codes, outputs):
            print(f"\n==> {audio_file.name} ({lang_code})")
            print("Prediction:", prediction)
    finally:
        # Clean up temporary converted files
        for temp_file in temp_files:
            try:
                temp_file.unlink()
            except Exception:
                pass  # Ignore cleanup errors


def run_inference(
    pipeline: ASRInferencePipeline,
    lang_examples: Iterable[dict],
) -> None:
    """
    Fetches one short sample per language, runs inference, and prints the output.
    """
    batch: List[dict] = []
    lang_codes: List[str] = []
    references: List[str] = []
    summaries: List[Tuple[str, str, float, str]] = []

    for example in lang_examples:
        print(
            f"Searching for {example['label']} sample "
            f"(ISO={example['iso_639_3']}, script={example.get('iso_15924')})..."
        )

        audio_dict, reference_text, duration = fetch_dataset_sample(
            example["iso_639_3"], example.get("iso_15924")
        )
        batch.append(audio_dict)
        lang_codes.append(example["lang_code"])
        references.append(reference_text)
        summaries.append((example["label"], example["lang_code"], duration, reference_text))

    print("Loaded samples:")
    for lang_label, lang_code, duration, reference in summaries:
        snippet = reference if len(reference) <= 80 else reference[:77] + "..."
        print(f"- {lang_label} ({lang_code}): {duration:.2f}s | Reference: {snippet}")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(batch, lang=lang_codes, batch_size=len(batch))

    for (lang_label, lang_code, _, reference), prediction in zip(
        summaries, outputs, strict=True
    ):
        print("\n==>", lang_label, f"({lang_code})")
        print("Reference :", reference)
        print("Prediction:", prediction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run omnilingual ASR inference using locally cached assets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use local audio files
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng --audio-file english.wav --lang-code eng_Latn

  # Use dataset samples (default)
  %(prog)s --model-card omniASR_LLM_1B_local
        """,
    )
    parser.add_argument(
        "--model-card",
        default="omniASR_LLM_1B_local",
        help=(
            "Name of the fairseq2 model card that points to the local checkpoint. "
            "Defaults to 'omniASR_LLM_1B_local'."
        ),
    )
    parser.add_argument(
        "--audio-file",
        action="append",
        dest="audio_files",
        type=Path,
        help="Path to an audio file to transcribe. Can be specified multiple times.",
    )
    parser.add_argument(
        "--lang-code",
        action="append",
        dest="lang_codes",
        help=(
            "Language code for the corresponding audio file (e.g., 'ben_Beng', 'eng_Latn'). "
            "Must be specified once per --audio-file. Can be specified multiple times."
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=38.0,
        help="Maximum clip duration (in seconds) when sampling from the dataset.",
    )

    args = parser.parse_args()

    print(f"Loading pipeline with model card '{args.model_card}'...")
    pipeline = ASRInferencePipeline(model_card=args.model_card)

    # If audio files are provided, use them instead of dataset
    if args.audio_files:
        if not args.lang_codes:
            parser.error("--lang-code must be specified when using --audio-file")
        if len(args.audio_files) != len(args.lang_codes):
            parser.error(
                f"Number of --audio-file ({len(args.audio_files)}) must match "
                f"number of --lang-code ({len(args.lang_codes)})"
            )
        # Verify files exist
        for audio_file in args.audio_files:
            if not audio_file.exists():
                parser.error(f"Audio file not found: {audio_file}")
        run_inference_from_files(pipeline, args.audio_files, args.lang_codes)
    else:
        # Default: use dataset samples
        run_inference(pipeline, LANG_EXAMPLES)


if __name__ == "__main__":
    main()


```

Updated the script to automatically convert unsupported audio formats (like `.m4a`) to WAV using `ffmpeg`. It will:

1. Detect if the file format is unsupported
2. Convert it to WAV format (16kHz, mono) using `ffmpeg`
3. Use the converted file for transcription
4. Clean up temporary files after processing

Run:

```bash
python scripts/run_local_inference.py --audio-file bangla.m4a --lang-code ben_Beng
```

The script will automatically convert your `.m4a` file to WAV before transcribing. You should see a message like:
```
Converting bangla.m4a (.m4a) to WAV format...
```

The conversion happens automatically, so you don't need to manually convert the file. The script supports common formats like `.wav`, `.flac`, `.ogg`, `.mp3`, and will convert any other format (like `.m4a`, `.aac`, `.m4v`, etc.) to WAV.

```shellscript
python scripts/run_local_inference.py --audio-file bangla.m4a --lang-code ben_Beng
```

```plaintext
Converting bangla.m4a (.m4a) to WAV format...
```

---

**User**

I see it transcribes bangla pretty well wtih bangla.m4a audi but can it hangle mixed language in a single audio file like mix of bangla and english in the banglish.m4a audio file ? @bash (991-1026) I have tried, it can but the poutput all bangla like english also becoming bangla, will be be as such or can it seperately write english on english characters and bangla as angla charaters?

---

**Cursor**

The model is transcribing everything in Bengali script because a single language code (`ben_Beng`) is provided. Checking if the pipeline supports code-switching or mixed-language transcription:


The pipeline applies one language code per audio file, so with `ben_Beng` it outputs in Bengali script. Checking if omitting the language code (auto-detect) helps with code-switching:


The model uses a single language code per file, so with `ben_Beng` it outputs in Bengali script. Trying without a language code (auto-detect) to see if it improves code-switching:


```python
#!/usr/bin/env python3
"""
Utility script to run Omnilingual ASR inference against locally downloaded
model/tokenizer assets.

The script expects that you have created custom fairseq2 asset cards pointing
to the local checkpoints (see README instructions added by the assistant) and
placed them under ~/.config/fairseq2/assets/cards/models. With that in place,
running this script will not trigger any remote downloads.

It performs two quick sanity checks by streaming short samples from the public
`facebook/omnilingual-asr-corpus` dataset (one Bangla, one English), runs them
through the local omniASR_LLM_7B checkpoint, and prints the reference vs.
predicted text to stdout.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from datasets import load_dataset

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


LANG_EXAMPLES: Tuple[dict, ...] = (
    {
        "label": "Bengali",
        "lang_code": "ben_Beng",
        "iso_639_3": "ben",
        "iso_15924": "Beng",
    },
    {
        "label": "English",
        "lang_code": "eng_Latn",
        "iso_639_3": "eng",
        "iso_15924": "Latn",
    },
)


def fetch_dataset_sample(
    iso_639_3: str,
    iso_15924: str | None = None,
    *,
    max_duration_s: float = 38.0,
    max_samples_to_scan: int = 50_000,
) -> Tuple[dict, str, float]:
    """
    Stream samples from the Hugging Face dataset until we find one shorter than
    ``max_duration_s`` seconds. The returned audio dictionary is already in the
    format accepted by :class:`ASRInferencePipeline`.
    """
    dataset = load_dataset("facebook/omnilingual-asr-corpus", split="train", streaming=True)

    for idx, sample in enumerate(dataset, start=1):
        if sample["iso_639_3"] != iso_639_3:
            continue
        if iso_15924 is not None and sample.get("iso_15924") != iso_15924:
            continue

        audio = sample["audio"]
        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]
        duration = len(waveform) / sample_rate

        if duration <= max_duration_s:
            return (
                {"waveform": waveform, "sample_rate": sample_rate},
                sample["raw_text"],
                duration,
            )

        if idx % 5000 == 0:
            print(
                f"  scanned {idx} samples for ISO={iso_639_3} script={iso_15924} "
                f"but none shorter than {max_duration_s}s yet..."
            )

        if idx >= max_samples_to_scan:
            break

    raise RuntimeError(
        f"Could not find an example shorter than {max_duration_s}s for ISO={iso_639_3} "
        f"script={iso_15924} within {max_samples_to_scan} samples."
    )


# Supported audio formats by libsndfile (used by fairseq2)
SUPPORTED_FORMATS = {".wav", ".flac", ".ogg", ".au", ".aiff", ".mp3"}


def convert_audio_to_wav(input_file: Path, output_file: Optional[Path] = None) -> Path:
    """
    Convert an audio file to WAV format using ffmpeg.
    
    Args:
        input_file: Path to the input audio file
        output_file: Optional path for output file. If None, creates a temp file.
    
    Returns:
        Path to the converted WAV file
    """
    if output_file is None:
        # Create a temporary WAV file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        output_file = Path(temp_path)
        # Close the file descriptor so ffmpeg can write to it
        os.close(temp_fd)
    
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(input_file),
                "-ar", "16000",  # Resample to 16kHz (model requirement)
                "-ac", "1",      # Convert to mono
                "-y",            # Overwrite output file
                str(output_file),
            ],
            check=True,
            capture_output=True,
        )
        return output_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to convert {input_file} to WAV format. "
            f"ffmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg to convert audio files, "
            "or convert your audio files to WAV format manually."
        )


def prepare_audio_files(audio_files: List[Path]) -> Tuple[List[Path], List[Path]]:
    """
    Prepare audio files for transcription by converting unsupported formats to WAV.
    
    Returns:
        Tuple of (list of file paths to use, list of temp files to clean up)
    """
    prepared_files: List[Path] = []
    temp_files: List[Path] = []
    
    for audio_file in audio_files:
        suffix = audio_file.suffix.lower()
        if suffix in SUPPORTED_FORMATS:
            # File is already in a supported format
            prepared_files.append(audio_file)
        else:
            # Need to convert
            print(f"  Converting {audio_file.name} ({suffix}) to WAV format...")
            converted = convert_audio_to_wav(audio_file)
            prepared_files.append(converted)
            temp_files.append(converted)
    
    return prepared_files, temp_files


def run_inference_from_files(
    pipeline: ASRInferencePipeline,
    audio_files: List[Path],
    lang_codes: List[str],
) -> None:
    """
    Runs inference on local audio files and prints the output.
    """
    print(f"Transcribing {len(audio_files)} audio file(s)...")
    for audio_file, lang_code in zip(audio_files, lang_codes):
        print(f"  - {audio_file.name} ({lang_code})")
    
    # Convert unsupported formats to WAV
    prepared_files, temp_files = prepare_audio_files(audio_files)
    
    try:
        print("\nRunning transcription...")
        outputs = pipeline.transcribe(
            [str(f) for f in prepared_files],
            lang=lang_codes,
            batch_size=len(prepared_files),
        )

        for audio_file, lang_code, prediction in zip(audio_files, lang_codes, outputs):
            print(f"\n==> {audio_file.name} ({lang_code})")
            print("Prediction:", prediction)
    finally:
        # Clean up temporary converted files
        for temp_file in temp_files:
            try:
                temp_file.unlink()
            except Exception:
                pass  # Ignore cleanup errors


def run_inference(
    pipeline: ASRInferencePipeline,
    lang_examples: Iterable[dict],
) -> None:
    """
    Fetches one short sample per language, runs inference, and prints the output.
    """
    batch: List[dict] = []
    lang_codes: List[str] = []
    references: List[str] = []
    summaries: List[Tuple[str, str, float, str]] = []

    for example in lang_examples:
        print(
            f"Searching for {example['label']} sample "
            f"(ISO={example['iso_639_3']}, script={example.get('iso_15924')})..."
        )

        audio_dict, reference_text, duration = fetch_dataset_sample(
            example["iso_639_3"], example.get("iso_15924")
        )
        batch.append(audio_dict)
        lang_codes.append(example["lang_code"])
        references.append(reference_text)
        summaries.append((example["label"], example["lang_code"], duration, reference_text))

    print("Loaded samples:")
    for lang_label, lang_code, duration, reference in summaries:
        snippet = reference if len(reference) <= 80 else reference[:77] + "..."
        print(f"- {lang_label} ({lang_code}): {duration:.2f}s | Reference: {snippet}")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(batch, lang=lang_codes, batch_size=len(batch))

    for (lang_label, lang_code, _, reference), prediction in zip(
        summaries, outputs, strict=True
    ):
        print("\n==>", lang_label, f"({lang_code})")
        print("Reference :", reference)
        print("Prediction:", prediction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run omnilingual ASR inference using locally cached assets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use local audio files
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng --audio-file english.wav --lang-code eng_Latn

  # Use dataset samples (default)
  %(prog)s --model-card omniASR_LLM_1B_local
        """,
    )
    parser.add_argument(
        "--model-card",
        default="omniASR_LLM_1B_local",
        help=(
            "Name of the fairseq2 model card that points to the local checkpoint. "
            "Defaults to 'omniASR_LLM_1B_local'."
        ),
    )
    parser.add_argument(
        "--audio-file",
        action="append",
        dest="audio_files",
        type=Path,
        help="Path to an audio file to transcribe. Can be specified multiple times.",
    )
    parser.add_argument(
        "--lang-code",
        action="append",
        dest="lang_codes",
        help=(
            "Language code for the corresponding audio file (e.g., 'ben_Beng', 'eng_Latn'). "
            "Must be specified once per --audio-file. Can be specified multiple times. "
            "Use 'auto' or omit to let the model auto-detect (may help with code-switching)."
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=38.0,
        help="Maximum clip duration (in seconds) when sampling from the dataset.",
    )

    args = parser.parse_args()

    print(f"Loading pipeline with model card '{args.model_card}'...")
    pipeline = ASRInferencePipeline(model_card=args.model_card)

    # If audio files are provided, use them instead of dataset
    if args.audio_files:
        if not args.lang_codes:
            parser.error("--lang-code must be specified when using --audio-file")
        if len(args.audio_files) != len(args.lang_codes):
            parser.error(
                f"Number of --audio-file ({len(args.audio_files)}) must match "
                f"number of --lang-code ({len(args.lang_codes)})"
            )
        # Verify files exist
        for audio_file in args.audio_files:
            if not audio_file.exists():
                parser.error(f"Audio file not found: {audio_file}")
        run_inference_from_files(pipeline, args.audio_files, args.lang_codes)
    else:
        # Default: use dataset samples
        run_inference(pipeline, LANG_EXAMPLES)


if __name__ == "__main__":
    main()


```

```python
#!/usr/bin/env python3
"""
Utility script to run Omnilingual ASR inference against locally downloaded
model/tokenizer assets.

The script expects that you have created custom fairseq2 asset cards pointing
to the local checkpoints (see README instructions added by the assistant) and
placed them under ~/.config/fairseq2/assets/cards/models. With that in place,
running this script will not trigger any remote downloads.

It performs two quick sanity checks by streaming short samples from the public
`facebook/omnilingual-asr-corpus` dataset (one Bangla, one English), runs them
through the local omniASR_LLM_7B checkpoint, and prints the reference vs.
predicted text to stdout.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from datasets import load_dataset

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


LANG_EXAMPLES: Tuple[dict, ...] = (
    {
        "label": "Bengali",
        "lang_code": "ben_Beng",
        "iso_639_3": "ben",
        "iso_15924": "Beng",
    },
    {
        "label": "English",
        "lang_code": "eng_Latn",
        "iso_639_3": "eng",
        "iso_15924": "Latn",
    },
)


def fetch_dataset_sample(
    iso_639_3: str,
    iso_15924: str | None = None,
    *,
    max_duration_s: float = 38.0,
    max_samples_to_scan: int = 50_000,
) -> Tuple[dict, str, float]:
    """
    Stream samples from the Hugging Face dataset until we find one shorter than
    ``max_duration_s`` seconds. The returned audio dictionary is already in the
    format accepted by :class:`ASRInferencePipeline`.
    """
    dataset = load_dataset("facebook/omnilingual-asr-corpus", split="train", streaming=True)

    for idx, sample in enumerate(dataset, start=1):
        if sample["iso_639_3"] != iso_639_3:
            continue
        if iso_15924 is not None and sample.get("iso_15924") != iso_15924:
            continue

        audio = sample["audio"]
        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]
        duration = len(waveform) / sample_rate

        if duration <= max_duration_s:
            return (
                {"waveform": waveform, "sample_rate": sample_rate},
                sample["raw_text"],
                duration,
            )

        if idx % 5000 == 0:
            print(
                f"  scanned {idx} samples for ISO={iso_639_3} script={iso_15924} "
                f"but none shorter than {max_duration_s}s yet..."
            )

        if idx >= max_samples_to_scan:
            break

    raise RuntimeError(
        f"Could not find an example shorter than {max_duration_s}s for ISO={iso_639_3} "
        f"script={iso_15924} within {max_samples_to_scan} samples."
    )


# Supported audio formats by libsndfile (used by fairseq2)
SUPPORTED_FORMATS = {".wav", ".flac", ".ogg", ".au", ".aiff", ".mp3"}


def convert_audio_to_wav(input_file: Path, output_file: Optional[Path] = None) -> Path:
    """
    Convert an audio file to WAV format using ffmpeg.
    
    Args:
        input_file: Path to the input audio file
        output_file: Optional path for output file. If None, creates a temp file.
    
    Returns:
        Path to the converted WAV file
    """
    if output_file is None:
        # Create a temporary WAV file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        output_file = Path(temp_path)
        # Close the file descriptor so ffmpeg can write to it
        os.close(temp_fd)
    
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(input_file),
                "-ar", "16000",  # Resample to 16kHz (model requirement)
                "-ac", "1",      # Convert to mono
                "-y",            # Overwrite output file
                str(output_file),
            ],
            check=True,
            capture_output=True,
        )
        return output_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to convert {input_file} to WAV format. "
            f"ffmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg to convert audio files, "
            "or convert your audio files to WAV format manually."
        )


def prepare_audio_files(audio_files: List[Path]) -> Tuple[List[Path], List[Path]]:
    """
    Prepare audio files for transcription by converting unsupported formats to WAV.
    
    Returns:
        Tuple of (list of file paths to use, list of temp files to clean up)
    """
    prepared_files: List[Path] = []
    temp_files: List[Path] = []
    
    for audio_file in audio_files:
        suffix = audio_file.suffix.lower()
        if suffix in SUPPORTED_FORMATS:
            # File is already in a supported format
            prepared_files.append(audio_file)
        else:
            # Need to convert
            print(f"  Converting {audio_file.name} ({suffix}) to WAV format...")
            converted = convert_audio_to_wav(audio_file)
            prepared_files.append(converted)
            temp_files.append(converted)
    
    return prepared_files, temp_files


def run_inference_from_files(
    pipeline: ASRInferencePipeline,
    audio_files: List[Path],
    lang_codes: List[str],
) -> None:
    """
    Runs inference on local audio files and prints the output.
    """
    print(f"Transcribing {len(audio_files)} audio file(s)...")
    for audio_file, lang_code in zip(audio_files, lang_codes):
        print(f"  - {audio_file.name} ({lang_code})")
    
    # Convert unsupported formats to WAV
    prepared_files, temp_files = prepare_audio_files(audio_files)
    
    try:
        print("\nRunning transcription...")
        outputs = pipeline.transcribe(
            [str(f) for f in prepared_files],
            lang=lang_codes,
            batch_size=len(prepared_files),
        )

        for audio_file, lang_code, prediction in zip(audio_files, lang_codes, outputs):
            print(f"\n==> {audio_file.name} ({lang_code})")
            print("Prediction:", prediction)
    finally:
        # Clean up temporary converted files
        for temp_file in temp_files:
            try:
                temp_file.unlink()
            except Exception:
                pass  # Ignore cleanup errors


def run_inference(
    pipeline: ASRInferencePipeline,
    lang_examples: Iterable[dict],
) -> None:
    """
    Fetches one short sample per language, runs inference, and prints the output.
    """
    batch: List[dict] = []
    lang_codes: List[str] = []
    references: List[str] = []
    summaries: List[Tuple[str, str, float, str]] = []

    for example in lang_examples:
        print(
            f"Searching for {example['label']} sample "
            f"(ISO={example['iso_639_3']}, script={example.get('iso_15924')})..."
        )

        audio_dict, reference_text, duration = fetch_dataset_sample(
            example["iso_639_3"], example.get("iso_15924")
        )
        batch.append(audio_dict)
        lang_codes.append(example["lang_code"])
        references.append(reference_text)
        summaries.append((example["label"], example["lang_code"], duration, reference_text))

    print("Loaded samples:")
    for lang_label, lang_code, duration, reference in summaries:
        snippet = reference if len(reference) <= 80 else reference[:77] + "..."
        print(f"- {lang_label} ({lang_code}): {duration:.2f}s | Reference: {snippet}")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(batch, lang=lang_codes, batch_size=len(batch))

    for (lang_label, lang_code, _, reference), prediction in zip(
        summaries, outputs, strict=True
    ):
        print("\n==>", lang_label, f"({lang_code})")
        print("Reference :", reference)
        print("Prediction:", prediction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run omnilingual ASR inference using locally cached assets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use local audio files
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng --audio-file english.wav --lang-code eng_Latn

  # Use dataset samples (default)
  %(prog)s --model-card omniASR_LLM_1B_local
        """,
    )
    parser.add_argument(
        "--model-card",
        default="omniASR_LLM_1B_local",
        help=(
            "Name of the fairseq2 model card that points to the local checkpoint. "
            "Defaults to 'omniASR_LLM_1B_local'."
        ),
    )
    parser.add_argument(
        "--audio-file",
        action="append",
        dest="audio_files",
        type=Path,
        help="Path to an audio file to transcribe. Can be specified multiple times.",
    )
    parser.add_argument(
        "--lang-code",
        action="append",
        dest="lang_codes",
        help=(
            "Language code for the corresponding audio file (e.g., 'ben_Beng', 'eng_Latn'). "
            "Must be specified once per --audio-file. Can be specified multiple times. "
            "Use 'auto' or omit to let the model auto-detect (may help with code-switching)."
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=38.0,
        help="Maximum clip duration (in seconds) when sampling from the dataset.",
    )

    args = parser.parse_args()

    print(f"Loading pipeline with model card '{args.model_card}'...")
    pipeline = ASRInferencePipeline(model_card=args.model_card)

    # If audio files are provided, use them instead of dataset
    if args.audio_files:
        # Handle lang codes: convert 'auto' to None, or use None if not provided
        if args.lang_codes:
            lang_codes = [None if lc.lower() == "auto" else lc for lc in args.lang_codes]
            if len(args.audio_files) != len(lang_codes):
                parser.error(
                    f"Number of --audio-file ({len(args.audio_files)}) must match "
                    f"number of --lang-code ({len(args.lang_codes)})"
                )
        else:
            # No lang codes provided - use None for all (auto-detect)
            lang_codes = [None] * len(args.audio_files)
            print(
                "Warning: No --lang-code specified. Using auto-detection "
                "(may help with code-switching but quality might be lower)."
            )
        
        # Verify files exist
        for audio_file in args.audio_files:
            if not audio_file.exists():
                parser.error(f"Audio file not found: {audio_file}")
        run_inference_from_files(pipeline, args.audio_files, lang_codes)
    else:
        # Default: use dataset samples
        run_inference(pipeline, LANG_EXAMPLES)


if __name__ == "__main__":
    main()


```

```python
#!/usr/bin/env python3
"""
Utility script to run Omnilingual ASR inference against locally downloaded
model/tokenizer assets.

The script expects that you have created custom fairseq2 asset cards pointing
to the local checkpoints (see README instructions added by the assistant) and
placed them under ~/.config/fairseq2/assets/cards/models. With that in place,
running this script will not trigger any remote downloads.

It performs two quick sanity checks by streaming short samples from the public
`facebook/omnilingual-asr-corpus` dataset (one Bangla, one English), runs them
through the local omniASR_LLM_7B checkpoint, and prints the reference vs.
predicted text to stdout.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from datasets import load_dataset

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


LANG_EXAMPLES: Tuple[dict, ...] = (
    {
        "label": "Bengali",
        "lang_code": "ben_Beng",
        "iso_639_3": "ben",
        "iso_15924": "Beng",
    },
    {
        "label": "English",
        "lang_code": "eng_Latn",
        "iso_639_3": "eng",
        "iso_15924": "Latn",
    },
)


def fetch_dataset_sample(
    iso_639_3: str,
    iso_15924: str | None = None,
    *,
    max_duration_s: float = 38.0,
    max_samples_to_scan: int = 50_000,
) -> Tuple[dict, str, float]:
    """
    Stream samples from the Hugging Face dataset until we find one shorter than
    ``max_duration_s`` seconds. The returned audio dictionary is already in the
    format accepted by :class:`ASRInferencePipeline`.
    """
    dataset = load_dataset("facebook/omnilingual-asr-corpus", split="train", streaming=True)

    for idx, sample in enumerate(dataset, start=1):
        if sample["iso_639_3"] != iso_639_3:
            continue
        if iso_15924 is not None and sample.get("iso_15924") != iso_15924:
            continue

        audio = sample["audio"]
        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]
        duration = len(waveform) / sample_rate

        if duration <= max_duration_s:
            return (
                {"waveform": waveform, "sample_rate": sample_rate},
                sample["raw_text"],
                duration,
            )

        if idx % 5000 == 0:
            print(
                f"  scanned {idx} samples for ISO={iso_639_3} script={iso_15924} "
                f"but none shorter than {max_duration_s}s yet..."
            )

        if idx >= max_samples_to_scan:
            break

    raise RuntimeError(
        f"Could not find an example shorter than {max_duration_s}s for ISO={iso_639_3} "
        f"script={iso_15924} within {max_samples_to_scan} samples."
    )


# Supported audio formats by libsndfile (used by fairseq2)
SUPPORTED_FORMATS = {".wav", ".flac", ".ogg", ".au", ".aiff", ".mp3"}


def convert_audio_to_wav(input_file: Path, output_file: Optional[Path] = None) -> Path:
    """
    Convert an audio file to WAV format using ffmpeg.
    
    Args:
        input_file: Path to the input audio file
        output_file: Optional path for output file. If None, creates a temp file.
    
    Returns:
        Path to the converted WAV file
    """
    if output_file is None:
        # Create a temporary WAV file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        output_file = Path(temp_path)
        # Close the file descriptor so ffmpeg can write to it
        os.close(temp_fd)
    
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(input_file),
                "-ar", "16000",  # Resample to 16kHz (model requirement)
                "-ac", "1",      # Convert to mono
                "-y",            # Overwrite output file
                str(output_file),
            ],
            check=True,
            capture_output=True,
        )
        return output_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to convert {input_file} to WAV format. "
            f"ffmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg to convert audio files, "
            "or convert your audio files to WAV format manually."
        )


def prepare_audio_files(audio_files: List[Path]) -> Tuple[List[Path], List[Path]]:
    """
    Prepare audio files for transcription by converting unsupported formats to WAV.
    
    Returns:
        Tuple of (list of file paths to use, list of temp files to clean up)
    """
    prepared_files: List[Path] = []
    temp_files: List[Path] = []
    
    for audio_file in audio_files:
        suffix = audio_file.suffix.lower()
        if suffix in SUPPORTED_FORMATS:
            # File is already in a supported format
            prepared_files.append(audio_file)
        else:
            # Need to convert
            print(f"  Converting {audio_file.name} ({suffix}) to WAV format...")
            converted = convert_audio_to_wav(audio_file)
            prepared_files.append(converted)
            temp_files.append(converted)
    
    return prepared_files, temp_files


def run_inference_from_files(
    pipeline: ASRInferencePipeline,
    audio_files: List[Path],
    lang_codes: List[str | None],
) -> None:
    """
    Runs inference on local audio files and prints the output.
    """
    print(f"Transcribing {len(audio_files)} audio file(s)...")
    for audio_file, lang_code in zip(audio_files, lang_codes):
        lang_display = lang_code if lang_code else "auto-detect"
        print(f"  - {audio_file.name} ({lang_display})")
    
    # Convert unsupported formats to WAV
    prepared_files, temp_files = prepare_audio_files(audio_files)
    
    try:
        print("\nRunning transcription...")
        outputs = pipeline.transcribe(
            [str(f) for f in prepared_files],
            lang=lang_codes,
            batch_size=len(prepared_files),
        )

        for audio_file, lang_code, prediction in zip(audio_files, lang_codes, outputs):
            print(f"\n==> {audio_file.name} ({lang_code})")
            print("Prediction:", prediction)
    finally:
        # Clean up temporary converted files
        for temp_file in temp_files:
            try:
                temp_file.unlink()
            except Exception:
                pass  # Ignore cleanup errors


def run_inference(
    pipeline: ASRInferencePipeline,
    lang_examples: Iterable[dict],
) -> None:
    """
    Fetches one short sample per language, runs inference, and prints the output.
    """
    batch: List[dict] = []
    lang_codes: List[str] = []
    references: List[str] = []
    summaries: List[Tuple[str, str, float, str]] = []

    for example in lang_examples:
        print(
            f"Searching for {example['label']} sample "
            f"(ISO={example['iso_639_3']}, script={example.get('iso_15924')})..."
        )

        audio_dict, reference_text, duration = fetch_dataset_sample(
            example["iso_639_3"], example.get("iso_15924")
        )
        batch.append(audio_dict)
        lang_codes.append(example["lang_code"])
        references.append(reference_text)
        summaries.append((example["label"], example["lang_code"], duration, reference_text))

    print("Loaded samples:")
    for lang_label, lang_code, duration, reference in summaries:
        snippet = reference if len(reference) <= 80 else reference[:77] + "..."
        print(f"- {lang_label} ({lang_code}): {duration:.2f}s | Reference: {snippet}")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(batch, lang=lang_codes, batch_size=len(batch))

    for (lang_label, lang_code, _, reference), prediction in zip(
        summaries, outputs, strict=True
    ):
        print("\n==>", lang_label, f"({lang_code})")
        print("Reference :", reference)
        print("Prediction:", prediction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run omnilingual ASR inference using locally cached assets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use local audio files
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng --audio-file english.wav --lang-code eng_Latn

  # Use dataset samples (default)
  %(prog)s --model-card omniASR_LLM_1B_local
        """,
    )
    parser.add_argument(
        "--model-card",
        default="omniASR_LLM_1B_local",
        help=(
            "Name of the fairseq2 model card that points to the local checkpoint. "
            "Defaults to 'omniASR_LLM_1B_local'."
        ),
    )
    parser.add_argument(
        "--audio-file",
        action="append",
        dest="audio_files",
        type=Path,
        help="Path to an audio file to transcribe. Can be specified multiple times.",
    )
    parser.add_argument(
        "--lang-code",
        action="append",
        dest="lang_codes",
        help=(
            "Language code for the corresponding audio file (e.g., 'ben_Beng', 'eng_Latn'). "
            "Must be specified once per --audio-file. Can be specified multiple times. "
            "Use 'auto' or omit to let the model auto-detect (may help with code-switching)."
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=38.0,
        help="Maximum clip duration (in seconds) when sampling from the dataset.",
    )

    args = parser.parse_args()

    print(f"Loading pipeline with model card '{args.model_card}'...")
    pipeline = ASRInferencePipeline(model_card=args.model_card)

    # If audio files are provided, use them instead of dataset
    if args.audio_files:
        # Handle lang codes: convert 'auto' to None, or use None if not provided
        if args.lang_codes:
            lang_codes = [None if lc.lower() == "auto" else lc for lc in args.lang_codes]
            if len(args.audio_files) != len(lang_codes):
                parser.error(
                    f"Number of --audio-file ({len(args.audio_files)}) must match "
                    f"number of --lang-code ({len(args.lang_codes)})"
                )
        else:
            # No lang codes provided - use None for all (auto-detect)
            lang_codes = [None] * len(args.audio_files)
            print(
                "Warning: No --lang-code specified. Using auto-detection "
                "(may help with code-switching but quality might be lower)."
            )
        
        # Verify files exist
        for audio_file in args.audio_files:
            if not audio_file.exists():
                parser.error(f"Audio file not found: {audio_file}")
        run_inference_from_files(pipeline, args.audio_files, lang_codes)
    else:
        # Default: use dataset samples
        run_inference(pipeline, LANG_EXAMPLES)


if __name__ == "__main__":
    main()


```

```python
#!/usr/bin/env python3
"""
Utility script to run Omnilingual ASR inference against locally downloaded
model/tokenizer assets.

The script expects that you have created custom fairseq2 asset cards pointing
to the local checkpoints (see README instructions added by the assistant) and
placed them under ~/.config/fairseq2/assets/cards/models. With that in place,
running this script will not trigger any remote downloads.

It performs two quick sanity checks by streaming short samples from the public
`facebook/omnilingual-asr-corpus` dataset (one Bangla, one English), runs them
through the local omniASR_LLM_7B checkpoint, and prints the reference vs.
predicted text to stdout.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from datasets import load_dataset

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


LANG_EXAMPLES: Tuple[dict, ...] = (
    {
        "label": "Bengali",
        "lang_code": "ben_Beng",
        "iso_639_3": "ben",
        "iso_15924": "Beng",
    },
    {
        "label": "English",
        "lang_code": "eng_Latn",
        "iso_639_3": "eng",
        "iso_15924": "Latn",
    },
)


def fetch_dataset_sample(
    iso_639_3: str,
    iso_15924: str | None = None,
    *,
    max_duration_s: float = 38.0,
    max_samples_to_scan: int = 50_000,
) -> Tuple[dict, str, float]:
    """
    Stream samples from the Hugging Face dataset until we find one shorter than
    ``max_duration_s`` seconds. The returned audio dictionary is already in the
    format accepted by :class:`ASRInferencePipeline`.
    """
    dataset = load_dataset("facebook/omnilingual-asr-corpus", split="train", streaming=True)

    for idx, sample in enumerate(dataset, start=1):
        if sample["iso_639_3"] != iso_639_3:
            continue
        if iso_15924 is not None and sample.get("iso_15924") != iso_15924:
            continue

        audio = sample["audio"]
        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]
        duration = len(waveform) / sample_rate

        if duration <= max_duration_s:
            return (
                {"waveform": waveform, "sample_rate": sample_rate},
                sample["raw_text"],
                duration,
            )

        if idx % 5000 == 0:
            print(
                f"  scanned {idx} samples for ISO={iso_639_3} script={iso_15924} "
                f"but none shorter than {max_duration_s}s yet..."
            )

        if idx >= max_samples_to_scan:
            break

    raise RuntimeError(
        f"Could not find an example shorter than {max_duration_s}s for ISO={iso_639_3} "
        f"script={iso_15924} within {max_samples_to_scan} samples."
    )


# Supported audio formats by libsndfile (used by fairseq2)
SUPPORTED_FORMATS = {".wav", ".flac", ".ogg", ".au", ".aiff", ".mp3"}


def convert_audio_to_wav(input_file: Path, output_file: Optional[Path] = None) -> Path:
    """
    Convert an audio file to WAV format using ffmpeg.
    
    Args:
        input_file: Path to the input audio file
        output_file: Optional path for output file. If None, creates a temp file.
    
    Returns:
        Path to the converted WAV file
    """
    if output_file is None:
        # Create a temporary WAV file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        output_file = Path(temp_path)
        # Close the file descriptor so ffmpeg can write to it
        os.close(temp_fd)
    
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(input_file),
                "-ar", "16000",  # Resample to 16kHz (model requirement)
                "-ac", "1",      # Convert to mono
                "-y",            # Overwrite output file
                str(output_file),
            ],
            check=True,
            capture_output=True,
        )
        return output_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to convert {input_file} to WAV format. "
            f"ffmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg to convert audio files, "
            "or convert your audio files to WAV format manually."
        )


def prepare_audio_files(audio_files: List[Path]) -> Tuple[List[Path], List[Path]]:
    """
    Prepare audio files for transcription by converting unsupported formats to WAV.
    
    Returns:
        Tuple of (list of file paths to use, list of temp files to clean up)
    """
    prepared_files: List[Path] = []
    temp_files: List[Path] = []
    
    for audio_file in audio_files:
        suffix = audio_file.suffix.lower()
        if suffix in SUPPORTED_FORMATS:
            # File is already in a supported format
            prepared_files.append(audio_file)
        else:
            # Need to convert
            print(f"  Converting {audio_file.name} ({suffix}) to WAV format...")
            converted = convert_audio_to_wav(audio_file)
            prepared_files.append(converted)
            temp_files.append(converted)
    
    return prepared_files, temp_files


def run_inference_from_files(
    pipeline: ASRInferencePipeline,
    audio_files: List[Path],
    lang_codes: List[str | None],
) -> None:
    """
    Runs inference on local audio files and prints the output.
    """
    print(f"Transcribing {len(audio_files)} audio file(s)...")
    for audio_file, lang_code in zip(audio_files, lang_codes):
        lang_display = lang_code if lang_code else "auto-detect"
        print(f"  - {audio_file.name} ({lang_display})")
    
    # Convert unsupported formats to WAV
    prepared_files, temp_files = prepare_audio_files(audio_files)
    
    try:
        print("\nRunning transcription...")
        outputs = pipeline.transcribe(
            [str(f) for f in prepared_files],
            lang=lang_codes,
            batch_size=len(prepared_files),
        )

        for audio_file, lang_code, prediction in zip(audio_files, lang_codes, outputs):
            lang_display = lang_code if lang_code else "auto-detect"
            print(f"\n==> {audio_file.name} ({lang_display})")
            print("Prediction:", prediction)
    finally:
        # Clean up temporary converted files
        for temp_file in temp_files:
            try:
                temp_file.unlink()
            except Exception:
                pass  # Ignore cleanup errors


def run_inference(
    pipeline: ASRInferencePipeline,
    lang_examples: Iterable[dict],
) -> None:
    """
    Fetches one short sample per language, runs inference, and prints the output.
    """
    batch: List[dict] = []
    lang_codes: List[str] = []
    references: List[str] = []
    summaries: List[Tuple[str, str, float, str]] = []

    for example in lang_examples:
        print(
            f"Searching for {example['label']} sample "
            f"(ISO={example['iso_639_3']}, script={example.get('iso_15924')})..."
        )

        audio_dict, reference_text, duration = fetch_dataset_sample(
            example["iso_639_3"], example.get("iso_15924")
        )
        batch.append(audio_dict)
        lang_codes.append(example["lang_code"])
        references.append(reference_text)
        summaries.append((example["label"], example["lang_code"], duration, reference_text))

    print("Loaded samples:")
    for lang_label, lang_code, duration, reference in summaries:
        snippet = reference if len(reference) <= 80 else reference[:77] + "..."
        print(f"- {lang_label} ({lang_code}): {duration:.2f}s | Reference: {snippet}")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(batch, lang=lang_codes, batch_size=len(batch))

    for (lang_label, lang_code, _, reference), prediction in zip(
        summaries, outputs, strict=True
    ):
        print("\n==>", lang_label, f"({lang_code})")
        print("Reference :", reference)
        print("Prediction:", prediction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run omnilingual ASR inference using locally cached assets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use local audio files
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng --audio-file english.wav --lang-code eng_Latn

  # Use dataset samples (default)
  %(prog)s --model-card omniASR_LLM_1B_local
        """,
    )
    parser.add_argument(
        "--model-card",
        default="omniASR_LLM_1B_local",
        help=(
            "Name of the fairseq2 model card that points to the local checkpoint. "
            "Defaults to 'omniASR_LLM_1B_local'."
        ),
    )
    parser.add_argument(
        "--audio-file",
        action="append",
        dest="audio_files",
        type=Path,
        help="Path to an audio file to transcribe. Can be specified multiple times.",
    )
    parser.add_argument(
        "--lang-code",
        action="append",
        dest="lang_codes",
        help=(
            "Language code for the corresponding audio file (e.g., 'ben_Beng', 'eng_Latn'). "
            "Must be specified once per --audio-file. Can be specified multiple times. "
            "Use 'auto' or omit to let the model auto-detect (may help with code-switching)."
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=38.0,
        help="Maximum clip duration (in seconds) when sampling from the dataset.",
    )

    args = parser.parse_args()

    print(f"Loading pipeline with model card '{args.model_card}'...")
    pipeline = ASRInferencePipeline(model_card=args.model_card)

    # If audio files are provided, use them instead of dataset
    if args.audio_files:
        # Handle lang codes: convert 'auto' to None, or use None if not provided
        if args.lang_codes:
            lang_codes = [None if lc.lower() == "auto" else lc for lc in args.lang_codes]
            if len(args.audio_files) != len(lang_codes):
                parser.error(
                    f"Number of --audio-file ({len(args.audio_files)}) must match "
                    f"number of --lang-code ({len(args.lang_codes)})"
                )
        else:
            # No lang codes provided - use None for all (auto-detect)
            lang_codes = [None] * len(args.audio_files)
            print(
                "Warning: No --lang-code specified. Using auto-detection "
                "(may help with code-switching but quality might be lower)."
            )
        
        # Verify files exist
        for audio_file in args.audio_files:
            if not audio_file.exists():
                parser.error(f"Audio file not found: {audio_file}")
        run_inference_from_files(pipeline, args.audio_files, lang_codes)
    else:
        # Default: use dataset samples
        run_inference(pipeline, LANG_EXAMPLES)


if __name__ == "__main__":
    main()


```

```python
#!/usr/bin/env python3
"""
Utility script to run Omnilingual ASR inference against locally downloaded
model/tokenizer assets.

The script expects that you have created custom fairseq2 asset cards pointing
to the local checkpoints (see README instructions added by the assistant) and
placed them under ~/.config/fairseq2/assets/cards/models. With that in place,
running this script will not trigger any remote downloads.

It performs two quick sanity checks by streaming short samples from the public
`facebook/omnilingual-asr-corpus` dataset (one Bangla, one English), runs them
through the local omniASR_LLM_7B checkpoint, and prints the reference vs.
predicted text to stdout.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from datasets import load_dataset

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


LANG_EXAMPLES: Tuple[dict, ...] = (
    {
        "label": "Bengali",
        "lang_code": "ben_Beng",
        "iso_639_3": "ben",
        "iso_15924": "Beng",
    },
    {
        "label": "English",
        "lang_code": "eng_Latn",
        "iso_639_3": "eng",
        "iso_15924": "Latn",
    },
)


def fetch_dataset_sample(
    iso_639_3: str,
    iso_15924: str | None = None,
    *,
    max_duration_s: float = 38.0,
    max_samples_to_scan: int = 50_000,
) -> Tuple[dict, str, float]:
    """
    Stream samples from the Hugging Face dataset until we find one shorter than
    ``max_duration_s`` seconds. The returned audio dictionary is already in the
    format accepted by :class:`ASRInferencePipeline`.
    """
    dataset = load_dataset("facebook/omnilingual-asr-corpus", split="train", streaming=True)

    for idx, sample in enumerate(dataset, start=1):
        if sample["iso_639_3"] != iso_639_3:
            continue
        if iso_15924 is not None and sample.get("iso_15924") != iso_15924:
            continue

        audio = sample["audio"]
        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]
        duration = len(waveform) / sample_rate

        if duration <= max_duration_s:
            return (
                {"waveform": waveform, "sample_rate": sample_rate},
                sample["raw_text"],
                duration,
            )

        if idx % 5000 == 0:
            print(
                f"  scanned {idx} samples for ISO={iso_639_3} script={iso_15924} "
                f"but none shorter than {max_duration_s}s yet..."
            )

        if idx >= max_samples_to_scan:
            break

    raise RuntimeError(
        f"Could not find an example shorter than {max_duration_s}s for ISO={iso_639_3} "
        f"script={iso_15924} within {max_samples_to_scan} samples."
    )


# Supported audio formats by libsndfile (used by fairseq2)
SUPPORTED_FORMATS = {".wav", ".flac", ".ogg", ".au", ".aiff", ".mp3"}


def convert_audio_to_wav(input_file: Path, output_file: Optional[Path] = None) -> Path:
    """
    Convert an audio file to WAV format using ffmpeg.
    
    Args:
        input_file: Path to the input audio file
        output_file: Optional path for output file. If None, creates a temp file.
    
    Returns:
        Path to the converted WAV file
    """
    if output_file is None:
        # Create a temporary WAV file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        output_file = Path(temp_path)
        # Close the file descriptor so ffmpeg can write to it
        os.close(temp_fd)
    
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(input_file),
                "-ar", "16000",  # Resample to 16kHz (model requirement)
                "-ac", "1",      # Convert to mono
                "-y",            # Overwrite output file
                str(output_file),
            ],
            check=True,
            capture_output=True,
        )
        return output_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to convert {input_file} to WAV format. "
            f"ffmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg to convert audio files, "
            "or convert your audio files to WAV format manually."
        )


def prepare_audio_files(audio_files: List[Path]) -> Tuple[List[Path], List[Path]]:
    """
    Prepare audio files for transcription by converting unsupported formats to WAV.
    
    Returns:
        Tuple of (list of file paths to use, list of temp files to clean up)
    """
    prepared_files: List[Path] = []
    temp_files: List[Path] = []
    
    for audio_file in audio_files:
        suffix = audio_file.suffix.lower()
        if suffix in SUPPORTED_FORMATS:
            # File is already in a supported format
            prepared_files.append(audio_file)
        else:
            # Need to convert
            print(f"  Converting {audio_file.name} ({suffix}) to WAV format...")
            converted = convert_audio_to_wav(audio_file)
            prepared_files.append(converted)
            temp_files.append(converted)
    
    return prepared_files, temp_files


def run_inference_from_files(
    pipeline: ASRInferencePipeline,
    audio_files: List[Path],
    lang_codes: List[Optional[str]],
) -> None:
    """
    Runs inference on local audio files and prints the output.
    """
    print(f"Transcribing {len(audio_files)} audio file(s)...")
    for audio_file, lang_code in zip(audio_files, lang_codes):
        lang_display = lang_code if lang_code else "auto-detect"
        print(f"  - {audio_file.name} ({lang_display})")
    
    # Convert unsupported formats to WAV
    prepared_files, temp_files = prepare_audio_files(audio_files)
    
    try:
        print("\nRunning transcription...")
        outputs = pipeline.transcribe(
            [str(f) for f in prepared_files],
            lang=lang_codes,
            batch_size=len(prepared_files),
        )

        for audio_file, lang_code, prediction in zip(audio_files, lang_codes, outputs):
            lang_display = lang_code if lang_code else "auto-detect"
            print(f"\n==> {audio_file.name} ({lang_display})")
            print("Prediction:", prediction)
    finally:
        # Clean up temporary converted files
        for temp_file in temp_files:
            try:
                temp_file.unlink()
            except Exception:
                pass  # Ignore cleanup errors


def run_inference(
    pipeline: ASRInferencePipeline,
    lang_examples: Iterable[dict],
) -> None:
    """
    Fetches one short sample per language, runs inference, and prints the output.
    """
    batch: List[dict] = []
    lang_codes: List[str] = []
    references: List[str] = []
    summaries: List[Tuple[str, str, float, str]] = []

    for example in lang_examples:
        print(
            f"Searching for {example['label']} sample "
            f"(ISO={example['iso_639_3']}, script={example.get('iso_15924')})..."
        )

        audio_dict, reference_text, duration = fetch_dataset_sample(
            example["iso_639_3"], example.get("iso_15924")
        )
        batch.append(audio_dict)
        lang_codes.append(example["lang_code"])
        references.append(reference_text)
        summaries.append((example["label"], example["lang_code"], duration, reference_text))

    print("Loaded samples:")
    for lang_label, lang_code, duration, reference in summaries:
        snippet = reference if len(reference) <= 80 else reference[:77] + "..."
        print(f"- {lang_label} ({lang_code}): {duration:.2f}s | Reference: {snippet}")

    print("\nRunning transcription...")
    outputs = pipeline.transcribe(batch, lang=lang_codes, batch_size=len(batch))

    for (lang_label, lang_code, _, reference), prediction in zip(
        summaries, outputs, strict=True
    ):
        print("\n==>", lang_label, f"({lang_code})")
        print("Reference :", reference)
        print("Prediction:", prediction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run omnilingual ASR inference using locally cached assets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use local audio files
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng
  %(prog)s --audio-file bangla.m4a --lang-code ben_Beng --audio-file english.wav --lang-code eng_Latn

  # Use dataset samples (default)
  %(prog)s --model-card omniASR_LLM_1B_local
        """,
    )
    parser.add_argument(
        "--model-card",
        default="omniASR_LLM_1B_local",
        help=(
            "Name of the fairseq2 model card that points to the local checkpoint. "
            "Defaults to 'omniASR_LLM_1B_local'."
        ),
    )
    parser.add_argument(
        "--audio-file",
        action="append",
        dest="audio_files",
        type=Path,
        help="Path to an audio file to transcribe. Can be specified multiple times.",
    )
    parser.add_argument(
        "--lang-code",
        action="append",
        dest="lang_codes",
        help=(
            "Language code for the corresponding audio file (e.g., 'ben_Beng', 'eng_Latn'). "
            "Must be specified once per --audio-file. Can be specified multiple times. "
            "Use 'auto' or omit to let the model auto-detect (may help with code-switching)."
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=38.0,
        help="Maximum clip duration (in seconds) when sampling from the dataset.",
    )

    args = parser.parse_args()

    print(f"Loading pipeline with model card '{args.model_card}'...")
    pipeline = ASRInferencePipeline(model_card=args.model_card)

    # If audio files are provided, use them instead of dataset
    if args.audio_files:
        # Handle lang codes: convert 'auto' to None, or use None if not provided
        if args.lang_codes:
            lang_codes = [None if lc.lower() == "auto" else lc for lc in args.lang_codes]
            if len(args.audio_files) != len(lang_codes):
                parser.error(
                    f"Number of --audio-file ({len(args.audio_files)}) must match "
                    f"number of --lang-code ({len(args.lang_codes)})"
                )
        else:
            # No lang codes provided - use None for all (auto-detect)
            lang_codes = [None] * len(args.audio_files)
            print(
                "Warning: No --lang-code specified. Using auto-detection "
                "(may help with code-switching but quality might be lower)."
            )
        
        # Verify files exist
        for audio_file in args.audio_files:
            if not audio_file.exists():
                parser.error(f"Audio file not found: {audio_file}")
        run_inference_from_files(pipeline, args.audio_files, lang_codes)
    else:
        # Default: use dataset samples
        run_inference(pipeline, LANG_EXAMPLES)


if __name__ == "__main__":
    main()


```

Updated the script to support auto-detection. Try transcribing your mixed-language file without specifying a language code:

```bash
python scripts/run_local_inference.py --audio-file banglish.m4a
```

Or explicitly use `auto`:

```bash
python scripts/run_local_inference.py --audio-file banglish.m4a --lang-code auto
```

**Important notes about code-switching:**

1. **Current limitation**: The model is designed to output in a single script per audio file. When you specify `ben_Beng`, it conditions the output to Bengali script, so English words get transliterated.

2. **Auto-detection**: Using `None` (auto) lets the model choose the script, which may help with code-switching, but:
   - Quality may be lower (as noted in the docs)
   - It may still not perfectly separate scripts for mixed content
   - The model may default to one script based on the dominant language

3. **Best approach for code-switching**: The model doesn't have built-in code-switching detection. For better results, you could:
   - Segment the audio into language-specific chunks and transcribe each with the appropriate language code
   - Use a separate code-switching detection model to identify language boundaries first

Try the auto-detection option and see if it improves the output. The model may handle some code-switching, but perfect script separation isn't guaranteed.

```shellscript
python scripts/run_local_inference.py --audio-file banglish.m4a
```

```shellscript
python scripts/run_local_inference.py --audio-file banglish.m4a --lang-code auto
```

---

**User**

I see Leverage Zero-Shot Learning: Bring your own language

Achieved zero-shot learning: our model suite can be taught a new language through in-context learning.

Can I build such banglish language zeros shot?

---

**Cursor**

Yes—with the zero-shot model (omniASR_LLM_7B_ZS) you can in-context “teach” the model your Banglish style so English stays in Latin and Bengali in Bengali.

What you need
- Model: omniASR_LLM_7B_ZS (zero-shot variant)
- 1–10 short context examples (≤30s each), each an audio + its desired transcript showing your mixed-style (Bangla in Bengali script, English in Latin).
- The target audio clip (≤40s).

How to run (conceptually)
- Use the zero-shot API: transcribe_with_context(...) on ASRInferencePipeline with model_card="omniASR_LLM_7B_ZS".
- Provide 1–10 ContextExample pairs per input; they get internally expanded/truncated to 10.

Python example

```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline, ContextExample

pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B_ZS")

# Your “Banglish” style examples:
context = [
    ContextExample(
        audio="ex1_banglish.wav",
        text="আমরা কাল মিটিং করব at 3 PM in the main office."
    ),
    ContextExample(
        audio="ex2_banglish.wav",
        text="Please send the report আজই, before EOD."
    ),
    # Add up to ~10 short examples
]

# The clip to transcribe in the same style:
inputs = ["banglish_test.wav"]

out = pipeline.transcribe_with_context(
    inputs,
    context_examples=[context],  # one list per input
    batch_size=1
)

print(out[0])
```

Notes
- The zero-shot model ignores the lang= parameter; it learns from your examples.
- Keep examples short (≤30s), clear, and representative of the mixed script you want. If you want English to remain Latin, make sure your example transcripts keep English in Latin.
- If you supply fewer than 10 examples, the pipeline internally repeats them to 10; more than 10 are truncated.
- For local files, you can reuse your conversion path (WAV/FLAC recommended).
- If you’re using local checkpoints, create a user card (like you did) pointing to `omniASR-LLM-7B-ZS.pt` and the standard tokenizer, then pass that name as model_card.

This is the best-supported way to get consistent mixed-script “Banglish” output.

```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline, ContextExample

pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B_ZS")

# Your “Banglish” style examples:
context = [
    ContextExample(
        audio="ex1_banglish.wav",
        text="আমরা কাল মিটিং করব at 3 PM in the main office."
    ),
    ContextExample(
        audio="ex2_banglish.wav",
        text="Please send the report আজই, before EOD."
    ),
    # Add up to ~10 short examples
]

# The clip to transcribe in the same style:
inputs = ["banglish_test.wav"]

out = pipeline.transcribe_with_context(
    inputs,
    context_examples=[context],  # one list per input
    batch_size=1
)

print(out[0])
```

---

**User**

whats the maximum length of audio file currently supported

---

**Cursor**

40 seconds per file.

Details:
- The inference pipeline enforces a 40s cap per input.
- Best accuracy is around ≤30s (training regime).

---

**User**

with the zero-shot model (omniASR_LLM_7B_ZS) you can in-context “teach” the model your Banglish style so English stays in Latin and Bengali in Bengali. Woudl the model run on my 8GB nvdia gpu> how much memory required for the tuning? I see the 1B model taking most of the VRAM 

@bash (4-12) 

---

**Cursor**


Checking the memory requirements for the zero-shot model:



The zero-shot model (omniASR_LLM_7B_ZS) requires ~20 GiB VRAM, which exceeds your 8GB GPU.

Memory requirements:
- omniASR_LLM_7B_ZS: ~20 GiB VRAM
- omniASR_LLM_7B: ~17 GiB VRAM
- omniASR_LLM_3B: ~10 GiB VRAM
- omniASR_LLM_1B: ~6 GiB VRAM (fits on 8GB)

Your GPU: 8GB (8192 MiB), currently using ~5762 MiB with the 1B model.

Options:
1) Use CPU inference (slower, but works)



You can run on CPU, but it will be slow. The pipeline supports `device="cpu"`.

2) Use the 1B model with language conditioning (best fit for your GPU)

The 1B model fits on your GPU and supports language conditioning. It won't do zero-shot in-context learning, but you can:
- Use `lang=None` (auto-detect) for mixed content
- Try segmenting audio and using different language codes per segment

3) Check if quantization is available

Checking if the model supports 8-bit or 4-bit quantization to reduce memory:



No built-in quantization found. The model uses bfloat16 by default.

## Summary & Recommendation

**The 7B zero-shot model will not fit on your 8GB GPU** — it needs ~20 GiB VRAM.

**Your options:**

1. **Use CPU for 7B zero-shot** (slow, but works):
   ```python
   pipeline = ASRInferencePipeline(
       model_card="omniASR_LLM_7B_ZS",
       device="cpu"  # Force CPU
   )
   ```
   Expect very slow inference (minutes per audio clip).

2. **Stick with 1B model** (fits on GPU, but no zero-shot):
   - Use `lang=None` (auto-detect) for mixed content
   - May still transliterate English to Bengali script
   - Fastest option that fits your GPU

3. **Upgrade GPU** (if zero-shot is required):
   - Need at least 20GB+ VRAM (e.g., RTX 3090, A4000, A5000, or A100)

4. **Post-process the output**:
   - Use the 1B model with auto-detect
   - Apply a post-processing step to detect and convert Bengali-transliterated English words back to Latin script

**Bottom line**: For Banglish with proper script separation on your current hardware, the zero-shot model isn't feasible. The 1B model with auto-detection is your best option, though it may still transliterate English words.

```python
   pipeline = ASRInferencePipeline(
       model_card="omniASR_LLM_7B_ZS",
       device="cpu"  # Force CPU
   )
```

---

**User**

How many hour it may take for rtxa5000 23gb gpu if I finetune on the cloud ?

---

**Cursor**


Checking the training recipes and documentation for finetuning time estimates:



Estimating finetuning time on an RTX A5000 (24GB VRAM):

## Model size that fits on A5000

**Training VRAM ≈ 3× inference VRAM** (model + gradients + optimizer states):

- CTC 300M: ~6-8GB → fits easily
- CTC 1B: ~9-12GB → fits comfortably
- CTC 3B: ~24-32GB → tight fit, small batch size
- LLM 300M: ~15-18GB → fits
- LLM 1B: ~18-24GB → tight fit, small batch
- LLM 3B: ~30-40GB → won't fit

## Training time estimates

Based on the configs:
- CTC finetune: 5,000 steps
- LLM finetune: 20,000 steps

### CTC models (5,000 steps)

| Model | Batch Size | Steps/sec | Estimated Time |
|-------|------------|-----------|----------------|
| CTC 300M | 8-16 | ~2-4 | 20-40 minutes |
| CTC 1B | 4-8 | ~1-2 | 40-80 minutes |
| CTC 3B | 1-2 | ~0.5-1 | 1.5-3 hours |

### LLM models (20,000 steps)

| Model | Batch Size | Steps/sec | Estimated Time |
|-------|------------|-----------|----------------|
| LLM 300M | 2-4 | ~0.5-1 | 5-11 hours |
| LLM 1B | 1-2 | ~0.3-0.6 | 9-18 hours |
| LLM 3B | N/A | N/A | Won't fit |

## Factors affecting time

1. Batch size: smaller batch = slower training
2. Audio length: longer clips = slower per step
3. Gradient accumulation: may need 4-8x to fit in memory
4. Dataset size: larger datasets = more time per epoch

## Recommendation

For Banglish finetuning on A5000:
- Best option: CTC 1B — ~1-2 hours
- Alternative: LLM 1B — ~10-15 hours (better quality, slower)
- Fastest: CTC 300M — ~30 minutes (may be less accurate)

Note: These are rough estimates. Actual time depends on your dataset size, audio lengths, and batch size. Start with CTC 1B for a good balance of speed and quality.

---

**User**

Start with CTC 1B for a good balance of speed and quality. but you said omniASR_LLM_7B_ZS) to Provide 1–10 ContextExample pairs per input; they get internally expanded/truncated to 10., say I provide 20 examples, so how shoudl I train the 1B or the 7B_ZS model in the cloud gpu? I need output as 1B model 

---

**Cursor**


Clarifying the approach: you want a 1B model that handles Banglish code-switching. Checking the training setup:



Clarifying the difference:

- Zero-shot (7B_ZS): no training; provide examples at inference time. Won't fit on A5000.
- Finetuning (1B): train on your examples to produce a 1B model that handles Banglish.

Recommendation: finetune LLM 1B on your 20 Banglish examples.

Why LLM 1B over CTC 1B:
- Better for code-switching
- Supports language conditioning
- Fits on A5000 (~18–24GB VRAM)

Training setup:



Step-by-step guide to finetune LLM 1B on your 20 Banglish examples:

## Training Plan for LLM 1B on A5000 (24GB)

### Step 1: Prepare Your Dataset

You need to convert your 20 Banglish audio files + transcripts into the parquet format. Create a script to prepare your data:

```python
# prepare_banglish_dataset.py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from omnilingual_asr.datasets.utils.audio import encode_audio_to_bytes

# Your 20 examples: list of (audio_path, transcript) tuples
banglish_examples = [
    ("banglish_1.wav", "আমরা কাল মিটিং করব at 3 PM"),
    ("banglish_2.wav", "Please send the report আজই"),
    # ... add all 20 examples
]

# Prepare data
data = []
for audio_path, text in banglish_examples:
    audio_bytes = encode_audio_to_bytes(audio_path)  # Converts to 16kHz mono FLAC
    audio_size = len(audio_bytes)  # Size in bytes
    
    data.append({
        "text": text,
        "audio_bytes": audio_bytes,
        "audio_size": audio_size,
        "corpus": "banglish",
        "split": "train",  # Use 15 for train, 5 for dev
        "language": "ben_Beng"  # Or create custom code
    })

# Create parquet file
df = pd.DataFrame(data)
table = pa.Table.from_pandas(df)
pq.write_table(table, "banglish_train.parquet", row_group_size=100)
```

### Step 2: Create Training Config

Create a config file for LLM 1B finetuning:

```yaml
# llm-1b-banglish-finetune.yaml
model:
  name: "omniASR_LLM_1B"

dataset:
  name: "banglish_dataset"
  train_split: "train"
  valid_split: "dev"
  storage_mode: "MIXTURE_PARQUET"
  task_mode: "ASR"
  mixture_parquet_storage_config:
    dataset_summary_path: "/path/to/banglish/language_distribution.tsv"
    beta_corpus: 1.0  # Only your corpus
    beta_language: 1.0
    fragment_loading:
      cache: True
  asr_task_config:
     min_audio_len: 32_000  # 2 seconds
     max_audio_len: 480_000  # 30 seconds (shorter for small dataset)
     max_num_elements: 480_000
     batch_shuffle_window: 1
     normalize_audio: true
     example_shuffle_window: 1

tokenizer:
  name: "omniASR_tokenizer"

optimizer:
  config:
    lr: 5e-05  # Lower LR for finetuning

trainer:
  freeze_encoder_for_n_steps: 0  # Don't freeze - finetune everything
  mixed_precision:
    dtype: "torch.bfloat16"
  grad_accumulation:
    num_batches: 8  # Increase if OOM, helps with small batch

regime:
  num_steps: 2_000  # Fewer steps for small dataset (20 examples)
  validate_after_n_steps: 200
  validate_every_n_steps: 200
  checkpoint_every_n_steps: 200
  publish_metrics_every_n_steps: 50
```

### Step 3: Training Time Estimate

With 20 examples on A5000:
- Batch size: 1-2 (due to memory)
- Steps: 2,000 (fewer than default since small dataset)
- Time per step: ~1-2 seconds
- Total time: ~1-2 hours

### Step 4: Run Training

```bash
export OUTPUT_DIR="/path/to/output"
python -m workflows.recipes.wav2vec2.asr $OUTPUT_DIR \
  --config-file llm-1b-banglish-finetune.yaml
```

### Important Notes:

1. Small dataset: With only 20 examples, you may need:
   - Data augmentation (speed, noise, pitch)
   - More epochs (increase `num_steps`)
   - Lower learning rate to avoid overfitting

2. Memory: If you get OOM errors:
   - Increase `grad_accumulation.num_batches` to 16 or 32
   - Reduce `max_audio_len` to 240,000 (15 seconds)
   - Use smaller batch size

3. Output: The finetuned model will be saved as a checkpoint you can load like:
   ```python
   pipeline = ASRInferencePipeline(
       model_card="omniASR_LLM_1B",  # Your finetuned checkpoint
       device="cuda"
   )
   ```

Should I create a complete training script for your 20 Banglish examples?

```python
# prepare_banglish_dataset.py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from omnilingual_asr.datasets.utils.audio import encode_audio_to_bytes

# Your 20 examples: list of (audio_path, transcript) tuples
banglish_examples = [
    ("banglish_1.wav", "আমরা কাল মিটিং করব at 3 PM"),
    ("banglish_2.wav", "Please send the report আজই"),
    # ... add all 20 examples
]

# Prepare data
data = []
for audio_path, text in banglish_examples:
    audio_bytes = encode_audio_to_bytes(audio_path)  # Converts to 16kHz mono FLAC
    audio_size = len(audio_bytes)  # Size in bytes
    
    data.append({
        "text": text,
        "audio_bytes": audio_bytes,
        "audio_size": audio_size,
        "corpus": "banglish",
        "split": "train",  # Use 15 for train, 5 for dev
        "language": "ben_Beng"  # Or create custom code
    })

# Create parquet file
df = pd.DataFrame(data)
table = pa.Table.from_pandas(df)
pq.write_table(table, "banglish_train.parquet", row_group_size=100)
```

```yaml
# llm-1b-banglish-finetune.yaml
model:
  name: "omniASR_LLM_1B"

dataset:
  name: "banglish_dataset"
  train_split: "train"
  valid_split: "dev"
  storage_mode: "MIXTURE_PARQUET"
  task_mode: "ASR"
  mixture_parquet_storage_config:
    dataset_summary_path: "/path/to/banglish/language_distribution.tsv"
    beta_corpus: 1.0  # Only your corpus
    beta_language: 1.0
    fragment_loading:
      cache: True
  asr_task_config:
     min_audio_len: 32_000  # 2 seconds
     max_audio_len: 480_000  # 30 seconds (shorter for small dataset)
     max_num_elements: 480_000
     batch_shuffle_window: 1
     normalize_audio: true
     example_shuffle_window: 1

tokenizer:
  name: "omniASR_tokenizer"

optimizer:
  config:
    lr: 5e-05  # Lower LR for finetuning

trainer:
  freeze_encoder_for_n_steps: 0  # Don't freeze - finetune everything
  mixed_precision:
    dtype: "torch.bfloat16"
  grad_accumulation:
    num_batches: 8  # Increase if OOM, helps with small batch

regime:
  num_steps: 2_000  # Fewer steps for small dataset (20 examples)
  validate_after_n_steps: 200
  validate_every_n_steps: 200
  checkpoint_every_n_steps: 200
  publish_metrics_every_n_steps: 50
```

```shellscript
export OUTPUT_DIR="/path/to/output"
python -m workflows.recipes.wav2vec2.asr $OUTPUT_DIR \
  --config-file llm-1b-banglish-finetune.yaml
```

```python
   pipeline = ASRInferencePipeline(
       model_card="omniASR_LLM_1B",  # Your finetuned checkpoint
       device="cuda"
   )
```

---

**User**

so  can I train the CTC 1B model on my 8GB local gpu for  20 Banglish audio files + transcripts into the parquet format? 

---

**Cursor**


Checking if CTC 1B training is feasible on an 8GB GPU:



## Memory analysis for CTC 1B training on 8GB GPU

Training memory breakdown:
- Model weights (BF16): ~1.85 GiB
- Gradients: ~1.85 GiB
- Optimizer states (Adam): ~3.7 GiB
- Activations (batch=1, 30s audio): ~1-2 GiB
- System overhead: ~0.5 GiB

Total: ~9-10 GiB minimum

Your GPU: 8GB (8192 MiB), with ~5.7GB already used = ~2.3GB free

## Verdict: CTC 1B won't fit on 8GB GPU

Even with batch_size=1 and aggressive gradient accumulation, you'd need ~9-10GB, which exceeds your available memory.

## Better option: CTC 300M

CTC 300M training memory:
- Model weights (BF16): ~0.6 GiB
- Gradients: ~0.6 GiB
- Optimizer states: ~1.2 GiB
- Activations: ~0.5-1 GiB
- Total: ~3-4 GiB

This fits on your 8GB GPU.

## Recommended config for CTC 300M on 8GB GPU

```yaml
# ctc-300m-banglish-8gb.yaml
model:
  name: "omniASR_CTC_300M"

dataset:
  name: "banglish_dataset"
  train_split: "train"
  valid_split: "dev"
  storage_mode: "MIXTURE_PARQUET"
  task_mode: "ASR"
  mixture_parquet_storage_config:
    dataset_summary_path: "/path/to/banglish/language_distribution.tsv"
    beta_corpus: 1.0
    beta_language: 1.0
    fragment_loading:
      cache: True
  asr_task_config:
     max_audio_len: 480_000  # 30 seconds (reduce if OOM)
     max_num_elements: 480_000
     batch_shuffle_window: 1
     normalize_audio: true
     example_shuffle_window: 1

tokenizer:
  name: "omniASR_tokenizer"

optimizer:
  config:
    lr: 1e-05

trainer:
  freeze_encoder_for_n_steps: 0
  mixed_precision:
    dtype: "torch.bfloat16"
  grad_accumulation:
    num_batches: 4  # Start with 4, increase if OOM

regime:
  num_steps: 2_000  # Fewer steps for 20 examples
  validate_after_n_steps: 200
  validate_every_n_steps: 200
  checkpoint_every_n_steps: 200
```

## Training time estimate for CTC 300M

- Batch size: 2-4
- Steps: 2,000
- Time: ~30-60 minutes on 8GB GPU

## If you still want to try CTC 1B

Extreme memory-saving options (may still fail):
1. Batch size: 1
2. Gradient accumulation: 16-32
3. Max audio length: 15 seconds (240,000 samples)
4. Freeze encoder: `freeze_encoder_for_n_steps: 1000` (finetune only decoder first)
5. Use gradient checkpointing (if supported)

Recommendation: Use CTC 300M for your 8GB GPU. It should handle Banglish reasonably well and will actually fit. If you need better quality, use cloud GPU (A5000 or similar) for CTC 1B or LLM 1B.

Should I create a complete training script for CTC 300M on your 8GB GPU?

```yaml
# ctc-300m-banglish-8gb.yaml
model:
  name: "omniASR_CTC_300M"

dataset:
  name: "banglish_dataset"
  train_split: "train"
  valid_split: "dev"
  storage_mode: "MIXTURE_PARQUET"
  task_mode: "ASR"
  mixture_parquet_storage_config:
    dataset_summary_path: "/path/to/banglish/language_distribution.tsv"
    beta_corpus: 1.0
    beta_language: 1.0
    fragment_loading:
      cache: True
  asr_task_config:
     max_audio_len: 480_000  # 30 seconds (reduce if OOM)
     max_num_elements: 480_000
     batch_shuffle_window: 1
     normalize_audio: true
     example_shuffle_window: 1

tokenizer:
  name: "omniASR_tokenizer"

optimizer:
  config:
    lr: 1e-05

trainer:
  freeze_encoder_for_n_steps: 0
  mixed_precision:
    dtype: "torch.bfloat16"
  grad_accumulation:
    num_batches: 4  # Start with 4, increase if OOM

regime:
  num_steps: 2_000  # Fewer steps for 20 examples
  validate_after_n_steps: 200
  validate_every_n_steps: 200
  checkpoint_every_n_steps: 200
```

---

**User**

But I also have another 12Gb RTX 2060 gpu, can I train there the CTC 1B model?

---

**Cursor**


Yes, you can train CTC 1B on a 12GB RTX 2060. Here's the analysis:

## Memory analysis for CTC 1B on 12GB RTX 2060

**Training memory breakdown:**
- Model weights (BF16): ~1.85 GiB
- Gradients: ~1.85 GiB
- Optimizer states (Adam): ~3.7 GiB
- Activations (batch=2-4, 30s audio): ~1-2 GiB
- System overhead: ~0.5 GiB
- **Total: ~9-10 GiB**

**Your RTX 2060: 12GB (12288 MiB)** — should fit with room to spare.

## Recommended config for CTC 1B on 12GB RTX 2060

```yaml
# ctc-1b-banglish-12gb.yaml
model:
  name: "omniASR_CTC_1B"

dataset:
  name: "banglish_dataset"
  train_split: "train"
  valid_split: "dev"
  storage_mode: "MIXTURE_PARQUET"
  task_mode: "ASR"
  mixture_parquet_storage_config:
    dataset_summary_path: "/path/to/banglish/language_distribution.tsv"
    beta_corpus: 1.0
    beta_language: 1.0
    fragment_loading:
      cache: True
  asr_task_config:
     max_audio_len: 480_000  # 30 seconds
     max_num_elements: 480_000
     batch_shuffle_window: 1
     normalize_audio: true
     example_shuffle_window: 1

tokenizer:
  name: "omniASR_tokenizer"

optimizer:
  config:
    lr: 1e-05

trainer:
  freeze_encoder_for_n_steps: 0
  mixed_precision:
    dtype: "torch.bfloat16"
  grad_accumulation:
    num_batches: 4  # Start with 4, can increase to 8 if needed

regime:
  num_steps: 2_000  # Adjust based on your 20 examples
  validate_after_n_steps: 200
  validate_every_n_steps: 200
  checkpoint_every_n_steps: 200
  publish_metrics_every_n_steps: 50
```

## Training time estimate

- Batch size: 2-4 (comfortable on 12GB)
- Steps: 2,000
- Time per step: ~1-2 seconds
- Total time: ~1-2 hours

## Notes for RTX 2060

1. BF16 support: RTX 2060 (Turing) supports BF16, so mixed precision should work.
2. If you hit OOM:
   - Increase `grad_accumulation.num_batches` to 8 or 16
   - Reduce `max_audio_len` to 240,000 (15 seconds)
   - Reduce batch size to 1
3. Performance: RTX 2060 is slower than A5000, but training time should still be reasonable for 20 examples.

## Multi-GPU option

If both GPUs are in the same system, you could:
- Use the 12GB RTX 2060 for training (CTC 1B)
- Keep the 8GB GPU free for other tasks or use it for inference

## Recommendation

Train CTC 1B on the 12GB RTX 2060. It should fit and give better results than CTC 300M. The training time for 20 examples should be manageable (1-2 hours).

Should I create a complete training script for your 20 Banglish examples on the 12GB RTX 2060?

```yaml
# ctc-1b-banglish-12gb.yaml
model:
  name: "omniASR_CTC_1B"

dataset:
  name: "banglish_dataset"
  train_split: "train"
  valid_split: "dev"
  storage_mode: "MIXTURE_PARQUET"
  task_mode: "ASR"
  mixture_parquet_storage_config:
    dataset_summary_path: "/path/to/banglish/language_distribution.tsv"
    beta_corpus: 1.0
    beta_language: 1.0
    fragment_loading:
      cache: True
  asr_task_config:
     max_audio_len: 480_000  # 30 seconds
     max_num_elements: 480_000
     batch_shuffle_window: 1
     normalize_audio: true
     example_shuffle_window: 1

tokenizer:
  name: "omniASR_tokenizer"

optimizer:
  config:
    lr: 1e-05

trainer:
  freeze_encoder_for_n_steps: 0
  mixed_precision:
    dtype: "torch.bfloat16"
  grad_accumulation:
    num_batches: 4  # Start with 4, can increase to 8 if needed

regime:
  num_steps: 2_000  # Adjust based on your 20 examples
  validate_after_n_steps: 200
  validate_every_n_steps: 200
  checkpoint_every_n_steps: 200
  publish_metrics_every_n_steps: 50
```

---

**User**

which model currently the  @app.py  using? CTC or LLM ? I see the output accuracy is good, for bangla, just the mmultilingual (eng + ben) gets problamatic sometimes, and numerics. So whats the difference totrain it on RTX 2060 12 GB vs A5000 24 Gb , only the bacth size aand time? Any effect on accuracy?

---

**Cursor**


Checking which model `app.py` uses:



## Current model in app.py

Your app is using LLM 1B (not CTC):
```python
DEFAULT_MODEL_CARD = "omniASR_LLM_1B_local"
```

This is the LLM (Language Model) variant, which explains:
- Good accuracy for Bangla
- Issues with mixed-language (English + Bengali) — it tends to transliterate English to Bengali script
- Numeric issues — common in ASR models

## RTX 2060 12GB vs A5000 24GB: impact on accuracy

Short answer: the GPU does not directly affect final model accuracy. It mainly affects batch size and training speed.

### What doesn't affect accuracy
- GPU model (RTX 2060 vs A5000)
- Training speed
- Memory capacity (as long as training fits)

### What does affect accuracy
- Training data quality and quantity
- Number of training steps/epochs
- Learning rate and other hyperparameters
- Model architecture (CTC vs LLM)
- Batch size (indirectly, via convergence)

### Practical differences

| Aspect | RTX 2060 12GB | A5000 24GB | Impact on Accuracy |
|--------|---------------|------------|-------------------|
| Batch size | 2-4 | 4-8 | Minimal (larger batches can help convergence slightly) |
| Training time | ~2-3 hours | ~1-1.5 hours | None (just convenience) |
| Memory headroom | Tight | Comfortable | None (if training fits) |
| Hyperparameter tuning | Limited | More flexible | Indirect (more experiments = better tuning) |

### Batch size and accuracy

Larger batches can:
- Provide more stable gradients (slightly better convergence)
- Allow higher effective learning rates
- Reduce training variance

For 20 examples, the difference is minimal. With larger datasets (1000+ examples), larger batches can help.

## Recommendations for your use case

### Option 1: Train on RTX 2060 12GB (local)
- Pros: Free, convenient, sufficient for 20 examples
- Cons: Slower, less headroom for experiments
- Accuracy: Same as A5000 with same hyperparameters

### Option 2: Train on A5000 24GB (cloud)
- Pros: Faster, more flexibility, can experiment more
- Cons: Costs money (~$0.50-1.50/hour)
- Accuracy: Same as RTX 2060 with same hyperparameters

## Addressing your specific issues

For mixed-language (English + Bengali) and numerics:

1. Mixed-language: Finetune LLM 1B on your 20 Banglish examples with proper script separation in transcripts:
   ```
   "আমরা কাল মিটING করব at 3 PM"  # English in Latin, Bengali in Bengali
   ```

2. Numerics: Include numeric examples in training:
   ```
   "আমার ফোন নম্বর 01712345678"
   "The price is $50.99"
   ```

## Final recommendation

For 20 examples, train on your RTX 2060 12GB:
- Accuracy will be the same
- Saves cloud costs
- Training time difference is small (1-2 hours vs 1.5-3 hours)
- Sufficient for experimentation

Use A5000 if:
- You plan to scale to 100+ examples
- You want to experiment with many hyperparameters
- Time is critical

Bottom line: GPU choice affects speed and convenience, not final accuracy. Focus on data quality and hyperparameter tuning.

```python
DEFAULT_MODEL_CARD = "omniASR_LLM_1B_local"
```

```plaintext
   "আমরা কাল মিটING করব at 3 PM"  # English in Latin, Bengali in Bengali
```

```plaintext
   "আমার ফোন নম্বর 01712345678"
   "The price is $50.99"
   ```

## Final recommendation

For 20 examples, train on your RTX 2060 12GB:
- Accuracy will be the same
- Saves cloud costs
- Training time difference is small (1-2 hours vs 1.5-3 hours)
- Sufficient for experimentation

Use A5000 if:
- You plan to scale to 100+ examples
- You want to experiment with many hyperparameters
- Time is critical

Bottom line: GPU choice affects speed and convenience, not final accuracy. Focus on data quality and hyperparameter tuning.
```

---

**User**

but would the 1B LLM or CTC model would be better for this use case? I havent tested CTC model yet, so cant compare 

---

**Cursor**


Comparing CTC vs LLM for your use case:



## CTC 1B vs LLM 1B for Banglish code-switching

### CTC 1B model

**Pros:**
- Faster inference (parallel generation)
- Lower memory usage (~3GB vs ~6GB)
- Can be competitive with larger LLM models when finetuned (per docs)
- Simpler architecture (no autoregressive decoding)

**Cons:**
- No language conditioning support
- Must learn code-switching entirely from training data
- May struggle with script separation without explicit guidance

### LLM 1B model

**Pros:**
- Supports language conditioning (though limited for mixed content)
- Autoregressive generation can help with context
- Generally better quality for complex scenarios
- More flexible for future enhancements

**Cons:**
- Slower inference (~1x real-time vs CTC's 48x)
- Higher memory usage (~6GB vs ~3GB)
- More complex training

## Recommendation for your use case

For Banglish code-switching with 20 examples, try CTC 1B first.

###
 Why CTC 1B might be better:
1. Finetuning can make CTC competitive: the docs note finetuned CTC models can match 7B LLM quality for specific languages.
2. Code-switching is learned from data: both models need to learn it from examples; CTC can learn it if the training data shows proper script separation.
3. Faster inference: useful for a web app.
4. Lower memory: fits better on your 8GB GPU for inference.

### When LLM 1B might be better:
- If you have 100+ examples (more data benefits LLM)
- If you need language conditioning for other use cases
- If you plan to use zero-shot capabilities later

## Quick test suggestion

Test CTC 1B before training:

```python
# Quick test script
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

# Test CTC 1B
pipeline_ctc = ASRInferencePipeline(model_card="omniASR_CTC_1B")
transcription_ctc = pipeline_ctc.transcribe(
    ["banglish.m4a"], 
    lang=None,  # CTC ignores lang anyway
    batch_size=1
)

# Compare with your current LLM 1B results
print("CTC 1B:", transcription_ctc[0])
```

## Final recommendation

1. Test CTC 1B first on your Banglish files (no training).
2. If results are close to LLM 1B, finetune CTC 1B (faster, lower memory).
3. If CTC 1B is significantly worse, finetune LLM 1B.

For code-switching with limited data (20 examples), CTC 1B finetuned on proper script-separated examples may perform similarly to LLM 1B, with faster inference and lower memory.

Bottom line: Test CTC 1B first. If it's close, prefer CTC for speed and efficiency. If it's clearly worse, use LLM 1B.

```python
# Quick test script
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

# Test CTC 1B
pipeline_ctc = ASRInferencePipeline(model_card="omniASR_CTC_1B_local")
transcription_ctc = pipeline_ctc.transcribe(
    ["banglish.m4a"], 
    lang=None,  # CTC ignores lang anyway
    batch_size=1
)

# Compare with your current LLM 1B results
print("CTC 1B:", transcription_ctc[0])
```

