# Bangla ASR Training Dataset Format Outline

To train or fine-tune the `omniASR-CTC-1B` (MMS-1B) model for Bangla and Banglish, you should prepare your data in one of the following formats.

## 1. Hugging Face Format (Recommended for LoRA/Accelerate)

This is the simplest format and is compatible with the industry-standard `transformers` and `datasets` libraries.

### Directory Structure
```text
dataset/
├── audios/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
├── metadata.csv (or metadata.jsonl)
```

### Metadata Format (`metadata.csv`)
The CSV file must have at least two columns: `audio_path` and `sentence` (transcription).

| audio_path | sentence |
| :--- | :--- |
| audios/sample1.wav | আসসালামু আলাইকুম, কেমন আছেন? |
| audios/sample2.wav | ami bhalo achi, dhonnobad. |
| audios/sample3.wav | My phone number is 01712345678. |

---

## 2. Manifest Format (Repo Recipes / fairseq2)

If you intend to use the recipes located in `workflows/recipes/wav2vec2/asr`, you must use the Manifest format.

### Directory Structure
```text
dataset/
├── train.tsv
├── train.wrd
├── val.tsv
├── val.wrd
└── audio/
    └── ... (audio files)
```

### Manifest Files

#### `{split}.tsv` (Audio Index)
- **Line 1:** The absolute path to the base audio directory.
- **Subsequent Lines:** Relative path to audio file and its length in frames.

```text
/home/sigmind/omnilingual-asr/dataset/bangla_sample/audio
sample1.wav 160000
sample2.wav 80000
```

#### `{split}.wrd` (Transcriptions)
Each line corresponds to the same line in the `.tsv` file (excluding the header).

```text
আসসালামু আলাইকুম, কেমন আছেন?
ami bhalo achi, dhonnobad.
```

---

## 3. Best Practices for Banglish & Numerics

### Mixed Script Strategy
- **Exact Match:** Ensure your text labels strictly match the spoken audio. If the speaker says "Internet", use "Internet" or "ইন্টারনেট" consistently based on your chosen convention.
- **Spaces for Pauses:** Use spaces or commas in the transcription to represent natural pauses in speech. This helps the CTC model learn temporal alignment.

### Numerics & Alphanumerics
- **Literal Digits:** Use digital characters (`0-9`) instead of words ("শূণ্য", "one").
- **Passport/IDs:** For alphanumeric sequences like passport numbers, group them as they are spoken (e.g., `A 123 456`).

### Audio Requirements
- **Format:** `.wav`, `.flac`, or `.m4a`.
- **Sampling Rate:** 16,000 Hz (Standard for MMS/Wav2Vec2).
- **Channels:** Mono (Single channel).
