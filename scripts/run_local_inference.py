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

