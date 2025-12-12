#!/usr/bin/env python3
"""
Flask web dashboard for Omnilingual ASR inference.
Provides a simple, beautiful interface for uploading audio files,
transcribing them, and managing transcription history.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

# Supported audio formats by libsndfile (used by fairseq2)
SUPPORTED_FORMATS = {".wav", ".flac", ".ogg", ".au", ".aiff", ".mp3"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max file size
app.config["UPLOAD_FOLDER"] = Path(tempfile.gettempdir()) / "omnilingual_asr_uploads"
app.config["HISTORY_FILE"] = Path.home() / ".omnilingual_asr_history.json"

# Ensure upload directory exists
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)

# Initialize pipeline at startup
_pipeline: Optional[ASRInferencePipeline] = None
DEFAULT_MODEL_CARD = "omniASR_LLM_1B_local"

# Common language codes for the dropdown
COMMON_LANGUAGES = [
    ("ben_Beng", "Bengali (বাংলা)"),
    ("eng_Latn", "English"),
    ("hin_Deva", "Hindi (हिन्दी)"),
    ("spa_Latn", "Spanish (Español)"),
    ("fra_Latn", "French (Français)"),
    ("deu_Latn", "German (Deutsch)"),
    ("jpn_Jpan", "Japanese (日本語)"),
    ("kor_Hang", "Korean (한국어)"),
    ("zho_Hans", "Chinese Simplified (简体中文)"),
    ("ara_Arab", "Arabic (العربية)"),
    ("por_Latn", "Portuguese (Português)"),
    ("rus_Cyrl", "Russian (Русский)"),
    ("ita_Latn", "Italian (Italiano)"),
    ("nld_Latn", "Dutch (Nederlands)"),
    ("pol_Latn", "Polish (Polski)"),
    ("tur_Latn", "Turkish (Türkçe)"),
    ("vie_Latn", "Vietnamese (Tiếng Việt)"),
    ("tha_Thai", "Thai (ไทย)"),
    ("ind_Latn", "Indonesian (Bahasa Indonesia)"),
    ("msa_Latn", "Malay (Bahasa Melayu)"),
]


def get_pipeline() -> ASRInferencePipeline:
    """Get the pipeline (should be initialized at startup)."""
    global _pipeline
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized. This should not happen.")
    return _pipeline


def initialize_pipeline(model_card: str = DEFAULT_MODEL_CARD) -> None:
    """Initialize the pipeline at startup."""
    global _pipeline, DEFAULT_MODEL_CARD
    if _pipeline is None:
        DEFAULT_MODEL_CARD = model_card
        print(f"Loading pipeline with model card '{model_card}'...")
        print("This may take a few moments...")
        try:
            _pipeline = ASRInferencePipeline(model_card=model_card)
            print("✓ Pipeline loaded successfully!")
        except Exception as e:
            print(f"✗ Failed to load pipeline: {e}")
            raise


def load_history() -> List[dict]:
    """Load transcription history from file."""
    if app.config["HISTORY_FILE"].exists():
        try:
            with open(app.config["HISTORY_FILE"], "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history: List[dict]) -> None:
    """Save transcription history to file."""
    # Keep only last 100 entries
    history = history[-100:]
    try:
        with open(app.config["HISTORY_FILE"], "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # Silently fail if we can't save history


def add_to_history(filename: str, lang_code: str, transcription: str) -> None:
    """Add a transcription to history."""
    history = load_history()
    history.append(
        {
            "id": len(history),
            "filename": filename,
            "lang_code": lang_code,
            "transcription": transcription,
            "timestamp": datetime.now().isoformat(),
        }
    )
    save_history(history)


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


@app.route("/")
def index():
    """Render the main dashboard page."""
    return render_template("index.html", languages=COMMON_LANGUAGES)


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    """Handle audio file upload and transcription."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    lang_code = request.form.get("lang_code", "ben_Beng")

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save uploaded file temporarily
    filename = secure_filename(file.filename)
    # Preserve original extension
    original_ext = Path(filename).suffix.lower()
    temp_path = app.config["UPLOAD_FOLDER"] / f"{Path(filename).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{original_ext}"
    file.save(str(temp_path))

    # Convert unsupported formats to WAV
    converted_path = None
    audio_path = temp_path
    
    if original_ext not in SUPPORTED_FORMATS:
        try:
            print(f"Converting {original_ext} file to WAV format...")
            converted_path = convert_audio_to_wav(temp_path)
            audio_path = converted_path
        except Exception as e:
            # Clean up original file
            try:
                temp_path.unlink()
            except Exception:
                pass
            return jsonify({"error": f"Audio conversion failed: {str(e)}"}), 400

    try:
        # Get pipeline and transcribe
        # The pipeline handles audio decoding automatically for supported formats
        pipeline = get_pipeline()
        transcriptions = pipeline.transcribe(
            [str(audio_path)], lang=[lang_code], batch_size=1
        )

        if not transcriptions:
            return jsonify({"error": "Transcription failed - no output generated"}), 500

        transcription = transcriptions[0]

        # Add to history and get the history ID
        history = load_history()
        history_id = len(history)
        add_to_history(filename, lang_code, transcription)

        return jsonify(
            {
                "success": True,
                "transcription": transcription,
                "filename": filename,
                "lang_code": lang_code,
                "history_id": history_id,
            }
        )
    except ValueError as e:
        # Handle audio length or format errors
        error_msg = str(e)
        if "Max audio length" in error_msg:
            return jsonify({"error": "Audio file is too long. Maximum length is 40 seconds."}), 400
        return jsonify({"error": f"Audio processing error: {error_msg}"}), 400
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Transcription error: {error_details}")
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
    finally:
        # Clean up temp files
        try:
            if temp_path.exists():
                temp_path.unlink()
            if converted_path and converted_path.exists():
                converted_path.unlink()
        except Exception:
            pass


@app.route("/api/history", methods=["GET"])
def get_history():
    """Get transcription history."""
    history = load_history()
    # Return most recent first
    return jsonify({"history": list(reversed(history))})


@app.route("/api/download/<int:history_id>", methods=["GET"])
def download_transcription(history_id: int):
    """Download a transcription as a text file."""
    history = load_history()
    entry = next((h for h in history if h["id"] == history_id), None)

    if not entry:
        return jsonify({"error": "History entry not found"}), 404

    # Create a temporary text file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".txt", text=True)
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            f.write(f"Filename: {entry['filename']}\n")
            f.write(f"Language: {entry['lang_code']}\n")
            f.write(f"Timestamp: {entry['timestamp']}\n")
            f.write("\n" + "=" * 50 + "\n\n")
            f.write(entry["transcription"])

        return send_file(
            temp_path,
            as_attachment=True,
            download_name=f"transcription_{entry['filename']}_{entry['id']}.txt",
            mimetype="text/plain",
        )
    except Exception:
        return jsonify({"error": "Failed to create download file"}), 500


@app.route("/api/languages", methods=["GET"])
def get_languages():
    """Get list of available languages."""
    return jsonify({"languages": [{"code": code, "name": name} for code, name in COMMON_LANGUAGES]})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Omnilingual ASR web dashboard")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to (default: 5000)",
    )
    parser.add_argument(
        "--model-card",
        default=DEFAULT_MODEL_CARD,
        help=f"Model card to use (default: {DEFAULT_MODEL_CARD})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )

    args = parser.parse_args()
    
    # Initialize pipeline BEFORE starting Flask server
    print("=" * 60)
    print("Initializing Omnilingual ASR Dashboard")
    print("=" * 60)
    try:
        initialize_pipeline(args.model_card)
    except Exception as e:
        print(f"\n❌ Failed to initialize pipeline: {e}")
        print("Please check your model card configuration and try again.")
        exit(1)
    
    print(f"\nStarting Flask server on http://{args.host}:{args.port}")
    print("=" * 60)
    print("Dashboard is ready! Open the URL in your browser.")
    print("=" * 60)
    
    # Disable reloader to avoid threading issues with fairseq2
    # The model is loaded in the main thread, and we want to keep it there
    # In production, use a proper WSGI server like gunicorn with a single worker
    app.run(
        host=args.host, 
        port=args.port, 
        debug=args.debug,
        use_reloader=False,  # Disable reloader to avoid threading issues
        threaded=False  # Disable threading to avoid fairseq2 gang issues
    )

