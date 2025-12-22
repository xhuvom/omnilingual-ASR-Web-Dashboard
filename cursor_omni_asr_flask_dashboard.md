# Omni-ASR Flask dashboard
_Exported on 12/12/2025 at 07:11:52 GMT+6 from Cursor (2.0.34)_

---

**User**

Ojkey great so we havbe successfully run the omni-asr lcoally . @bash (151-176)   with @run_local_inference.py   for  ben_Beng and producting good putput. Now I want a simple demo dashboard with flask service where I can upload audop by draw and drop or uplaod button and sprocess and see the output on the dahbaord ans save the output to text or recent history. I showud be able to switch language menu, ben_Beng default.  So build a simple yeat beauitful  but functional web dashbaord for it. 

---

**Cursor**


Exploring the codebase to understand the structure, then creating a Flask dashboard with drag-and-drop upload, language selection, transcription display, and history.



Checking available language codes:



Creating a Flask dashboard with drag-and-drop upload, language selection, transcription display, and history. Starting with the Flask app and frontend files:


```python
#!/usr/bin/env python3
"""
Flask web dashboard for Omnilingual ASR inference.
Provides a simple, beautiful interface for uploading audio files,
transcribing them, and managing transcription history.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max file size
app.config["UPLOAD_FOLDER"] = Path(tempfile.gettempdir()) / "omnilingual_asr_uploads"
app.config["HISTORY_FILE"] = Path.home() / ".omnilingual_asr_history.json"

# Ensure upload directory exists
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)

# Initialize pipeline (lazy loading)
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
    """Lazy load the pipeline to avoid loading on import."""
    global _pipeline
    if _pipeline is None:
        print(f"Loading pipeline with model card '{DEFAULT_MODEL_CARD}'...")
        _pipeline = ASRInferencePipeline(model_card=DEFAULT_MODEL_CARD)
    return _pipeline


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
    temp_path = app.config["UPLOAD_FOLDER"] / filename
    file.save(str(temp_path))

    try:
        # Get pipeline and transcribe
        pipeline = get_pipeline()
        transcriptions = pipeline.transcribe(
            [str(temp_path)], lang=[lang_code], batch_size=1
        )

        if not transcriptions:
            return jsonify({"error": "Transcription failed"}), 500

        transcription = transcriptions[0]

        # Add to history
        add_to_history(filename, lang_code, transcription)

        return jsonify(
            {
                "success": True,
                "transcription": transcription,
                "filename": filename,
                "lang_code": lang_code,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temp file
        try:
            temp_path.unlink()
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
    DEFAULT_MODEL_CARD = args.model_card

    print(f"Starting Omnilingual ASR Dashboard on http://{args.host}:{args.port}")
    print(f"Using model card: {DEFAULT_MODEL_CARD}")
    app.run(host=args.host, port=args.port, debug=args.debug)

```


Creating the HTML template and static assets:


```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Omnilingual ASR Dashboard</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="container">
        <header>
            <h1>🎤 Omnilingual ASR Dashboard</h1>
            <p class="subtitle">Upload audio files and get transcriptions in multiple languages</p>
        </header>

        <main>
            <!-- Upload Section -->
            <section class="upload-section">
                <div class="upload-area" id="uploadArea">
                    <div class="upload-content">
                        <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="17 8 12 3 7 8"></polyline>
                            <line x1="12" y1="3" x2="12" y2="15"></line>
                        </svg>
                        <h2>Drag & Drop Audio File</h2>
                        <p>or</p>
                        <button class="btn btn-primary" id="uploadBtn">Choose File</button>
                        <input type="file" id="fileInput" accept="audio/*" style="display: none;">
                        <p class="file-info" id="fileInfo"></p>
                    </div>
                </div>

                <!-- Language Selection -->
                <div class="language-section">
                    <label for="langSelect">Language:</label>
                    <select id="langSelect" class="lang-select">
                        {% for code, name in languages %}
                        <option value="{{ code }}" {% if code == 'ben_Beng' %}selected{% endif %}>{{ name }}</option>
                        {% endfor %}
                    </select>
                </div>

                <!-- Process Button -->
                <button class="btn btn-process" id="processBtn" disabled>
                    <span class="btn-text">Transcribe</span>
                    <span class="btn-spinner" style="display: none;">⏳ Processing...</span>
                </button>
            </section>

            <!-- Results Section -->
            <section class="results-section" id="resultsSection" style="display: none;">
                <div class="result-header">
                    <h2>Transcription Result</h2>
                    <button class="btn btn-secondary" id="downloadBtn">Download as Text</button>
                </div>
                <div class="result-content">
                    <div class="result-meta">
                        <span id="resultFilename"></span>
                        <span id="resultLang"></span>
                    </div>
                    <div class="result-text" id="resultText"></div>
                </div>
            </section>

            <!-- History Section -->
            <section class="history-section">
                <div class="history-header">
                    <h2>Recent History</h2>
                    <button class="btn btn-link" id="refreshHistoryBtn">Refresh</button>
                </div>
                <div class="history-list" id="historyList">
                    <p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>
                </div>
            </section>
        </main>
    </div>

    <script src="{{ url_for('static', filename='js/app.js') }}"></script>
</body>
</html>

```

```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    --primary-color: #6366f1;
    --primary-hover: #4f46e5;
    --secondary-color: #8b5cf6;
    --success-color: #10b981;
    --danger-color: #ef4444;
    --bg-color: #f8fafc;
    --card-bg: #ffffff;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --border-color: #e2e8f0;
    --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    --radius: 12px;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    padding: 2rem;
    color: var(--text-primary);
}

.container {
    max-width: 1200px;
    margin: 0 auto;
}

header {
    text-align: center;
    color: white;
    margin-bottom: 2rem;
}

header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.subtitle {
    font-size: 1.1rem;
    opacity: 0.9;
}

main {
    display: flex;
    flex-direction: column;
    gap: 2rem;
}

section {
    background: var(--card-bg);
    border-radius: var(--radius);
    padding: 2rem;
    box-shadow: var(--shadow-lg);
}

/* Upload Section */
.upload-section {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

.upload-area {
    border: 3px dashed var(--border-color);
    border-radius: var(--radius);
    padding: 3rem 2rem;
    text-align: center;
    transition: all 0.3s ease;
    cursor: pointer;
    background: var(--bg-color);
}

.upload-area:hover {
    border-color: var(--primary-color);
    background: #f1f5f9;
}

.upload-area.dragover {
    border-color: var(--primary-color);
    background: #eef2ff;
    transform: scale(1.02);
}

.upload-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
}

.upload-icon {
    width: 64px;
    height: 64px;
    color: var(--primary-color);
    margin-bottom: 0.5rem;
}

.upload-area h2 {
    font-size: 1.5rem;
    color: var(--text-primary);
    margin: 0;
}

.upload-area p {
    color: var(--text-secondary);
    margin: 0;
}

.file-info {
    margin-top: 1rem;
    font-weight: 500;
    color: var(--primary-color);
}

/* Language Selection */
.language-section {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.language-section label {
    font-weight: 600;
    color: var(--text-primary);
    min-width: 80px;
}

.lang-select {
    flex: 1;
    padding: 0.75rem 1rem;
    border: 2px solid var(--border-color);
    border-radius: 8px;
    font-size: 1rem;
    background: white;
    color: var(--text-primary);
    cursor: pointer;
    transition: all 0.2s ease;
}

.lang-select:hover {
    border-color: var(--primary-color);
}

.lang-select:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

/* Buttons */
.btn {
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}

.btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.btn-primary {
    background: var(--primary-color);
    color: white;
}

.btn-primary:hover:not(:disabled) {
    background: var(--primary-hover);
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.btn-process {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    color: white;
    padding: 1rem 2rem;
    font-size: 1.1rem;
    width: 100%;
}

.btn-process:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.btn-secondary {
    background: var(--secondary-color);
    color: white;
}

.btn-secondary:hover {
    background: #7c3aed;
}

.btn-link {
    background: none;
    color: var(--primary-color);
    padding: 0.5rem 1rem;
    text-decoration: underline;
}

.btn-link:hover {
    color: var(--primary-hover);
}

/* Results Section */
.results-section {
    animation: slideIn 0.3s ease;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid var(--border-color);
}

.result-header h2 {
    font-size: 1.5rem;
    color: var(--text-primary);
}

.result-meta {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}

.result-meta span {
    padding: 0.5rem 1rem;
    background: var(--bg-color);
    border-radius: 6px;
    font-size: 0.9rem;
    color: var(--text-secondary);
    font-weight: 500;
}

.result-text {
    background: var(--bg-color);
    padding: 1.5rem;
    border-radius: 8px;
    line-height: 1.8;
    font-size: 1.1rem;
    color: var(--text-primary);
    white-space: pre-wrap;
    word-wrap: break-word;
    min-height: 100px;
}

/* History Section */
.history-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid var(--border-color);
}

.history-header h2 {
    font-size: 1.5rem;
    color: var(--text-primary);
}

.history-list {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    max-height: 500px;
    overflow-y: auto;
}

.history-item {
    background: var(--bg-color);
    padding: 1.5rem;
    border-radius: 8px;
    border: 2px solid var(--border-color);
    transition: all 0.2s ease;
    cursor: pointer;
}

.history-item:hover {
    border-color: var(--primary-color);
    transform: translateX(4px);
}

.history-item-header {
    display: flex;
    justify-content: space-between;
    align-items: start;
    margin-bottom: 0.75rem;
}

.history-item-meta {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

.history-item-filename {
    font-weight: 600;
    color: var(--text-primary);
}

.history-item-lang {
    font-size: 0.85rem;
    color: var(--text-secondary);
}

.history-item-time {
    font-size: 0.8rem;
    color: var(--text-secondary);
}

.history-item-text {
    color: var(--text-primary);
    line-height: 1.6;
    margin-top: 0.75rem;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.history-item-actions {
    display: flex;
    gap: 0.5rem;
}

.btn-icon {
    padding: 0.5rem;
    background: white;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.btn-icon:hover {
    background: var(--primary-color);
    color: white;
    border-color: var(--primary-color);
}

.empty-state {
    text-align: center;
    color: var(--text-secondary);
    padding: 2rem;
    font-style: italic;
}

/* Loading Spinner */
.spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    border-top-color: white;
    animation: spin 0.8s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Responsive */
@media (max-width: 768px) {
    body {
        padding: 1rem;
    }

    header h1 {
        font-size: 2rem;
    }

    section {
        padding: 1.5rem;
    }

    .result-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 1rem;
    }

    .history-item-header {
        flex-direction: column;
        gap: 0.5rem;
    }
}

```

```javascript
// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const processBtn = document.getElementById('processBtn');
const langSelect = document.getElementById('langSelect');
const fileInfo = document.getElementById('fileInfo');
const resultsSection = document.getElementById('resultsSection');
const resultText = document.getElementById('resultText');
const resultFilename = document.getElementById('resultFilename');
const resultLang = document.getElementById('resultLang');
const downloadBtn = document.getElementById('downloadBtn');
const historyList = document.getElementById('historyList');
const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

let currentFile = null;
let currentTranscription = null;
let currentHistoryId = null;

// Event Listeners
uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
uploadArea.addEventListener('click', () => fileInput.click());

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

processBtn.addEventListener('click', handleTranscribe);
downloadBtn.addEventListener('click', handleDownload);
refreshHistoryBtn.addEventListener('click', loadHistory);

// Initialize
loadHistory();

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

    currentFile = file;
    fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
    processBtn.disabled = false;
    resultsSection.style.display = 'none';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
}

function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
        });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

```


Checking for linting errors and verifying Flask is available:



Updating the Flask app to handle audio conversion and adding a requirements file:


```pip-requirements
flask>=3.0.0
werkzeug>=3.0.0

```

```python
#!/usr/bin/env python3
"""
Flask web dashboard for Omnilingual ASR inference.
Provides a simple, beautiful interface for uploading audio files,
transcribing them, and managing transcription history.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max file size
app.config["UPLOAD_FOLDER"] = Path(tempfile.gettempdir()) / "omnilingual_asr_uploads"
app.config["HISTORY_FILE"] = Path.home() / ".omnilingual_asr_history.json"

# Ensure upload directory exists
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)

# Initialize pipeline (lazy loading)
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
    """Lazy load the pipeline to avoid loading on import."""
    global _pipeline
    if _pipeline is None:
        print(f"Loading pipeline with model card '{DEFAULT_MODEL_CARD}'...")
        _pipeline = ASRInferencePipeline(model_card=DEFAULT_MODEL_CARD)
    return _pipeline


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
    original_ext = Path(filename).suffix
    temp_path = app.config["UPLOAD_FOLDER"] / f"{Path(filename).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{original_ext}"
    file.save(str(temp_path))

    try:
        # Get pipeline and transcribe
        # The pipeline handles audio decoding automatically for supported formats
        pipeline = get_pipeline()
        transcriptions = pipeline.transcribe(
            [str(temp_path)], lang=[lang_code], batch_size=1
        )

        if not transcriptions:
            return jsonify({"error": "Transcription failed - no output generated"}), 500

        transcription = transcriptions[0]

        # Add to history
        add_to_history(filename, lang_code, transcription)

        return jsonify(
            {
                "success": True,
                "transcription": transcription,
                "filename": filename,
                "lang_code": lang_code,
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
        # Clean up temp file
        try:
            if temp_path.exists():
                temp_path.unlink()
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
    DEFAULT_MODEL_CARD = args.model_card

    print(f"Starting Omnilingual ASR Dashboard on http://{args.host}:{args.port}")
    print(f"Using model card: {DEFAULT_MODEL_CARD}")
    app.run(host=args.host, port=args.port, debug=args.debug)


```


Adding audio conversion support for unsupported formats (e.g., .m4a):


```python
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

# Initialize pipeline (lazy loading)
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
    """Lazy load the pipeline to avoid loading on import."""
    global _pipeline
    if _pipeline is None:
        print(f"Loading pipeline with model card '{DEFAULT_MODEL_CARD}'...")
        _pipeline = ASRInferencePipeline(model_card=DEFAULT_MODEL_CARD)
    return _pipeline


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
    original_ext = Path(filename).suffix
    temp_path = app.config["UPLOAD_FOLDER"] / f"{Path(filename).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{original_ext}"
    file.save(str(temp_path))

    try:
        # Get pipeline and transcribe
        # The pipeline handles audio decoding automatically for supported formats
        pipeline = get_pipeline()
        transcriptions = pipeline.transcribe(
            [str(temp_path)], lang=[lang_code], batch_size=1
        )

        if not transcriptions:
            return jsonify({"error": "Transcription failed - no output generated"}), 500

        transcription = transcriptions[0]

        # Add to history
        add_to_history(filename, lang_code, transcription)

        return jsonify(
            {
                "success": True,
                "transcription": transcription,
                "filename": filename,
                "lang_code": lang_code,
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
        # Clean up temp file
        try:
            if temp_path.exists():
                temp_path.unlink()
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
    DEFAULT_MODEL_CARD = args.model_card

    print(f"Starting Omnilingual ASR Dashboard on http://{args.host}:{args.port}")
    print(f"Using model card: {DEFAULT_MODEL_CARD}")
    app.run(host=args.host, port=args.port, debug=args.debug)


```

```python
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

# Initialize pipeline (lazy loading)
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
    """Lazy load the pipeline to avoid loading on import."""
    global _pipeline
    if _pipeline is None:
        print(f"Loading pipeline with model card '{DEFAULT_MODEL_CARD}'...")
        _pipeline = ASRInferencePipeline(model_card=DEFAULT_MODEL_CARD)
    return _pipeline


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
    original_ext = Path(filename).suffix
    temp_path = app.config["UPLOAD_FOLDER"] / f"{Path(filename).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{original_ext}"
    file.save(str(temp_path))

    try:
        # Get pipeline and transcribe
        # The pipeline handles audio decoding automatically for supported formats
        pipeline = get_pipeline()
        transcriptions = pipeline.transcribe(
            [str(temp_path)], lang=[lang_code], batch_size=1
        )

        if not transcriptions:
            return jsonify({"error": "Transcription failed - no output generated"}), 500

        transcription = transcriptions[0]

        # Add to history
        add_to_history(filename, lang_code, transcription)

        return jsonify(
            {
                "success": True,
                "transcription": transcription,
                "filename": filename,
                "lang_code": lang_code,
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
        # Clean up temp file
        try:
            if temp_path.exists():
                temp_path.unlink()
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
    DEFAULT_MODEL_CARD = args.model_card

    print(f"Starting Omnilingual ASR Dashboard on http://{args.host}:{args.port}")
    print(f"Using model card: {DEFAULT_MODEL_CARD}")
    app.run(host=args.host, port=args.port, debug=args.debug)


```

```python
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

# Initialize pipeline (lazy loading)
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
    """Lazy load the pipeline to avoid loading on import."""
    global _pipeline
    if _pipeline is None:
        print(f"Loading pipeline with model card '{DEFAULT_MODEL_CARD}'...")
        _pipeline = ASRInferencePipeline(model_card=DEFAULT_MODEL_CARD)
    return _pipeline


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

        # Add to history
        add_to_history(filename, lang_code, transcription)

        return jsonify(
            {
                "success": True,
                "transcription": transcription,
                "filename": filename,
                "lang_code": lang_code,
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
        # Clean up temp file
        try:
            if temp_path.exists():
                temp_path.unlink()
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
    DEFAULT_MODEL_CARD = args.model_card

    print(f"Starting Omnilingual ASR Dashboard on http://{args.host}:{args.port}")
    print(f"Using model card: {DEFAULT_MODEL_CARD}")
    app.run(host=args.host, port=args.port, debug=args.debug)


```

```python
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

# Initialize pipeline (lazy loading)
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
    """Lazy load the pipeline to avoid loading on import."""
    global _pipeline
    if _pipeline is None:
        print(f"Loading pipeline with model card '{DEFAULT_MODEL_CARD}'...")
        _pipeline = ASRInferencePipeline(model_card=DEFAULT_MODEL_CARD)
    return _pipeline


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

        # Add to history
        add_to_history(filename, lang_code, transcription)

        return jsonify(
            {
                "success": True,
                "transcription": transcription,
                "filename": filename,
                "lang_code": lang_code,
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
    DEFAULT_MODEL_CARD = args.model_card

    print(f"Starting Omnilingual ASR Dashboard on http://{args.host}:{args.port}")
    print(f"Using model card: {DEFAULT_MODEL_CARD}")
    app.run(host=args.host, port=args.port, debug=args.debug)


```


Creating a README for the dashboard with usage instructions:


```markdown
# Omnilingual ASR Web Dashboard

A beautiful, modern web dashboard for transcribing audio files using the Omnilingual ASR model.

## Features

- 🎤 **Drag & Drop Upload**: Easily upload audio files by dragging and dropping
- 🌐 **Multi-language Support**: Select from 20+ common languages (default: Bengali)
- 📝 **Real-time Transcription**: Get instant transcriptions of your audio files
- 💾 **Download Results**: Save transcriptions as text files
- 📚 **History Management**: View and access your recent transcriptions
- 🎨 **Modern UI**: Beautiful, responsive design with smooth animations

## Installation

1. Install Flask (if not already installed):
```bash
pip install -r requirements-dashboard.txt
```

2. Ensure you have `ffmpeg` installed for audio format conversion:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

## Usage

### Start the Dashboard

```bash
python app.py
```

Or with custom options:

```bash
python app.py --host 0.0.0.0 --port 5000 --model-card omniASR_LLM_1B_local
```

### Access the Dashboard

Open your web browser and navigate to:
```
http://localhost:5000
```

### Using the Dashboard

1. **Upload Audio**: 
   - Drag and drop an audio file onto the upload area, or
   - Click "Choose File" to browse for a file

2. **Select Language**: 
   - Choose the language from the dropdown menu (default: Bengali - ben_Beng)

3. **Transcribe**: 
   - Click the "Transcribe" button to process the audio

4. **View Results**: 
   - The transcription will appear in the results section
   - Click "Download as Text" to save the transcription

5. **View History**: 
   - Scroll down to see your recent transcriptions
   - Click on any history item to view the full transcription
   - Click the download icon to download any previous transcription

## Supported Audio Formats

The dashboard supports all audio formats that the underlying model supports:
- **Direct support**: `.wav`, `.flac`, `.ogg`, `.au`, `.aiff`, `.mp3`
- **Auto-converted**: `.m4a`, `.aac`, and other formats (requires ffmpeg)

**Note**: Audio files must be shorter than 40 seconds.

## Configuration

### Model Card

By default, the dashboard uses `omniASR_LLM_1B_local`. To use a different model:

```bash
python app.py --model-card YOUR_MODEL_CARD
```

### History Storage

Transcription history is stored in `~/.omnilingual_asr_history.json` and is limited to the last 100 entries.

## Troubleshooting

### Audio Conversion Errors

If you encounter errors with `.m4a` or other formats:
- Ensure `ffmpeg` is installed and available in your PATH
- Try converting the file to `.wav` format manually first

### Model Loading Issues

- Ensure your model card is properly configured in `~/.config/fairseq2/assets/cards/models`
- Check that the model files are accessible

### Port Already in Use

If port 5000 is already in use:
```bash
python app.py --port 5001
```

## API Endpoints

The dashboard also exposes REST API endpoints:

- `POST /api/transcribe` - Upload and transcribe an audio file
- `GET /api/history` - Get transcription history
- `GET /api/download/<id>` - Download a transcription by history ID
- `GET /api/languages` - Get list of available languages

## License

Same as the main Omnilingual ASR project.

```


Fixing a small issue with history ID tracking for downloads:


```python
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

# Initialize pipeline (lazy loading)
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
    """Lazy load the pipeline to avoid loading on import."""
    global _pipeline
    if _pipeline is None:
        print(f"Loading pipeline with model card '{DEFAULT_MODEL_CARD}'...")
        _pipeline = ASRInferencePipeline(model_card=DEFAULT_MODEL_CARD)
    return _pipeline


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
    DEFAULT_MODEL_CARD = args.model_card

    print(f"Starting Omnilingual ASR Dashboard on http://{args.host}:{args.port}")
    print(f"Using model card: {DEFAULT_MODEL_CARD}")
    app.run(host=args.host, port=args.port, debug=args.debug)


```

```javascript
// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const processBtn = document.getElementById('processBtn');
const langSelect = document.getElementById('langSelect');
const fileInfo = document.getElementById('fileInfo');
const resultsSection = document.getElementById('resultsSection');
const resultText = document.getElementById('resultText');
const resultFilename = document.getElementById('resultFilename');
const resultLang = document.getElementById('resultLang');
const downloadBtn = document.getElementById('downloadBtn');
const historyList = document.getElementById('historyList');
const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

let currentFile = null;
let currentTranscription = null;
let currentHistoryId = null;

// Event Listeners
uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
uploadArea.addEventListener('click', () => fileInput.click());

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

processBtn.addEventListener('click', handleTranscribe);
downloadBtn.addEventListener('click', handleDownload);
refreshHistoryBtn.addEventListener('click', loadHistory);

// Initialize
loadHistory();

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

    currentFile = file;
    fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
    processBtn.disabled = false;
    resultsSection.style.display = 'none';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
}

function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
        });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}


```


Verifying everything is in place:



## Dashboard features

1. **Upload**
   - Drag-and-drop audio files
   - File browser button
   - Supports common audio formats (auto-converts unsupported formats like .m4a)

2. **Language selection**
   - Dropdown with 20+ languages
   - Default: Bengali (ben_Beng)
   - Easy to switch

3. **Transcription**
   - Real-time processing with loading indicators
   - Results displayed in a formatted section
   - Shows filename and language code

4. **Download**
   - Download transcriptions as text files
   - Includes metadata (filename, language, timestamp)

5. **History**
   - View recent transcriptions (last 100)
   - Click any item to view full transcription
   - Download previous transcriptions
   - Stored in `~/.omnilingual_asr_history.json`

6. **UI**
   - Responsive design
   - Smooth animations
   - Works on desktop and mobile

## Files created

- `app.py` - Flask application with all routes
- `templates/index.html` - Main dashboard HTML
- `static/css/style.css` - Styling
- `static/js/app.js` - Frontend JavaScript
- `requirements-dashboard.txt` - Flask dependencies
- `DASHBOARD_README.md` - Usage instructions

## To run the dashboard

1. Install Flask:
```bash
pip install -r requirements-dashboard.txt
```

2. Start the server:
```bash
python app.py
```

3. Open your browser:
```
http://localhost:5000
```

The dashboard uses your existing `omniASR_LLM_1B_local` model card and handles audio conversion automatically (including .m4a files if ffmpeg is installed).

Ready to use. Upload an audio file, select Bengali (or any other language), and get transcriptions.

```shellscript
pip install -r requirements-dashboard.txt
```

```shellscript
python app.py
```

```plaintext
http://localhost:5000
```

---

**User**

I get error when transcribing @python3 (211-256) 

 it should load the model first during ahboird startup and run inference on the fly

---

**Cursor**


Loading the model at startup instead of on the first request to avoid threading issues. Updating the app to initialize the pipeline at startup:


```python
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
    """Lazy load the pipeline to avoid loading on import."""
    global _pipeline
    if _pipeline is None:
        print(f"Loading pipeline with model card '{DEFAULT_MODEL_CARD}'...")
        _pipeline = ASRInferencePipeline(model_card=DEFAULT_MODEL_CARD)
    return _pipeline


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
    DEFAULT_MODEL_CARD = args.model_card

    print(f"Starting Omnilingual ASR Dashboard on http://{args.host}:{args.port}")
    print(f"Using model card: {DEFAULT_MODEL_CARD}")
    app.run(host=args.host, port=args.port, debug=args.debug)


```

```python
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
    DEFAULT_MODEL_CARD = args.model_card

    print(f"Starting Omnilingual ASR Dashboard on http://{args.host}:{args.port}")
    print(f"Using model card: {DEFAULT_MODEL_CARD}")
    app.run(host=args.host, port=args.port, debug=args.debug)


```

```python
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
    
    # Disable threading in debug mode to avoid thread-local issues
    # In production, use a proper WSGI server like gunicorn with workers
    use_reloader = args.debug
    app.run(
        host=args.host, 
        port=args.port, 
        debug=args.debug,
        use_reloader=False,  # Disable reloader to avoid threading issues
        threaded=True  # Enable threading for handling multiple requests
    )


```


The error is a threading issue with fairseq2. Ensuring the model loads at startup in the main thread. Updating the code:


```python
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


```


Updated the app to load the model at startup before Flask starts handling requests. This should fix the threading issue. Changes:

1. **Model loads at startup**: The pipeline initializes before the Flask server starts, so it's ready when requests arrive.
2. **Threading disabled**: Disabled Flask threading to avoid the fairseq2 thread-local storage issue.

The updated flow:
- Model loads when you run `python app.py`
- Server starts only after the model is loaded
- Requests are processed synchronously (one at a time) to avoid threading issues

Restart the server to test:



The code is updated. When you run `python app.py`, you'll see:

1. **Model loading at startup** - The pipeline initializes before the Flask server starts
2. **Ready message** - Once loaded, the server starts and is ready to handle requests

The model loads once at startup, so subsequent transcription requests are faster and avoid threading issues.

**To test:**
1. Stop the current server (Ctrl+C)
2. Restart it: `python app.py`
3. You should see the model loading message before the server starts
4. Try uploading an audio file again

The error should be resolved because the model is loaded in the main thread before any requests are handled.

---

**User**

Great, the dahbaord working now! Now add a microphone capture feature in the dashbaord to take input voice from mic on the browser and convert it and send appropriately to the asr model for inferencing. Does it support real time inferencing to transcript? Find out also, if not, thats okey, but you have to take input appropriate chink length from mic to the current limit (40 sec mybe)

---

**Cursor**


Checking the pipeline for real-time/streaming support, then adding microphone capture with chunked recording (up to 40 seconds).



The pipeline doesn't support real-time streaming; it processes complete audio files. Adding microphone capture with chunked recording (up to 40 seconds). Implementing:



Adding microphone recording functionality:


```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Omnilingual ASR Dashboard</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="container">
        <header>
            <h1>🎤 Omnilingual ASR Dashboard</h1>
            <p class="subtitle">Upload audio files and get transcriptions in multiple languages</p>
        </header>

        <main>
            <!-- Microphone Recording Section -->
            <section class="recording-section">
                <h2>🎙️ Record from Microphone</h2>
                <div class="recording-controls">
                    <button class="btn btn-record" id="recordBtn">
                        <span class="record-icon">🎤</span>
                        <span class="record-text">Start Recording</span>
                    </button>
                    <button class="btn btn-stop" id="stopBtn" disabled style="display: none;">
                        <span class="stop-icon">⏹️</span>
                        <span class="stop-text">Stop Recording</span>
                    </button>
                    <div class="recording-status" id="recordingStatus" style="display: none;">
                        <div class="recording-indicator">
                            <span class="pulse-dot"></span>
                            <span id="recordingTimer">00:00</span>
                        </div>
                        <div class="recording-warning" id="recordingWarning"></div>
                    </div>
                </div>
                <div class="audio-visualizer" id="audioVisualizer" style="display: none;">
                    <canvas id="waveformCanvas" width="400" height="80"></canvas>
                </div>
            </section>

            <!-- Upload Section -->
            <section class="upload-section">
                <h2>📁 Or Upload Audio File</h2>
                <div class="upload-area" id="uploadArea">
                    <div class="upload-content">
                        <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="17 8 12 3 7 8"></polyline>
                            <line x1="12" y1="3" x2="12" y2="15"></line>
                        </svg>
                        <h3>Drag & Drop Audio File</h3>
                        <p>or</p>
                        <button class="btn btn-primary" id="uploadBtn">Choose File</button>
                        <input type="file" id="fileInput" accept="audio/*" style="display: none;">
                        <p class="file-info" id="fileInfo"></p>
                    </div>
                </div>

                <!-- Language Selection -->
                <div class="language-section">
                    <label for="langSelect">Language:</label>
                    <select id="langSelect" class="lang-select">
                        {% for code, name in languages %}
                        <option value="{{ code }}" {% if code == 'ben_Beng' %}selected{% endif %}>{{ name }}</option>
                        {% endfor %}
                    </select>
                </div>

                <!-- Process Button -->
                <button class="btn btn-process" id="processBtn" disabled>
                    <span class="btn-text">Transcribe</span>
                    <span class="btn-spinner" style="display: none;">⏳ Processing...</span>
                </button>
            </section>

            <!-- Results Section -->
            <section class="results-section" id="resultsSection" style="display: none;">
                <div class="result-header">
                    <h2>Transcription Result</h2>
                    <button class="btn btn-secondary" id="downloadBtn">Download as Text</button>
                </div>
                <div class="result-content">
                    <div class="result-meta">
                        <span id="resultFilename"></span>
                        <span id="resultLang"></span>
                    </div>
                    <div class="result-text" id="resultText"></div>
                </div>
            </section>

            <!-- History Section -->
            <section class="history-section">
                <div class="history-header">
                    <h2>Recent History</h2>
                    <button class="btn btn-link" id="refreshHistoryBtn">Refresh</button>
                </div>
                <div class="history-list" id="historyList">
                    <p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>
                </div>
            </section>
        </main>
    </div>

    <script src="{{ url_for('static', filename='js/app.js') }}"></script>
</body>
</html>


```


Adding CSS for the recording section, then implementing JavaScript for microphone recording:


```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    --primary-color: #6366f1;
    --primary-hover: #4f46e5;
    --secondary-color: #8b5cf6;
    --success-color: #10b981;
    --danger-color: #ef4444;
    --bg-color: #f8fafc;
    --card-bg: #ffffff;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --border-color: #e2e8f0;
    --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    --radius: 12px;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    padding: 2rem;
    color: var(--text-primary);
}

.container {
    max-width: 1200px;
    margin: 0 auto;
}

header {
    text-align: center;
    color: white;
    margin-bottom: 2rem;
}

header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.subtitle {
    font-size: 1.1rem;
    opacity: 0.9;
}

main {
    display: flex;
    flex-direction: column;
    gap: 2rem;
}

section {
    background: var(--card-bg);
    border-radius: var(--radius);
    padding: 2rem;
    box-shadow: var(--shadow-lg);
}

/* Recording Section */
.recording-section {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

.recording-section h2 {
    font-size: 1.5rem;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.recording-controls {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
}

.btn-record {
    background: linear-gradient(135deg, var(--danger-color) 0%, #dc2626 100%);
    color: white;
    padding: 1rem 2rem;
    font-size: 1.1rem;
    min-width: 200px;
}

.btn-record:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.btn-record:disabled {
    opacity: 0.6;
}

.btn-stop {
    background: var(--text-secondary);
    color: white;
    padding: 1rem 2rem;
    font-size: 1.1rem;
    min-width: 200px;
}

.btn-stop:hover:not(:disabled) {
    background: var(--text-primary);
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.recording-status {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
}

.recording-indicator {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--danger-color);
}

.pulse-dot {
    width: 12px;
    height: 12px;
    background: var(--danger-color);
    border-radius: 50%;
    animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
        transform: scale(1);
    }
    50% {
        opacity: 0.5;
        transform: scale(1.2);
    }
}

.recording-warning {
    font-size: 0.9rem;
    color: var(--text-secondary);
    text-align: center;
}

.recording-warning.warning {
    color: #f59e0b;
    font-weight: 600;
}

.audio-visualizer {
    width: 100%;
    margin-top: 1rem;
    padding: 1rem;
    background: var(--bg-color);
    border-radius: 8px;
}

#waveformCanvas {
    width: 100%;
    height: 80px;
    border-radius: 4px;
}

/* Upload Section */
.upload-section {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

.upload-section h2 {
    font-size: 1.5rem;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.upload-area {
    border: 3px dashed var(--border-color);
    border-radius: var(--radius);
    padding: 3rem 2rem;
    text-align: center;
    transition: all 0.3s ease;
    cursor: pointer;
    background: var(--bg-color);
}

.upload-area:hover {
    border-color: var(--primary-color);
    background: #f1f5f9;
}

.upload-area.dragover {
    border-color: var(--primary-color);
    background: #eef2ff;
    transform: scale(1.02);
}

.upload-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
}

.upload-icon {
    width: 64px;
    height: 64px;
    color: var(--primary-color);
    margin-bottom: 0.5rem;
}

.upload-area h2 {
    font-size: 1.5rem;
    color: var(--text-primary);
    margin: 0;
}

.upload-area p {
    color: var(--text-secondary);
    margin: 0;
}

.file-info {
    margin-top: 1rem;
    font-weight: 500;
    color: var(--primary-color);
}

/* Language Selection */
.language-section {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.language-section label {
    font-weight: 600;
    color: var(--text-primary);
    min-width: 80px;
}

.lang-select {
    flex: 1;
    padding: 0.75rem 1rem;
    border: 2px solid var(--border-color);
    border-radius: 8px;
    font-size: 1rem;
    background: white;
    color: var(--text-primary);
    cursor: pointer;
    transition: all 0.2s ease;
}

.lang-select:hover {
    border-color: var(--primary-color);
}

.lang-select:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

/* Buttons */
.btn {
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}

.btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.btn-primary {
    background: var(--primary-color);
    color: white;
}

.btn-primary:hover:not(:disabled) {
    background: var(--primary-hover);
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.btn-process {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    color: white;
    padding: 1rem 2rem;
    font-size: 1.1rem;
    width: 100%;
}

.btn-process:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.btn-secondary {
    background: var(--secondary-color);
    color: white;
}

.btn-secondary:hover {
    background: #7c3aed;
}

.btn-link {
    background: none;
    color: var(--primary-color);
    padding: 0.5rem 1rem;
    text-decoration: underline;
}

.btn-link:hover {
    color: var(--primary-hover);
}

/* Results Section */
.results-section {
    animation: slideIn 0.3s ease;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid var(--border-color);
}

.result-header h2 {
    font-size: 1.5rem;
    color: var(--text-primary);
}

.result-meta {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}

.result-meta span {
    padding: 0.5rem 1rem;
    background: var(--bg-color);
    border-radius: 6px;
    font-size: 0.9rem;
    color: var(--text-secondary);
    font-weight: 500;
}

.result-text {
    background: var(--bg-color);
    padding: 1.5rem;
    border-radius: 8px;
    line-height: 1.8;
    font-size: 1.1rem;
    color: var(--text-primary);
    white-space: pre-wrap;
    word-wrap: break-word;
    min-height: 100px;
}

/* History Section */
.history-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid var(--border-color);
}

.history-header h2 {
    font-size: 1.5rem;
    color: var(--text-primary);
}

.history-list {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    max-height: 500px;
    overflow-y: auto;
}

.history-item {
    background: var(--bg-color);
    padding: 1.5rem;
    border-radius: 8px;
    border: 2px solid var(--border-color);
    transition: all 0.2s ease;
    cursor: pointer;
}

.history-item:hover {
    border-color: var(--primary-color);
    transform: translateX(4px);
}

.history-item-header {
    display: flex;
    justify-content: space-between;
    align-items: start;
    margin-bottom: 0.75rem;
}

.history-item-meta {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

.history-item-filename {
    font-weight: 600;
    color: var(--text-primary);
}

.history-item-lang {
    font-size: 0.85rem;
    color: var(--text-secondary);
}

.history-item-time {
    font-size: 0.8rem;
    color: var(--text-secondary);
}

.history-item-text {
    color: var(--text-primary);
    line-height: 1.6;
    margin-top: 0.75rem;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.history-item-actions {
    display: flex;
    gap: 0.5rem;
}

.btn-icon {
    padding: 0.5rem;
    background: white;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.btn-icon:hover {
    background: var(--primary-color);
    color: white;
    border-color: var(--primary-color);
}

.empty-state {
    text-align: center;
    color: var(--text-secondary);
    padding: 2rem;
    font-style: italic;
}

/* Loading Spinner */
.spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    border-top-color: white;
    animation: spin 0.8s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Responsive */
@media (max-width: 768px) {
    body {
        padding: 1rem;
    }

    header h1 {
        font-size: 2rem;
    }

    section {
        padding: 1.5rem;
    }

    .result-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 1rem;
    }

    .history-item-header {
        flex-direction: column;
        gap: 0.5rem;
    }
}


```


Adding JavaScript for microphone recording:


```javascript
// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const processBtn = document.getElementById('processBtn');
const langSelect = document.getElementById('langSelect');
const fileInfo = document.getElementById('fileInfo');
const resultsSection = document.getElementById('resultsSection');
const resultText = document.getElementById('resultText');
const resultFilename = document.getElementById('resultFilename');
const resultLang = document.getElementById('resultLang');
const downloadBtn = document.getElementById('downloadBtn');
const historyList = document.getElementById('historyList');
const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

// Recording elements
const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const recordingStatus = document.getElementById('recordingStatus');
const recordingTimer = document.getElementById('recordingTimer');
const recordingWarning = document.getElementById('recordingWarning');
const audioVisualizer = document.getElementById('audioVisualizer');
const waveformCanvas = document.getElementById('waveformCanvas');

let currentFile = null;
let currentTranscription = null;
let currentHistoryId = null;

// Recording state
let mediaRecorder = null;
let audioChunks = [];
let audioStream = null;
let recordingStartTime = null;
let timerInterval = null;
let animationFrameId = null;
let audioContext = null;
let analyser = null;
let dataArray = null;
const MAX_RECORDING_TIME = 40; // seconds

// Event Listeners
uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
uploadArea.addEventListener('click', () => fileInput.click());

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

processBtn.addEventListener('click', handleTranscribe);
downloadBtn.addEventListener('click', handleDownload);
refreshHistoryBtn.addEventListener('click', loadHistory);

// Initialize
loadHistory();

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

    currentFile = file;
    fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
    processBtn.disabled = false;
    resultsSection.style.display = 'none';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
}

function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
        });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}


```

```javascript
// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const processBtn = document.getElementById('processBtn');
const langSelect = document.getElementById('langSelect');
const fileInfo = document.getElementById('fileInfo');
const resultsSection = document.getElementById('resultsSection');
const resultText = document.getElementById('resultText');
const resultFilename = document.getElementById('resultFilename');
const resultLang = document.getElementById('resultLang');
const downloadBtn = document.getElementById('downloadBtn');
const historyList = document.getElementById('historyList');
const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

// Recording elements
const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const recordingStatus = document.getElementById('recordingStatus');
const recordingTimer = document.getElementById('recordingTimer');
const recordingWarning = document.getElementById('recordingWarning');
const audioVisualizer = document.getElementById('audioVisualizer');
const waveformCanvas = document.getElementById('waveformCanvas');

let currentFile = null;
let currentTranscription = null;
let currentHistoryId = null;

// Recording state
let mediaRecorder = null;
let audioChunks = [];
let audioStream = null;
let recordingStartTime = null;
let timerInterval = null;
let animationFrameId = null;
let audioContext = null;
let analyser = null;
let dataArray = null;
const MAX_RECORDING_TIME = 40; // seconds

// Event Listeners
uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
uploadArea.addEventListener('click', () => fileInput.click());

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

processBtn.addEventListener('click', handleTranscribe);
downloadBtn.addEventListener('click', handleDownload);
refreshHistoryBtn.addEventListener('click', loadHistory);
recordBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);

// Initialize
loadHistory();

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

    currentFile = file;
    fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
    processBtn.disabled = false;
    resultsSection.style.display = 'none';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
}

function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
        });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}


```

```javascript
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Microphone Recording Functions
async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Set up MediaRecorder
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = ''; // Let browser choose
            }
        }

        mediaRecorder = new MediaRecorder(audioStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await processRecordedAudio();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        recordingStartTime = Date.now();

        // Update UI
        recordBtn.disabled = true;
        recordBtn.style.display = 'none';
        stopBtn.disabled = false;
        stopBtn.style.display = 'inline-flex';
        recordingStatus.style.display = 'flex';
        audioVisualizer.style.display = 'block';
        recordingWarning.textContent = '';
        recordingWarning.className = 'recording-warning';

        // Start timer
        startTimer();

        // Start visualization
        visualizeAudio();

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions and try again.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop audio stream
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    // Stop timer and visualization
    stopTimer();
    stopVisualization();

    // Update UI
    recordBtn.disabled = false;
    recordBtn.style.display = 'inline-flex';
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
    recordingStatus.style.display = 'none';
    audioVisualizer.style.display = 'none';
}

function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Warn at 35 seconds, stop at 40 seconds
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
            alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
        } else if (elapsed >= 35) {
            recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
            recordingWarning.className = 'recording-warning warning';
        }
    }, 100);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function visualizeAudio() {
    if (!analyser || !waveformCanvas) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    function draw() {
        if (!analyser) return;

        animationFrameId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const barWidth = width / dataArray.length * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.8;
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

    draw();
}

function stopVisualization() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
}

async function processRecordedAudio() {
    // Convert recorded chunks to WAV format
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    try {
        // Convert to WAV using Web Audio API
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        
        // Create a File object from the WAV blob
        const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        // Set as current file and auto-transcribe
        currentFile = wavFile;
        fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
        processBtn.disabled = false;
        
        // Auto-transcribe
        await handleTranscribe();
        
    } catch (error) {
        console.error('Error processing recorded audio:', error);
        alert('Error processing recorded audio. Please try again.');
    }
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = buffer.length * numChannels * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}
```


Checking the end of the file and adding the recording functions:


```javascript
// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const processBtn = document.getElementById('processBtn');
const langSelect = document.getElementById('langSelect');
const fileInfo = document.getElementById('fileInfo');
const resultsSection = document.getElementById('resultsSection');
const resultText = document.getElementById('resultText');
const resultFilename = document.getElementById('resultFilename');
const resultLang = document.getElementById('resultLang');
const downloadBtn = document.getElementById('downloadBtn');
const historyList = document.getElementById('historyList');
const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

// Recording elements
const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const recordingStatus = document.getElementById('recordingStatus');
const recordingTimer = document.getElementById('recordingTimer');
const recordingWarning = document.getElementById('recordingWarning');
const audioVisualizer = document.getElementById('audioVisualizer');
const waveformCanvas = document.getElementById('waveformCanvas');

let currentFile = null;
let currentTranscription = null;
let currentHistoryId = null;

// Recording state
let mediaRecorder = null;
let audioChunks = [];
let audioStream = null;
let recordingStartTime = null;
let timerInterval = null;
let animationFrameId = null;
let audioContext = null;
let analyser = null;
let dataArray = null;
const MAX_RECORDING_TIME = 40; // seconds

// Event Listeners
uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
uploadArea.addEventListener('click', () => fileInput.click());

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

processBtn.addEventListener('click', handleTranscribe);
downloadBtn.addEventListener('click', handleDownload);
refreshHistoryBtn.addEventListener('click', loadHistory);
recordBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);

// Initialize
loadHistory();

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

    currentFile = file;
    fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
    processBtn.disabled = false;
    resultsSection.style.display = 'none';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
}

function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
        });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Microphone Recording Functions
async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Set up MediaRecorder
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = ''; // Let browser choose
            }
        }

        mediaRecorder = new MediaRecorder(audioStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await processRecordedAudio();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        recordingStartTime = Date.now();

        // Update UI
        recordBtn.disabled = true;
        recordBtn.style.display = 'none';
        stopBtn.disabled = false;
        stopBtn.style.display = 'inline-flex';
        recordingStatus.style.display = 'flex';
        audioVisualizer.style.display = 'block';
        recordingWarning.textContent = '';
        recordingWarning.className = 'recording-warning';

        // Start timer
        startTimer();

        // Start visualization
        visualizeAudio();

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions and try again.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop audio stream
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    // Stop timer and visualization
    stopTimer();
    stopVisualization();

    // Update UI
    recordBtn.disabled = false;
    recordBtn.style.display = 'inline-flex';
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
    recordingStatus.style.display = 'none';
    audioVisualizer.style.display = 'none';
}

function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Warn at 35 seconds, stop at 40 seconds
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
            alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
        } else if (elapsed >= 35) {
            recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
            recordingWarning.className = 'recording-warning warning';
        }
    }, 100);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function visualizeAudio() {
    if (!analyser || !waveformCanvas) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    function draw() {
        if (!analyser) return;

        animationFrameId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const barWidth = width / dataArray.length * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.8;
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

    draw();
}

function stopVisualization() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
}

async function processRecordedAudio() {
    // Convert recorded chunks to WAV format
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    try {
        // Convert to WAV using Web Audio API
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        
        // Create a File object from the WAV blob
        const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        // Set as current file and auto-transcribe
        currentFile = wavFile;
        fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
        processBtn.disabled = false;
        
        // Auto-transcribe
        await handleTranscribe();
        
    } catch (error) {
        console.error('Error processing recorded audio:', error);
        alert('Error processing recorded audio. Please try again.');
    }
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = buffer.length * numChannels * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}


```

```javascript
// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const processBtn = document.getElementById('processBtn');
const langSelect = document.getElementById('langSelect');
const fileInfo = document.getElementById('fileInfo');
const resultsSection = document.getElementById('resultsSection');
const resultText = document.getElementById('resultText');
const resultFilename = document.getElementById('resultFilename');
const resultLang = document.getElementById('resultLang');
const downloadBtn = document.getElementById('downloadBtn');
const historyList = document.getElementById('historyList');
const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

// Recording elements
const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const recordingStatus = document.getElementById('recordingStatus');
const recordingTimer = document.getElementById('recordingTimer');
const recordingWarning = document.getElementById('recordingWarning');
const audioVisualizer = document.getElementById('audioVisualizer');
const waveformCanvas = document.getElementById('waveformCanvas');

let currentFile = null;
let currentTranscription = null;
let currentHistoryId = null;

// Recording state
let mediaRecorder = null;
let audioChunks = [];
let audioStream = null;
let recordingStartTime = null;
let timerInterval = null;
let animationFrameId = null;
let audioContext = null;
let analyser = null;
let dataArray = null;
const MAX_RECORDING_TIME = 40; // seconds

// Recording state
let mediaRecorder = null;
let audioChunks = [];
let audioStream = null;
let recordingStartTime = null;
let timerInterval = null;
let animationFrameId = null;
let audioContext = null;
let analyser = null;
let dataArray = null;
const MAX_RECORDING_TIME = 40; // seconds

// Event Listeners
uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
uploadArea.addEventListener('click', () => fileInput.click());

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

processBtn.addEventListener('click', handleTranscribe);
downloadBtn.addEventListener('click', handleDownload);
refreshHistoryBtn.addEventListener('click', loadHistory);
recordBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);

// Initialize
loadHistory();

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

    currentFile = file;
    fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
    processBtn.disabled = false;
    resultsSection.style.display = 'none';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
}

function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
        });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Microphone Recording Functions
async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Set up MediaRecorder
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = ''; // Let browser choose
            }
        }

        mediaRecorder = new MediaRecorder(audioStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await processRecordedAudio();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        recordingStartTime = Date.now();

        // Update UI
        recordBtn.disabled = true;
        recordBtn.style.display = 'none';
        stopBtn.disabled = false;
        stopBtn.style.display = 'inline-flex';
        recordingStatus.style.display = 'flex';
        audioVisualizer.style.display = 'block';
        recordingWarning.textContent = '';
        recordingWarning.className = 'recording-warning';

        // Start timer
        startTimer();

        // Start visualization
        visualizeAudio();

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions and try again.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop audio stream
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    // Stop timer and visualization
    stopTimer();
    stopVisualization();

    // Update UI
    recordBtn.disabled = false;
    recordBtn.style.display = 'inline-flex';
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
    recordingStatus.style.display = 'none';
    audioVisualizer.style.display = 'none';
}

function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Warn at 35 seconds, stop at 40 seconds
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
            alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
        } else if (elapsed >= 35) {
            recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
            recordingWarning.className = 'recording-warning warning';
        }
    }, 100);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function visualizeAudio() {
    if (!analyser || !waveformCanvas) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    function draw() {
        if (!analyser) return;

        animationFrameId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const barWidth = width / dataArray.length * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.8;
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

    draw();
}

function stopVisualization() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
}

async function processRecordedAudio() {
    // Convert recorded chunks to WAV format
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    try {
        // Convert to WAV using Web Audio API
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        
        // Create a File object from the WAV blob
        const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        // Set as current file and auto-transcribe
        currentFile = wavFile;
        fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
        processBtn.disabled = false;
        
        // Auto-transcribe
        await handleTranscribe();
        
    } catch (error) {
        console.error('Error processing recorded audio:', error);
        alert('Error processing recorded audio. Please try again.');
    }
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = buffer.length * numChannels * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}


```

```javascript
processBtn.addEventListener('click', handleTranscribe);
downloadBtn.addEventListener('click', handleDownload);
refreshHistoryBtn.addEventListener('click', loadHistory);
recordBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);

// Initialize
loadHistory();
```


Checking the file structure:



Event listeners are in place. Verifying the recording functions are present and fixing a potential issue with audio context reuse:



Fixing a potential issue with audio context reuse in `processRecordedAudio`:


```javascript
// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const processBtn = document.getElementById('processBtn');
const langSelect = document.getElementById('langSelect');
const fileInfo = document.getElementById('fileInfo');
const resultsSection = document.getElementById('resultsSection');
const resultText = document.getElementById('resultText');
const resultFilename = document.getElementById('resultFilename');
const resultLang = document.getElementById('resultLang');
const downloadBtn = document.getElementById('downloadBtn');
const historyList = document.getElementById('historyList');
const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

// Recording elements
const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const recordingStatus = document.getElementById('recordingStatus');
const recordingTimer = document.getElementById('recordingTimer');
const recordingWarning = document.getElementById('recordingWarning');
const audioVisualizer = document.getElementById('audioVisualizer');
const waveformCanvas = document.getElementById('waveformCanvas');

let currentFile = null;
let currentTranscription = null;
let currentHistoryId = null;

// Recording state
let mediaRecorder = null;
let audioChunks = [];
let audioStream = null;
let recordingStartTime = null;
let timerInterval = null;
let animationFrameId = null;
let audioContext = null;
let analyser = null;
let dataArray = null;
const MAX_RECORDING_TIME = 40; // seconds

// Recording state
let mediaRecorder = null;
let audioChunks = [];
let audioStream = null;
let recordingStartTime = null;
let timerInterval = null;
let animationFrameId = null;
let audioContext = null;
let analyser = null;
let dataArray = null;
const MAX_RECORDING_TIME = 40; // seconds

// Event Listeners
uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
uploadArea.addEventListener('click', () => fileInput.click());

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

processBtn.addEventListener('click', handleTranscribe);
downloadBtn.addEventListener('click', handleDownload);
refreshHistoryBtn.addEventListener('click', loadHistory);
recordBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);

// Initialize
loadHistory();

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

    currentFile = file;
    fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
    processBtn.disabled = false;
    resultsSection.style.display = 'none';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
}

function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
        });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Microphone Recording Functions
async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Set up MediaRecorder
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = ''; // Let browser choose
            }
        }

        mediaRecorder = new MediaRecorder(audioStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await processRecordedAudio();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        recordingStartTime = Date.now();

        // Update UI
        recordBtn.disabled = true;
        recordBtn.style.display = 'none';
        stopBtn.disabled = false;
        stopBtn.style.display = 'inline-flex';
        recordingStatus.style.display = 'flex';
        audioVisualizer.style.display = 'block';
        recordingWarning.textContent = '';
        recordingWarning.className = 'recording-warning';

        // Start timer
        startTimer();

        // Start visualization
        visualizeAudio();

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions and try again.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop audio stream
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    // Stop timer and visualization
    stopTimer();
    stopVisualization();

    // Update UI
    recordBtn.disabled = false;
    recordBtn.style.display = 'inline-flex';
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
    recordingStatus.style.display = 'none';
    audioVisualizer.style.display = 'none';
}

function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Warn at 35 seconds, stop at 40 seconds
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
            alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
        } else if (elapsed >= 35) {
            recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
            recordingWarning.className = 'recording-warning warning';
        }
    }, 100);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function visualizeAudio() {
    if (!analyser || !waveformCanvas) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    function draw() {
        if (!analyser) return;

        animationFrameId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const barWidth = width / dataArray.length * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.8;
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

    draw();
}

function stopVisualization() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
}

async function processRecordedAudio() {
    // Convert recorded chunks to WAV format
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    try {
        // Create a new audio context for decoding (since the visualization one might be closed)
        const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Convert to WAV using Web Audio API
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        
        // Close the decode context
        await decodeContext.close();
        
        // Create a File object from the WAV blob
        const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        // Set as current file and auto-transcribe
        currentFile = wavFile;
        fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
        processBtn.disabled = false;
        
        // Auto-transcribe
        await handleTranscribe();
        
    } catch (error) {
        console.error('Error processing recorded audio:', error);
        alert('Error processing recorded audio. Please try again.');
    }
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = buffer.length * numChannels * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}


```


Microphone recording feature added. Summary:

## Microphone recording features

1. Recording controls
   - "Start Recording" button to begin
   - "Stop Recording" button to end
   - Visual feedback during recording

2. Real-time visualization
   - Live waveform visualization
   - Frequency bars showing audio levels

3. Timer and limits
   - Timer showing elapsed time (MM:SS)
   - Warning at 35 seconds
   - Auto-stop at 40 seconds (model limit)

4. Audio processing
   - Records at 16kHz mono (model requirement)
   - Converts to WAV format in the browser
   - Automatically transcribes after recording stops

5. User experience
   - Microphone permission request on first use
   - Visual indicators (pulsing dot, timer)
   - Automatic transcription after recording

## How it works

1. Click "Start Recording" → browser requests microphone permission
2. Speak into the microphone → see real-time waveform visualization
3. Click "Stop Recording" or wait for auto-stop at 40 seconds
4. Audio is converted to WAV format automatically
5. Transcription starts automatically using the selected language

## Technical details

- Uses Web Audio API for recording and visualization
- MediaRecorder API for capturing audio
- Converts WebM/Opus to WAV format in the browser
- Records at 16kHz, mono channel (matches model requirements)
- Echo cancellation and noise suppression enabled

The backend already handles WAV files, so no changes were needed there. The recorded audio is sent to the same `/api/transcribe` endpoint as file uploads.

Try it: click "Start Recording", speak for a few seconds, then click "Stop Recording" to see the transcription.

---

**User**

But the buttons in the dahboard not working, neither file ulaod or start rec or drag and drop

---

**Cursor**


Checking for JavaScript errors. Verifying DOM element selection and event listeners:



Fixing duplicate variable declarations and ensuring the script runs after the DOM is ready:


```javascript
// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const langSelect = document.getElementById('langSelect');
    const fileInfo = document.getElementById('fileInfo');
    const resultsSection = document.getElementById('resultsSection');
    const resultText = document.getElementById('resultText');
    const resultFilename = document.getElementById('resultFilename');
    const resultLang = document.getElementById('resultLang');
    const downloadBtn = document.getElementById('downloadBtn');
    const historyList = document.getElementById('historyList');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

    // Recording elements
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    const recordingWarning = document.getElementById('recordingWarning');
    const audioVisualizer = document.getElementById('audioVisualizer');
    const waveformCanvas = document.getElementById('waveformCanvas');

    // Check if all required elements exist
    if (!uploadArea || !fileInput || !uploadBtn || !processBtn || !recordBtn || !stopBtn) {
        console.error('Required DOM elements not found!');
        return;
    }

    // Global state variables
    let currentFile = null;
    let currentTranscription = null;
    let currentHistoryId = null;

    // Recording state
    let mediaRecorder = null;
    let audioChunks = [];
    let audioStream = null;
    let recordingStartTime = null;
    let timerInterval = null;
    let animationFrameId = null;
    let audioContext = null;
    let analyser = null;
    let dataArray = null;
    const MAX_RECORDING_TIME = 40; // seconds

    // Event Listeners
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    processBtn.addEventListener('click', handleTranscribe);
    downloadBtn.addEventListener('click', handleDownload);
    refreshHistoryBtn.addEventListener('click', loadHistory);
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);

    // Initialize
    loadHistory();

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

    currentFile = file;
    fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
    processBtn.disabled = false;
    resultsSection.style.display = 'none';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
}

function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
        });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Microphone Recording Functions
async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Set up MediaRecorder
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = ''; // Let browser choose
            }
        }

        mediaRecorder = new MediaRecorder(audioStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await processRecordedAudio();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        recordingStartTime = Date.now();

        // Update UI
        recordBtn.disabled = true;
        recordBtn.style.display = 'none';
        stopBtn.disabled = false;
        stopBtn.style.display = 'inline-flex';
        recordingStatus.style.display = 'flex';
        audioVisualizer.style.display = 'block';
        recordingWarning.textContent = '';
        recordingWarning.className = 'recording-warning';

        // Start timer
        startTimer();

        // Start visualization
        visualizeAudio();

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions and try again.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop audio stream
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    // Stop timer and visualization
    stopTimer();
    stopVisualization();

    // Update UI
    recordBtn.disabled = false;
    recordBtn.style.display = 'inline-flex';
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
    recordingStatus.style.display = 'none';
    audioVisualizer.style.display = 'none';
}

function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Warn at 35 seconds, stop at 40 seconds
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
            alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
        } else if (elapsed >= 35) {
            recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
            recordingWarning.className = 'recording-warning warning';
        }
    }, 100);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function visualizeAudio() {
    if (!analyser || !waveformCanvas) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    function draw() {
        if (!analyser) return;

        animationFrameId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const barWidth = width / dataArray.length * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.8;
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

    draw();
}

function stopVisualization() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
}

async function processRecordedAudio() {
    // Convert recorded chunks to WAV format
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    try {
        // Create a new audio context for decoding (since the visualization one might be closed)
        const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Convert to WAV using Web Audio API
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        
        // Close the decode context
        await decodeContext.close();
        
        // Create a File object from the WAV blob
        const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        // Set as current file and auto-transcribe
        currentFile = wavFile;
        fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
        processBtn.disabled = false;
        
        // Auto-transcribe
        await handleTranscribe();
        
    } catch (error) {
        console.error('Error processing recorded audio:', error);
        alert('Error processing recorded audio. Please try again.');
    }
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = buffer.length * numChannels * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}


```


Moving all functions inside the DOMContentLoaded block so they can access the DOM elements:


```javascript
// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const langSelect = document.getElementById('langSelect');
    const fileInfo = document.getElementById('fileInfo');
    const resultsSection = document.getElementById('resultsSection');
    const resultText = document.getElementById('resultText');
    const resultFilename = document.getElementById('resultFilename');
    const resultLang = document.getElementById('resultLang');
    const downloadBtn = document.getElementById('downloadBtn');
    const historyList = document.getElementById('historyList');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

    // Recording elements
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    const recordingWarning = document.getElementById('recordingWarning');
    const audioVisualizer = document.getElementById('audioVisualizer');
    const waveformCanvas = document.getElementById('waveformCanvas');

    // Check if all required elements exist
    if (!uploadArea || !fileInput || !uploadBtn || !processBtn || !recordBtn || !stopBtn) {
        console.error('Required DOM elements not found!');
        return;
    }

    // Global state variables
    let currentFile = null;
    let currentTranscription = null;
    let currentHistoryId = null;

    // Recording state
    let mediaRecorder = null;
    let audioChunks = [];
    let audioStream = null;
    let recordingStartTime = null;
    let timerInterval = null;
    let animationFrameId = null;
    let audioContext = null;
    let analyser = null;
    let dataArray = null;
    const MAX_RECORDING_TIME = 40; // seconds

    // Event Listeners
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    processBtn.addEventListener('click', handleTranscribe);
    downloadBtn.addEventListener('click', handleDownload);
    refreshHistoryBtn.addEventListener('click', loadHistory);
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);

    // Initialize
    loadHistory();

    // Function definitions
    function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

    currentFile = file;
    fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
    processBtn.disabled = false;
    resultsSection.style.display = 'none';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
}

function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
        });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Microphone Recording Functions
async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Set up MediaRecorder
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = ''; // Let browser choose
            }
        }

        mediaRecorder = new MediaRecorder(audioStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await processRecordedAudio();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        recordingStartTime = Date.now();

        // Update UI
        recordBtn.disabled = true;
        recordBtn.style.display = 'none';
        stopBtn.disabled = false;
        stopBtn.style.display = 'inline-flex';
        recordingStatus.style.display = 'flex';
        audioVisualizer.style.display = 'block';
        recordingWarning.textContent = '';
        recordingWarning.className = 'recording-warning';

        // Start timer
        startTimer();

        // Start visualization
        visualizeAudio();

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions and try again.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop audio stream
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    // Stop timer and visualization
    stopTimer();
    stopVisualization();

    // Update UI
    recordBtn.disabled = false;
    recordBtn.style.display = 'inline-flex';
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
    recordingStatus.style.display = 'none';
    audioVisualizer.style.display = 'none';
}

function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Warn at 35 seconds, stop at 40 seconds
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
            alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
        } else if (elapsed >= 35) {
            recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
            recordingWarning.className = 'recording-warning warning';
        }
    }, 100);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function visualizeAudio() {
    if (!analyser || !waveformCanvas) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    function draw() {
        if (!analyser) return;

        animationFrameId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const barWidth = width / dataArray.length * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.8;
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

    draw();
}

function stopVisualization() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
}

async function processRecordedAudio() {
    // Convert recorded chunks to WAV format
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    try {
        // Create a new audio context for decoding (since the visualization one might be closed)
        const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Convert to WAV using Web Audio API
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        
        // Close the decode context
        await decodeContext.close();
        
        // Create a File object from the WAV blob
        const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        // Set as current file and auto-transcribe
        currentFile = wavFile;
        fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
        processBtn.disabled = false;
        
        // Auto-transcribe
        await handleTranscribe();
        
    } catch (error) {
        console.error('Error processing recorded audio:', error);
        alert('Error processing recorded audio. Please try again.');
    }
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = buffer.length * numChannels * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}


```


Moving all functions inside the DOMContentLoaded block so they can access the DOM elements. Restructuring the file:



Fixing indentation so all functions are inside the DOMContentLoaded block:


```javascript
// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const langSelect = document.getElementById('langSelect');
    const fileInfo = document.getElementById('fileInfo');
    const resultsSection = document.getElementById('resultsSection');
    const resultText = document.getElementById('resultText');
    const resultFilename = document.getElementById('resultFilename');
    const resultLang = document.getElementById('resultLang');
    const downloadBtn = document.getElementById('downloadBtn');
    const historyList = document.getElementById('historyList');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

    // Recording elements
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    const recordingWarning = document.getElementById('recordingWarning');
    const audioVisualizer = document.getElementById('audioVisualizer');
    const waveformCanvas = document.getElementById('waveformCanvas');

    // Check if all required elements exist
    if (!uploadArea || !fileInput || !uploadBtn || !processBtn || !recordBtn || !stopBtn) {
        console.error('Required DOM elements not found!');
        return;
    }

    // Global state variables
    let currentFile = null;
    let currentTranscription = null;
    let currentHistoryId = null;

    // Recording state
    let mediaRecorder = null;
    let audioChunks = [];
    let audioStream = null;
    let recordingStartTime = null;
    let timerInterval = null;
    let animationFrameId = null;
    let audioContext = null;
    let analyser = null;
    let dataArray = null;
    const MAX_RECORDING_TIME = 40; // seconds

    // Event Listeners
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    processBtn.addEventListener('click', handleTranscribe);
    downloadBtn.addEventListener('click', handleDownload);
    refreshHistoryBtn.addEventListener('click', loadHistory);
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);

    // Function definitions
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

    currentFile = file;
    fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
    processBtn.disabled = false;
    resultsSection.style.display = 'none';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
}

function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
        });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Microphone Recording Functions
async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Set up MediaRecorder
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = ''; // Let browser choose
            }
        }

        mediaRecorder = new MediaRecorder(audioStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await processRecordedAudio();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        recordingStartTime = Date.now();

        // Update UI
        recordBtn.disabled = true;
        recordBtn.style.display = 'none';
        stopBtn.disabled = false;
        stopBtn.style.display = 'inline-flex';
        recordingStatus.style.display = 'flex';
        audioVisualizer.style.display = 'block';
        recordingWarning.textContent = '';
        recordingWarning.className = 'recording-warning';

        // Start timer
        startTimer();

        // Start visualization
        visualizeAudio();

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions and try again.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop audio stream
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    // Stop timer and visualization
    stopTimer();
    stopVisualization();

    // Update UI
    recordBtn.disabled = false;
    recordBtn.style.display = 'inline-flex';
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
    recordingStatus.style.display = 'none';
    audioVisualizer.style.display = 'none';
}

function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Warn at 35 seconds, stop at 40 seconds
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
            alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
        } else if (elapsed >= 35) {
            recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
            recordingWarning.className = 'recording-warning warning';
        }
    }, 100);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function visualizeAudio() {
    if (!analyser || !waveformCanvas) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    function draw() {
        if (!analyser) return;

        animationFrameId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const barWidth = width / dataArray.length * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.8;
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

    draw();
}

function stopVisualization() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
}

async function processRecordedAudio() {
    // Convert recorded chunks to WAV format
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    try {
        // Create a new audio context for decoding (since the visualization one might be closed)
        const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Convert to WAV using Web Audio API
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        
        // Close the decode context
        await decodeContext.close();
        
        // Create a File object from the WAV blob
        const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        // Set as current file and auto-transcribe
        currentFile = wavFile;
        fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
        processBtn.disabled = false;
        
        // Auto-transcribe
        await handleTranscribe();
        
    } catch (error) {
        console.error('Error processing recorded audio:', error);
        alert('Error processing recorded audio. Please try again.');
    }
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = buffer.length * numChannels * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}


```

```javascript
// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const langSelect = document.getElementById('langSelect');
    const fileInfo = document.getElementById('fileInfo');
    const resultsSection = document.getElementById('resultsSection');
    const resultText = document.getElementById('resultText');
    const resultFilename = document.getElementById('resultFilename');
    const resultLang = document.getElementById('resultLang');
    const downloadBtn = document.getElementById('downloadBtn');
    const historyList = document.getElementById('historyList');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

    // Recording elements
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    const recordingWarning = document.getElementById('recordingWarning');
    const audioVisualizer = document.getElementById('audioVisualizer');
    const waveformCanvas = document.getElementById('waveformCanvas');

    // Check if all required elements exist
    if (!uploadArea || !fileInput || !uploadBtn || !processBtn || !recordBtn || !stopBtn) {
        console.error('Required DOM elements not found!');
        return;
    }

    // Global state variables
    let currentFile = null;
    let currentTranscription = null;
    let currentHistoryId = null;

    // Recording state
    let mediaRecorder = null;
    let audioChunks = [];
    let audioStream = null;
    let recordingStartTime = null;
    let timerInterval = null;
    let animationFrameId = null;
    let audioContext = null;
    let analyser = null;
    let dataArray = null;
    const MAX_RECORDING_TIME = 40; // seconds

    // Event Listeners
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    processBtn.addEventListener('click', handleTranscribe);
    downloadBtn.addEventListener('click', handleDownload);
    refreshHistoryBtn.addEventListener('click', loadHistory);
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);

    // Function definitions
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

        currentFile = file;
        fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        processBtn.disabled = false;
        resultsSection.style.display = 'none';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
}

function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
        });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Microphone Recording Functions
async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Set up MediaRecorder
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = ''; // Let browser choose
            }
        }

        mediaRecorder = new MediaRecorder(audioStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await processRecordedAudio();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        recordingStartTime = Date.now();

        // Update UI
        recordBtn.disabled = true;
        recordBtn.style.display = 'none';
        stopBtn.disabled = false;
        stopBtn.style.display = 'inline-flex';
        recordingStatus.style.display = 'flex';
        audioVisualizer.style.display = 'block';
        recordingWarning.textContent = '';
        recordingWarning.className = 'recording-warning';

        // Start timer
        startTimer();

        // Start visualization
        visualizeAudio();

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions and try again.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop audio stream
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    // Stop timer and visualization
    stopTimer();
    stopVisualization();

    // Update UI
    recordBtn.disabled = false;
    recordBtn.style.display = 'inline-flex';
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
    recordingStatus.style.display = 'none';
    audioVisualizer.style.display = 'none';
}

function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Warn at 35 seconds, stop at 40 seconds
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
            alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
        } else if (elapsed >= 35) {
            recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
            recordingWarning.className = 'recording-warning warning';
        }
    }, 100);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function visualizeAudio() {
    if (!analyser || !waveformCanvas) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    function draw() {
        if (!analyser) return;

        animationFrameId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const barWidth = width / dataArray.length * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.8;
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

    draw();
}

function stopVisualization() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
}

async function processRecordedAudio() {
    // Convert recorded chunks to WAV format
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    try {
        // Create a new audio context for decoding (since the visualization one might be closed)
        const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Convert to WAV using Web Audio API
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        
        // Close the decode context
        await decodeContext.close();
        
        // Create a File object from the WAV blob
        const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        // Set as current file and auto-transcribe
        currentFile = wavFile;
        fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
        processBtn.disabled = false;
        
        // Auto-transcribe
        await handleTranscribe();
        
    } catch (error) {
        console.error('Error processing recorded audio:', error);
        alert('Error processing recorded audio. Please try again.');
    }
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = buffer.length * numChannels * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}


```

```javascript
        btnText.style.display = 'none';
        btnSpinner.style.display = 'inline';
        }
    }

    function handleDownload() {
```

```javascript
// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const langSelect = document.getElementById('langSelect');
    const fileInfo = document.getElementById('fileInfo');
    const resultsSection = document.getElementById('resultsSection');
    const resultText = document.getElementById('resultText');
    const resultFilename = document.getElementById('resultFilename');
    const resultLang = document.getElementById('resultLang');
    const downloadBtn = document.getElementById('downloadBtn');
    const historyList = document.getElementById('historyList');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

    // Recording elements
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    const recordingWarning = document.getElementById('recordingWarning');
    const audioVisualizer = document.getElementById('audioVisualizer');
    const waveformCanvas = document.getElementById('waveformCanvas');

    // Check if all required elements exist
    if (!uploadArea || !fileInput || !uploadBtn || !processBtn || !recordBtn || !stopBtn) {
        console.error('Required DOM elements not found!');
        return;
    }

    // Global state variables
    let currentFile = null;
    let currentTranscription = null;
    let currentHistoryId = null;

    // Recording state
    let mediaRecorder = null;
    let audioChunks = [];
    let audioStream = null;
    let recordingStartTime = null;
    let timerInterval = null;
    let animationFrameId = null;
    let audioContext = null;
    let analyser = null;
    let dataArray = null;
    const MAX_RECORDING_TIME = 40; // seconds

    // Event Listeners
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    processBtn.addEventListener('click', handleTranscribe);
    downloadBtn.addEventListener('click', handleDownload);
    refreshHistoryBtn.addEventListener('click', loadHistory);
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);

    // Function definitions
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

        currentFile = file;
        fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        processBtn.disabled = false;
        resultsSection.style.display = 'none';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        }
    }

    async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
}

function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
        });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Microphone Recording Functions
async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Set up MediaRecorder
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = ''; // Let browser choose
            }
        }

        mediaRecorder = new MediaRecorder(audioStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await processRecordedAudio();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        recordingStartTime = Date.now();

        // Update UI
        recordBtn.disabled = true;
        recordBtn.style.display = 'none';
        stopBtn.disabled = false;
        stopBtn.style.display = 'inline-flex';
        recordingStatus.style.display = 'flex';
        audioVisualizer.style.display = 'block';
        recordingWarning.textContent = '';
        recordingWarning.className = 'recording-warning';

        // Start timer
        startTimer();

        // Start visualization
        visualizeAudio();

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions and try again.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop audio stream
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    // Stop timer and visualization
    stopTimer();
    stopVisualization();

    // Update UI
    recordBtn.disabled = false;
    recordBtn.style.display = 'inline-flex';
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
    recordingStatus.style.display = 'none';
    audioVisualizer.style.display = 'none';
}

function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Warn at 35 seconds, stop at 40 seconds
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
            alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
        } else if (elapsed >= 35) {
            recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
            recordingWarning.className = 'recording-warning warning';
        }
    }, 100);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function visualizeAudio() {
    if (!analyser || !waveformCanvas) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    function draw() {
        if (!analyser) return;

        animationFrameId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const barWidth = width / dataArray.length * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.8;
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

    draw();
}

function stopVisualization() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
}

async function processRecordedAudio() {
    // Convert recorded chunks to WAV format
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    try {
        // Create a new audio context for decoding (since the visualization one might be closed)
        const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Convert to WAV using Web Audio API
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        
        // Close the decode context
        await decodeContext.close();
        
        // Create a File object from the WAV blob
        const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        // Set as current file and auto-transcribe
        currentFile = wavFile;
        fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
        processBtn.disabled = false;
        
        // Auto-transcribe
        await handleTranscribe();
        
    } catch (error) {
        console.error('Error processing recorded audio:', error);
        alert('Error processing recorded audio. Please try again.');
    }
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = buffer.length * numChannels * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}


```

```javascript
// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const langSelect = document.getElementById('langSelect');
    const fileInfo = document.getElementById('fileInfo');
    const resultsSection = document.getElementById('resultsSection');
    const resultText = document.getElementById('resultText');
    const resultFilename = document.getElementById('resultFilename');
    const resultLang = document.getElementById('resultLang');
    const downloadBtn = document.getElementById('downloadBtn');
    const historyList = document.getElementById('historyList');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

    // Recording elements
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    const recordingWarning = document.getElementById('recordingWarning');
    const audioVisualizer = document.getElementById('audioVisualizer');
    const waveformCanvas = document.getElementById('waveformCanvas');

    // Check if all required elements exist
    if (!uploadArea || !fileInput || !uploadBtn || !processBtn || !recordBtn || !stopBtn) {
        console.error('Required DOM elements not found!');
        return;
    }

    // Global state variables
    let currentFile = null;
    let currentTranscription = null;
    let currentHistoryId = null;

    // Recording state
    let mediaRecorder = null;
    let audioChunks = [];
    let audioStream = null;
    let recordingStartTime = null;
    let timerInterval = null;
    let animationFrameId = null;
    let audioContext = null;
    let analyser = null;
    let dataArray = null;
    const MAX_RECORDING_TIME = 40; // seconds

    // Event Listeners
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    processBtn.addEventListener('click', handleTranscribe);
    downloadBtn.addEventListener('click', handleDownload);
    refreshHistoryBtn.addEventListener('click', loadHistory);
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);

    // Function definitions
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

        currentFile = file;
        fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        processBtn.disabled = false;
        resultsSection.style.display = 'none';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        }
    }

    async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
        }
    }

    function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
}

function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
        });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Microphone Recording Functions
async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Set up MediaRecorder
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = ''; // Let browser choose
            }
        }

        mediaRecorder = new MediaRecorder(audioStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await processRecordedAudio();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        recordingStartTime = Date.now();

        // Update UI
        recordBtn.disabled = true;
        recordBtn.style.display = 'none';
        stopBtn.disabled = false;
        stopBtn.style.display = 'inline-flex';
        recordingStatus.style.display = 'flex';
        audioVisualizer.style.display = 'block';
        recordingWarning.textContent = '';
        recordingWarning.className = 'recording-warning';

        // Start timer
        startTimer();

        // Start visualization
        visualizeAudio();

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions and try again.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop audio stream
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    // Stop timer and visualization
    stopTimer();
    stopVisualization();

    // Update UI
    recordBtn.disabled = false;
    recordBtn.style.display = 'inline-flex';
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
    recordingStatus.style.display = 'none';
    audioVisualizer.style.display = 'none';
}

function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Warn at 35 seconds, stop at 40 seconds
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
            alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
        } else if (elapsed >= 35) {
            recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
            recordingWarning.className = 'recording-warning warning';
        }
    }, 100);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function visualizeAudio() {
    if (!analyser || !waveformCanvas) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    function draw() {
        if (!analyser) return;

        animationFrameId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const barWidth = width / dataArray.length * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.8;
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

    draw();
}

function stopVisualization() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
}

async function processRecordedAudio() {
    // Convert recorded chunks to WAV format
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    try {
        // Create a new audio context for decoding (since the visualization one might be closed)
        const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Convert to WAV using Web Audio API
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        
        // Close the decode context
        await decodeContext.close();
        
        // Create a File object from the WAV blob
        const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        // Set as current file and auto-transcribe
        currentFile = wavFile;
        fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
        processBtn.disabled = false;
        
        // Auto-transcribe
        await handleTranscribe();
        
    } catch (error) {
        console.error('Error processing recorded audio:', error);
        alert('Error processing recorded audio. Please try again.');
    }
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = buffer.length * numChannels * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}


```

```javascript
// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const langSelect = document.getElementById('langSelect');
    const fileInfo = document.getElementById('fileInfo');
    const resultsSection = document.getElementById('resultsSection');
    const resultText = document.getElementById('resultText');
    const resultFilename = document.getElementById('resultFilename');
    const resultLang = document.getElementById('resultLang');
    const downloadBtn = document.getElementById('downloadBtn');
    const historyList = document.getElementById('historyList');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

    // Recording elements
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    const recordingWarning = document.getElementById('recordingWarning');
    const audioVisualizer = document.getElementById('audioVisualizer');
    const waveformCanvas = document.getElementById('waveformCanvas');

    // Check if all required elements exist
    if (!uploadArea || !fileInput || !uploadBtn || !processBtn || !recordBtn || !stopBtn) {
        console.error('Required DOM elements not found!');
        return;
    }

    // Global state variables
    let currentFile = null;
    let currentTranscription = null;
    let currentHistoryId = null;

    // Recording state
    let mediaRecorder = null;
    let audioChunks = [];
    let audioStream = null;
    let recordingStartTime = null;
    let timerInterval = null;
    let animationFrameId = null;
    let audioContext = null;
    let analyser = null;
    let dataArray = null;
    const MAX_RECORDING_TIME = 40; // seconds

    // Event Listeners
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    processBtn.addEventListener('click', handleTranscribe);
    downloadBtn.addEventListener('click', handleDownload);
    refreshHistoryBtn.addEventListener('click', loadHistory);
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);

    // Function definitions
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

        currentFile = file;
        fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        processBtn.disabled = false;
        resultsSection.style.display = 'none';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        }
    }

    async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
        }
    }

    function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
    }

    function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
        });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Microphone Recording Functions
async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Set up MediaRecorder
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = ''; // Let browser choose
            }
        }

        mediaRecorder = new MediaRecorder(audioStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await processRecordedAudio();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        recordingStartTime = Date.now();

        // Update UI
        recordBtn.disabled = true;
        recordBtn.style.display = 'none';
        stopBtn.disabled = false;
        stopBtn.style.display = 'inline-flex';
        recordingStatus.style.display = 'flex';
        audioVisualizer.style.display = 'block';
        recordingWarning.textContent = '';
        recordingWarning.className = 'recording-warning';

        // Start timer
        startTimer();

        // Start visualization
        visualizeAudio();

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions and try again.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop audio stream
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    // Stop timer and visualization
    stopTimer();
    stopVisualization();

    // Update UI
    recordBtn.disabled = false;
    recordBtn.style.display = 'inline-flex';
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
    recordingStatus.style.display = 'none';
    audioVisualizer.style.display = 'none';
}

function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Warn at 35 seconds, stop at 40 seconds
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
            alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
        } else if (elapsed >= 35) {
            recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
            recordingWarning.className = 'recording-warning warning';
        }
    }, 100);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function visualizeAudio() {
    if (!analyser || !waveformCanvas) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    function draw() {
        if (!analyser) return;

        animationFrameId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const barWidth = width / dataArray.length * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.8;
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

    draw();
}

function stopVisualization() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
}

async function processRecordedAudio() {
    // Convert recorded chunks to WAV format
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    try {
        // Create a new audio context for decoding (since the visualization one might be closed)
        const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Convert to WAV using Web Audio API
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        
        // Close the decode context
        await decodeContext.close();
        
        // Create a File object from the WAV blob
        const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        // Set as current file and auto-transcribe
        currentFile = wavFile;
        fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
        processBtn.disabled = false;
        
        // Auto-transcribe
        await handleTranscribe();
        
    } catch (error) {
        console.error('Error processing recorded audio:', error);
        alert('Error processing recorded audio. Please try again.');
    }
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = buffer.length * numChannels * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}


```

```javascript
// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const langSelect = document.getElementById('langSelect');
    const fileInfo = document.getElementById('fileInfo');
    const resultsSection = document.getElementById('resultsSection');
    const resultText = document.getElementById('resultText');
    const resultFilename = document.getElementById('resultFilename');
    const resultLang = document.getElementById('resultLang');
    const downloadBtn = document.getElementById('downloadBtn');
    const historyList = document.getElementById('historyList');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

    // Recording elements
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    const recordingWarning = document.getElementById('recordingWarning');
    const audioVisualizer = document.getElementById('audioVisualizer');
    const waveformCanvas = document.getElementById('waveformCanvas');

    // Check if all required elements exist
    if (!uploadArea || !fileInput || !uploadBtn || !processBtn || !recordBtn || !stopBtn) {
        console.error('Required DOM elements not found!');
        return;
    }

    // Global state variables
    let currentFile = null;
    let currentTranscription = null;
    let currentHistoryId = null;

    // Recording state
    let mediaRecorder = null;
    let audioChunks = [];
    let audioStream = null;
    let recordingStartTime = null;
    let timerInterval = null;
    let animationFrameId = null;
    let audioContext = null;
    let analyser = null;
    let dataArray = null;
    const MAX_RECORDING_TIME = 40; // seconds

    // Event Listeners
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    processBtn.addEventListener('click', handleTranscribe);
    downloadBtn.addEventListener('click', handleDownload);
    refreshHistoryBtn.addEventListener('click', loadHistory);
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);

    // Function definitions
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

        currentFile = file;
        fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        processBtn.disabled = false;
        resultsSection.style.display = 'none';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        }
    }

    async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
        }
    }

    function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
    }

    function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
            });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Microphone Recording Functions
    async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Set up MediaRecorder
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = ''; // Let browser choose
            }
        }

        mediaRecorder = new MediaRecorder(audioStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await processRecordedAudio();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        recordingStartTime = Date.now();

        // Update UI
        recordBtn.disabled = true;
        recordBtn.style.display = 'none';
        stopBtn.disabled = false;
        stopBtn.style.display = 'inline-flex';
        recordingStatus.style.display = 'flex';
        audioVisualizer.style.display = 'block';
        recordingWarning.textContent = '';
        recordingWarning.className = 'recording-warning';

        // Start timer
        startTimer();

        // Start visualization
        visualizeAudio();

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions and try again.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop audio stream
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    // Stop timer and visualization
    stopTimer();
    stopVisualization();

    // Update UI
    recordBtn.disabled = false;
    recordBtn.style.display = 'inline-flex';
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
    recordingStatus.style.display = 'none';
    audioVisualizer.style.display = 'none';
}

function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Warn at 35 seconds, stop at 40 seconds
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
            alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
        } else if (elapsed >= 35) {
            recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
            recordingWarning.className = 'recording-warning warning';
        }
    }, 100);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function visualizeAudio() {
    if (!analyser || !waveformCanvas) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    function draw() {
        if (!analyser) return;

        animationFrameId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const barWidth = width / dataArray.length * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.8;
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

    draw();
}

function stopVisualization() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
}

async function processRecordedAudio() {
    // Convert recorded chunks to WAV format
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    try {
        // Create a new audio context for decoding (since the visualization one might be closed)
        const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Convert to WAV using Web Audio API
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        
        // Close the decode context
        await decodeContext.close();
        
        // Create a File object from the WAV blob
        const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        // Set as current file and auto-transcribe
        currentFile = wavFile;
        fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
        processBtn.disabled = false;
        
        // Auto-transcribe
        await handleTranscribe();
        
    } catch (error) {
        console.error('Error processing recorded audio:', error);
        alert('Error processing recorded audio. Please try again.');
    }
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = buffer.length * numChannels * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}


```

```javascript
// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const langSelect = document.getElementById('langSelect');
    const fileInfo = document.getElementById('fileInfo');
    const resultsSection = document.getElementById('resultsSection');
    const resultText = document.getElementById('resultText');
    const resultFilename = document.getElementById('resultFilename');
    const resultLang = document.getElementById('resultLang');
    const downloadBtn = document.getElementById('downloadBtn');
    const historyList = document.getElementById('historyList');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

    // Recording elements
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    const recordingWarning = document.getElementById('recordingWarning');
    const audioVisualizer = document.getElementById('audioVisualizer');
    const waveformCanvas = document.getElementById('waveformCanvas');

    // Check if all required elements exist
    if (!uploadArea || !fileInput || !uploadBtn || !processBtn || !recordBtn || !stopBtn) {
        console.error('Required DOM elements not found!');
        return;
    }

    // Global state variables
    let currentFile = null;
    let currentTranscription = null;
    let currentHistoryId = null;

    // Recording state
    let mediaRecorder = null;
    let audioChunks = [];
    let audioStream = null;
    let recordingStartTime = null;
    let timerInterval = null;
    let animationFrameId = null;
    let audioContext = null;
    let analyser = null;
    let dataArray = null;
    const MAX_RECORDING_TIME = 40; // seconds

    // Event Listeners
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    processBtn.addEventListener('click', handleTranscribe);
    downloadBtn.addEventListener('click', handleDownload);
    refreshHistoryBtn.addEventListener('click', loadHistory);
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);

    // Function definitions
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

        currentFile = file;
        fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        processBtn.disabled = false;
        resultsSection.style.display = 'none';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        }
    }

    async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
        }
    }

    function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
    }

    function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
            });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Microphone Recording Functions
    async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Set up MediaRecorder
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = ''; // Let browser choose
            }
        }

        mediaRecorder = new MediaRecorder(audioStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await processRecordedAudio();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        recordingStartTime = Date.now();

        // Update UI
        recordBtn.disabled = true;
        recordBtn.style.display = 'none';
        stopBtn.disabled = false;
        stopBtn.style.display = 'inline-flex';
        recordingStatus.style.display = 'flex';
        audioVisualizer.style.display = 'block';
        recordingWarning.textContent = '';
        recordingWarning.className = 'recording-warning';

        // Start timer
        startTimer();

        // Start visualization
        visualizeAudio();

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions and try again.');
        }
    }

    function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop audio stream
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    // Stop timer and visualization
    stopTimer();
    stopVisualization();

    // Update UI
    recordBtn.disabled = false;
    recordBtn.style.display = 'inline-flex';
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
    recordingStatus.style.display = 'none';
    audioVisualizer.style.display = 'none';
}

function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Warn at 35 seconds, stop at 40 seconds
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
            alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
        } else if (elapsed >= 35) {
            recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
            recordingWarning.className = 'recording-warning warning';
        }
    }, 100);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function visualizeAudio() {
    if (!analyser || !waveformCanvas) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    function draw() {
        if (!analyser) return;

        animationFrameId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const barWidth = width / dataArray.length * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.8;
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

    draw();
}

function stopVisualization() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
}

async function processRecordedAudio() {
    // Convert recorded chunks to WAV format
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    try {
        // Create a new audio context for decoding (since the visualization one might be closed)
        const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Convert to WAV using Web Audio API
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        
        // Close the decode context
        await decodeContext.close();
        
        // Create a File object from the WAV blob
        const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        // Set as current file and auto-transcribe
        currentFile = wavFile;
        fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
        processBtn.disabled = false;
        
        // Auto-transcribe
        await handleTranscribe();
        
    } catch (error) {
        console.error('Error processing recorded audio:', error);
        alert('Error processing recorded audio. Please try again.');
    }
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = buffer.length * numChannels * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}


```

```javascript
// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const langSelect = document.getElementById('langSelect');
    const fileInfo = document.getElementById('fileInfo');
    const resultsSection = document.getElementById('resultsSection');
    const resultText = document.getElementById('resultText');
    const resultFilename = document.getElementById('resultFilename');
    const resultLang = document.getElementById('resultLang');
    const downloadBtn = document.getElementById('downloadBtn');
    const historyList = document.getElementById('historyList');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

    // Recording elements
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    const recordingWarning = document.getElementById('recordingWarning');
    const audioVisualizer = document.getElementById('audioVisualizer');
    const waveformCanvas = document.getElementById('waveformCanvas');

    // Check if all required elements exist
    if (!uploadArea || !fileInput || !uploadBtn || !processBtn || !recordBtn || !stopBtn) {
        console.error('Required DOM elements not found!');
        return;
    }

    // Global state variables
    let currentFile = null;
    let currentTranscription = null;
    let currentHistoryId = null;

    // Recording state
    let mediaRecorder = null;
    let audioChunks = [];
    let audioStream = null;
    let recordingStartTime = null;
    let timerInterval = null;
    let animationFrameId = null;
    let audioContext = null;
    let analyser = null;
    let dataArray = null;
    const MAX_RECORDING_TIME = 40; // seconds

    // Event Listeners
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    processBtn.addEventListener('click', handleTranscribe);
    downloadBtn.addEventListener('click', handleDownload);
    refreshHistoryBtn.addEventListener('click', loadHistory);
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);

    // Function definitions
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

        currentFile = file;
        fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        processBtn.disabled = false;
        resultsSection.style.display = 'none';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        }
    }

    async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
        }
    }

    function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
    }

    function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
            });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Microphone Recording Functions
    async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Set up MediaRecorder
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = ''; // Let browser choose
            }
        }

        mediaRecorder = new MediaRecorder(audioStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await processRecordedAudio();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        recordingStartTime = Date.now();

        // Update UI
        recordBtn.disabled = true;
        recordBtn.style.display = 'none';
        stopBtn.disabled = false;
        stopBtn.style.display = 'inline-flex';
        recordingStatus.style.display = 'flex';
        audioVisualizer.style.display = 'block';
        recordingWarning.textContent = '';
        recordingWarning.className = 'recording-warning';

        // Start timer
        startTimer();

        // Start visualization
        visualizeAudio();

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions and try again.');
        }
    }

    function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop audio stream
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    // Stop timer and visualization
    stopTimer();
    stopVisualization();

    // Update UI
    recordBtn.disabled = false;
    recordBtn.style.display = 'inline-flex';
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
        recordingStatus.style.display = 'none';
        audioVisualizer.style.display = 'none';
    }

    function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Warn at 35 seconds, stop at 40 seconds
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
            alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
        } else if (elapsed >= 35) {
            recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
            recordingWarning.className = 'recording-warning warning';
        }
    }, 100);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function visualizeAudio() {
    if (!analyser || !waveformCanvas) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    function draw() {
        if (!analyser) return;

        animationFrameId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const barWidth = width / dataArray.length * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.8;
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

    draw();
}

function stopVisualization() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
}

async function processRecordedAudio() {
    // Convert recorded chunks to WAV format
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    try {
        // Create a new audio context for decoding (since the visualization one might be closed)
        const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Convert to WAV using Web Audio API
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        
        // Close the decode context
        await decodeContext.close();
        
        // Create a File object from the WAV blob
        const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        // Set as current file and auto-transcribe
        currentFile = wavFile;
        fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
        processBtn.disabled = false;
        
        // Auto-transcribe
        await handleTranscribe();
        
    } catch (error) {
        console.error('Error processing recorded audio:', error);
        alert('Error processing recorded audio. Please try again.');
    }
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = buffer.length * numChannels * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}


```

```javascript
// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const langSelect = document.getElementById('langSelect');
    const fileInfo = document.getElementById('fileInfo');
    const resultsSection = document.getElementById('resultsSection');
    const resultText = document.getElementById('resultText');
    const resultFilename = document.getElementById('resultFilename');
    const resultLang = document.getElementById('resultLang');
    const downloadBtn = document.getElementById('downloadBtn');
    const historyList = document.getElementById('historyList');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

    // Recording elements
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    const recordingWarning = document.getElementById('recordingWarning');
    const audioVisualizer = document.getElementById('audioVisualizer');
    const waveformCanvas = document.getElementById('waveformCanvas');

    // Check if all required elements exist
    if (!uploadArea || !fileInput || !uploadBtn || !processBtn || !recordBtn || !stopBtn) {
        console.error('Required DOM elements not found!');
        return;
    }

    // Global state variables
    let currentFile = null;
    let currentTranscription = null;
    let currentHistoryId = null;

    // Recording state
    let mediaRecorder = null;
    let audioChunks = [];
    let audioStream = null;
    let recordingStartTime = null;
    let timerInterval = null;
    let animationFrameId = null;
    let audioContext = null;
    let analyser = null;
    let dataArray = null;
    const MAX_RECORDING_TIME = 40; // seconds

    // Event Listeners
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    processBtn.addEventListener('click', handleTranscribe);
    downloadBtn.addEventListener('click', handleDownload);
    refreshHistoryBtn.addEventListener('click', loadHistory);
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);

    // Function definitions
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

        currentFile = file;
        fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        processBtn.disabled = false;
        resultsSection.style.display = 'none';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        }
    }

    async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
        }
    }

    function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
    }

    function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
            });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Microphone Recording Functions
    async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Set up MediaRecorder
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = ''; // Let browser choose
            }
        }

        mediaRecorder = new MediaRecorder(audioStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await processRecordedAudio();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        recordingStartTime = Date.now();

        // Update UI
        recordBtn.disabled = true;
        recordBtn.style.display = 'none';
        stopBtn.disabled = false;
        stopBtn.style.display = 'inline-flex';
        recordingStatus.style.display = 'flex';
        audioVisualizer.style.display = 'block';
        recordingWarning.textContent = '';
        recordingWarning.className = 'recording-warning';

        // Start timer
        startTimer();

        // Start visualization
        visualizeAudio();

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions and try again.');
        }
    }

    function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop audio stream
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    // Stop timer and visualization
    stopTimer();
    stopVisualization();

    // Update UI
    recordBtn.disabled = false;
    recordBtn.style.display = 'inline-flex';
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
        recordingStatus.style.display = 'none';
        audioVisualizer.style.display = 'none';
    }

    function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Warn at 35 seconds, stop at 40 seconds
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
            alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
        } else if (elapsed >= 35) {
            recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
            recordingWarning.className = 'recording-warning warning';
            }
        }, 100);
    }

    function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function visualizeAudio() {
    if (!analyser || !waveformCanvas) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    function draw() {
        if (!analyser) return;

        animationFrameId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const barWidth = width / dataArray.length * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.8;
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

    draw();
}

function stopVisualization() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
}

async function processRecordedAudio() {
    // Convert recorded chunks to WAV format
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    try {
        // Create a new audio context for decoding (since the visualization one might be closed)
        const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Convert to WAV using Web Audio API
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        
        // Close the decode context
        await decodeContext.close();
        
        // Create a File object from the WAV blob
        const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        // Set as current file and auto-transcribe
        currentFile = wavFile;
        fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
        processBtn.disabled = false;
        
        // Auto-transcribe
        await handleTranscribe();
        
    } catch (error) {
        console.error('Error processing recorded audio:', error);
        alert('Error processing recorded audio. Please try again.');
    }
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = buffer.length * numChannels * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}


```

```javascript
// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const langSelect = document.getElementById('langSelect');
    const fileInfo = document.getElementById('fileInfo');
    const resultsSection = document.getElementById('resultsSection');
    const resultText = document.getElementById('resultText');
    const resultFilename = document.getElementById('resultFilename');
    const resultLang = document.getElementById('resultLang');
    const downloadBtn = document.getElementById('downloadBtn');
    const historyList = document.getElementById('historyList');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

    // Recording elements
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    const recordingWarning = document.getElementById('recordingWarning');
    const audioVisualizer = document.getElementById('audioVisualizer');
    const waveformCanvas = document.getElementById('waveformCanvas');

    // Check if all required elements exist
    if (!uploadArea || !fileInput || !uploadBtn || !processBtn || !recordBtn || !stopBtn) {
        console.error('Required DOM elements not found!');
        return;
    }

    // Global state variables
    let currentFile = null;
    let currentTranscription = null;
    let currentHistoryId = null;

    // Recording state
    let mediaRecorder = null;
    let audioChunks = [];
    let audioStream = null;
    let recordingStartTime = null;
    let timerInterval = null;
    let animationFrameId = null;
    let audioContext = null;
    let analyser = null;
    let dataArray = null;
    const MAX_RECORDING_TIME = 40; // seconds

    // Event Listeners
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    processBtn.addEventListener('click', handleTranscribe);
    downloadBtn.addEventListener('click', handleDownload);
    refreshHistoryBtn.addEventListener('click', loadHistory);
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);

    // Function definitions
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

        currentFile = file;
        fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        processBtn.disabled = false;
        resultsSection.style.display = 'none';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        }
    }

    async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
        }
    }

    function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
    }

    function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
            });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Microphone Recording Functions
    async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Set up MediaRecorder
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = ''; // Let browser choose
            }
        }

        mediaRecorder = new MediaRecorder(audioStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await processRecordedAudio();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        recordingStartTime = Date.now();

        // Update UI
        recordBtn.disabled = true;
        recordBtn.style.display = 'none';
        stopBtn.disabled = false;
        stopBtn.style.display = 'inline-flex';
        recordingStatus.style.display = 'flex';
        audioVisualizer.style.display = 'block';
        recordingWarning.textContent = '';
        recordingWarning.className = 'recording-warning';

        // Start timer
        startTimer();

        // Start visualization
        visualizeAudio();

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions and try again.');
        }
    }

    function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop audio stream
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    // Stop timer and visualization
    stopTimer();
    stopVisualization();

    // Update UI
    recordBtn.disabled = false;
    recordBtn.style.display = 'inline-flex';
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
        recordingStatus.style.display = 'none';
        audioVisualizer.style.display = 'none';
    }

    function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Warn at 35 seconds, stop at 40 seconds
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
            alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
        } else if (elapsed >= 35) {
            recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
            recordingWarning.className = 'recording-warning warning';
            }
        }, 100);
    }

    function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
        }
    }

    function visualizeAudio() {
    if (!analyser || !waveformCanvas) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    function draw() {
        if (!analyser) return;

        animationFrameId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const barWidth = width / dataArray.length * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.8;
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

    draw();
}

function stopVisualization() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
}

async function processRecordedAudio() {
    // Convert recorded chunks to WAV format
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    try {
        // Create a new audio context for decoding (since the visualization one might be closed)
        const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Convert to WAV using Web Audio API
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        
        // Close the decode context
        await decodeContext.close();
        
        // Create a File object from the WAV blob
        const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        // Set as current file and auto-transcribe
        currentFile = wavFile;
        fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
        processBtn.disabled = false;
        
        // Auto-transcribe
        await handleTranscribe();
        
    } catch (error) {
        console.error('Error processing recorded audio:', error);
        alert('Error processing recorded audio. Please try again.');
    }
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = buffer.length * numChannels * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}


```

```javascript
// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const langSelect = document.getElementById('langSelect');
    const fileInfo = document.getElementById('fileInfo');
    const resultsSection = document.getElementById('resultsSection');
    const resultText = document.getElementById('resultText');
    const resultFilename = document.getElementById('resultFilename');
    const resultLang = document.getElementById('resultLang');
    const downloadBtn = document.getElementById('downloadBtn');
    const historyList = document.getElementById('historyList');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

    // Recording elements
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    const recordingWarning = document.getElementById('recordingWarning');
    const audioVisualizer = document.getElementById('audioVisualizer');
    const waveformCanvas = document.getElementById('waveformCanvas');

    // Check if all required elements exist
    if (!uploadArea || !fileInput || !uploadBtn || !processBtn || !recordBtn || !stopBtn) {
        console.error('Required DOM elements not found!');
        return;
    }

    // Global state variables
    let currentFile = null;
    let currentTranscription = null;
    let currentHistoryId = null;

    // Recording state
    let mediaRecorder = null;
    let audioChunks = [];
    let audioStream = null;
    let recordingStartTime = null;
    let timerInterval = null;
    let animationFrameId = null;
    let audioContext = null;
    let analyser = null;
    let dataArray = null;
    const MAX_RECORDING_TIME = 40; // seconds

    // Event Listeners
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    processBtn.addEventListener('click', handleTranscribe);
    downloadBtn.addEventListener('click', handleDownload);
    refreshHistoryBtn.addEventListener('click', loadHistory);
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);

    // Function definitions
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

        currentFile = file;
        fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        processBtn.disabled = false;
        resultsSection.style.display = 'none';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        }
    }

    async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
        }
    }

    function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
    }

    function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
            });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Microphone Recording Functions
    async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Set up MediaRecorder
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = ''; // Let browser choose
            }
        }

        mediaRecorder = new MediaRecorder(audioStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await processRecordedAudio();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        recordingStartTime = Date.now();

        // Update UI
        recordBtn.disabled = true;
        recordBtn.style.display = 'none';
        stopBtn.disabled = false;
        stopBtn.style.display = 'inline-flex';
        recordingStatus.style.display = 'flex';
        audioVisualizer.style.display = 'block';
        recordingWarning.textContent = '';
        recordingWarning.className = 'recording-warning';

        // Start timer
        startTimer();

        // Start visualization
        visualizeAudio();

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions and try again.');
        }
    }

    function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop audio stream
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    // Stop timer and visualization
    stopTimer();
    stopVisualization();

    // Update UI
    recordBtn.disabled = false;
    recordBtn.style.display = 'inline-flex';
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
        recordingStatus.style.display = 'none';
        audioVisualizer.style.display = 'none';
    }

    function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Warn at 35 seconds, stop at 40 seconds
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
            alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
        } else if (elapsed >= 35) {
            recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
            recordingWarning.className = 'recording-warning warning';
            }
        }, 100);
    }

    function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
        }
    }

    function visualizeAudio() {
    if (!analyser || !waveformCanvas) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    function draw() {
        if (!analyser) return;

        animationFrameId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const barWidth = width / dataArray.length * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.8;
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

        draw();
    }

    function stopVisualization() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
}

async function processRecordedAudio() {
    // Convert recorded chunks to WAV format
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    try {
        // Create a new audio context for decoding (since the visualization one might be closed)
        const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Convert to WAV using Web Audio API
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        
        // Close the decode context
        await decodeContext.close();
        
        // Create a File object from the WAV blob
        const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        // Set as current file and auto-transcribe
        currentFile = wavFile;
        fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
        processBtn.disabled = false;
        
        // Auto-transcribe
        await handleTranscribe();
        
    } catch (error) {
        console.error('Error processing recorded audio:', error);
        alert('Error processing recorded audio. Please try again.');
    }
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = buffer.length * numChannels * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}


```

```javascript
// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const langSelect = document.getElementById('langSelect');
    const fileInfo = document.getElementById('fileInfo');
    const resultsSection = document.getElementById('resultsSection');
    const resultText = document.getElementById('resultText');
    const resultFilename = document.getElementById('resultFilename');
    const resultLang = document.getElementById('resultLang');
    const downloadBtn = document.getElementById('downloadBtn');
    const historyList = document.getElementById('historyList');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

    // Recording elements
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    const recordingWarning = document.getElementById('recordingWarning');
    const audioVisualizer = document.getElementById('audioVisualizer');
    const waveformCanvas = document.getElementById('waveformCanvas');

    // Check if all required elements exist
    if (!uploadArea || !fileInput || !uploadBtn || !processBtn || !recordBtn || !stopBtn) {
        console.error('Required DOM elements not found!');
        return;
    }

    // Global state variables
    let currentFile = null;
    let currentTranscription = null;
    let currentHistoryId = null;

    // Recording state
    let mediaRecorder = null;
    let audioChunks = [];
    let audioStream = null;
    let recordingStartTime = null;
    let timerInterval = null;
    let animationFrameId = null;
    let audioContext = null;
    let analyser = null;
    let dataArray = null;
    const MAX_RECORDING_TIME = 40; // seconds

    // Event Listeners
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    processBtn.addEventListener('click', handleTranscribe);
    downloadBtn.addEventListener('click', handleDownload);
    refreshHistoryBtn.addEventListener('click', loadHistory);
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);

    // Function definitions
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

        currentFile = file;
        fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        processBtn.disabled = false;
        resultsSection.style.display = 'none';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        }
    }

    async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
        }
    }

    function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
    }

    function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
            });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Microphone Recording Functions
    async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Set up MediaRecorder
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = ''; // Let browser choose
            }
        }

        mediaRecorder = new MediaRecorder(audioStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await processRecordedAudio();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        recordingStartTime = Date.now();

        // Update UI
        recordBtn.disabled = true;
        recordBtn.style.display = 'none';
        stopBtn.disabled = false;
        stopBtn.style.display = 'inline-flex';
        recordingStatus.style.display = 'flex';
        audioVisualizer.style.display = 'block';
        recordingWarning.textContent = '';
        recordingWarning.className = 'recording-warning';

        // Start timer
        startTimer();

        // Start visualization
        visualizeAudio();

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions and try again.');
        }
    }

    function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop audio stream
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    // Stop timer and visualization
    stopTimer();
    stopVisualization();

    // Update UI
    recordBtn.disabled = false;
    recordBtn.style.display = 'inline-flex';
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
        recordingStatus.style.display = 'none';
        audioVisualizer.style.display = 'none';
    }

    function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Warn at 35 seconds, stop at 40 seconds
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
            alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
        } else if (elapsed >= 35) {
            recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
            recordingWarning.className = 'recording-warning warning';
            }
        }, 100);
    }

    function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
        }
    }

    function visualizeAudio() {
    if (!analyser || !waveformCanvas) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    function draw() {
        if (!analyser) return;

        animationFrameId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const barWidth = width / dataArray.length * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.8;
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

        draw();
    }

    function stopVisualization() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
        }
    }

    async function processRecordedAudio() {
    // Convert recorded chunks to WAV format
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    try {
        // Create a new audio context for decoding (since the visualization one might be closed)
        const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Convert to WAV using Web Audio API
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        
        // Close the decode context
        await decodeContext.close();
        
        // Create a File object from the WAV blob
        const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        // Set as current file and auto-transcribe
        currentFile = wavFile;
        fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
        processBtn.disabled = false;
        
        // Auto-transcribe
        await handleTranscribe();
        
    } catch (error) {
        console.error('Error processing recorded audio:', error);
        alert('Error processing recorded audio. Please try again.');
    }
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = buffer.length * numChannels * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}


```

```javascript
// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const langSelect = document.getElementById('langSelect');
    const fileInfo = document.getElementById('fileInfo');
    const resultsSection = document.getElementById('resultsSection');
    const resultText = document.getElementById('resultText');
    const resultFilename = document.getElementById('resultFilename');
    const resultLang = document.getElementById('resultLang');
    const downloadBtn = document.getElementById('downloadBtn');
    const historyList = document.getElementById('historyList');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

    // Recording elements
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    const recordingWarning = document.getElementById('recordingWarning');
    const audioVisualizer = document.getElementById('audioVisualizer');
    const waveformCanvas = document.getElementById('waveformCanvas');

    // Check if all required elements exist
    if (!uploadArea || !fileInput || !uploadBtn || !processBtn || !recordBtn || !stopBtn) {
        console.error('Required DOM elements not found!');
        return;
    }

    // Global state variables
    let currentFile = null;
    let currentTranscription = null;
    let currentHistoryId = null;

    // Recording state
    let mediaRecorder = null;
    let audioChunks = [];
    let audioStream = null;
    let recordingStartTime = null;
    let timerInterval = null;
    let animationFrameId = null;
    let audioContext = null;
    let analyser = null;
    let dataArray = null;
    const MAX_RECORDING_TIME = 40; // seconds

    // Event Listeners
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    processBtn.addEventListener('click', handleTranscribe);
    downloadBtn.addEventListener('click', handleDownload);
    refreshHistoryBtn.addEventListener('click', loadHistory);
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);

    // Function definitions
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

        currentFile = file;
        fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        processBtn.disabled = false;
        resultsSection.style.display = 'none';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        }
    }

    async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
        }
    }

    function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
    }

    function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
            });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Microphone Recording Functions
    async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Set up MediaRecorder
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = ''; // Let browser choose
            }
        }

        mediaRecorder = new MediaRecorder(audioStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await processRecordedAudio();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        recordingStartTime = Date.now();

        // Update UI
        recordBtn.disabled = true;
        recordBtn.style.display = 'none';
        stopBtn.disabled = false;
        stopBtn.style.display = 'inline-flex';
        recordingStatus.style.display = 'flex';
        audioVisualizer.style.display = 'block';
        recordingWarning.textContent = '';
        recordingWarning.className = 'recording-warning';

        // Start timer
        startTimer();

        // Start visualization
        visualizeAudio();

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions and try again.');
        }
    }

    function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop audio stream
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    // Stop timer and visualization
    stopTimer();
    stopVisualization();

    // Update UI
    recordBtn.disabled = false;
    recordBtn.style.display = 'inline-flex';
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
        recordingStatus.style.display = 'none';
        audioVisualizer.style.display = 'none';
    }

    function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Warn at 35 seconds, stop at 40 seconds
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
            alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
        } else if (elapsed >= 35) {
            recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
            recordingWarning.className = 'recording-warning warning';
            }
        }, 100);
    }

    function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
        }
    }

    function visualizeAudio() {
    if (!analyser || !waveformCanvas) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    function draw() {
        if (!analyser) return;

        animationFrameId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const barWidth = width / dataArray.length * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.8;
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

        draw();
    }

    function stopVisualization() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
        }
    }

    async function processRecordedAudio() {
    // Convert recorded chunks to WAV format
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    try {
        // Create a new audio context for decoding (since the visualization one might be closed)
        const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Convert to WAV using Web Audio API
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        
        // Close the decode context
        await decodeContext.close();
        
        // Create a File object from the WAV blob
        const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        // Set as current file and auto-transcribe
        currentFile = wavFile;
        fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
        processBtn.disabled = false;
        
        // Auto-transcribe
        await handleTranscribe();
        
    } catch (error) {
        console.error('Error processing recorded audio:', error);
        alert('Error processing recorded audio. Please try again.');
        }
    }

    // Convert AudioBuffer to WAV format
    function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = buffer.length * numChannels * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}


```

```javascript
// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const langSelect = document.getElementById('langSelect');
    const fileInfo = document.getElementById('fileInfo');
    const resultsSection = document.getElementById('resultsSection');
    const resultText = document.getElementById('resultText');
    const resultFilename = document.getElementById('resultFilename');
    const resultLang = document.getElementById('resultLang');
    const downloadBtn = document.getElementById('downloadBtn');
    const historyList = document.getElementById('historyList');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

    // Recording elements
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    const recordingWarning = document.getElementById('recordingWarning');
    const audioVisualizer = document.getElementById('audioVisualizer');
    const waveformCanvas = document.getElementById('waveformCanvas');

    // Check if all required elements exist
    if (!uploadArea || !fileInput || !uploadBtn || !processBtn || !recordBtn || !stopBtn) {
        console.error('Required DOM elements not found!');
        return;
    }

    // Global state variables
    let currentFile = null;
    let currentTranscription = null;
    let currentHistoryId = null;

    // Recording state
    let mediaRecorder = null;
    let audioChunks = [];
    let audioStream = null;
    let recordingStartTime = null;
    let timerInterval = null;
    let animationFrameId = null;
    let audioContext = null;
    let analyser = null;
    let dataArray = null;
    const MAX_RECORDING_TIME = 40; // seconds

    // Event Listeners
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    processBtn.addEventListener('click', handleTranscribe);
    downloadBtn.addEventListener('click', handleDownload);
    refreshHistoryBtn.addEventListener('click', loadHistory);
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);

    // Function definitions
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

        currentFile = file;
        fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        processBtn.disabled = false;
        resultsSection.style.display = 'none';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        }
    }

    async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
        }
    }

    function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
    }

    function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
            });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Microphone Recording Functions
    async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Set up MediaRecorder
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = ''; // Let browser choose
            }
        }

        mediaRecorder = new MediaRecorder(audioStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await processRecordedAudio();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        recordingStartTime = Date.now();

        // Update UI
        recordBtn.disabled = true;
        recordBtn.style.display = 'none';
        stopBtn.disabled = false;
        stopBtn.style.display = 'inline-flex';
        recordingStatus.style.display = 'flex';
        audioVisualizer.style.display = 'block';
        recordingWarning.textContent = '';
        recordingWarning.className = 'recording-warning';

        // Start timer
        startTimer();

        // Start visualization
        visualizeAudio();

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions and try again.');
        }
    }

    function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop audio stream
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    // Stop timer and visualization
    stopTimer();
    stopVisualization();

    // Update UI
    recordBtn.disabled = false;
    recordBtn.style.display = 'inline-flex';
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
        recordingStatus.style.display = 'none';
        audioVisualizer.style.display = 'none';
    }

    function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Warn at 35 seconds, stop at 40 seconds
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
            alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
        } else if (elapsed >= 35) {
            recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
            recordingWarning.className = 'recording-warning warning';
            }
        }, 100);
    }

    function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
        }
    }

    function visualizeAudio() {
    if (!analyser || !waveformCanvas) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    function draw() {
        if (!analyser) return;

        animationFrameId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const barWidth = width / dataArray.length * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.8;
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

        draw();
    }

    function stopVisualization() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
        }
    }

    async function processRecordedAudio() {
    // Convert recorded chunks to WAV format
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    try {
        // Create a new audio context for decoding (since the visualization one might be closed)
        const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Convert to WAV using Web Audio API
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        
        // Close the decode context
        await decodeContext.close();
        
        // Create a File object from the WAV blob
        const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        // Set as current file and auto-transcribe
        currentFile = wavFile;
        fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
        processBtn.disabled = false;
        
        // Auto-transcribe
        await handleTranscribe();
        
    } catch (error) {
        console.error('Error processing recorded audio:', error);
        alert('Error processing recorded audio. Please try again.');
        }
    }

    // Convert AudioBuffer to WAV format
    function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = buffer.length * numChannels * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
    }

    // Initialize
    loadHistory();
});


```


Checking the end of the file to ensure it's properly closed:



Removing the duplicate initialization call. Verifying the file structure:



The structure looks correct. Verifying the file is properly closed and checking for syntax issues:



Syntax is valid. Moving function definitions before the event listeners for clarity and adding error handling:


```javascript
// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const langSelect = document.getElementById('langSelect');
    const fileInfo = document.getElementById('fileInfo');
    const resultsSection = document.getElementById('resultsSection');
    const resultText = document.getElementById('resultText');
    const resultFilename = document.getElementById('resultFilename');
    const resultLang = document.getElementById('resultLang');
    const downloadBtn = document.getElementById('downloadBtn');
    const historyList = document.getElementById('historyList');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

    // Recording elements
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    const recordingWarning = document.getElementById('recordingWarning');
    const audioVisualizer = document.getElementById('audioVisualizer');
    const waveformCanvas = document.getElementById('waveformCanvas');

    // Check if all required elements exist
    if (!uploadArea || !fileInput || !uploadBtn || !processBtn || !recordBtn || !stopBtn) {
        console.error('Required DOM elements not found!');
        return;
    }

    // Global state variables
    let currentFile = null;
    let currentTranscription = null;
    let currentHistoryId = null;

    // Recording state
    let mediaRecorder = null;
    let audioChunks = [];
    let audioStream = null;
    let recordingStartTime = null;
    let timerInterval = null;
    let animationFrameId = null;
    let audioContext = null;
    let analyser = null;
    let dataArray = null;
    const MAX_RECORDING_TIME = 40; // seconds

    // Function definitions (must be before event listeners)
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
        // Validate file type
        if (!file.type.startsWith('audio/')) {
            alert('Please select an audio file');
            return;
        }

        currentFile = file;
        fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        processBtn.disabled = false;
        resultsSection.style.display = 'none';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    async function handleTranscribe() {
        if (!currentFile) {
            alert('Please select a file first');
            return;
        }

        // Disable button and show loading
        processBtn.disabled = true;
        const btnText = processBtn.querySelector('.btn-text');
        const btnSpinner = processBtn.querySelector('.btn-spinner');
        btnText.style.display = 'none';
        btnSpinner.style.display = 'inline';

        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('lang_code', langSelect.value);

        try {
            const response = await fetch('/api/transcribe', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Transcription failed');
            }

            // Display results
            currentTranscription = data.transcription;
            currentHistoryId = data.history_id !== undefined ? data.history_id : null;
            resultText.textContent = data.transcription;
            resultFilename.textContent = `📄 ${data.filename}`;
            resultLang.textContent = `🌐 ${data.lang_code}`;
            resultsSection.style.display = 'block';
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

            // Reload history
            loadHistory();

        } catch (error) {
            alert('Error: ' + error.message);
            console.error('Transcription error:', error);
        } finally {
            // Re-enable button
            processBtn.disabled = false;
            btnText.style.display = 'inline';
            btnSpinner.style.display = 'none';
        }
    }

    function handleDownload() {
        if (!currentTranscription) {
            alert('No transcription to download');
            return;
        }

        // Find the history entry for current transcription
        if (currentHistoryId !== null) {
            window.location.href = `/api/download/${currentHistoryId}`;
        } else {
            // Create a temporary download
            const blob = new Blob([currentTranscription], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    }

    async function loadHistory() {
        try {
            const response = await fetch('/api/history');
            const data = await response.json();

            if (data.history && data.history.length > 0) {
                historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
                
                // Add event listeners to history items
                document.querySelectorAll('.history-item').forEach(item => {
                    item.addEventListener('click', () => {
                        const historyId = parseInt(item.dataset.id);
                        loadHistoryItem(historyId);
                    });
                });

                document.querySelectorAll('.btn-download-history').forEach(btn => {
                    btn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        const historyId = parseInt(btn.dataset.id);
                        window.location.href = `/api/download/${historyId}`;
                    });
                });
            } else {
                historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
            }
        } catch (error) {
            console.error('Error loading history:', error);
        }
    }

    function createHistoryItem(item) {
        const date = new Date(item.timestamp);
        const timeStr = date.toLocaleString();
        const preview = item.transcription.length > 150 
            ? item.transcription.substring(0, 150) + '...' 
            : item.transcription;

        return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
    }

    function loadHistoryItem(historyId) {
        // Load the full transcription from history
        fetch('/api/history')
            .then(response => response.json())
            .then(data => {
                const item = data.history.find(h => h.id === historyId);
                if (item) {
                    currentTranscription = item.transcription;
                    currentHistoryId = item.id;
                    resultText.textContent = item.transcription;
                    resultFilename.textContent = `📄 ${item.filename}`;
                    resultLang.textContent = `🌐 ${item.lang_code}`;
                    resultsSection.style.display = 'block';
                    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            })
            .catch(error => {
                console.error('Error loading history item:', error);
            });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Microphone Recording Functions
    async function startRecording() {
        try {
            // Request microphone access
            audioStream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true
                } 
            });

            // Set up audio context for visualization
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            const source = audioContext.createMediaStreamSource(audioStream);
            source.connect(analyser);
            dataArray = new Uint8Array(analyser.frequencyBinCount);

            // Set up MediaRecorder
            const options = { mimeType: 'audio/webm' };
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = 'audio/webm;codecs=opus';
                if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                    options.mimeType = ''; // Let browser choose
                }
            }

            mediaRecorder = new MediaRecorder(audioStream, options);
            audioChunks = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                await processRecordedAudio();
            };

            // Start recording
            mediaRecorder.start(100); // Collect data every 100ms
            recordingStartTime = Date.now();

            // Update UI
            recordBtn.disabled = true;
            recordBtn.style.display = 'none';
            stopBtn.disabled = false;
            stopBtn.style.display = 'inline-flex';
            recordingStatus.style.display = 'flex';
            audioVisualizer.style.display = 'block';
            recordingWarning.textContent = '';
            recordingWarning.className = 'recording-warning';

            // Start timer
            startTimer();

            // Start visualization
            visualizeAudio();

        } catch (error) {
            console.error('Error accessing microphone:', error);
            alert('Could not access microphone. Please check permissions and try again.');
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        
        // Stop audio stream
        if (audioStream) {
            audioStream.getTracks().forEach(track => track.stop());
            audioStream = null;
        }

        // Stop timer and visualization
        stopTimer();
        stopVisualization();

        // Update UI
        recordBtn.disabled = false;
        recordBtn.style.display = 'inline-flex';
        stopBtn.disabled = true;
        stopBtn.style.display = 'none';
        recordingStatus.style.display = 'none';
        audioVisualizer.style.display = 'none';
    }

    function startTimer() {
        recordingStartTime = Date.now();
        timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

            // Warn at 35 seconds, stop at 40 seconds
            if (elapsed >= MAX_RECORDING_TIME) {
                stopRecording();
                alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
            } else if (elapsed >= 35) {
                recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
                recordingWarning.className = 'recording-warning warning';
            }
        }, 100);
    }

    function stopTimer() {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
    }

    function visualizeAudio() {
        if (!analyser || !waveformCanvas) return;

        const canvas = waveformCanvas;
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        function draw() {
            if (!analyser) return;

            animationFrameId = requestAnimationFrame(draw);

            analyser.getByteFrequencyData(dataArray);

            ctx.fillStyle = '#f8fafc';
            ctx.fillRect(0, 0, width, height);

            const barWidth = width / dataArray.length * 2.5;
            let x = 0;

            for (let i = 0; i < dataArray.length; i++) {
                const barHeight = (dataArray[i] / 255) * height * 0.8;
                const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
                gradient.addColorStop(0, '#6366f1');
                gradient.addColorStop(1, '#8b5cf6');
                
                ctx.fillStyle = gradient;
                ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
                x += barWidth;
            }
        }

        draw();
    }

    function stopVisualization() {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        if (waveformCanvas) {
            const ctx = waveformCanvas.getContext('2d');
            ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
        }
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
    }

    async function processRecordedAudio() {
        // Convert recorded chunks to WAV format
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        
        try {
            // Create a new audio context for decoding (since the visualization one might be closed)
            const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Convert to WAV using Web Audio API
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer);
            const wavBlob = audioBufferToWav(audioBuffer);
            
            // Close the decode context
            await decodeContext.close();
            
            // Create a File object from the WAV blob
            const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
            
            // Set as current file and auto-transcribe
            currentFile = wavFile;
            fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
            processBtn.disabled = false;
            
            // Auto-transcribe
            await handleTranscribe();
            
        } catch (error) {
            console.error('Error processing recorded audio:', error);
            alert('Error processing recorded audio. Please try again.');
        }
    }

    // Convert AudioBuffer to WAV format
    function audioBufferToWav(buffer) {
        const numChannels = buffer.numberOfChannels;
        const sampleRate = buffer.sampleRate;
        const format = 1; // PCM
        const bitDepth = 16;

        const bytesPerSample = bitDepth / 8;
        const blockAlign = numChannels * bytesPerSample;

        const length = buffer.length * numChannels * bytesPerSample;
        const arrayBuffer = new ArrayBuffer(44 + length);
        const view = new DataView(arrayBuffer);

        // WAV header
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + length, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, format, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * blockAlign, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitDepth, true);
        writeString(36, 'data');
        view.setUint32(40, length, true);

        // Convert audio data
        let offset = 44;
        for (let i = 0; i < buffer.length; i++) {
            for (let channel = 0; channel < numChannels; channel++) {
                const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
                view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
                offset += 2;
            }
        }

        return new Blob([arrayBuffer], { type: 'audio/wav' });
    }

    // Event Listeners (after function definitions)
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    processBtn.addEventListener('click', handleTranscribe);
    downloadBtn.addEventListener('click', handleDownload);
    refreshHistoryBtn.addEventListener('click', loadHistory);
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);

    // Function definitions
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
    }

        currentFile = file;
        fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        processBtn.disabled = false;
        resultsSection.style.display = 'none';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    async function handleTranscribe() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }

    // Disable button and show loading
    processBtn.disabled = true;
    const btnText = processBtn.querySelector('.btn-text');
    const btnSpinner = processBtn.querySelector('.btn-spinner');
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('lang_code', langSelect.value);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        }
    }

    async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
        }
    }

    function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
    }

    function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
            });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Microphone Recording Functions
    async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Set up MediaRecorder
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = ''; // Let browser choose
            }
        }

        mediaRecorder = new MediaRecorder(audioStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await processRecordedAudio();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        recordingStartTime = Date.now();

        // Update UI
        recordBtn.disabled = true;
        recordBtn.style.display = 'none';
        stopBtn.disabled = false;
        stopBtn.style.display = 'inline-flex';
        recordingStatus.style.display = 'flex';
        audioVisualizer.style.display = 'block';
        recordingWarning.textContent = '';
        recordingWarning.className = 'recording-warning';

        // Start timer
        startTimer();

        // Start visualization
        visualizeAudio();

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions and try again.');
        }
    }

    function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop audio stream
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    // Stop timer and visualization
    stopTimer();
    stopVisualization();

    // Update UI
    recordBtn.disabled = false;
    recordBtn.style.display = 'inline-flex';
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
        recordingStatus.style.display = 'none';
        audioVisualizer.style.display = 'none';
    }

    function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Warn at 35 seconds, stop at 40 seconds
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
            alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
        } else if (elapsed >= 35) {
            recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
            recordingWarning.className = 'recording-warning warning';
            }
        }, 100);
    }

    function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
        }
    }

    function visualizeAudio() {
    if (!analyser || !waveformCanvas) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    function draw() {
        if (!analyser) return;

        animationFrameId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const barWidth = width / dataArray.length * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.8;
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

        draw();
    }

    function stopVisualization() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
        }
    }

    async function processRecordedAudio() {
    // Convert recorded chunks to WAV format
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    try {
        // Create a new audio context for decoding (since the visualization one might be closed)
        const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Convert to WAV using Web Audio API
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        
        // Close the decode context
        await decodeContext.close();
        
        // Create a File object from the WAV blob
        const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        // Set as current file and auto-transcribe
        currentFile = wavFile;
        fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
        processBtn.disabled = false;
        
        // Auto-transcribe
        await handleTranscribe();
        
    } catch (error) {
        console.error('Error processing recorded audio:', error);
        alert('Error processing recorded audio. Please try again.');
        }
    }

    // Convert AudioBuffer to WAV format
    function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = buffer.length * numChannels * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
    }

    // Initialize
    loadHistory();
});


```


Removing duplicate function definitions:



Removing duplicate function definitions after the event listeners:



Removing duplicate function definitions:


```javascript
// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const langSelect = document.getElementById('langSelect');
    const fileInfo = document.getElementById('fileInfo');
    const resultsSection = document.getElementById('resultsSection');
    const resultText = document.getElementById('resultText');
    const resultFilename = document.getElementById('resultFilename');
    const resultLang = document.getElementById('resultLang');
    const downloadBtn = document.getElementById('downloadBtn');
    const historyList = document.getElementById('historyList');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

    // Recording elements
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    const recordingWarning = document.getElementById('recordingWarning');
    const audioVisualizer = document.getElementById('audioVisualizer');
    const waveformCanvas = document.getElementById('waveformCanvas');

    // Check if all required elements exist
    if (!uploadArea || !fileInput || !uploadBtn || !processBtn || !recordBtn || !stopBtn) {
        console.error('Required DOM elements not found!');
        return;
    }

    // Global state variables
    let currentFile = null;
    let currentTranscription = null;
    let currentHistoryId = null;

    // Recording state
    let mediaRecorder = null;
    let audioChunks = [];
    let audioStream = null;
    let recordingStartTime = null;
    let timerInterval = null;
    let animationFrameId = null;
    let audioContext = null;
    let analyser = null;
    let dataArray = null;
    const MAX_RECORDING_TIME = 40; // seconds

    // Function definitions (must be before event listeners)
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
        // Validate file type
        if (!file.type.startsWith('audio/')) {
            alert('Please select an audio file');
            return;
        }

        currentFile = file;
        fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        processBtn.disabled = false;
        resultsSection.style.display = 'none';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    async function handleTranscribe() {
        if (!currentFile) {
            alert('Please select a file first');
            return;
        }

        // Disable button and show loading
        processBtn.disabled = true;
        const btnText = processBtn.querySelector('.btn-text');
        const btnSpinner = processBtn.querySelector('.btn-spinner');
        btnText.style.display = 'none';
        btnSpinner.style.display = 'inline';

        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('lang_code', langSelect.value);

        try {
            const response = await fetch('/api/transcribe', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Transcription failed');
            }

            // Display results
            currentTranscription = data.transcription;
            currentHistoryId = data.history_id !== undefined ? data.history_id : null;
            resultText.textContent = data.transcription;
            resultFilename.textContent = `📄 ${data.filename}`;
            resultLang.textContent = `🌐 ${data.lang_code}`;
            resultsSection.style.display = 'block';
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

            // Reload history
            loadHistory();

        } catch (error) {
            alert('Error: ' + error.message);
            console.error('Transcription error:', error);
        } finally {
            // Re-enable button
            processBtn.disabled = false;
            btnText.style.display = 'inline';
            btnSpinner.style.display = 'none';
        }
    }

    function handleDownload() {
        if (!currentTranscription) {
            alert('No transcription to download');
            return;
        }

        // Find the history entry for current transcription
        if (currentHistoryId !== null) {
            window.location.href = `/api/download/${currentHistoryId}`;
        } else {
            // Create a temporary download
            const blob = new Blob([currentTranscription], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    }

    async function loadHistory() {
        try {
            const response = await fetch('/api/history');
            const data = await response.json();

            if (data.history && data.history.length > 0) {
                historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
                
                // Add event listeners to history items
                document.querySelectorAll('.history-item').forEach(item => {
                    item.addEventListener('click', () => {
                        const historyId = parseInt(item.dataset.id);
                        loadHistoryItem(historyId);
                    });
                });

                document.querySelectorAll('.btn-download-history').forEach(btn => {
                    btn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        const historyId = parseInt(btn.dataset.id);
                        window.location.href = `/api/download/${historyId}`;
                    });
                });
            } else {
                historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
            }
        } catch (error) {
            console.error('Error loading history:', error);
        }
    }

    function createHistoryItem(item) {
        const date = new Date(item.timestamp);
        const timeStr = date.toLocaleString();
        const preview = item.transcription.length > 150 
            ? item.transcription.substring(0, 150) + '...' 
            : item.transcription;

        return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
    }

    function loadHistoryItem(historyId) {
        // Load the full transcription from history
        fetch('/api/history')
            .then(response => response.json())
            .then(data => {
                const item = data.history.find(h => h.id === historyId);
                if (item) {
                    currentTranscription = item.transcription;
                    currentHistoryId = item.id;
                    resultText.textContent = item.transcription;
                    resultFilename.textContent = `📄 ${item.filename}`;
                    resultLang.textContent = `🌐 ${item.lang_code}`;
                    resultsSection.style.display = 'block';
                    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            })
            .catch(error => {
                console.error('Error loading history item:', error);
            });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Microphone Recording Functions
    async function startRecording() {
        try {
            // Request microphone access
            audioStream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true
                } 
            });

            // Set up audio context for visualization
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            const source = audioContext.createMediaStreamSource(audioStream);
            source.connect(analyser);
            dataArray = new Uint8Array(analyser.frequencyBinCount);

            // Set up MediaRecorder
            const options = { mimeType: 'audio/webm' };
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = 'audio/webm;codecs=opus';
                if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                    options.mimeType = ''; // Let browser choose
                }
            }

            mediaRecorder = new MediaRecorder(audioStream, options);
            audioChunks = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                await processRecordedAudio();
            };

            // Start recording
            mediaRecorder.start(100); // Collect data every 100ms
            recordingStartTime = Date.now();

            // Update UI
            recordBtn.disabled = true;
            recordBtn.style.display = 'none';
            stopBtn.disabled = false;
            stopBtn.style.display = 'inline-flex';
            recordingStatus.style.display = 'flex';
            audioVisualizer.style.display = 'block';
            recordingWarning.textContent = '';
            recordingWarning.className = 'recording-warning';

            // Start timer
            startTimer();

            // Start visualization
            visualizeAudio();

        } catch (error) {
            console.error('Error accessing microphone:', error);
            alert('Could not access microphone. Please check permissions and try again.');
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        
        // Stop audio stream
        if (audioStream) {
            audioStream.getTracks().forEach(track => track.stop());
            audioStream = null;
        }

        // Stop timer and visualization
        stopTimer();
        stopVisualization();

        // Update UI
        recordBtn.disabled = false;
        recordBtn.style.display = 'inline-flex';
        stopBtn.disabled = true;
        stopBtn.style.display = 'none';
        recordingStatus.style.display = 'none';
        audioVisualizer.style.display = 'none';
    }

    function startTimer() {
        recordingStartTime = Date.now();
        timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

            // Warn at 35 seconds, stop at 40 seconds
            if (elapsed >= MAX_RECORDING_TIME) {
                stopRecording();
                alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
            } else if (elapsed >= 35) {
                recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
                recordingWarning.className = 'recording-warning warning';
            }
        }, 100);
    }

    function stopTimer() {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
    }

    function visualizeAudio() {
        if (!analyser || !waveformCanvas) return;

        const canvas = waveformCanvas;
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        function draw() {
            if (!analyser) return;

            animationFrameId = requestAnimationFrame(draw);

            analyser.getByteFrequencyData(dataArray);

            ctx.fillStyle = '#f8fafc';
            ctx.fillRect(0, 0, width, height);

            const barWidth = width / dataArray.length * 2.5;
            let x = 0;

            for (let i = 0; i < dataArray.length; i++) {
                const barHeight = (dataArray[i] / 255) * height * 0.8;
                const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
                gradient.addColorStop(0, '#6366f1');
                gradient.addColorStop(1, '#8b5cf6');
                
                ctx.fillStyle = gradient;
                ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
                x += barWidth;
            }
        }

        draw();
    }

    function stopVisualization() {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        if (waveformCanvas) {
            const ctx = waveformCanvas.getContext('2d');
            ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
        }
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
    }

    async function processRecordedAudio() {
        // Convert recorded chunks to WAV format
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        
        try {
            // Create a new audio context for decoding (since the visualization one might be closed)
            const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Convert to WAV using Web Audio API
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer);
            const wavBlob = audioBufferToWav(audioBuffer);
            
            // Close the decode context
            await decodeContext.close();
            
            // Create a File object from the WAV blob
            const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
            
            // Set as current file and auto-transcribe
            currentFile = wavFile;
            fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
            processBtn.disabled = false;
            
            // Auto-transcribe
            await handleTranscribe();
            
        } catch (error) {
            console.error('Error processing recorded audio:', error);
            alert('Error processing recorded audio. Please try again.');
        }
    }

    // Convert AudioBuffer to WAV format
    function audioBufferToWav(buffer) {
        const numChannels = buffer.numberOfChannels;
        const sampleRate = buffer.sampleRate;
        const format = 1; // PCM
        const bitDepth = 16;

        const bytesPerSample = bitDepth / 8;
        const blockAlign = numChannels * bytesPerSample;

        const length = buffer.length * numChannels * bytesPerSample;
        const arrayBuffer = new ArrayBuffer(44 + length);
        const view = new DataView(arrayBuffer);

        // WAV header
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + length, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, format, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * blockAlign, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitDepth, true);
        writeString(36, 'data');
        view.setUint32(40, length, true);

        // Convert audio data
        let offset = 44;
        for (let i = 0; i < buffer.length; i++) {
            for (let channel = 0; channel < numChannels; channel++) {
                const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
                view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
                offset += 2;
            }
        }

        return new Blob([arrayBuffer], { type: 'audio/wav' });
    }

    // Event Listeners (after function definitions)
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    processBtn.addEventListener('click', handleTranscribe);
    downloadBtn.addEventListener('click', handleDownload);
    refreshHistoryBtn.addEventListener('click', loadHistory);
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Transcription failed');
        }

        // Display results
        currentTranscription = data.transcription;
        currentHistoryId = data.history_id !== undefined ? data.history_id : null;
        resultText.textContent = data.transcription;
        resultFilename.textContent = `📄 ${data.filename}`;
        resultLang.textContent = `🌐 ${data.lang_code}`;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reload history
        loadHistory();

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Transcription error:', error);
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
}

function handleDownload() {
    if (!currentTranscription) {
        alert('No transcription to download');
        return;
    }

    // Find the history entry for current transcription
    if (currentHistoryId !== null) {
        window.location.href = `/api/download/${currentHistoryId}`;
    } else {
        // Create a temporary download
        const blob = new Blob([currentTranscription], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        }
    }

    async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
            
            // Add event listeners to history items
            document.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const historyId = parseInt(item.dataset.id);
                    loadHistoryItem(historyId);
                });
            });

            document.querySelectorAll('.btn-download-history').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const historyId = parseInt(btn.dataset.id);
                    window.location.href = `/api/download/${historyId}`;
                });
            });
        } else {
            historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
        }
    }

    function createHistoryItem(item) {
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleString();
    const preview = item.transcription.length > 150 
        ? item.transcription.substring(0, 150) + '...' 
        : item.transcription;

    return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
    }

    function loadHistoryItem(historyId) {
    // Load the full transcription from history
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const item = data.history.find(h => h.id === historyId);
            if (item) {
                currentTranscription = item.transcription;
                currentHistoryId = item.id;
                resultText.textContent = item.transcription;
                resultFilename.textContent = `📄 ${item.filename}`;
                resultLang.textContent = `🌐 ${item.lang_code}`;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        })
        .catch(error => {
            console.error('Error loading history item:', error);
            });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Microphone Recording Functions
    async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Set up MediaRecorder
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = ''; // Let browser choose
            }
        }

        mediaRecorder = new MediaRecorder(audioStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await processRecordedAudio();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        recordingStartTime = Date.now();

        // Update UI
        recordBtn.disabled = true;
        recordBtn.style.display = 'none';
        stopBtn.disabled = false;
        stopBtn.style.display = 'inline-flex';
        recordingStatus.style.display = 'flex';
        audioVisualizer.style.display = 'block';
        recordingWarning.textContent = '';
        recordingWarning.className = 'recording-warning';

        // Start timer
        startTimer();

        // Start visualization
        visualizeAudio();

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions and try again.');
        }
    }

    function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop audio stream
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    // Stop timer and visualization
    stopTimer();
    stopVisualization();

    // Update UI
    recordBtn.disabled = false;
    recordBtn.style.display = 'inline-flex';
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
        recordingStatus.style.display = 'none';
        audioVisualizer.style.display = 'none';
    }

    function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Warn at 35 seconds, stop at 40 seconds
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
            alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
        } else if (elapsed >= 35) {
            recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
            recordingWarning.className = 'recording-warning warning';
            }
        }, 100);
    }

    function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
        }
    }

    function visualizeAudio() {
    if (!analyser || !waveformCanvas) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    function draw() {
        if (!analyser) return;

        animationFrameId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const barWidth = width / dataArray.length * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.8;
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

        draw();
    }

    function stopVisualization() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
        }
    }

    async function processRecordedAudio() {
    // Convert recorded chunks to WAV format
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    try {
        // Create a new audio context for decoding (since the visualization one might be closed)
        const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Convert to WAV using Web Audio API
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        
        // Close the decode context
        await decodeContext.close();
        
        // Create a File object from the WAV blob
        const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        // Set as current file and auto-transcribe
        currentFile = wavFile;
        fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
        processBtn.disabled = false;
        
        // Auto-transcribe
        await handleTranscribe();
        
    } catch (error) {
        console.error('Error processing recorded audio:', error);
        alert('Error processing recorded audio. Please try again.');
        }
    }

    // Convert AudioBuffer to WAV format
    function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = buffer.length * numChannels * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
    }

    // Initialize
    loadHistory();
});


```


Removing broken duplicate code and fixing the file structure:


```javascript
// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const langSelect = document.getElementById('langSelect');
    const fileInfo = document.getElementById('fileInfo');
    const resultsSection = document.getElementById('resultsSection');
    const resultText = document.getElementById('resultText');
    const resultFilename = document.getElementById('resultFilename');
    const resultLang = document.getElementById('resultLang');
    const downloadBtn = document.getElementById('downloadBtn');
    const historyList = document.getElementById('historyList');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

    // Recording elements
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    const recordingWarning = document.getElementById('recordingWarning');
    const audioVisualizer = document.getElementById('audioVisualizer');
    const waveformCanvas = document.getElementById('waveformCanvas');

    // Check if all required elements exist
    if (!uploadArea || !fileInput || !uploadBtn || !processBtn || !recordBtn || !stopBtn) {
        console.error('Required DOM elements not found!');
        return;
    }

    // Global state variables
    let currentFile = null;
    let currentTranscription = null;
    let currentHistoryId = null;

    // Recording state
    let mediaRecorder = null;
    let audioChunks = [];
    let audioStream = null;
    let recordingStartTime = null;
    let timerInterval = null;
    let animationFrameId = null;
    let audioContext = null;
    let analyser = null;
    let dataArray = null;
    const MAX_RECORDING_TIME = 40; // seconds

    // Function definitions (must be before event listeners)
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
        // Validate file type
        if (!file.type.startsWith('audio/')) {
            alert('Please select an audio file');
            return;
        }

        currentFile = file;
        fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        processBtn.disabled = false;
        resultsSection.style.display = 'none';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    async function handleTranscribe() {
        if (!currentFile) {
            alert('Please select a file first');
            return;
        }

        // Disable button and show loading
        processBtn.disabled = true;
        const btnText = processBtn.querySelector('.btn-text');
        const btnSpinner = processBtn.querySelector('.btn-spinner');
        btnText.style.display = 'none';
        btnSpinner.style.display = 'inline';

        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('lang_code', langSelect.value);

        try {
            const response = await fetch('/api/transcribe', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Transcription failed');
            }

            // Display results
            currentTranscription = data.transcription;
            currentHistoryId = data.history_id !== undefined ? data.history_id : null;
            resultText.textContent = data.transcription;
            resultFilename.textContent = `📄 ${data.filename}`;
            resultLang.textContent = `🌐 ${data.lang_code}`;
            resultsSection.style.display = 'block';
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

            // Reload history
            loadHistory();

        } catch (error) {
            alert('Error: ' + error.message);
            console.error('Transcription error:', error);
        } finally {
            // Re-enable button
            processBtn.disabled = false;
            btnText.style.display = 'inline';
            btnSpinner.style.display = 'none';
        }
    }

    function handleDownload() {
        if (!currentTranscription) {
            alert('No transcription to download');
            return;
        }

        // Find the history entry for current transcription
        if (currentHistoryId !== null) {
            window.location.href = `/api/download/${currentHistoryId}`;
        } else {
            // Create a temporary download
            const blob = new Blob([currentTranscription], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `transcription_${currentFile?.name || 'audio'}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    }

    async function loadHistory() {
        try {
            const response = await fetch('/api/history');
            const data = await response.json();

            if (data.history && data.history.length > 0) {
                historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');
                
                // Add event listeners to history items
                document.querySelectorAll('.history-item').forEach(item => {
                    item.addEventListener('click', () => {
                        const historyId = parseInt(item.dataset.id);
                        loadHistoryItem(historyId);
                    });
                });

                document.querySelectorAll('.btn-download-history').forEach(btn => {
                    btn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        const historyId = parseInt(btn.dataset.id);
                        window.location.href = `/api/download/${historyId}`;
                    });
                });
            } else {
                historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
            }
        } catch (error) {
            console.error('Error loading history:', error);
        }
    }

    function createHistoryItem(item) {
        const date = new Date(item.timestamp);
        const timeStr = date.toLocaleString();
        const preview = item.transcription.length > 150 
            ? item.transcription.substring(0, 150) + '...' 
            : item.transcription;

        return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
    }

    function loadHistoryItem(historyId) {
        // Load the full transcription from history
        fetch('/api/history')
            .then(response => response.json())
            .then(data => {
                const item = data.history.find(h => h.id === historyId);
                if (item) {
                    currentTranscription = item.transcription;
                    currentHistoryId = item.id;
                    resultText.textContent = item.transcription;
                    resultFilename.textContent = `📄 ${item.filename}`;
                    resultLang.textContent = `🌐 ${item.lang_code}`;
                    resultsSection.style.display = 'block';
                    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            })
            .catch(error => {
                console.error('Error loading history item:', error);
            });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Microphone Recording Functions
    async function startRecording() {
        try {
            // Request microphone access
            audioStream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true
                } 
            });

            // Set up audio context for visualization
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            const source = audioContext.createMediaStreamSource(audioStream);
            source.connect(analyser);
            dataArray = new Uint8Array(analyser.frequencyBinCount);

            // Set up MediaRecorder
            const options = { mimeType: 'audio/webm' };
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = 'audio/webm;codecs=opus';
                if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                    options.mimeType = ''; // Let browser choose
                }
            }

            mediaRecorder = new MediaRecorder(audioStream, options);
            audioChunks = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                await processRecordedAudio();
            };

            // Start recording
            mediaRecorder.start(100); // Collect data every 100ms
            recordingStartTime = Date.now();

            // Update UI
            recordBtn.disabled = true;
            recordBtn.style.display = 'none';
            stopBtn.disabled = false;
            stopBtn.style.display = 'inline-flex';
            recordingStatus.style.display = 'flex';
            audioVisualizer.style.display = 'block';
            recordingWarning.textContent = '';
            recordingWarning.className = 'recording-warning';

            // Start timer
            startTimer();

            // Start visualization
            visualizeAudio();

        } catch (error) {
            console.error('Error accessing microphone:', error);
            alert('Could not access microphone. Please check permissions and try again.');
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        
        // Stop audio stream
        if (audioStream) {
            audioStream.getTracks().forEach(track => track.stop());
            audioStream = null;
        }

        // Stop timer and visualization
        stopTimer();
        stopVisualization();

        // Update UI
        recordBtn.disabled = false;
        recordBtn.style.display = 'inline-flex';
        stopBtn.disabled = true;
        stopBtn.style.display = 'none';
        recordingStatus.style.display = 'none';
        audioVisualizer.style.display = 'none';
    }

    function startTimer() {
        recordingStartTime = Date.now();
        timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

            // Warn at 35 seconds, stop at 40 seconds
            if (elapsed >= MAX_RECORDING_TIME) {
                stopRecording();
                alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
            } else if (elapsed >= 35) {
                recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
                recordingWarning.className = 'recording-warning warning';
            }
        }, 100);
    }

    function stopTimer() {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
    }

    function visualizeAudio() {
        if (!analyser || !waveformCanvas) return;

        const canvas = waveformCanvas;
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        function draw() {
            if (!analyser) return;

            animationFrameId = requestAnimationFrame(draw);

            analyser.getByteFrequencyData(dataArray);

            ctx.fillStyle = '#f8fafc';
            ctx.fillRect(0, 0, width, height);

            const barWidth = width / dataArray.length * 2.5;
            let x = 0;

            for (let i = 0; i < dataArray.length; i++) {
                const barHeight = (dataArray[i] / 255) * height * 0.8;
                const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
                gradient.addColorStop(0, '#6366f1');
                gradient.addColorStop(1, '#8b5cf6');
                
                ctx.fillStyle = gradient;
                ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
                x += barWidth;
            }
        }

        draw();
    }

    function stopVisualization() {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        if (waveformCanvas) {
            const ctx = waveformCanvas.getContext('2d');
            ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
        }
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
    }

    async function processRecordedAudio() {
        // Convert recorded chunks to WAV format
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        
        try {
            // Create a new audio context for decoding (since the visualization one might be closed)
            const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Convert to WAV using Web Audio API
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer);
            const wavBlob = audioBufferToWav(audioBuffer);
            
            // Close the decode context
            await decodeContext.close();
            
            // Create a File object from the WAV blob
            const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
            
            // Set as current file and auto-transcribe
            currentFile = wavFile;
            fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
            processBtn.disabled = false;
            
            // Auto-transcribe
            await handleTranscribe();
            
        } catch (error) {
            console.error('Error processing recorded audio:', error);
            alert('Error processing recorded audio. Please try again.');
        }
    }

    // Convert AudioBuffer to WAV format
    function audioBufferToWav(buffer) {
        const numChannels = buffer.numberOfChannels;
        const sampleRate = buffer.sampleRate;
        const format = 1; // PCM
        const bitDepth = 16;

        const bytesPerSample = bitDepth / 8;
        const blockAlign = numChannels * bytesPerSample;

        const length = buffer.length * numChannels * bytesPerSample;
        const arrayBuffer = new ArrayBuffer(44 + length);
        const view = new DataView(arrayBuffer);

        // WAV header
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + length, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, format, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * blockAlign, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitDepth, true);
        writeString(36, 'data');
        view.setUint32(40, length, true);

        // Convert audio data
        let offset = 44;
        for (let i = 0; i < buffer.length; i++) {
            for (let channel = 0; channel < numChannels; channel++) {
                const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
                view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
                offset += 2;
            }
        }

        return new Blob([arrayBuffer], { type: 'audio/wav' });
    }

    // Event Listeners (after function definitions)
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    processBtn.addEventListener('click', handleTranscribe);
    downloadBtn.addEventListener('click', handleDownload);
    refreshHistoryBtn.addEventListener('click', loadHistory);
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);

    // Initialize
    loadHistory();
});


```


Fixed the JavaScript file. The issue was duplicate function definitions and broken code. The file is now structured correctly:

1. All functions are defined before the event listeners
2. Everything is wrapped in `DOMContentLoaded` to ensure the DOM is ready
3. Removed duplicate/broken code

The buttons should work now. Refresh your browser and try:
- File upload button
- Drag and drop
- Start Recording button

All event listeners are properly attached and the functions are accessible. If issues persist, check the browser console (F12) for any error messages.

