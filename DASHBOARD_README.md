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

