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

