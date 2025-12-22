// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function () {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const langSelect = document.getElementById('langSelect');
    const modelSelect = document.getElementById('modelSelect');
    const fileInfo = document.getElementById('fileInfo');
    const resultsSection = document.getElementById('resultsSection');
    const resultText = document.getElementById('resultText');
    const resultFilename = document.getElementById('resultFilename');
    const resultLang = document.getElementById('resultLang');
    const downloadBtn = document.getElementById('downloadBtn');
    const historyList = document.getElementById('historyList');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

    // Transcribe Tab Elements
    const transcribeRecordBtn = document.getElementById('transcribeRecordBtn');
    const transcribeStopBtn = document.getElementById('transcribeStopBtn');
    const transcribeRecordingStatus = document.getElementById('transcribeRecordingStatus');
    const transcribeRecordingTimer = document.getElementById('transcribeRecordingTimer');
    const transcribeRecordingWarning = document.getElementById('transcribeRecordingWarning');
    const transcribeAudioVisualizer = document.getElementById('transcribeAudioVisualizer');
    const transcribeWaveformCanvas = document.getElementById('transcribeWaveformCanvas');

    // Contribute Tab Elements
    // Recording elements (Transcription Tab) - Removed/Replaced by Data Collection
    // const recordBtn = document.getElementById('recordBtn'); 
    // ...

    // Data Collection Elements
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const playBtn = document.getElementById('playBtn');
    const submitRecordingBtn = document.getElementById('submitRecordingBtn');
    const skipBtn = document.getElementById('skipBtn');

    // Mode Toggles
    const modePresetBtn = document.getElementById('modePresetBtn');
    const modeCustomBtn = document.getElementById('modeCustomBtn');
    const customPromptInput = document.getElementById('customPromptInput');
    const progressIndicator = document.getElementById('progressIndicator');
    const instructionText = document.getElementById('instructionText');

    const promptText = document.getElementById('promptText');
    const promptCategory = document.getElementById('promptCategory');
    const currentPromptIndexSpan = document.getElementById('currentPromptIndex');
    const totalPromptsSpan = document.getElementById('totalPrompts');
    const contributionMessage = document.getElementById('contributionMessage');
    const audioPlayback = document.getElementById('audioPlayback');
    const contributionList = document.getElementById('contributionList');

    // Visualizer
    const visualizer = document.getElementById('visualizer');

    // Tab Elements
    const tabs = document.querySelectorAll('.nav-tab');
    const tabContents = document.querySelectorAll('.tab-content');

    // Check if all required transcription elements exist
    if (!uploadArea || !fileInput || !uploadBtn || !processBtn) {
        console.error('Required Transcription DOM elements not found!');
        return;
    }

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
    let recordingBlob = null;

    // Data Collection State
    let prompts = [];
    let currentPromptIndex = 0;
    let isCustomMode = false;

    // Navigation Logic
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            const tabId = tab.dataset.tab;

            if (tabId === 'transcribe') {
                document.getElementById('transcribeTab').classList.add('active');
                fetchModels(); // Refresh models when switching to transcribe
            } else if (tabId === 'long_audio') {
                document.getElementById('longAudioTab').classList.add('active');
                fetchModels(); // Refresh models when switching to long audio
            } else {
                document.getElementById('contributeTab').classList.add('active');
                if (prompts.length === 0) loadPrompts();
            }
        });
    });

    // --- Long Audio Elements ---
    const longAudioUploadArea = document.getElementById('longAudioUploadArea');
    const longAudioFileInput = document.getElementById('longAudioFileInput');
    const longAudioUploadBtn = document.getElementById('longAudioUploadBtn');
    const longAudioProcessBtn = document.getElementById('longAudioProcessBtn');
    const longAudioLangSelect = document.getElementById('longAudioLangSelect');
    const longAudioModelSelect = document.getElementById('longAudioModelSelect');
    const longAudioFileInfo = document.getElementById('longAudioFileInfo');
    const longAudioResultsSection = document.getElementById('longAudioResultsSection');
    const longAudioResultsTable = document.getElementById('longAudioResultsTable');
    const longAudioResultFilename = document.getElementById('longAudioResultFilename');
    const longAudioResultLang = document.getElementById('longAudioResultLang');
    const longAudioDownloadBtn = document.getElementById('longAudioDownloadBtn');

    let currentLongAudioFile = null;
    let currentLongAudioResults = [];

    // Initialize Long Audio Model Select (Sync with main model select logic)
    // We'll reuse the fetchModels function but populate both selects if they exist
    // Modified fetchModels below to handle multiple selects or just call it again/copy options.
    // For simplicity, let's just make sure both get populated.

    if (longAudioModelSelect) {
        longAudioModelSelect.addEventListener('change', handleModelSwitch);
    }

    // Long Audio Event Listeners
    if (longAudioUploadBtn && longAudioFileInput) {
        longAudioUploadBtn.addEventListener('click', () => longAudioFileInput.click());
        longAudioFileInput.addEventListener('change', handleLongAudioFileSelect);

        longAudioUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            longAudioUploadArea.classList.add('dragover');
        });
        longAudioUploadArea.addEventListener('dragleave', () => {
            longAudioUploadArea.classList.remove('dragover');
        });
        longAudioUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            longAudioUploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file) handleLongAudioFile(file);
        });
    }

    if (longAudioProcessBtn) {
        longAudioProcessBtn.addEventListener('click', handleLongAudioTranscribe);
    }

    if (longAudioDownloadBtn) {
        longAudioDownloadBtn.addEventListener('click', handleLongAudioDownload);
    }

    function handleLongAudioFileSelect(e) {
        const file = e.target.files[0];
        if (file) handleLongAudioFile(file);
    }

    function handleLongAudioFile(file) {
        if (!file.type.startsWith('audio/')) {
            alert('Please select an audio file');
            return;
        }
        currentLongAudioFile = file;
        longAudioFileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        longAudioProcessBtn.disabled = false;
        longAudioResultsSection.style.display = 'none';
    }

    async function handleLongAudioTranscribe() {
        if (!currentLongAudioFile) return;

        longAudioProcessBtn.disabled = true;
        const btnText = longAudioProcessBtn.querySelector('.btn-text');
        const btnSpinner = longAudioProcessBtn.querySelector('.btn-spinner');
        btnText.style.display = 'none';
        btnSpinner.style.display = 'inline';

        const formData = new FormData();
        formData.append('file', currentLongAudioFile);
        formData.append('lang_code', longAudioLangSelect.value);

        try {
            const response = await fetch('/api/transcribe_long', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Transcription failed');

            currentLongAudioResults = data.results; // List of {segment, filename, text}

            // Render Results
            renderLongAudioResults(data.results);

            longAudioResultFilename.textContent = `📄 ${data.filename}`;
            longAudioResultLang.textContent = `🌐 ${data.lang_code}`;
            longAudioResultsSection.style.display = 'block';
            longAudioResultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            longAudioProcessBtn.disabled = false;
            btnText.style.display = 'inline';
            btnSpinner.style.display = 'none';
        }
    }

    function renderLongAudioResults(results) {
        if (!results || results.length === 0) {
            longAudioResultsTable.innerHTML = '<p>No results found.</p>';
            return;
        }

        let html = '<div class="table-container" style="overflow-x:auto;"><table style="width:100%; border-collapse: collapse; margin-top: 1rem;">';
        html += '<thead><tr style="background:#f1f5f9; text-align:left;"><th style="padding:0.75rem; border-bottom:1px solid #e2e8f0;">Segment</th><th style="padding:0.75rem; border-bottom:1px solid #e2e8f0;">Text</th></tr></thead><tbody>';

        results.forEach(item => {
            html += `<tr style="border-bottom:1px solid #e2e8f0;">
                <td style="padding:0.75rem; white-space:nowrap; vertical-align:top; font-weight:bold; color:#64748b;">${item.segment}</td>
                <td style="padding:0.75rem;">${item.text}</td>
            </tr>`;
        });

        html += '</tbody></table></div>';
        longAudioResultsTable.innerHTML = html;
    }

    function handleLongAudioDownload() {
        if (!currentLongAudioResults || currentLongAudioResults.length === 0) return;

        const fullText = currentLongAudioResults.map(r => `[Segment ${r.segment}] ${r.text}`).join('\n\n');
        const blob = new Blob([fullText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `long_transcription_${currentLongAudioFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // Model switching event
    if (modelSelect) {
        modelSelect.addEventListener('change', handleModelSwitch);
    }

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

    // Data Collection Functions
    async function loadPrompts() {
        try {
            promptText.textContent = "Loading prompts...";
            const response = await fetch('/api/prompts');
            const data = await response.json();
            if (data.prompts) {
                prompts = data.prompts;
                totalPromptsSpan.textContent = prompts.length;
                loadPrompt(0);
            }
        } catch (e) {
            console.error("Failed to load prompts", e);
            promptText.textContent = "Error loading prompts.";
        }
    }

    function loadPrompt(index) {
        if (index >= prompts.length) {
            promptText.textContent = "All prompts completed! Thank you!";
            promptCategory.textContent = "Complete";
            recordBtn.disabled = true;
            return;
        }
        currentPromptIndex = index;
        const prompt = prompts[index];
        promptText.textContent = prompt.text;
        promptCategory.textContent = prompt.category; // + " (" + prompt.lang_code + ")"
        currentPromptIndexSpan.textContent = index + 1;

        // Reset UI
        resetRecordingUI();
    }

    function resetRecordingUI() {
        recordingBlob = null;
        audioChunks = [];
        audioPlayback.src = "";

        recordBtn.disabled = false;
        recordBtn.style.display = 'inline-flex';
        stopBtn.disabled = true;
        stopBtn.style.display = 'none';
        playBtn.disabled = true;
        submitRecordingBtn.disabled = true;

        visualizer.classList.remove('recording');
        document.querySelector('.recording-status-text').textContent = "Ready to Record";
    }

    /* --- Model Switching Logic --- */

    async function fetchModels() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();

            if (data.models) {
                [modelSelect, longAudioModelSelect].forEach(select => {
                    if (!select) return;
                    select.innerHTML = '';
                    Object.entries(data.models).forEach(([displayName, modelCard]) => {
                        const option = document.createElement('option');
                        option.value = modelCard;
                        option.textContent = displayName;
                        if (modelCard === data.current_model) {
                            option.selected = true;
                        }
                        select.appendChild(option);
                    });
                });
            }
        } catch (e) {
            console.error("Failed to load models", e);
            [modelSelect, longAudioModelSelect].forEach(select => {
                if (select) select.innerHTML = '<option disabled>Error loading models</option>';
            });
        }
    }

    async function handleModelSwitch(e) {
        const selectedModel = e.target.value;
        const previousModel = e.target.getAttribute('data-prev') || selectedModel; // fallback

        // Confirmation or direct switch? Let's just switch with UI feedback
        const confirmSwitch = confirm("Switching models will take a few seconds and clear current memory. Continue?");
        if (!confirmSwitch) {
            e.target.value = previousModel; // Revert
            return;
        }

        // Disable UI
        document.body.style.cursor = 'wait';
        modelSelect.disabled = true;
        processBtn.disabled = true;

        // Store current as previous for next time
        e.target.setAttribute('data-prev', selectedModel);

        try {
            const response = await fetch('/api/model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_card: selectedModel })
            });
            const data = await response.json();

            if (data.success) {
                alert(`Successfully switched to model: ${selectedModel}`);
            } else {
                throw new Error(data.error || "Unknown error switching model");
            }
        } catch (error) {
            alert(`Error switching model: ${error.message}`);
            // Revert selection if possible, though strict sync with backend might differ
            // We'll just fetch active model again to be sure
            fetchModels();
        } finally {
            document.body.style.cursor = 'default';
            modelSelect.disabled = false;
            if (currentFile) processBtn.disabled = false;
        }
    }

    /* --- Mode Switching Logic --- */

    function setCollectionMode(mode) {
        isCustomMode = (mode === 'custom');

        // Update Buttons
        if (isCustomMode) {
            modePresetBtn.classList.remove('active');
            modeCustomBtn.classList.add('active');

            // UI Changes
            promptText.style.display = 'none';
            customPromptInput.style.display = 'block';
            progressIndicator.style.display = 'none';
            skipBtn.style.display = 'none';
            instructionText.innerText = "Type your text, then record.";
            promptCategory.textContent = "Custom Input";

            // Reset for new custom input
            resetRecordingUI();

        } else {
            modePresetBtn.classList.add('active');
            modeCustomBtn.classList.remove('active');

            // UI Changes
            promptText.style.display = 'block';
            customPromptInput.style.display = 'none';
            progressIndicator.style.display = 'inline-block';
            skipBtn.style.display = 'inline-block';
            instructionText.innerText = "Read the prompt below and record your voice.";

            // Reload current preset
            loadPrompt(currentPromptIndex);
        }
    }

    /* --- Contribute Recording Logic --- */

    async function startContributeRecording() {
        if (isCustomMode && !customPromptInput.value.trim()) {
            alert("Please type some text first!");
            return;
        }

        try {
            audioStream = await navigator.mediaDevices.getUserMedia({
                audio: { channelCount: 1, sampleRate: 16000 }
            });
            mediaRecorder = new MediaRecorder(audioStream, { mimeType: 'audio/webm' });
            audioChunks = [];
            mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
            mediaRecorder.onstop = () => {
                recordingBlob = new Blob(audioChunks, { type: 'audio/webm' });
                audioPlayback.src = URL.createObjectURL(recordingBlob);
                playBtn.disabled = false;
                submitRecordingBtn.disabled = false;
                visualizer.classList.remove('recording');
                document.querySelector('.recording-status-text').textContent = "Recording Stopped";
            };
            mediaRecorder.start();

            recordBtn.style.display = 'none';
            stopBtn.style.display = 'inline-flex';
            stopBtn.disabled = false;
            visualizer.classList.add('recording');
            document.querySelector('.recording-status-text').textContent = "Recording...";

            // Disable input while recording
            if (isCustomMode) customPromptInput.disabled = true;

        } catch (e) {
            alert("Microphone error: " + e.message);
        }
    }

    function stopContributeRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
        if (audioStream) audioStream.getTracks().forEach(t => t.stop());
        if (isCustomMode) customPromptInput.disabled = false;
    }

    function playRecording() {
        if (audioPlayback.src) {
            audioPlayback.play();
        }
    }

    async function submitRecording() {
        if (!recordingBlob) return;

        let id, text, lang;

        if (isCustomMode) {
            text = customPromptInput.value.trim();
            id = "custom_" + Date.now();
            lang = "ben_Beng"; // Default to Bangla for custom, or could add selector
            if (!text) return alert("Text required!");
            customPromptInput.value = ""; // Clear after submit
        } else {
            const prompt = prompts[currentPromptIndex];
            text = prompt.text;
            id = prompt.id;
            lang = prompt.lang_code;
        }

        const formData = new FormData();
        formData.append("audio", recordingBlob, "recording.webm");
        formData.append("prompt_id", id);
        formData.append("transcript", text);
        formData.append("lang_code", lang);

        submitRecordingBtn.disabled = true;
        submitRecordingBtn.textContent = "Submitting...";

        try {
            const response = await fetch('/api/dataset/submit', { method: 'POST', body: formData });
            const result = await response.json();

            if (result.success) {
                contributionMessage.textContent = "Saved successfully!";
                contributionMessage.className = "contribution-message success";
                addToContributionList(text, lang);

                // Only advance if in preset mode
                if (!isCustomMode) {
                    setTimeout(() => {
                        loadPrompt(currentPromptIndex + 1);
                    }, 1000);
                } else {
                    resetRecordingUI(); // Just reset for next custom input
                }
            } else {
                throw new Error(result.error);
            }
        } catch (e) {
            contributionMessage.textContent = "Error: " + e.message;
            contributionMessage.className = "contribution-message error";
        } finally {
            submitRecordingBtn.disabled = false;
            submitRecordingBtn.textContent = "Submit & Next";
        }
    }

    function skipPrompt() {
        loadPrompt(currentPromptIndex + 1);
    }

    function addToContributionList(text, lang) {
        const div = document.createElement('div');
        div.className = "history-item";
        div.style.padding = "1rem";
        div.innerHTML = `<div class="history-item-text">✅ ${text} <small>(${lang})</small></div>`;
        contributionList.prepend(div);

        const emptyState = contributionList.querySelector('.empty-state');
        if (emptyState) emptyState.remove();
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

    /* --- Transcription Recording Logic --- */

    async function startTranscribeRecording() {
        try {
            audioStream = await navigator.mediaDevices.getUserMedia({
                audio: { channelCount: 1, sampleRate: 16000, echoCancellation: true, noiseSuppression: true }
            });

            // Visualizer setup
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            const source = audioContext.createMediaStreamSource(audioStream);
            source.connect(analyser);
            dataArray = new Uint8Array(analyser.frequencyBinCount);

            mediaRecorder = new MediaRecorder(audioStream, { mimeType: 'audio/webm' });
            audioChunks = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) audioChunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                await processTranscribeRecording();
            };

            mediaRecorder.start(100);
            recordingStartTime = Date.now();

            // UI Updates
            transcribeRecordBtn.style.display = 'none';
            transcribeStopBtn.style.display = 'inline-flex';
            transcribeStopBtn.disabled = false;
            transcribeRecordingStatus.style.display = 'flex';
            transcribeAudioVisualizer.style.display = 'block';

            startTranscribeTimer();
            visualizeTranscribeAudio();

        } catch (error) {
            console.error('Error accessing microphone:', error);
            alert('Could not access microphone.');
        }
    }

    function stopTranscribeRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        if (audioStream) {
            audioStream.getTracks().forEach(track => track.stop());
            // Don't nullify yet if we want to reuse? No, better get fresh stream next time.
        }
        stopTranscribeTimer();
        stopTranscribeVisualization();

        // UI Reset
        transcribeRecordBtn.style.display = 'inline-flex';
        transcribeStopBtn.style.display = 'none';
        transcribeRecordingStatus.style.display = 'none';
        transcribeAudioVisualizer.style.display = 'none';
    }

    function startTranscribeTimer() {
        timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            transcribeRecordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
            if (elapsed >= 40) stopTranscribeRecording();
        }, 100);
    }

    function stopTranscribeTimer() {
        clearInterval(timerInterval);
    }

    function visualizeTranscribeAudio() {
        if (!analyser || !transcribeWaveformCanvas) return;
        const ctx = transcribeWaveformCanvas.getContext('2d');
        const width = transcribeWaveformCanvas.width;
        const height = transcribeWaveformCanvas.height;

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

    function stopTranscribeVisualization() {
        cancelAnimationFrame(animationFrameId);
        if (audioContext) audioContext.close();
    }

    async function processTranscribeRecording() {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        try {
            const decodeCtx = new (window.AudioContext || window.webkitAudioContext)();
            const ab = await audioBlob.arrayBuffer();
            const audioBuffer = await decodeCtx.decodeAudioData(ab);
            const wavBlob = audioBufferToWav(audioBuffer);
            await decodeCtx.close();

            const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
            currentFile = wavFile;
            fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
            processBtn.disabled = false;

            // Auto-transcribe
            await handleTranscribe();
        } catch (e) {
            console.error(e);
            alert("Error processing recording.");
        }
    }

    /* --- Contribute Recording Logic (Simplified) --- */

    async function startContributeRecording() {
        try {
            audioStream = await navigator.mediaDevices.getUserMedia({
                audio: { channelCount: 1, sampleRate: 16000 }
            });
            mediaRecorder = new MediaRecorder(audioStream, { mimeType: 'audio/webm' });
            audioChunks = [];
            mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
            mediaRecorder.onstop = () => {
                recordingBlob = new Blob(audioChunks, { type: 'audio/webm' });
                audioPlayback.src = URL.createObjectURL(recordingBlob);
                playBtn.disabled = false;
                submitRecordingBtn.disabled = false;
                visualizer.classList.remove('recording');
                document.querySelector('.recording-status-text').textContent = "Recording Stopped";
            };
            mediaRecorder.start();

            recordBtn.style.display = 'none';
            stopBtn.style.display = 'inline-flex';
            stopBtn.disabled = false;
            visualizer.classList.add('recording');
            document.querySelector('.recording-status-text').textContent = "Recording...";
        } catch (e) {
            alert("Microphone error: " + e.message);
        }
    }

    function stopContributeRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
        if (audioStream) audioStream.getTracks().forEach(t => t.stop());
    }

    // Event Listeners
    processBtn.addEventListener('click', handleTranscribe);
    downloadBtn.addEventListener('click', handleDownload);
    refreshHistoryBtn.addEventListener('click', loadHistory);

    // Transcribe Recording Events
    if (transcribeRecordBtn) {
        transcribeRecordBtn.addEventListener('click', startTranscribeRecording);
        transcribeStopBtn.addEventListener('click', stopTranscribeRecording);
    }

    // Contribute Recording Events
    recordBtn.addEventListener('click', startContributeRecording);
    stopBtn.addEventListener('click', stopContributeRecording);
    playBtn.addEventListener('click', playRecording);
    submitRecordingBtn.addEventListener('click', submitRecording);
    skipBtn.addEventListener('click', skipPrompt);

    modePresetBtn.addEventListener('click', () => setCollectionMode('preset'));
    modeCustomBtn.addEventListener('click', () => setCollectionMode('custom'));

    // Initialize
    fetchModels();
    loadHistory();
});
