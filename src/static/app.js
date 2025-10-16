// Variáveis Globais
let currentSessionId = null;
let progressInterval = null;
let currentMode = 'auto';
let videoInfo = null;
let timelineStartTime = 0;
let timelineEndTime = 0;
let uploadedFileName = null;
let watermarkPath = '';

// Inicialização
document.addEventListener('DOMContentLoaded', function() {
    console.log('✓ Globinho AI inicializado');
    initializeEventListeners();
    loadSettings();
});

function initializeEventListeners() {
    // Navegação
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', function() {
            switchTab(this.dataset.tab);
        });
    });
    
    // Upload de Vídeo
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const uploadBtn = dropArea.querySelector('.upload-btn');
    
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('drag-over'), false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('drag-over'), false);
    });
    
    dropArea.addEventListener('drop', handleDrop);
    
    // Upload de Marca d'água
    const watermarkUpload = document.getElementById('watermark-upload');
    const watermarkInput = document.getElementById('watermark-input');
    const removeWatermark = document.getElementById('remove-watermark');
    
    watermarkUpload.addEventListener('click', () => watermarkInput.click());
    watermarkInput.addEventListener('change', handleWatermarkUpload);
    removeWatermark.addEventListener('click', removeWatermarkFile);
    
    // Timeline
    initializeTimeline();
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
        document.getElementById('file-input').files = files;
        handleFileSelect();
    }
}

// ========== UPLOAD DE MARCA D'ÁGUA ==========
async function handleWatermarkUpload() {
    const watermarkInput = document.getElementById('watermark-input');
    if (!watermarkInput.files || !watermarkInput.files[0]) return;
    
    const file = watermarkInput.files[0];
    const formData = new FormData();
    formData.append('watermark', file);
    
    try {
        const response = await fetch('/upload-watermark', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            watermarkPath = data.path;
            
            // Mostrar preview
            const preview = document.getElementById('watermark-preview');
            preview.src = URL.createObjectURL(file);
            preview.style.display = 'block';
            
            // Mostrar botão de remover
            document.getElementById('remove-watermark').classList.remove('hidden');
            
            console.log('✓ Marca d\'água carregada:', watermarkPath);
        }
    } catch (error) {
        alert('Erro ao fazer upload da marca d\'água: ' + error.message);
        console.error(error);
    }
}

function removeWatermarkFile() {
    watermarkPath = '';
    document.getElementById('watermark-preview').style.display = 'none';
    document.getElementById('watermark-input').value = '';
    document.getElementById('remove-watermark').classList.add('hidden');
    console.log('✓ Marca d\'água removida');
}

// ========== GERENCIAMENTO DE ABAS ==========
function switchTab(tabName) {
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById('tab-' + tabName).classList.add('active');
    
    if (tabName === 'dashboard') loadDashboard();
    else if (tabName === 'history') loadHistory();
}

// ========== UPLOAD E PREVIEW ==========
async function handleFileSelect() {
    const fileInput = document.getElementById('file-input');
    if (!fileInput.files || !fileInput.files[0]) return;
    
    const file = fileInput.files[0];
    uploadedFileName = file.name;
    
    // Preparar preferências antes do upload
    const preferences = getPreferences();
    
    // Fazer upload do arquivo
    const formData = new FormData();
    formData.append('video', file);
    formData.append('preferences', JSON.stringify(preferences));
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            alert('Erro no upload: ' + (data.error || 'Erro desconhecido'));
            return;
        }
        
        currentSessionId = data.session_id;
        
        // Carregar preview
        await loadVideoPreview();
        
        // Mostrar tela de preview
        document.getElementById('settings-card').classList.add('hidden');
        document.getElementById('upload-view').classList.add('hidden');
        document.getElementById('preview-view').classList.remove('hidden');
        
    } catch (error) {
        alert('Erro ao enviar arquivo: ' + error.message);
        console.error(error);
    }
}

async function loadVideoPreview() {
    try {
        const response = await fetch(`/preview/${currentSessionId}`);
        const data = await response.json();
        
        if (data.error) {
            console.error('Erro ao carregar preview:', data.error);
            return;
        }
        
        videoInfo = data;
        
        // Configurar player de vídeo
        const videoPlayer = document.getElementById('video-preview');
        videoPlayer.src = `/video/${data.filename}`;
        
        // Configurar timeline
        timelineEndTime = data.duration;
        updateTimelineDisplay();
        updateManualTimeInputs();
        drawTimelineFrames();
        
        console.log('✓ Preview carregado:', data);
        
    } catch (error) {
        console.error('Erro ao carregar preview:', error);
    }
}

// ========== SELEÇÃO DE MODO ==========
function selectMode(mode) {
    currentMode = mode;
    
    document.getElementById('mode-auto').classList.remove('active');
    document.getElementById('mode-manual').classList.remove('active');
    
    if (mode === 'auto') {
        document.getElementById('mode-auto').classList.add('active');
        document.getElementById('timeline-section').classList.add('hidden');
    } else {
        document.getElementById('mode-manual').classList.add('active');
        document.getElementById('timeline-section').classList.remove('hidden');
    }
}

// ========== TIMELINE INTERATIVA ==========
function initializeTimeline() {
    const canvas = document.getElementById('timeline-canvas');
    const markerStart = document.getElementById('marker-start');
    const markerEnd = document.getElementById('marker-end');
    
    // Arrastar marcadores
    let isDragging = false;
    let currentMarker = null;
    
    markerStart.addEventListener('mousedown', (e) => {
        isDragging = true;
        currentMarker = 'start';
        e.preventDefault();
    });
    
    markerEnd.addEventListener('mousedown', (e) => {
        isDragging = true;
        currentMarker = 'end';
        e.preventDefault();
    });
    
    // Suporte para toque em dispositivos móveis
    markerStart.addEventListener('touchstart', (e) => {
        isDragging = true;
        currentMarker = 'start';
        e.preventDefault();
    });
    
    markerEnd.addEventListener('touchstart', (e) => {
        isDragging = true;
        currentMarker = 'end';
        e.preventDefault();
    });
    
    document.addEventListener('mousemove', handleMarkerDrag);
    document.addEventListener('touchmove', handleMarkerDrag);
    
    document.addEventListener('mouseup', () => {
        isDragging = false;
        currentMarker = null;
    });
    
    document.addEventListener('touchend', () => {
        isDragging = false;
        currentMarker = null;
    });
}

function handleMarkerDrag(e) {
    if (!isDragging || !currentMarker || !videoInfo) return;
    
    const wrapper = document.querySelector('.timeline-canvas-wrapper');
    const rect = wrapper.getBoundingClientRect();
    
    // Suportar tanto mouse quanto toque
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const x = clientX - rect.left;
    const percent = Math.max(0, Math.min(100, (x / rect.width) * 100));
    
    const markerStart = document.getElementById('marker-start');
    const markerEnd = document.getElementById('marker-end');
    
    if (currentMarker === 'start') {
        const endPercent = parseFloat(markerEnd.style.right) || 0;
        if (percent < (100 - endPercent)) {
            markerStart.style.left = percent + '%';
            timelineStartTime = (percent / 100) * videoInfo.duration;
            updateTimelineDisplay();
            updateManualTimeInputs();
        }
    } else if (currentMarker === 'end') {
        const startPercent = parseFloat(markerStart.style.left) || 0;
        if (percent > startPercent) {
            markerEnd.style.right = (100 - percent) + '%';
            timelineEndTime = (percent / 100) * videoInfo.duration;
            updateTimelineDisplay();
            updateManualTimeInputs();
        }
    }
}

function setMarkerFromVideo(type) {
    const video = document.getElementById('video-preview');
    const currentTime = video.currentTime;
    const percent = (currentTime / videoInfo.duration) * 100;
    
    if (type === 'start') {
        document.getElementById('marker-start').style.left = percent + '%';
        timelineStartTime = currentTime;
    } else {
        document.getElementById('marker-end').style.right = (100 - percent) + '%';
        timelineEndTime = currentTime;
    }
    
    updateTimelineDisplay();
    updateManualTimeInputs();
}

function applyManualTimes() {
    const startInput = document.getElementById('manual-start-time').value;
    const endInput = document.getElementById('manual-end-time').value;
    
    const startSeconds = parseTimeString(startInput);
    const endSeconds = parseTimeString(endInput);
    
    if (startSeconds === null || endSeconds === null) {
        alert('❌ Formato de tempo inválido. Use HH:MM:SS ou MM:SS');
        return;
    }
    
    if (startSeconds >= endSeconds) {
        alert('❌ O tempo de início deve ser menor que o tempo de fim');
        return;
    }
    
    if (endSeconds > videoInfo.duration) {
        alert('❌ O tempo de fim excede a duração do vídeo');
        return;
    }
    
    timelineStartTime = startSeconds;
    timelineEndTime = endSeconds;
    
    const startPercent = (startSeconds / videoInfo.duration) * 100;
    const endPercent = (endSeconds / videoInfo.duration) * 100;
    
    document.getElementById('marker-start').style.left = startPercent + '%';
    document.getElementById('marker-end').style.right = (100 - endPercent) + '%';
    
    updateTimelineDisplay();
    alert('✅ Tempos aplicados com sucesso!');
}

function parseTimeString(timeStr) {
    if (!timeStr || timeStr.trim() === '') return null;
    
    const parts = timeStr.split(':').map(p => parseInt(p));
    
    if (parts.some(isNaN)) return null;
    
    if (parts.length === 3) {
        const [h, m, s] = parts;
        return h * 3600 + m * 60 + s;
    } else if (parts.length === 2) {
        const [m, s] = parts;
        return m * 60 + s;
    } else if (parts.length === 1) {
        return parts[0];
    }
    
    return null;
}

function updateManualTimeInputs() {
    const startInput = document.getElementById('manual-start-time');
    const endInput = document.getElementById('manual-end-time');
    
    if (startInput && endInput) {
        startInput.value = formatTime(timelineStartTime);
        endInput.value = formatTime(timelineEndTime);
    }
}

function applyManualTimes() {
    const startInput = document.getElementById('manual-start-time').value;
    const endInput = document.getElementById('manual-end-time').value;
    
    if (!startInput || !endInput) {
        alert('❌ Por favor, preencha os dois campos de tempo (início e fim)');
        return;
    }
    
    const startSeconds = parseTimeString(startInput);
    const endSeconds = parseTimeString(endInput);
    
    if (startSeconds === null || endSeconds === null) {
        alert('❌ Formato de tempo inválido. Use HH:MM:SS ou MM:SS\n\nExemplos:\n• 00:01:30 (1 minuto e 30 segundos)\n• 01:30 (1 minuto e 30 segundos)');
        return;
    }
    
    if (startSeconds >= endSeconds) {
        alert('❌ O tempo de início deve ser MENOR que o tempo de fim');
        return;
    }
    
    if (endSeconds > videoInfo.duration) {
        alert(`❌ O tempo de fim (${formatTime(endSeconds)}) excede a duração do vídeo (${formatTime(videoInfo.duration)})`);
        return;
    }
    
    // Aplicar os tempos
    timelineStartTime = startSeconds;
    timelineEndTime = endSeconds;
    
    // Atualizar marcadores visuais
    const startPercent = (startSeconds / videoInfo.duration) * 100;
    const endPercent = (endSeconds / videoInfo.duration) * 100;
    
    document.getElementById('marker-start').style.left = startPercent + '%';
    document.getElementById('marker-end').style.right = (100 - endPercent) + '%';
    
    updateTimelineDisplay();
    
    // Atualizar o player de vídeo
    const videoPlayer = document.getElementById('video-preview');
    videoPlayer.currentTime = startSeconds;
    
    alert(`✅ Tempos aplicados com sucesso!\n\n⏮️ Início: ${formatTime(startSeconds)}\n⏭️ Fim: ${formatTime(endSeconds)}\n⏱️ Duração: ${formatTime(endSeconds - startSeconds)}`);
}

function resetTimeline() {
    document.getElementById('marker-start').style.left = '0%';
    document.getElementById('marker-end').style.right = '0%';
    timelineStartTime = 0;
    timelineEndTime = videoInfo.duration;
    updateTimelineDisplay();
    updateManualTimeInputs();
}

function updateTimelineDisplay() {
    if (!videoInfo) return;
    
    const startPercent = (timelineStartTime / videoInfo.duration) * 100;
    const endPercent = (timelineEndTime / videoInfo.duration) * 100;
    const selection = document.getElementById('timeline-selection');
    
    selection.style.left = startPercent + '%';
    selection.style.width = (endPercent - startPercent) + '%';
    
    document.getElementById('time-start').textContent = formatTime(timelineStartTime);
    document.getElementById('time-end').textContent = formatTime(timelineEndTime);
    document.getElementById('time-duration').textContent = formatTime(timelineEndTime - timelineStartTime);
}

function drawTimelineFrames() {
    const canvas = document.getElementById('timeline-canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    // Desenhar fundo gradiente
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
    gradient.addColorStop(0, '#06A4E2');
    gradient.addColorStop(0.5, '#90C43D');
    gradient.addColorStop(1, '#06A4E2');
    
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Desenhar marcadores de tempo
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    const numMarkers = 10;
    for (let i = 0; i <= numMarkers; i++) {
        const x = (canvas.width / numMarkers) * i;
        ctx.fillRect(x, 0, 2, canvas.height);
    }
}

function formatTime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

// ========== PROCESSAMENTO ==========
async function startProcessing() {
    const preferences = getPreferences();
    
    // Adicionar configurações de corte se modo manual
    if (currentMode === 'manual') {
        preferences.start_time = formatTime(timelineStartTime);
        preferences.end_time = formatTime(timelineEndTime);
    }
    
    // Esconder preview, mostrar processamento
    document.getElementById('preview-view').classList.add('hidden');
    document.getElementById('processing-view').classList.remove('hidden');
    
    // Iniciar polling de progresso
    startProgressPolling(currentSessionId);
}

function startProgressPolling(sessionId) {
    let logUpdateCount = 0;
    
    progressInterval = setInterval(async () => {
        try {
            const response = await fetch('/status/' + sessionId);
            const data = await response.json();
            
            // Atualizar progresso
            if (data.progress !== undefined) {
                document.getElementById('progress-fill').style.width = data.progress + '%';
                document.getElementById('progress-percent').textContent = Math.round(data.progress) + '%';
                document.getElementById('status-text').textContent = data.stage || 'Processando...';
            }
            
            // Atualizar logs
            if (data.logs && data.logs.length > logUpdateCount) {
                const logContainer = document.getElementById('log-container');
                const newLogs = data.logs.slice(logUpdateCount);
                
                newLogs.forEach(log => {
                    const entry = document.createElement('div');
                    entry.className = 'log-entry';
                    entry.innerHTML = `
                        <span class="log-time">${log.time}</span>
                        <span class="log-message">${log.message}</span>
                    `;
                    logContainer.appendChild(entry);
                });
                
                logUpdateCount = data.logs.length;
                logContainer.scrollTop = logContainer.scrollHeight;
            }
            
            // Verificar conclusão
            if (data.done) {
                clearInterval(progressInterval);
                
                if (data.error) {
                    alert('❌ Erro no processamento: ' + data.error);
                    resetUI();
                } else if (data.result && data.result.files && data.result.files.length > 0) {
                    showResults(data.result);
                } else {
                    alert('⚠️ Nenhum clipe foi gerado. Verifique se o vídeo contém fala ou ajuste as configurações.');
                    resetUI();
                }
            }
            
        } catch (error) {
            clearInterval(progressInterval);
            alert('Erro de conexão: ' + error.message);
            resetUI();
        }
    }, 1500);
}

function showResults(result) {
    document.getElementById('processing-view').classList.add('hidden');
    document.getElementById('results-view').classList.remove('hidden');
    
    const resultsList = document.getElementById('results-list');
    resultsList.innerHTML = '';
    
    // Mostrar TODOS os clipes gerados
    result.files.forEach((file, index) => {
        const card = document.createElement('div');
        card.className = 'result-card';
        card.innerHTML = `
            <div class="result-title">🎬 Clipe ${index + 1}</div>
            <video class="video-preview" controls preload="metadata">
                <source src="/done/${file}" type="video/mp4">
            </video>
            <div style="margin-top:1rem; display: flex; flex-wrap: wrap; gap: 0.5rem;">
                <a href="/done/${file}" download class="btn btn-primary">⬇️ Vídeo</a>
                <a href="/done/${file.replace('.mp4','_post.txt')}" download class="btn btn-secondary">📱 Post Instagram</a>
                <a href="/done/${file.replace('.mp4','_analise.txt')}" download class="btn btn-secondary">📊 Insights</a>
            </div>
        `;
        resultsList.appendChild(card);
    });
    
    // Botão de download geral
    const downloadAllBtn = document.getElementById('download-all-btn');
    downloadAllBtn.classList.remove('hidden');
    downloadAllBtn.onclick = () => {
        window.location.href = '/download-all/' + result.session_id;
    };
    
    // Mostrar analytics
    if (result.analytics) {
        document.getElementById('analytics-section').classList.remove('hidden');
        document.getElementById('stat-clips').textContent = result.analytics.total_clips;
        document.getElementById('stat-score').textContent = Math.round(result.analytics.avg_score);
        document.getElementById('stat-duration').textContent = Math.round(result.analytics.total_duration) + 's';
        document.getElementById('stat-sentiment').textContent = result.analytics.sentiment.sentiment;
        document.getElementById('stat-time').textContent = Math.round(result.analytics.processing_time || 0) + 's';
    }
    
    loadHistory();
    loadDashboard();
}

function resetUI() {
    document.getElementById('settings-card').classList.remove('hidden');
    document.getElementById('upload-view').classList.remove('hidden');
    document.getElementById('preview-view').classList.add('hidden');
    document.getElementById('processing-view').classList.add('hidden');
    document.getElementById('results-view').classList.add('hidden');
    document.getElementById('file-input').value = '';
    
    // Resetar valores
    currentSessionId = null;
    videoInfo = null;
    currentMode = 'auto';
    timelineStartTime = 0;
    timelineEndTime = 0;
    
    // Limpar log
    document.getElementById('log-container').innerHTML = `
        <div class="log-entry">
            <span class="log-time">--:--:--</span>
            <span class="log-message">Aguardando início do processamento...</span>
        </div>
    `;
}

// ========== PREFERÊNCIAS ==========
function getPreferences() {
    return {
        subtitle_font: document.getElementById('subtitle-font')?.value || 'Courier-New-Bold',
        subtitle_size: parseInt(document.getElementById('subtitle-size')?.value || 70),
        with_subtitles: document.getElementById('with-subtitles')?.checked || true,
        video_speed: parseFloat(document.getElementById('video-speed')?.value || 1.0),
        num_clips: parseInt(document.getElementById('num-clips')?.value || 3),
        min_duration: 30,
        max_duration: 120,
        whisper_model: document.getElementById('whisper-model')?.value || 'base',
        fast_mode: false,
        watermark_path: watermarkPath
    };
}

async function savePreferences(scope) {
    const prefs = getPreferences();
    
    if (scope === 'all') {
        try {
            await fetch('/save-preferences', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(prefs)
            });
            alert('✅ Configurações salvas para todos os próximos vídeos!');
        } catch {
            alert('❌ Erro ao salvar configurações.');
        }
    } else {
        // Apenas para este vídeo (não salvar permanentemente)
        alert('✅ Configurações aplicadas apenas para este vídeo!');
    }
}

async function loadSettings() {
    try {
        const response = await fetch('/get-preferences');
        const prefs = await response.json();
        
        if (document.getElementById('subtitle-size')) {
            document.getElementById('subtitle-size').value = prefs.subtitle_size || 70;
        }
        if (document.getElementById('num-clips')) {
            document.getElementById('num-clips').value = prefs.num_clips || 3;
        }
        if (document.getElementById('whisper-model')) {
            document.getElementById('whisper-model').value = prefs.whisper_model || 'base';
        }
        if (document.getElementById('video-speed')) {
            document.getElementById('video-speed').value = prefs.video_speed || 1.0;
        }
        if (document.getElementById('subtitle-font')) {
            document.getElementById('subtitle-font').value = prefs.subtitle_font || 'Courier-New-Bold';
        }
        if (document.getElementById('with-subtitles')) {
            document.getElementById('with-subtitles').checked = prefs.with_subtitles !== false;
        }
    } catch (error) {
        console.error('Erro ao carregar configurações:', error);
    }
}

// ========== HISTÓRICO ==========
async function loadHistory() {
    try {
        const response = await fetch('/history');
        const history = await response.json();
        const container = document.getElementById('history-list');
        
        if (!history || history.length === 0) {
            container.innerHTML = '<p style="text-align:center;padding:2rem;color:rgba(255,255,255,0.5)">📭 Nenhum processamento ainda.</p>';
            return;
        }
        
        container.innerHTML = '';
        
        history.forEach(item => {
            const date = new Date(item.date);
            const div = document.createElement('div');
            div.className = 'history-item';
            div.innerHTML = `
                <div style="color:rgba(255,255,255,0.6);font-size:0.85rem">${date.toLocaleString('pt-BR')}</div>
                <div style="font-weight:700;color:#06A4E2;font-size:1.1rem;margin:0.5rem 0">
                    ${item.video_name}
                </div>
                <p style="color:rgba(255,255,255,0.7)">
                    ${item.clips_count} clipes • Score: ${Math.round(item.avg_score)}
                </p>
                <div style="margin-top:1rem">
                    ${item.clips.map(c => `
                        <a href="/done/${c.file}" download class="btn btn-secondary">
                            ⬇️ ${c.file.split('_').pop()}
                        </a>
                    `).join('')}
                </div>
            `;
            container.appendChild(div);
        });
        
    } catch (error) {
        console.error('Erro ao carregar histórico:', error);
    }
}

// ========== DASHBOARD ==========
async function loadDashboard() {
    try {
        const response = await fetch('/analytics');
        const data = await response.json();
        
        // Atualizar estatísticas
        document.getElementById('total-clips').textContent = data.total_clips || 0;
        document.getElementById('avg-score').textContent = Math.round(data.avg_score || 0);
        document.getElementById('total-sessions').textContent = data.sessions?.length || 0;
        document.getElementById('avg-duration-stat').textContent = Math.round(data.avg_duration || 0) + 's';
        
        // Gráfico de Palavras-Chave
        renderKeywordsChart(data);
        
        // Gráfico de Narrativa
        renderNarrativeChart(data);
        
        // Gráfico de Sentimento
        renderSentimentChart(data);
        
    } catch (error) {
        console.error('Erro ao carregar dashboard:', error);
    }
}

function renderKeywordsChart(data) {
    const ctx = document.getElementById('keywordsChart')?.getContext('2d');
    if (!ctx) return;
    
    // Destruir gráfico anterior se existir
    if (window.keywordsChartInstance) {
        window.keywordsChartInstance.destroy();
    }
    
    const keywords = data.keywords || [];
    const keywordCount = {};
    
    keywords.forEach(word => {
        keywordCount[word] = (keywordCount[word] || 0) + 1;
    });
    
    const sortedKeywords = Object.entries(keywordCount)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10);
    
    const labels = sortedKeywords.map(k => k[0]);
    const values = sortedKeywords.map(k => k[1]);
    
    if (labels.length === 0) {
        ctx.font = '16px Montserrat';
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.textAlign = 'center';
        ctx.fillText('Nenhum dado disponível ainda', ctx.canvas.width / 2, ctx.canvas.height / 2);
        return;
    }
    
    window.keywordsChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Frequência',
                data: values,
                backgroundColor: 'rgba(144, 196, 61, 0.8)',
                borderColor: '#90C43D',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { 
                legend: { display: false },
                title: {
                    display: true,
                    text: 'Top 10 Palavras-Chave',
                    color: 'white',
                    font: { size: 16 }
                }
            },
            scales: {
                y: { 
                    beginAtZero: true, 
                    ticks: { color: 'white' }, 
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                x: { 
                    ticks: { color: 'white' }, 
                    grid: { display: false }
                }
            }
        }
    });
}

function renderNarrativeChart(data) {
    const ctx = document.getElementById('narrativeChart')?.getContext('2d');
    if (!ctx) return;
    
    // Destruir gráfico anterior se existir
    if (window.narrativeChartInstance) {
        window.narrativeChartInstance.destroy();
    }
    
    const narratives = data.narratives || {INTRODUCAO: 0, CONTEXTO: 0, CLIMAX: 0};
    const labels = Object.keys(narratives);
    const values = Object.values(narratives);
    
    const total = values.reduce((a, b) => a + b, 0);
    
    if (total === 0) {
        ctx.font = '16px Montserrat';
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.textAlign = 'center';
        ctx.fillText('Nenhum dado disponível ainda', ctx.canvas.width / 2, ctx.canvas.height / 2);
        return;
    }
    
    window.narrativeChartInstance = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels.map(l => {
                const map = {
                    'INTRODUCAO': 'Introdução',
                    'CONTEXTO': 'Contexto',
                    'CLIMAX': 'Clímax'
                };
                return map[l] || l;
            }),
            datasets: [{
                data: values,
                backgroundColor: ['#06A4E2', '#90C43D', '#FFA500']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { 
                legend: { 
                    position: 'right', 
                    labels: { color: 'white', font: { size: 14 } }
                },
                title: {
                    display: true,
                    text: 'Distribuição dos Tipos Narrativos',
                    color: 'white',
                    font: { size: 16 }
                }
            }
        }
    });
}

function renderSentimentChart(data) {
    const ctx = document.getElementById('sentimentChart')?.getContext('2d');
    if (!ctx) return;
    
    // Destruir gráfico anterior se existir
    if (window.sentimentChartInstance) {
        window.sentimentChartInstance.destroy();
    }
    
    const sentiments = data.sentiments || {};
    const labels = Object.keys(sentiments);
    const values = Object.values(sentiments);
    
    const total = values.reduce((a, b) => a + b, 0);
    
    if (total === 0) {
        ctx.font = '16px Montserrat';
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.textAlign = 'center';
        ctx.fillText('Nenhum dado disponível ainda', ctx.canvas.width / 2, ctx.canvas.height / 2);
        return;
    }
    
    const sentimentColors = {
        'URGENTE': '#FF6B6B',
        'ALERTA': '#FFA500',
        'SERIO': '#4ECDC4',
        'POSITIVO': '#90C43D',
        'NEUTRO': '#A0A0A0'
    };
    
    const colors = labels.map(label => sentimentColors[label] || '#A0A0A0');
    
    window.sentimentChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderWidth: 2,
                borderColor: '#161B22'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { 
                legend: { 
                    position: 'right', 
                    labels: { color: 'white', font: { size: 14 } }
                },
                title: {
                    display: true,
                    text: 'Análise de Sentimentos dos Conteúdos',
                    color: 'white',
                    font: { size: 16 }
                }
            }
        }
    });
}

console.log('✓ Globinho AI carregado com sucesso');