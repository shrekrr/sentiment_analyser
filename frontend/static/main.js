const API_BASE = "http://localhost:8000";
let timelineChartInstance = null;
let lastResult = null;

function populate(text) { document.getElementById('textInput').value = text; }

function switchTab(tabId, event) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
    
    event.target.classList.add('active');
    document.getElementById(tabId).style.display = 'block';
}

function rgbaAccent(opacity) { return `rgba(232, 255, 71, ${opacity})`; }

async function analyzeText() {
    const text = document.getElementById('textInput').value.trim();
    if (!text) return;
    
    const btn = document.getElementById('analyzeBtn');
    const btnText = document.getElementById('btnText');
    const resultsSec = document.getElementById('resultsSection');
    
    btn.classList.add('loading');
    btnText.innerText = 'Analysing...';
    btn.disabled = true;
    resultsSec.classList.remove('visible');
    
    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text: text})
        });
        
        if (!response.ok) throw new Error('API Error');
        const data = await response.json();
        renderResults(data);
        
        setTimeout(() => { resultsSec.classList.add('visible'); triggerAnimations(); }, 100);
    } catch (e) {
        alert("Error connecting to API.");
    } finally {
        btn.classList.remove('loading');
        btnText.innerText = 'Analyse';
        btn.disabled = false;
    }
}

function renderResults(data) {
    const sentLabel = document.getElementById('resLabel');
    sentLabel.innerText = data.sentiment.label;
    sentLabel.className = `sentiment-label serif ${data.sentiment.label}`;
    document.getElementById('resConfidence').innerText = `${(data.sentiment.confidence * 100).toFixed(1)}% confidence`;
    
    const sScores = data.sentiment.scores;
    let compoundScore = (sScores.positive - sScores.negative); 
    const bar = document.getElementById('scoreBar');
    bar.className = 'score-bar';
    bar.dataset.width = Math.abs(compoundScore * 100) + '%';
    if(compoundScore < 0) bar.classList.add('neg');
    else if(compoundScore > 0) bar.classList.add('pos');
    
    const bc = document.getElementById('badgesContainer');
    bc.innerHTML = '';
    if (data.sarcasm.detected) bc.innerHTML += `<div class="badge sarcasm">⚠ Sarcasm detected · ${(data.sarcasm.confidence * 100).toFixed(0)}%</div>`;
    if (data.contrastive_shift) bc.innerHTML += `<div class="badge contrast">⇅ Contrastive shift detected</div>`;
    
    document.getElementById('resReasoning').innerText = data.reasoning;
    
    const eb = document.getElementById('emotionBars');
    eb.innerHTML = '';
    Object.entries(data.emotions).sort((a,b) => b[1] - a[1]).forEach(([emo, score], idx) => {
        const isDom = emo === data.dominant_emotion;
        const width = (score * 100).toFixed(1);
        eb.innerHTML += `
            <div class="emotion-row ${isDom ? 'dominant' : ''}" id="emo-row-${idx}">
                <div class="emotion-name">${emo}</div>
                <div class="emotion-bar-bg"><div class="emotion-bar-fill" data-width="${width}%" id="emo-fill-${idx}"></div></div>
                <div class="emotion-score">${width}%</div>
            </div>`;
    });
    
    const attn = document.getElementById('attnHeatmap');
    attn.innerHTML = '';
    data.attention.tokens.forEach((tok, i) => {
        let cleanTok = tok.replace('##', '');
        const bg = tok !== '[CLS]' && tok !== '[SEP]' ? rgbaAccent(data.attention.weights[i] * 0.8) : 'transparent';
        attn.innerHTML += `<span class="attn-token" style="background-color: ${bg}; ${tok.startsWith('##') ? 'margin-left:-4px;' : ''}">${cleanTok}</span>`;
    });
    
    document.getElementById('jsonOutput').innerText = JSON.stringify(data, null, 2);
}

function triggerAnimations() {
    setTimeout(() => { const b=document.getElementById('scoreBar'); b.style.width=b.dataset.width; }, 300);
    document.querySelectorAll('.emotion-row').forEach((row, i) => {
        setTimeout(() => {
            row.style.opacity = 1; row.style.transform = 'translateX(0)';
            const fill = document.getElementById(`emo-fill-${i}`);
            fill.style.width = fill.dataset.width;
        }, 100 + (i * 50));
    });
}

async function analyzeBatch() {
    const texts = document.getElementById('batchInput').value.split('\n').map(t => t.trim()).filter(t => t.length > 0);
    if(texts.length === 0) return;
    
    const btn = document.getElementById('batchBtn');
    btn.innerText = "Analyzing..."; btn.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/predict-batch`, {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({texts: texts})
        });
        const data = await response.json();
        renderTimeline(data);
    } catch(e) { alert("Batch error."); } finally { btn.innerText = "Analyze Batch"; btn.disabled = false; }
}

function renderTimeline(results) {
    const ctx = document.getElementById('timelineChart').getContext('2d');
    const labels = results.map((_, i) => `T${i+1}`);
    const dataPoints = results.map(r => r.sentiment.scores.positive - r.sentiment.scores.negative);
    const pointColors = results.map(r => r.sarcasm.detected ? '#FB923C' : (r.sentiment.label === 'positive' ? '#4ADE80' : (r.sentiment.label === 'negative' ? '#F87171' : '#94A3B8')));
    const tooltips = results.map(r => r.text.length > 50 ? r.text.substring(0,47)+'...' : r.text);
    
    if(timelineChartInstance) timelineChartInstance.destroy();
    
    Chart.defaults.color = '#888888';
    Chart.defaults.font.family = "'DM Mono', monospace";
    
    timelineChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Score', data: dataPoints, borderColor: '#2A2A2A', borderWidth: 2,
                pointBackgroundColor: pointColors, pointBorderColor: '#0A0A0A', pointRadius: 6,
                pointHoverRadius: 8, fill: true, backgroundColor: 'rgba(26, 26, 26, 0.5)'
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: { y: { min: -1, max: 1, grid: { color: '#1A1A1A' } }, x: { grid: { color: '#1A1A1A' } } },
            plugins: { tooltip: { callbacks: { title: (items) => tooltips[items[0].dataIndex], label: (ctx) => `Score: ${ctx.raw.toFixed(2)}` } }, legend: { display: false } }
        }
    });
}
