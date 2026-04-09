const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
    ? 'http://127.0.0.1:8787' 
    : 'https://rag-worker.abhijaydutta123.workers.dev';

const chatHistory = document.getElementById('chatHistory');
const chatInput   = document.getElementById('chatInput');
const sendBtn     = document.getElementById('sendBtn');
const kbStatus    = document.getElementById('kbStatus');

// Configure Marked.js for safe rendering
marked.setOptions({
    headerIds: false,
    mangle: false,
    breaks: true,
    gfm: true
});

// ── Suggestion Items ──
document.querySelectorAll('.suggestion-item').forEach(item => {
    item.addEventListener('click', () => {
        chatInput.value = item.dataset.q;
        chatInput.focus();
        sendMessage();
    });
});

// ── Append Message ──
function appendMessage(role, content, meta = null) {
    // Remove welcome block on first message
    const welcome = chatHistory.querySelector('.welcome-msg');
    if (welcome) welcome.remove();

    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    let html = '';
    
    // Avatar column
    if (role === 'ai') {
        html += `<div class="msg-avatar">
                   <img src="momo.gif" alt="Miss MoMo" class="momo-idle-anim">
                 </div>`;
    }

    // Content column
    html += `<div class="msg-content">`;
    if (role === 'user') {
        html += `<div class="msg-role">You</div>`;
    }

    const renderedContent = role === 'ai' ? marked.parse(content) : content;
    html += `<div class="msg-bubble">${renderedContent}</div>`;

    if (role === 'ai' && meta) {
        // Source badges
        if (meta.sources && meta.sources.length > 0) {
            html += `<div class="source-badges">`;
            meta.sources.forEach(s => {
                html += `<span class="badge">📄 ${s}</span>`;
            });
            html += `</div>`;
        }
        
        // Metadata chips (Tokens)
        html += `<div class="msg-meta">
            <span>Input: ${meta.input_tokens} tok</span>
            <span>Output: ${meta.output_tokens} tok</span>
        </div>`;

        // Embedded images from retrieved chunks (Docling images)
        if (meta.images && meta.images.length > 0) {
            html += `<div class="msg-images">
                <div class="images-label">📐 Related Diagrams</div>`;
            meta.images.forEach((src, idx) => {
                html += `<img src="${src}" alt="Diagram ${idx + 1}" class="inline-diagram" loading="lazy">`;
            });
            html += `</div>`;
        }
    }

    html += `</div>`; // Close msg-content
    msgDiv.innerHTML = html;
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return msgDiv;
}

// ── Thinking Indicator ──
function showThinking() {
    const thinkingDiv = document.createElement('div');
    thinkingDiv.className = 'message ai';
    thinkingDiv.id = 'thinking-indicator';
    thinkingDiv.innerHTML = `
        <div class="msg-avatar">
            <img src="Thinking_Momo.gif" alt="Miss MoMo Thinking" class="momo-idle-anim">
        </div>
        <div class="msg-content">
            <div class="msg-bubble">
                <div class="thinking-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>`;
    chatHistory.appendChild(thinkingDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function hideThinking() {
    const el = document.getElementById('thinking-indicator');
    if (el) el.remove();
}

// ── Send Message ──
async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    appendMessage('user', query);
    chatInput.value = '';
    sendBtn.disabled = true;
    showThinking();

    // Update status dot (briefly)
    const dot = kbStatus.querySelector('.status-dot');
    dot.className = 'status-dot orange';

    try {
        const res = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query }),
        });
        const data = await res.json();
        hideThinking();

        if (res.ok) {
            const images = [];
            // Parse AI's intentional image tags: [SHOW_IMAGE: 1], [SHOW_IMAGE: 2]
            const showRegex = /\[SHOW_IMAGE:\s*(\d+)\]/g;
            let mShow;
            const requestedIndices = [];
            while ((mShow = showRegex.exec(data.answer)) !== null) {
                requestedIndices.push(parseInt(mShow[1], 10) - 1);
            }
            
            // Remove the raw tags from the text so the user doesn't see them
            const cleanAnswer = data.answer.replace(/\[SHOW_IMAGE:\s*\d+\]/g, '').trim();

            if (data.rich_chunks && requestedIndices.length > 0) {
                // Collect ALL available images in context sequentially
                const allContextImages = [];
                const imgRegex = /!\[[^\]]*\]\((data:image\/[^)]+)\)/g;
                for (const chunk of data.rich_chunks) {
                    let mImg;
                    while ((mImg = imgRegex.exec(chunk)) !== null) {
                        allContextImages.push(mImg[1]);
                    }
                }
                // Only attach the ones the AI explicitly requested
                for (const idx of requestedIndices) {
                    if (allContextImages[idx] && !images.includes(allContextImages[idx])) {
                        images.push(allContextImages[idx]);
                    }
                }
            }

            appendMessage('ai', cleanAnswer, {
                input_tokens:  data.input_tokens  ?? 0,
                output_tokens: data.output_tokens ?? 0,
                sources: data.sources ?? [],
                images,
            });
        } else {
            appendMessage('ai', `**Error**: ${data.error || 'The AI service encountered an issue.'}`);
        }
    } catch (err) {
        hideThinking();
        appendMessage('ai', `**Connection Error**: Could not reach the AI gateway. Please ensure the backend server is running.`);
    } finally {
        sendBtn.disabled = false;
        dot.className = 'status-dot green';
        chatInput.focus();
    }
}

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// ── Initial Health Check ──
(async () => {
    try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (!res.ok) throw new Error();
    } catch {
        const dot = kbStatus.querySelector('.status-dot');
        dot.className = 'status-dot red';
        kbStatus.innerHTML = '<span class="status-dot red"></span> Service Offline';
    }
})();
