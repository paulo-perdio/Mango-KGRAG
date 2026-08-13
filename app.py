from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.mango_kgrag import smart_answer

app = FastAPI()

class Query(BaseModel):
    message: str

@app.post("/ask")
def ask(query: Query):
    result = smart_answer(query.message)
    return result

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🥭 Mango AI</title>
        <meta charset="utf-8">
        <style>
            :root {
                --bg-color: #ffffff;
                --sidebar-bg: #f0f4f9;
                --debug-bg: #f8f9fa;
                --text-main: #1f1f1f;
                --text-secondary: #444746;
                --border-color: #dadce0;
                --user-msg-bg: #f0f4f9;
                --hover-bg: #e8eaed;
            }

            body {
                background: var(--bg-color);
                font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                color: var(--text-main);
                margin: 0;
                display: flex;
                height: 100vh;
                overflow: hidden;
            }

            /* Sidebar */
            #sidebar {
                width: 260px;
                background: var(--sidebar-bg);
                padding: 20px;
                display: flex;
                flex-direction: column;
                border-right: 1px solid var(--border-color);
            }
            .brand {
                font-size: 20px;
                font-weight: 500;
                margin-bottom: 20px;
                color: var(--text-main);
            }
            #new-chat-btn {
                background: #ffffff;
                border: 1px solid var(--border-color);
                padding: 12px 20px;
                border-radius: 24px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                color: var(--text-main);
                transition: background 0.2s, box-shadow 0.2s;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                margin-bottom: 20px;
            }
            #new-chat-btn:hover {
                background: var(--hover-bg);
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            
            /* History List */
            .history-title {
                font-size: 12px;
                font-weight: 600;
                color: var(--text-secondary);
                margin-bottom: 10px;
                text-transform: uppercase;
            }
            #history-list {
                flex: 1;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 5px;
            }
            .history-item {
                padding: 10px 12px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                color: var(--text-main);
                transition: background 0.2s;
            }
            .history-item:hover {
                background: var(--hover-bg);
            }
            .history-item.active {
                background: #d3e3fd;
                color: #041e49;
                font-weight: 500;
            }

            /* Main Chat Area */
            #main-content {
                flex: 1;
                display: flex;
                flex-direction: column;
                position: relative;
                background: var(--bg-color);
            }
            
            #chat {
                flex: 1;
                padding: 40px 10%;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 24px;
                padding-bottom: 120px;
            }

            /* Message Bubbles */
            .message-row {
                display: flex;
                width: 100%;
            }
            .message-row.user-row { justify-content: flex-end; }
            .message-row.bot-row { justify-content: flex-start; }

            .user {
                background: var(--user-msg-bg);
                padding: 12px 20px;
                border-radius: 24px 24px 4px 24px;
                max-width: 70%;
                line-height: 1.5;
            }
            .bot {
                background: transparent;
                padding: 12px 20px;
                border-radius: 12px;
                max-width: 85%;
                line-height: 1.6;
            }
            .thinking {
                color: #888;
                font-style: italic;
            }

            /* Curved Input Box */
            #input-area {
                position: absolute;
                bottom: 30px;
                left: 50%;
                transform: translateX(-50%);
                width: 70%;
                max-width: 800px;
                background: #ffffff;
                border: 1px solid var(--border-color);
                border-radius: 32px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                display: flex;
                align-items: center;
                padding: 6px 12px;
                box-sizing: border-box;
            }
            #input-area input {
                flex: 1;
                padding: 12px 16px;
                font-size: 16px;
                background: transparent;
                color: var(--text-main);
                border: none;
                outline: none;
            }
            #input-area button {
                background: transparent;
                color: #1a73e8;
                border: none;
                padding: 10px 16px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                border-radius: 20px;
                transition: background 0.2s;
            }
            #input-area button:hover {
                background: #f0f4f9;
            }

            /* Debug Panel */
            #debug {
                width: 320px;
                background: var(--debug-bg);
                padding: 20px;
                overflow-y: auto;
                border-left: 1px solid var(--border-color);
                font-size: 13px;
                color: var(--text-secondary);
            }
            #debug h3 { margin-top: 0; color: var(--text-main); }
            #debug h4 { margin-bottom: 4px; color: var(--text-main); }
            #debug pre {
                white-space: pre-wrap;
                word-break: break-word;
                background: #ffffff;
                border: 1px solid var(--border-color);
                padding: 10px;
                border-radius: 8px;
                max-height: 200px;
                overflow-y: auto;
            }
        </style>
    </head>
    <body>

    <!-- Sidebar -->
    <div id="sidebar">
        <div class="brand">🥭 Mango AI</div>
        <button id="new-chat-btn" onclick="startNewChat()">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="12" y1="5" x2="12" y2="19"></line>
                <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
            New Chat
        </button>
        <div class="history-title">Recent Chats</div>
        <div id="history-list"></div>
    </div>

    <!-- Main Chat Area -->
    <div id="main-content">
        <div id="chat"></div>

        <div id="input-area">
            <input id="msg" 
                placeholder="พิมพ์คำถามเกี่ยวกับมะม่วง..."
                autocomplete="off"
                onkeydown="if(event.key==='Enter') sendMessage()" />
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <!-- Debug Panel -->
    <div id="debug">
        <h3>📊 Debug Info</h3>
        <p id="mode">Mode used: -</p>
        <h4>Triples used:</h4>
        <pre id="triples"></pre>
        <h4>RAG chunks (Expanded → KG-RAG):</h4>
        <pre id="rag"></pre>
        <h4>Plain RAG chunks (Original):</h4>
        <pre id="plain_rag"></pre>
    </div>

    <script>
        // State Management
        let sessions = JSON.parse(localStorage.getItem('mango_sessions')) || {};
        let currentSessionId = null;

        const defaultChatHTML = `
            <div class="message-row bot-row">
                <div class="bot">สวัสดีค่ะ มีอะไรให้ฉันช่วยเกี่ยวกับมะม่วงไหมคะ?</div>
            </div>
        `;

        const defaultDebug = { mode: "-", triples: "", rag: "", plain_rag: "" };

        // Initialize on load
        window.onload = () => {
            renderHistoryList();
            
            // Load the most recent session if it exists, otherwise create a new one
            const sessionIds = Object.keys(sessions);
            if (sessionIds.length > 0) {
                const latestId = sessionIds.sort((a, b) => b - a)[0];
                loadSession(latestId);
            } else {
                startNewChat();
            }
        };

        function startNewChat() {
            currentSessionId = Date.now().toString();
            sessions[currentSessionId] = {
                id: currentSessionId,
                title: "New Chat",
                html: defaultChatHTML,
                debug: { ...defaultDebug }
            };
            saveSessions();
            loadSession(currentSessionId);
        }

        function saveSessions() {
            localStorage.setItem('mango_sessions', JSON.stringify(sessions));
            renderHistoryList();
        }

        function renderHistoryList() {
            const list = document.getElementById("history-list");
            list.innerHTML = "";
            
            // Sort newest first
            const sortedIds = Object.keys(sessions).sort((a, b) => b - a);
            
            sortedIds.forEach(id => {
                const session = sessions[id];
                const div = document.createElement("div");
                div.className = `history-item ${id === currentSessionId ? 'active' : ''}`;
                div.innerText = session.title;
                div.onclick = () => loadSession(id);
                list.appendChild(div);
            });
        }

        function loadSession(id) {
            currentSessionId = id;
            const session = sessions[id];
            
            // Load Chat UI
            const chat = document.getElementById("chat");
            chat.innerHTML = session.html;
            chat.scrollTo(0, chat.scrollHeight);
            
            // Load Debug UI
            document.getElementById("mode").innerText = "Mode used: " + (session.debug.mode || "-");
            document.getElementById("triples").innerText = session.debug.triples || "";
            document.getElementById("rag").innerText = session.debug.rag || "";
            document.getElementById("plain_rag").innerText = session.debug.plain_rag || "";
            
            // Update Active State in Sidebar
            renderHistoryList();
        }

        async function sendMessage() {
            let input = document.getElementById("msg");
            let message = input.value.trim();
            if (!message) return;

            let chat = document.getElementById("chat");

            // Update title if this is the first message in a "New Chat"
            if (sessions[currentSessionId].title === "New Chat") {
                sessions[currentSessionId].title = message.substring(0, 25) + (message.length > 25 ? "..." : "");
            }

            // Render User Message
            chat.innerHTML += `
                <div class="message-row user-row">
                    <div class="user">${message}</div>
                </div>`;
                
            // Render Loading State
            let loadingId = 'loading-' + Date.now();
            chat.innerHTML += `
                <div class="message-row bot-row" id="${loadingId}">
                    <div class="bot thinking">กำลังคิด...</div>
                </div>`;
            
            input.value = "";
            chat.scrollTo(0, chat.scrollHeight);

            // Save state immediately with user message
            sessions[currentSessionId].html = chat.innerHTML;
            saveSessions();

            try {
                let response = await fetch("/ask", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({message: message})
                });

                let data = await response.json();

                // Replace loading state with bot response
                let loadingEl = document.getElementById(loadingId);
                if (loadingEl) {
                    loadingEl.outerHTML = `
                        <div class="message-row bot-row">
                            <div class="bot">${data.answer}</div>
                        </div>`;
                }

                // Update debug state
                sessions[currentSessionId].debug = {
                    mode: data.mode || "-",
                    triples: data.triple_texts || "",
                    rag: data.rag_chunks || "",
                    plain_rag: data.plain_rag_chunks || ""
                };
                
            } catch (error) {
                let loadingEl = document.getElementById(loadingId);
                if (loadingEl) {
                    loadingEl.outerHTML = `
                        <div class="message-row bot-row">
                            <div class="bot" style="color: red;">Error: Could not connect to server.</div>
                        </div>`;
                }
            }
            
            // Final save after bot response
            sessions[currentSessionId].html = document.getElementById("chat").innerHTML;
            saveSessions();
            
            // Refresh UI to show debug data
            loadSession(currentSessionId); 
        }
    </script>
    </body>
    </html>
    """