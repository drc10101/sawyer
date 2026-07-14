"""FAQ page HTML for Sawyer onboarding."""

FAQ_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sawyer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface-2: #1a1a25;
    --border: #2a2a3a;
    --text: #e4e4ef;
    --text-dim: #8888a0;
    --accent: #12c7ef;
    --accent-dim: #0e9fc3;
    --green: #22c55e;
  }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }
  header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 14px 24px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
  }
  .header-right {
    font-size: 12px;
    color: var(--text-dim);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
  }
  .status-dot.online { background: #22c55e; }
  .status-dot.offline { background: #ef4444; }

  main {
    max-width: 780px;
    margin: 0 auto;
    padding: 32px 24px 64px;
  }

  .hero {
    text-align: center;
    margin-bottom: 48px;
    padding-top: 16px;
  }
  .hero-logo {
    width: 500px;
    max-width: 100%;
    height: auto;
    margin-bottom: 24px;
  }
  .hero h2 {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 12px;
    color: var(--text);
  }
  .hero p {
    font-size: 15px;
    color: var(--text-dim);
    line-height: 1.6;
  }

  /* The pitch */
  .pitch {
    background: var(--surface);
    border: 1px solid var(--accent);
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    margin-bottom: 48px;
  }
  .pitch h3 {
    font-size: 20px;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 8px;
  }
  .pitch p {
    font-size: 15px;
    color: var(--text-dim);
    line-height: 1.6;
  }
  .pitch code {
    display: inline-block;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 12px;
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-size: 13px;
    color: var(--green);
    margin-top: 12px;
  }

  /* Quickstart cards */
  .quickstart {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 16px;
    margin-bottom: 48px;
  }
  .quickstart-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    transition: border-color 0.2s;
  }
  .quickstart-card:hover {
    border-color: var(--accent);
  }
  .quickstart-card h3 {
    font-size: 13px;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
  }
  .os-badges {
    display: flex;
    gap: 6px;
    margin-bottom: 10px;
  }
  .os-badge {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    padding: 2px 8px;
    border-radius: 4px;
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text-dim);
  }
  .quickstart-card p {
    font-size: 13px;
    color: var(--text-dim);
    line-height: 1.5;
    margin-bottom: 10px;
  }
  .quickstart-card code {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 12px;
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-size: 12px;
    color: var(--accent);
    white-space: nowrap;
    overflow-x: auto;
  }
  .quickstart-card code .copy-btn {
    background: none;
    border: none;
    color: var(--text-dim);
    cursor: pointer;
    padding: 2px 4px;
    margin-left: 8px;
    font-size: 14px;
    line-height: 1;
    opacity: 0.5;
    transition: opacity 0.2s, color 0.2s;
    flex-shrink: 0;
  }
  .quickstart-card code .copy-btn:hover {
    opacity: 1;
    color: var(--accent);
  }
  .quickstart-card code .copy-btn.copied {
    color: var(--green);
    opacity: 1;
  }

  /* Features strip */
  .features {
    display: flex;
    justify-content: center;
    gap: 24px;
    flex-wrap: wrap;
    margin-bottom: 48px;
    padding: 16px 0;
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
  }
  .feature {
    font-size: 13px;
    color: var(--text-dim);
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .feature-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    display: inline-block;
  }

  /* FAQ sections */
  .faq-section {
    margin-bottom: 40px;
  }
  .faq-section h2 {
    font-size: 18px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }
  .faq-item {
    margin-bottom: 16px;
  }
  .faq-item summary {
    font-size: 14px;
    font-weight: 500;
    color: var(--text);
    cursor: pointer;
    padding: 12px 16px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    list-style: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: border-color 0.2s, background 0.2s;
  }
  .faq-item summary:hover {
    border-color: var(--accent);
    background: var(--surface-2);
  }
  .faq-item summary::after {
    content: '+';
    font-size: 18px;
    color: var(--text-dim);
    transition: transform 0.2s;
  }
  .faq-item[open] summary::after {
    content: '-';
  }
  .faq-item summary::-webkit-details-marker { display: none; }
  .faq-answer {
    padding: 12px 16px;
    font-size: 14px;
    line-height: 1.7;
    color: var(--text-dim);
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: none;
    border-radius: 0 0 8px 8px;
  }
  .faq-answer code {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 6px;
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-size: 12px;
    color: var(--accent);
  }
  .faq-answer pre {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0;
    overflow-x: auto;
    margin: 8px 0;
    position: relative;
  }
  .faq-answer pre code {
    display: block;
    background: none;
    border: none;
    padding: 12px 40px 12px 16px;
    font-size: 12px;
    color: var(--text);
  }
  .faq-answer pre .copy-btn {
    position: absolute;
    top: 8px;
    right: 8px;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text-dim);
    cursor: pointer;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    line-height: 1;
    opacity: 0.5;
    transition: opacity 0.2s, color 0.2s, border-color 0.2s;
  }
  .faq-answer pre .copy-btn:hover {
    opacity: 1;
    color: var(--accent);
    border-color: var(--accent);
  }
  .faq-answer pre .copy-btn.copied {
    color: var(--green);
    opacity: 1;
    border-color: var(--green);
  }
  .faq-answer a {
    color: var(--accent);
    text-decoration: none;
  }
  .faq-answer a:hover {
    text-decoration: underline;
  }
  .faq-answer ul {
    margin: 8px 0;
    padding-left: 20px;
  }
  .faq-answer li {
    margin-bottom: 4px;
  }

  footer {
    text-align: center;
    padding: 24px;
    font-size: 12px;
    color: var(--text-dim);
    border-top: 1px solid var(--border);
  }

  @media (max-width: 600px) {
    .quickstart {
      grid-template-columns: 1fr;
    }
    .hero h2 {
      font-size: 22px;
    }
  }
</style>
</head>
<body>

<header>
  <div class="header-right">
    <span class="status-dot" id="status-dot"></span>
    <span id="status-text">Checking...</span>
    <span id="model-display" style="margin-left: 8px;"></span>
  </div>
</header>

<main>
  <div class="hero">
    <img class="hero-logo" src="https://raw.githubusercontent.com/drc10101/sawyer-network/main/Sawyer_Agent_Github_Readme_Logo.png" alt="Sawyer">
    <h2>Point your GPU at Sawyer.</h2>
    <p>Your inference is always free. Earn from the compute you share. Run locally or join the network.</p>
  </div>

  <div class="pitch">
    <h3>Your hardware pays for itself.</h3>
    <p>Run <code>sawyer serve</code> and your GPU earns tokens from every inference request it handles. Your own inference costs nothing. The more you serve, the more you earn.</p>
  </div>

  <div class="quickstart">
    <div class="quickstart-card">
      <h3>Install</h3>
      <div class="os-badges"><span class="os-badge">Windows</span><span class="os-badge">Linux</span><span class="os-badge">macOS</span></div>
      <p>One command to get started:</p>
      <code><span>pip install sawyer-core</span><button class="copy-btn" onclick="copyCmd(this, 'pip install sawyer-core')" title="Copy">&#x2398;</button></code>
    </div>
    <div class="quickstart-card">
      <h3>Serve</h3>
      <div class="os-badges"><span class="os-badge">Windows</span><span class="os-badge">Linux</span><span class="os-badge">macOS</span></div>
      <p>Point your GPU at Sawyer:</p>
      <code><span>sawyer serve</span><button class="copy-btn" onclick="copyCmd(this, 'sawyer serve')" title="Copy">&#x2398;</button></code>
    </div>
    <div class="quickstart-card">
      <h3>Earn</h3>
      <div class="os-badges"><span class="os-badge">Windows</span><span class="os-badge">Linux</span><span class="os-badge">macOS</span></div>
      <p>Your inference is free. You earn from what you serve.</p>
      <code><span>sawyer status</span><button class="copy-btn" onclick="copyCmd(this, 'sawyer status')" title="Copy">&#x2398;</button></code>
    </div>
  </div>

  <div class="features">
    <span class="feature"><span class="feature-dot"></span>Free inference</span>
    <span class="feature"><span class="feature-dot"></span>Earn from idle GPU</span>
    <span class="feature"><span class="feature-dot"></span>MoE routing</span>
    <span class="feature"><span class="feature-dot"></span>Open source</span>
  </div>

  <div class="faq-section">
    <h2>Getting Started</h2>

    <details class="faq-item">
      <summary>What is Sawyer?</summary>
      <div class="faq-answer">
        Sawyer is a distributed inference network. You point your GPU at it, and in exchange your own inference is always free. When your hardware is idle, it handles requests from other users on the network and you earn tokens for every token you serve. The more compute you contribute, the more you earn.
      </div>
    </details>

    <details class="faq-item">
      <summary>How do I start earning?</summary>
      <div class="faq-answer">
        <pre><code>pip install sawyer-core
sawyer serve</code><button class="copy-btn" onclick="copyCmd(this, 'pip install sawyer-core &amp;&amp; sawyer serve')" title="Copy">&#x2398;</button></pre>
        That is it. Sawyer auto-detects your GPU and any models you have running. Once you are online, the network starts routing inference requests to your node. You earn tokens for every request you handle. Your own inference is free.
      </div>
    </details>

    <details class="faq-item">
      <summary>What do I need to run a node?</summary>
      <div class="faq-answer">
        A GPU with CUDA 12.4+ and Python 3.10+. That is it. If you already have Ollama, llama.cpp, vLLM, or LM Studio running, Sawyer finds them automatically. No configuration needed.
      </div>
    </details>

    <details class="faq-item">
      <summary>How much can I earn?</summary>
      <div class="faq-answer">
        Earnings depend on your GPU, uptime, and demand. A single RTX 4090 serving 24/7 can earn well beyond the cost of inference. The more you serve, the more you earn. Your own inference is always free regardless of how much you earn.
      </div>
    </details>

    <details class="faq-item">
      <summary>How do I run a model locally?</summary>
      <div class="faq-answer">
        <pre><code>sawyer serve --offline</code><button class="copy-btn" onclick="copyCmd(this, 'sawyer serve --offline')" title="Copy">&#x2398;</button></pre>
        This starts a local inference server using whatever models are available on your machine. No network, no earnings, but your inference still works.
      </div>
    </details>
  </div>

  <div class="faq-section">
    <h2>How It Works</h2>

    <details class="faq-item">
      <summary>What is MoE routing?</summary>
      <div class="faq-answer">
        Mixture-of-Experts routing means each request is sent to the node best suited for it. Sawyer tracks latency, throughput, and availability across all nodes, then routes requests to whichever node can deliver the result fastest. Your GPU handles what it is good at.
      </div>
    </details>

    <details class="faq-item">
      <summary>Do I need a GPU?</summary>
      <div class="faq-answer">
        To serve requests and earn, yes — you need a GPU with CUDA support. To just use inference without earning, any machine works. But if you have a GPU sitting idle, you are leaving money on the table.
      </div>
    </details>

    <details class="faq-item">
      <summary>What models are supported?</summary>
      <div class="faq-answer">
        Sawyer auto-detects any models running in Ollama, llama.cpp, vLLM, or LM Studio. The network routes requests to whatever models your node is serving.
      </div>
    </details>

    <details class="faq-item">
      <summary>Is my data private?</summary>
      <div class="faq-answer">
        When you run locally with <code>--offline</code>, nothing leaves your machine. On the network, requests are routed between Sawyer nodes — no third-party cloud involved. You see every request your node handles.
      </div>
    </details>
  </div>

  <div class="faq-section">
    <h2>Troubleshooting</h2>

    <details class="faq-item">
      <summary>Sawyer says "No models available"</summary>
      <div class="faq-answer">
        Make sure Ollama, llama.cpp, vLLM, or LM Studio is running. Sawyer checks the standard ports on startup:
        <ul>
          <li>Ollama: <code>http://localhost:11434</code></li>
          <li>llama.cpp: <code>http://localhost:8080</code></li>
          <li>vLLM: <code>http://localhost:8000</code></li>
          <li>LM Studio: <code>http://localhost:1234</code></li>
        </ul>
      </div>
    </details>

    <details class="faq-item">
      <summary>My model is running but Sawyer can not find it</summary>
      <div class="faq-answer">
        If your backend is on a non-default port, set it in your Sawyer config. You can also check what Sawyer sees:
        <pre><code>sawyer status</code><button class="copy-btn" onclick="copyCmd(this, 'sawyer status')" title="Copy">&#x2398;</button></pre>
      </div>
    </details>

    <details class="faq-item">
      <summary>How do I check my earnings?</summary>
      <div class="faq-answer">
        <pre><code>sawyer status</code><button class="copy-btn" onclick="copyCmd(this, 'sawyer status')" title="Copy">&#x2398;</button></pre>
        This shows your node health, models served, tokens processed, and current earnings balance.
      </div>
    </details>

    <details class="faq-item">
      <summary>How do I report a bug or request a feature?</summary>
      <div class="faq-answer">
        Open an issue on <a href="https://github.com/drc10101/sawyer-network">GitHub</a> or reach out in the community chat. Sawyer is open source — PRs welcome.
      </div>
    </details>
  </div>
</main>

<footer>
  Sawyer &mdash; Open source distributed inference
</footer>

<script>
  function copyCmd(btn, text) {
    navigator.clipboard.writeText(text).then(function() {
      btn.classList.add('copied');
      btn.innerHTML = '&#x2713;';
      setTimeout(function() {
        btn.classList.remove('copied');
        btn.innerHTML = '&#x2398;';
      }, 2000);
    });
  }

  // Status check
  async function checkStatus() {
    try {
      const resp = await fetch('/v1/models');
      if (!resp.ok) throw new Error('not ok');
      const data = await resp.json();
      const models = data.data || [];
      const dot = document.getElementById('status-dot');
      const text = document.getElementById('status-text');
      const display = document.getElementById('model-display');
      if (models.length > 0) {
        dot.className = 'status-dot online';
        text.textContent = models.length + ' model' + (models.length !== 1 ? 's' : '') + ' available';
        display.textContent = models.map(m => m.id).join(', ');
      } else {
        dot.className = 'status-dot offline';
        text.textContent = 'No models available';
      }
    } catch (e) {
      const dot = document.getElementById('status-dot');
      const text = document.getElementById('status-text');
      dot.className = 'status-dot offline';
      text.textContent = 'Server error';
    }
  }
  checkStatus();
  setInterval(checkStatus, 15000);
</script>

</body>
</html>"""