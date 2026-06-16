/* Welcome dashboard rendered when no messages are present.
 * Self-contained — only depends on _esc from util.js. */

Object.assign(ChatApp.prototype, {

  _showWelcome() {
    const el = document.getElementById('messages');
    el.innerHTML = '<div style="flex:1"></div>';
    const dash = document.createElement('div');
    dash.style.cssText = 'max-width:680px;margin:0 auto;padding:20px 0;';
    dash.innerHTML = `
      <div style="text-align:center;margin-bottom:20px;">
        <div style="font-size:28px;font-weight:700;color:var(--accent);margin-bottom:4px;">CheetahClaws</div>
        <div style="font-size:13px;color:var(--text-muted);">Personal AI Assistant &bull; Support Any Model &bull; Autonomous 24/7</div>
      </div>

      ${this._dashSection('Core', [
        this._dashCard('Write Code',    'Generate, edit, debug any language',     'Help me write a Python web scraper',  'var(--blue)'),
        this._dashCard('Read & Search', 'Glob, Grep, Read files intelligently',   'Find all TODO comments in this repo', 'var(--green)'),
        this._dashCard('Terminal',      'Run shell commands with safety checks',  'Show git log --oneline -10',          'var(--accent)'),
        this._dashCard('Analyze',       'Understand codebases, explain logic',    'Explain the architecture of this project', 'var(--purple)'),
      ])}

      ${this._dashSection('Agent Features', [
        this._dashCmd('/plan',       'Plan Mode',       'Read-only analysis, then implement step by step',  'var(--blue)'),
        this._dashCmd('/brainstorm', 'Brainstorm',      'Multi-persona debate with synthesis & auto-tasks', 'var(--purple)'),
        this._dashCmd('/worker',     'Auto Worker',     'Batch-implement pending tasks automatically',      'var(--green)'),
        this._dashCmd('/agent',      'Autonomous Agent', 'Self-directed agent loop with templates',         'var(--accent)'),
      ])}

      ${this._dashSection('Session & Memory', [
        this._dashCmd('/memory',     'Memory',      'Persistent context recalled across sessions',  'var(--blue)'),
        this._dashCmd('/checkpoint', 'Checkpoints', 'Snapshot & rewind file changes',               'var(--green)'),
        this._dashCmd('/save',       'Save/Load',   'Export and resume conversation sessions',      'var(--text-dim)'),
        this._dashCmd('/cloudsave',  'Cloud Sync',  'Encrypted backup to GitHub Gist',              'var(--purple)'),
      ])}

      ${this._dashSection('Multi-Model', [
        this._dashCmd('/model',    'Switch Model', 'Claude, GPT, Gemini, Ollama, DeepSeek, Qwen...', 'var(--accent)'),
        this._dashCmd('/thinking', 'Thinking',     'Extended reasoning for complex problems',         'var(--blue)'),
        this._dashCmd('/compact',  'Compact',      'Compress context when window is full',            'var(--green)'),
        this._dashCmd('/config',   'Config',       'View and modify all runtime settings',            'var(--text-dim)'),
      ])}

      ${this._dashSection('Development Tools', [
        this._dashCmd('/ssj',     'SSJ Mode',     'Integrated developer dashboard with power menu',  'var(--red)'),
        this._dashCmd('/tasks',   'Task Manager', 'Create, track, complete development tasks',       'var(--accent)'),
        this._dashCmd('/init',    'Init Project', 'Generate CLAUDE.md for your codebase',            'var(--blue)'),
        this._dashCmd('/cwd',     'Working Dir',  'View or change the working directory',            'var(--text-dim)'),
      ])}

      ${this._dashSection('Bridges', [
        this._dashCmd('/telegram', 'Telegram', 'Chat via Telegram bot bridge',           'var(--blue)'),
        this._dashCmd('/wechat',   'WeChat',   'Scan QR to chat via WeChat iLink bot',   'var(--green)'),
        this._dashCmd('/slack',    'Slack',    'Chat via Slack workspace integration',   'var(--purple)'),
        this._dashCmd('/monitor',  'Monitor',  'Watch RSS / webhooks, alert via bridges','var(--accent)'),
      ])}

      ${this._dashSection('Multi-Modal Media', [
        this._dashCmd('/voice',  'Voice Input', 'Speak to code with STT transcription',     'var(--green)'),
        this._dashCmd('/image',  'Vision',      'Analyze screenshots and UI designs',       'var(--accent)'),
        this._dashCmd('/copy',   'Copy Output', 'Copy the latest assistant reply to clipboard', 'var(--blue)'),
        this._dashCmd('/export', 'Export',      'Save the conversation as a Markdown file', 'var(--text-dim)'),
      ])}

      <div style="border:1px solid var(--border);border-radius:var(--radius);background:var(--surface);padding:12px 14px;margin-top:12px;">
        <div style="font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--text-muted);margin-bottom:8px;">Quick Status</div>
        <div style="display:flex;flex-wrap:wrap;gap:6px;">
          ${['/status','/context','/cost','/permissions','/doctor','/skills','/help']
            .map(c => this._cmdChip(c)).join('')}
        </div>
      </div>
      <div style="text-align:center;margin-top:10px;font-size:11px;color:var(--text-muted);">
        &#9881; Settings for model, API keys, permissions &bull; &#9790; Toggle dark/light theme
      </div>`;
    el.appendChild(dash);
  },

  _dashSection(title, cards) {
    return `<div style="margin-bottom:12px;">
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:.6px;
        color:var(--text-muted);margin-bottom:6px;padding-left:2px;">${title}</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">${cards.join('')}</div>
    </div>`;
  },

  _dashCard(title, desc, example, color) {
    return `<div style="background:var(--surface);border:1px solid var(--border);
      border-radius:var(--radius-sm);padding:10px 12px;cursor:pointer;
      border-left:3px solid ${color};transition:background .15s;"
      onmouseenter="this.style.background='var(--panel)'"
      onmouseleave="this.style.background='var(--surface)'"
      onclick="document.getElementById('prompt-input').value='${example}';document.getElementById('prompt-input').focus();">
      <div style="font-size:12px;font-weight:600;color:var(--text);margin-bottom:2px;">${title}</div>
      <div style="font-size:11px;color:var(--text-muted);line-height:1.4;">${desc}</div>
    </div>`;
  },

  _dashCmd(cmd, title, desc, color) {
    return `<div style="background:var(--surface);border:1px solid var(--border);
      border-radius:var(--radius-sm);padding:10px 12px;cursor:pointer;
      border-left:3px solid ${color};transition:background .15s;"
      onmouseenter="this.style.background='var(--panel)'"
      onmouseleave="this.style.background='var(--surface)'"
      onclick="document.getElementById('prompt-input').value='${cmd}';app.send()">
      <div style="display:flex;align-items:center;gap:6px;margin-bottom:2px;">
        <span style="font-family:var(--mono);font-size:11px;color:${color};font-weight:600;">${cmd}</span>
        <span style="font-size:12px;font-weight:600;color:var(--text);">${title}</span>
      </div>
      <div style="font-size:11px;color:var(--text-muted);line-height:1.4;">${desc}</div>
    </div>`;
  },

  _cmdChip(cmd) {
    return `<span style="font-family:var(--mono);font-size:11px;padding:3px 8px;
      background:var(--panel);border:1px solid var(--border);border-radius:4px;
      cursor:pointer;color:var(--text-dim);transition:border-color .15s;"
      onmouseenter="this.style.borderColor='var(--accent)'"
      onmouseleave="this.style.borderColor='var(--border)'"
      onclick="document.getElementById('prompt-input').value='${cmd}';app.send()">${cmd}</span>`;
  },
});
