/* Tool cards, activity spinner, slash-command results, input requests,
 * interactive menus. Everything that renders agent activity. */

Object.assign(ChatApp.prototype, {

  _addToolCard(name, inputs, status, result) {
    const id = 'tool-' + (this._toolCounter++);
    const card = document.createElement('details');
    card.className = 'tool-card'
      + (status === 'done' ? ' done' : '')
      + (status === 'denied' ? ' denied' : '');
    card.id = id;
    const inputStr = typeof inputs === 'string'
      ? inputs : JSON.stringify(inputs || {}, null, 2);
    card.innerHTML = `
      <summary>
        ${status === 'running' ? '<div class="spinner"></div>' : ''}
        <span class="tool-name">${this._esc(name)}</span>
        <span class="tool-badge ${status}">${status}</span>
      </summary>
      <div class="tool-body">
        <div class="label">Input</div>
        <pre>${this._esc(inputStr)}</pre>
        ${result ? `<div class="label">Output</div><pre>${this._esc(result)}</pre>` : ''}
      </div>`;
    document.getElementById('messages').appendChild(card);
    this._toolCards[name + ':' + (this._toolCounter - 1)] = card;
    this._toolCards['__last__' + name] = card;
    this._scrollBottom();
  },

  _completeToolCard(name, result, permitted) {
    const card = this._toolCards['__last__' + name];
    if (!card) return;
    const status = permitted ? 'done' : 'denied';
    card.className = 'tool-card ' + status;
    const summary = card.querySelector('summary');
    const spinner = summary.querySelector('.spinner');
    if (spinner) spinner.remove();
    const badge = summary.querySelector('.tool-badge');
    badge.className = 'tool-badge ' + status;
    badge.textContent = status;
    if (result) {
      const body = card.querySelector('.tool-body');
      const existing = body.querySelectorAll('.label');
      if (existing.length < 2) {
        body.innerHTML += `<div class="label">Output</div><pre>${this._esc(result)}</pre>`;
      }
    }
  },

  _showActivity(type, label, detail) {
    if (!this._activityEl) {
      this._activityEl = document.createElement('div');
      this._activityEl.className = 'activity-indicator';
      this._activityEl.innerHTML = `
        <div class="ai-spinner"></div>
        <div class="ai-text">
          <span class="ai-label"></span>
          <span class="ai-dots"></span>
          <span class="ai-detail"></span>
        </div>
        <div class="ai-progress"><div class="ai-fill"></div></div>`;
      document.getElementById('messages').appendChild(this._activityEl);
      this._scrollBottom();
    }
    this._activityEl.className = 'activity-indicator' + (type ? ' ' + type : '');
    this._activityEl.querySelector('.ai-label').textContent = label || 'Working';
    const detailEl = this._activityEl.querySelector('.ai-detail');
    if (detail) { detailEl.textContent = detail; detailEl.style.display = ''; }
    else { detailEl.style.display = 'none'; }
    this._scrollBottom();
  },

  _removeActivity() {
    if (this._activityEl) { this._activityEl.remove(); this._activityEl = null; }
    if (this._thinkEl) { this._thinkEl.remove(); this._thinkEl = null; }
  },

  _addInputRequest(data) {
    const el = document.createElement('div');
    el.className = 'msg assistant';
    const uid = 'ir-' + Date.now();
    el.innerHTML = `
      <div class="role-tag" style="color:var(--accent)">Input Required</div>
      <div style="background:var(--surface);border:1px solid var(--accent);border-radius:var(--radius-sm);
        padding:12px 14px;max-width:min(500px,90%);">
        <div style="font-size:13px;color:var(--text);margin-bottom:8px;">${this._esc(data.prompt)}</div>
        <div style="display:flex;gap:6px;">
          <input id="${uid}" type="text" placeholder="${this._esc(data.placeholder || '')}"
            style="flex:1;background:var(--panel);border:1px solid var(--border);color:var(--text);
            border-radius:var(--radius-sm);padding:6px 10px;font-size:13px;font-family:var(--font);outline:none;"
            onkeydown="if(event.key==='Enter'){document.getElementById('${uid}-go').click()}">
          <button id="${uid}-go" style="background:var(--accent);color:#000;border:none;padding:6px 14px;
            border-radius:var(--radius-sm);font-weight:600;font-size:12px;cursor:pointer;"
            onclick="(function(){
              var v=document.getElementById('${uid}').value.trim();
              var cmd='${data.command}' + (v ? ' ' + v : ' general project improvement');
              document.getElementById('prompt-input').value=cmd;
              app.send();
              this.parentElement.parentElement.style.opacity='0.5';
              this.parentElement.parentElement.style.pointerEvents='none';
            }).call(this)">Go</button>
        </div>
      </div>`;
    document.getElementById('messages').appendChild(el);
    this._scrollBottom();
    setTimeout(() => { const inp = document.getElementById(uid); if (inp) inp.focus(); }, 100);
  },

  _addInteractiveMenu(data) {
    const icons = {bulb:'&#128161;',clipboard:'&#128203;',worker:'&#128119;',
      brain:'&#129504;',sparkle:'&#10024;',search:'&#128270;',book:'&#128214;',
      chat:'&#128172;',test:'&#129514;',note:'&#128221;',monitor:'&#128225;',
      robot:'&#129302;'};
    const el = document.createElement('div');
    el.className = 'msg assistant';
    const items = (data.items || []).map(it =>
      `<div style="display:flex;align-items:center;gap:8px;padding:8px 12px;cursor:pointer;
        border-radius:var(--radius-sm);border:1px solid var(--border);background:var(--surface);
        transition:background .15s;"
        onmouseenter="this.style.background='var(--panel)'"
        onmouseleave="this.style.background='var(--surface)'"
        onclick="document.getElementById('prompt-input').value='${it.cmd}';app.send()">
        <span style="font-size:16px;">${icons[it.icon]||'&#9654;'}</span>
        <div>
          <div style="font-size:12px;font-weight:600;color:var(--text);">${this._esc(it.label)}</div>
          <div style="font-size:10px;font-family:var(--mono);color:var(--text-muted);">${this._esc(it.cmd)}</div>
        </div>
      </div>`).join('');
    el.innerHTML = `<div class="role-tag" style="color:var(--accent)">SSJ Developer Mode</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:min(640px,95%);
        margin-top:6px;">${items}</div>`;
    document.getElementById('messages').appendChild(el);
    this._scrollBottom();
  },

  _addCommandResult(command, output) {
    const el = document.createElement('div');
    el.className = 'msg assistant';
    el.innerHTML = `<div class="role-tag" style="color:var(--accent)">System</div>
      <div class="bubble" style="background:var(--surface);border:1px solid var(--border);
        border-left:3px solid var(--accent);border-radius:var(--radius-sm);padding:12px 14px;">
        <div style="font-family:var(--mono);font-size:11px;color:var(--accent);margin-bottom:6px;">${this._esc(command)}</div>
        <pre style="white-space:pre-wrap;font-family:var(--mono);font-size:12px;color:var(--text-dim);
          margin:0;background:none;border:none;padding:0;line-height:1.5;">${this._esc(output)}</pre>
      </div>`;
    document.getElementById('messages').appendChild(el);
    this._scrollBottom();
  },
});
