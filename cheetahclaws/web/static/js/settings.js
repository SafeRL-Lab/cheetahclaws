/* Settings panel: theme, model picker, behavior toggles, API keys,
 * welcome dashboard. */

Object.assign(ChatApp.prototype, {

  // ── Theme (light / dark / system) ──────────────────────────────
  // Stored preference in localStorage.cc-theme:
  //   'light'  → force light (data-theme="light")
  //   'dark'   → force dark  (data-theme="dark")
  //   null or 'system' → follow OS (no attribute; CSS @media takes over)
  // First-visit default is 'system', so new users see whatever their OS
  // is set to — including light, which is the answer to "default to light
  // unless the user has their system in dark mode."

  initTheme() {
    const saved = localStorage.getItem('cc-theme');
    this._applyTheme(saved === 'light' || saved === 'dark' ? saved : 'system');
    // Live-update the button icon when the OS switches themes while we're
    // in 'system' mode.
    if (window.matchMedia) {
      const mm = window.matchMedia('(prefers-color-scheme: dark)');
      const listener = () => {
        if ((localStorage.getItem('cc-theme') || 'system') === 'system') {
          this._updateThemeBtn('system');
        }
      };
      mm.addEventListener ? mm.addEventListener('change', listener)
                          : mm.addListener(listener);
    }
  },

  toggleTheme() {
    // Cycle: system → light → dark → system ...
    const current = localStorage.getItem('cc-theme') || 'system';
    const next = current === 'system' ? 'light'
               : current === 'light'  ? 'dark'
               :                         'system';
    this._applyTheme(next);
    if (next === 'system') localStorage.removeItem('cc-theme');
    else                   localStorage.setItem('cc-theme', next);
  },

  _applyTheme(theme) {
    if (theme === 'system') {
      document.documentElement.removeAttribute('data-theme');
    } else {
      document.documentElement.setAttribute('data-theme', theme);
    }
    this._updateThemeBtn(theme);
  },

  _updateThemeBtn(theme) {
    const btn = document.getElementById('theme-btn');
    if (!btn) return;
    const sysDark = window.matchMedia
      && window.matchMedia('(prefers-color-scheme: dark)').matches;
    // Icon reflects what you'll see NOW (effective theme), with a tiny
    // glyph that also indicates the current MODE.
    //   system + OS-light → ☀ (showing light, will follow OS)
    //   system + OS-dark  → ☾ (showing dark, will follow OS)
    //   forced light      → ☀
    //   forced dark       → ☾
    // Tooltip spells out the exact mode so users aren't guessing.
    const effectiveDark = theme === 'dark'
      || (theme === 'system' && sysDark);
    btn.textContent = effectiveDark ? '\u263E' : '\u2600';  // moon / sun
    btn.title =
      theme === 'system' ? `Theme: system (${sysDark ? 'dark' : 'light'}) — click for light`
      : theme === 'light' ? 'Theme: light — click for dark'
      :                     'Theme: dark — click for system';
  },

  // ── Settings panel ──────────────────────────────────────────────

  toggleSettings() {
    const panel = document.getElementById('settings-panel');
    if (panel.classList.contains('open')) {
      panel.classList.remove('open');
    } else {
      panel.classList.add('open');
      this._loadSettings();
    }
  },

  async _loadSettings() {
    if (this.sessionId) {
      try {
        const r = await this._fetchAuth(`/api/config?sid=${this.sessionId}`);
        const cfg = await r.json();
        this._renderConfig(cfg);
      } catch(e) { console.error('loadSettings:', e); }
    }
    try {
      const r = await this._fetchAuth('/api/models');
      const data = await r.json();
      this._renderModels(data.providers || []);
    } catch(e) { console.error('loadModels:', e); }
  },

  _renderConfig(cfg) {
    document.getElementById('sp-current-model').textContent = cfg.model || '(not set)';
    document.getElementById('sp-permission').value = cfg.permission_mode || 'auto';
    document.getElementById('sp-thinking').className =
      'sp-toggle' + (cfg.thinking ? ' on' : '');
    document.getElementById('sp-verbose').className =
      'sp-toggle' + (cfg.verbose ? ' on' : '');
    document.getElementById('sp-max-tokens').value = cfg.max_tokens || 40000;
    document.getElementById('sp-thinking-budget').value = cfg.thinking_budget || 10000;
    const keysEl = document.getElementById('sp-api-keys');
    const configured = cfg.api_keys_configured || {};
    const providers = ['anthropic','openai','gemini','deepseek','ollama',
                       'qwen','kimi','zhipu','minimax','custom'];
    keysEl.innerHTML = providers.map(p => {
      const has = configured[p] || p === 'ollama' || p === 'lmstudio';
      return `<div class="sp-key-row">
        <span class="key-dot ${has?'ok':'missing'}"></span>
        <span class="provider-name">${p}</span>
        ${p!=='ollama' && p!=='lmstudio'
          ? `<input class="sp-input" type="password" placeholder="${has?'(configured)':'API key...'}"
              onchange="app.setApiKey('${p}', this.value)">`
          : `<span style="font-size:11px;color:var(--text-muted)">no key needed</span>`}
      </div>`;
    }).join('');
  },

  _renderModels(providers) {
    const listEl = document.getElementById('sp-model-list');
    const currentModel = document.getElementById('sp-current-model').textContent;
    // Use data-model + delegated handler so a model name returned by the
    // server (deep-trust attack surface) can't break out of the onclick
    // string literal.
    listEl.innerHTML = providers.map(p => {
      if (!p.models.length && p.provider !== 'ollama') return '';
      const models = p.models.map(m => {
        const full = `${p.provider}/${m}`;
        const isActive = currentModel === full || currentModel === m;
        return `<div class="sp-model-item${isActive?' active':''}"
          data-model="${this._esc(full)}">${this._esc(m)}</div>`;
      }).join('');
      return `<details class="sp-model-group">
        <summary>${this._esc(p.provider)} (${p.models.length})
          ${!p.has_api_key && p.needs_api_key
            ? '<span style="color:var(--red);font-size:10px"> no key</span>' : ''}
        </summary>
        ${models}
      </details>`;
    }).join('');
    listEl.querySelectorAll('.sp-model-item').forEach(el => {
      el.onclick = () => this.selectModel(el.dataset.model);
    });
  },

  async selectModel(model) {
    await this.updateConfig('model', model);
    document.getElementById('sp-current-model').textContent = model;
    document.querySelectorAll('.sp-model-item').forEach(el => {
      el.classList.toggle('active', el.textContent.trim() === model.split('/').pop());
    });
  },

  async toggleConfig(key) {
    const el = document.getElementById('sp-' + key);
    const isOn = el.classList.contains('on');
    await this.updateConfig(key, !isOn);
    el.classList.toggle('on');
  },

  async updateConfig(key, value) {
    if (!this.sessionId) return;
    try {
      await this._fetchAuth('/api/config', {
        method: 'PATCH',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({session_id: this.sessionId, config: {[key]: value}}),
      });
    } catch(e) { console.error('updateConfig:', e); }
  },

  async setApiKey(provider, value) {
    if (!value || !this.sessionId) return;
    const keyMap = {
      anthropic:'anthropic_api_key', openai:'openai_api_key',
      gemini:'gemini_api_key', kimi:'kimi_api_key', qwen:'qwen_api_key',
      zhipu:'zhipu_api_key', deepseek:'deepseek_api_key',
      minimax:'minimax_api_key', custom:'custom_api_key',
    };
    await this.updateConfig(keyMap[provider], value);
    this._loadSettings();
  },

  sendSlash(cmd) {
    const input = document.getElementById('prompt-input');
    input.value = cmd;
    this.send();
    this.toggleSettings();
  },
});
