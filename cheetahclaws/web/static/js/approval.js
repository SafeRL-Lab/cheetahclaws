/* Permission approval card — shown when the agent needs confirmation
 * for a sensitive tool call. */

Object.assign(ChatApp.prototype, {

  _showApproval(description) {
    const card = document.createElement('div');
    card.className = 'approval-card';
    card.innerHTML = `
      <div class="ap-hdr">&#128274; Permission Required</div>
      <div class="ap-desc">${this._esc(description)}</div>
      <div class="ap-actions">
        <button class="ap-btn allow" onclick="app.approve(true)">Allow</button>
        <button class="ap-btn deny" onclick="app.approve(false)">Deny</button>
      </div>`;
    this._approvalEl = card;
    this._pendingApproval = true;
    document.getElementById('messages').appendChild(card);
    this._scrollBottom();
  },

  _resolveApproval(granted) {
    if (this._approvalEl) {
      this._approvalEl.classList.add('resolved');
      const badge = document.createElement('span');
      badge.style.cssText = `font-size:11px;color:${granted?'var(--green)':'var(--red)'};margin-left:8px;`;
      badge.textContent = granted ? '(allowed)' : '(denied)';
      this._approvalEl.querySelector('.ap-hdr').appendChild(badge);
      this._approvalEl = null;
    }
    this._pendingApproval = false;
  },

  approve(granted) {
    if (!this._pendingApproval) return;
    if (this.ws && this.ws.readyState === 1) {
      this.ws.send(JSON.stringify({type: 'approve', granted}));
    } else {
      fetch('/api/approve', {
        method: 'POST', credentials: 'same-origin',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({session_id: this.sessionId, granted}),
      }).catch(e => console.error('approve:', e));
    }
  },
});
