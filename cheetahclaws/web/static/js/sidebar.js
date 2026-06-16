/* Sidebar: session list, search filter, context menu (rename/delete/export),
 * new session, switch session. */

Object.assign(ChatApp.prototype, {

  async loadSessions() {
    // Fetch sessions and folders independently. If /api/folders is missing
    // (older server) or errors, the session list still renders flat.
    try {
      const r = await this._fetchAuth('/api/sessions');
      if (r.ok) {
        const sd = await r.json();
        this._sessions = sd.sessions || [];
      }
    } catch(e) { console.error('loadSessions sessions:', e); }
    try {
      const r = await this._fetchAuth('/api/folders');
      if (r.ok) {
        const fd = await r.json();
        this._folders = fd.folders || [];
      } else {
        this._folders = [];
      }
    } catch(e) {
      console.warn('loadSessions folders (non-fatal):', e);
      this._folders = [];
    }
    this._renderSessionList();
  },

  _collapsedFolders() {
    if (!this.__collapsedFolders) {
      try {
        const raw = localStorage.getItem('cc-collapsed-folders') || '[]';
        this.__collapsedFolders = new Set(JSON.parse(raw));
      } catch(e) { this.__collapsedFolders = new Set(); }
    }
    return this.__collapsedFolders;
  },

  _saveCollapsed() {
    try {
      localStorage.setItem('cc-collapsed-folders',
        JSON.stringify([...this._collapsedFolders()]));
    } catch(e) {}
  },

  _toggleCollapse(key) {
    const set = this._collapsedFolders();
    if (set.has(key)) set.delete(key); else set.add(key);
    this._saveCollapsed();
    this._renderSessionList();
  },

  _getActiveFolderId() {
    if (this._activeFolderId === undefined) {
      const raw = localStorage.getItem('cc-active-folder');
      this._activeFolderId = raw ? parseInt(raw, 10) : null;
    }
    return this._activeFolderId || null;
  },

  _setActiveFolder(fid) {
    this._activeFolderId = fid || null;
    if (fid) {
      localStorage.setItem('cc-active-folder', String(fid));
      // Expand the folder so its sessions are visible
      this._collapsedFolders().delete(`f:${fid}`);
      this._saveCollapsed();
    } else {
      localStorage.removeItem('cc-active-folder');
    }
    this._renderSessionList();
    this._updateTopbarFolder();
  },

  _updateTopbarFolder() {
    const titleEl = document.querySelector('#topbar .title');
    if (!titleEl) return;
    let badge = document.querySelector('#topbar .title-folder');
    const fid = this._getActiveFolderId();
    const folder = (this._folders || []).find(f => f.id === fid);
    if (folder) {
      if (!badge) {
        badge = document.createElement('span');
        badge.className = 'title-folder';
        titleEl.parentNode.insertBefore(badge, titleEl.nextSibling);
      }
      badge.textContent = `· in ${folder.name}`;
    } else if (badge) {
      badge.remove();
    }
  },

  _renderSessionList() {
    const list = document.getElementById('session-list');
    if (!list) return;
    if (!this._selectedIds) this._selectedIds = new Set();
    if (!this._folders) this._folders = [];
    const sel = this._selectMode;
    const q = (document.getElementById('sess-search-input')?.value || '')
      .trim().toLowerCase();
    const allSessions = (this._sessions || []).filter(s =>
      !q || (s.title || '').toLowerCase().includes(q) || s.id.includes(q)
    );
    list.innerHTML = '';
    if (allSessions.length === 0 && this._folders.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'sess-empty';
      empty.textContent = q
        ? 'No sessions match.'
        : 'No sessions yet — click + New.';
      list.appendChild(empty);
      this._renderBatchBar();
      return;
    }
    // Group sessions by folder_id
    const byFolder = new Map();
    const ungrouped = [];
    allSessions.forEach(s => {
      if (s.folder_id == null) ungrouped.push(s);
      else {
        if (!byFolder.has(s.folder_id)) byFolder.set(s.folder_id, []);
        byFolder.get(s.folder_id).push(s);
      }
    });
    const collapsed = this._collapsedFolders();
    const activeFid = this._getActiveFolderId();
    // Drop stale active reference if folder was deleted
    if (activeFid && !this._folders.some(f => f.id === activeFid)) {
      this._activeFolderId = null;
      localStorage.removeItem('cc-active-folder');
    }
    // Render each named folder
    this._folders.forEach(f => {
      const inside = byFolder.get(f.id) || [];
      const isCollapsed = collapsed.has(`f:${f.id}`);
      const isActive = this._getActiveFolderId() === f.id;
      const row = document.createElement('div');
      row.className = 'folder-row'
        + (isCollapsed ? ' collapsed' : '')
        + (isActive ? ' active-folder' : '');
      row.dataset.folderId = String(f.id);
      row.innerHTML = `
        <span class="arrow">${isCollapsed ? '▸' : '▾'}</span>
        <span class="folder-name">${this._escapeHtml(f.name)}</span>
        <span class="folder-count">${inside.length}</span>`;
      row.onclick = (e) => {
        // Click on the arrow: only toggle collapse. Click anywhere else on
        // the row: enter the folder (set as active). Mirrors how IDE-style
        // tree views separate the disclosure triangle from the row body.
        if (e.target.classList.contains('arrow')) {
          this._toggleCollapse(`f:${f.id}`);
        } else if (this._getActiveFolderId() === f.id) {
          // Already active → exit folder context (clear active)
          this._setActiveFolder(null);
        } else {
          this._setActiveFolder(f.id);
        }
      };
      row.oncontextmenu = (e) => {
        e.preventDefault();
        this._showFolderMenu(e.clientX, e.clientY, f);
      };
      this._wireDropTarget(row, f.id);
      list.appendChild(row);
      const wrap = document.createElement('div');
      wrap.className = 'folder-children' + (isCollapsed ? ' hidden' : '');
      inside.forEach(s => wrap.appendChild(this._renderSessItem(s, sel)));
      list.appendChild(wrap);
    });
    // Render Ungrouped header (always shown, even empty, when folders exist)
    const showUngrouped = this._folders.length > 0
      ? true                           // header always shown as a drop target
      : ungrouped.length > 0;          // no folders → just sessions, no header
    if (showUngrouped && this._folders.length > 0) {
      const isCollapsed = collapsed.has('ungrouped');
      const row = document.createElement('div');
      row.className = 'folder-row ungrouped'
        + (isCollapsed ? ' collapsed' : '');
      row.innerHTML = `
        <span class="arrow">${isCollapsed ? '▸' : '▾'}</span>
        <span class="folder-name">Ungrouped</span>
        <span class="folder-count">${ungrouped.length}</span>`;
      row.onclick = () => this._toggleCollapse('ungrouped');
      this._wireDropTarget(row, null);
      list.appendChild(row);
      const wrap = document.createElement('div');
      wrap.className = 'folder-children' + (isCollapsed ? ' hidden' : '');
      ungrouped.forEach(s => wrap.appendChild(this._renderSessItem(s, sel)));
      list.appendChild(wrap);
    } else {
      // No folders at all → render sessions flat (legacy layout)
      ungrouped.forEach(s =>
        list.appendChild(this._renderSessItem(s, sel)));
    }
    this._renderBatchBar();
    this._updateTopbarFolder();
  },

  _renderSessItem(s, sel) {
    const checked = sel && this._selectedIds.has(s.id);
    const el = document.createElement('div');
    el.className = 'sess-item'
      + (s.id === this.sessionId && !sel ? ' active' : '')
      + (checked ? ' selected' : '');
    el.dataset.sessionId = s.id;
    if (!sel) el.draggable = true;   // drag disabled in batch-select mode
    const title = s.title && s.title !== 'New chat'
      ? s.title : `Untitled (${s.id.slice(0, 6)})`;
    const checkboxHtml = sel ? '<span class="sess-checkbox"></span>' : '';
    el.innerHTML = `
      <div class="sess-title">
        ${checkboxHtml}
        <span class="sess-dot ${s.busy ? '' : 'idle'}"></span>
        <span>${this._escapeHtml(title)}</span>
      </div>
      <div class="sess-info">
        <span>${s.message_count || 0} msg</span>
        <span>${this._fmtRelTime(s.last_active)}</span>
      </div>`;
    el.onclick = () => {
      if (this._selectMode) this._toggleSelected(s.id);
      else this.switchSession(s.id);
    };
    el.oncontextmenu = (e) => {
      e.preventDefault();
      if (this._selectMode) return;
      this._showSessMenu(e.clientX, e.clientY, s);
    };
    el.ondragstart = (e) => {
      e.dataTransfer.setData('text/cc-session-id', s.id);
      e.dataTransfer.effectAllowed = 'move';
      el.classList.add('dragging');
    };
    el.ondragend = () => el.classList.remove('dragging');
    return el;
  },

  _wireDropTarget(row, folderId) {
    row.ondragover = (e) => {
      const sid = e.dataTransfer.types.includes('text/cc-session-id');
      if (!sid) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      row.classList.add('drop-target');
    };
    row.ondragleave = () => row.classList.remove('drop-target');
    row.ondrop = (e) => {
      e.preventDefault();
      row.classList.remove('drop-target');
      const sid = e.dataTransfer.getData('text/cc-session-id');
      if (sid) this.moveSessionToFolder(sid, folderId);
    };
  },

  toggleSelectMode() {
    this._selectMode = !this._selectMode;
    if (!this._selectedIds) this._selectedIds = new Set();
    if (!this._selectMode) this._selectedIds.clear();
    const btn = document.getElementById('select-btn');
    if (btn) btn.classList.toggle('active', this._selectMode);
    this._renderSessionList();
  },

  _toggleSelected(sid) {
    if (!this._selectedIds) this._selectedIds = new Set();
    if (this._selectedIds.has(sid)) this._selectedIds.delete(sid);
    else this._selectedIds.add(sid);
    this._renderSessionList();
  },

  _renderBatchBar() {
    const bar = document.getElementById('batch-bar');
    if (!bar) return;
    if (!this._selectMode) {
      bar.style.display = 'none';
      bar.innerHTML = '';
      return;
    }
    const sel = this._selectedIds || new Set();
    const visibleIds = this._visibleSessionIds();
    const allSelected = visibleIds.length > 0
      && visibleIds.every(id => sel.has(id));
    const toggleLabel = allSelected ? 'Deselect all' : 'Select all';
    const dis = sel.size === 0 ? 'disabled' : '';
    bar.style.display = '';
    bar.innerHTML = `
      <div class="batch-count">
        <span>${sel.size} selected</span>
        <button class="batch-link" onclick="app._toggleSelectAll()"
                ${visibleIds.length === 0 ? 'disabled' : ''}>${toggleLabel}</button>
      </div>
      <div class="batch-actions">
        <button class="btn-delete" ${dis} onclick="app.batchDelete()">Delete</button>
        <button class="btn-export" ${dis} onclick="app.batchExport()">Export</button>
        <button onclick="app.toggleSelectMode()">Cancel</button>
      </div>`;
  },

  _visibleSessionIds() {
    const q = (document.getElementById('sess-search-input')?.value || '')
      .trim().toLowerCase();
    return (this._sessions || [])
      .filter(s => !q
        || (s.title || '').toLowerCase().includes(q)
        || s.id.includes(q))
      .map(s => s.id);
  },

  _toggleSelectAll() {
    if (!this._selectedIds) this._selectedIds = new Set();
    const visible = this._visibleSessionIds();
    if (visible.length === 0) return;
    const allSelected = visible.every(id => this._selectedIds.has(id));
    if (allSelected) {
      visible.forEach(id => this._selectedIds.delete(id));
    } else {
      visible.forEach(id => this._selectedIds.add(id));
    }
    this._renderSessionList();
  },

  async batchDelete() {
    const ids = Array.from(this._selectedIds || []);
    if (ids.length === 0) return;
    const totalMsgs = (this._sessions || [])
      .filter(s => this._selectedIds.has(s.id))
      .reduce((sum, s) => sum + (s.message_count || 0), 0);
    if (!confirm(
      `Delete ${ids.length} session${ids.length === 1 ? '' : 's'}?\n\n` +
      `This removes ${totalMsgs} messages permanently.`
    )) return;
    try {
      const r = await this._fetchAuth('/api/sessions/batch_delete', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ids}),
      });
      const data = await r.json();
      if (!r.ok) {
        alert(data.error || `Server error (${r.status})`);
        return;
      }
      // If we deleted the currently active session, clear chat
      if (this.sessionId && this._selectedIds.has(this.sessionId)) {
        this._disconnectWS();
        this.sessionId = null;
        this._clearChat();
        this._showWelcome();
      }
      this._selectedIds.clear();
      this._selectMode = false;
      const btn = document.getElementById('select-btn');
      if (btn) btn.classList.remove('active');
      this.loadSessions();
    } catch(e) { alert('Delete failed: ' + e.message); }
  },

  async batchExport() {
    const ids = Array.from(this._selectedIds || []);
    if (ids.length === 0) return;
    try {
      const r = await this._fetchAuth('/api/sessions/batch_export', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ids}),
      });
      if (!r.ok) {
        const data = await r.json().catch(() => ({}));
        alert(data.error || `Server error (${r.status})`);
        return;
      }
      const blob = await r.blob();
      const cd = r.headers.get('Content-Disposition') || '';
      const m = cd.match(/filename="?([^"]+)"?/);
      const fname = (m && m[1]) || `chats-${ids.length}-sessions.md`;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = fname;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch(e) { alert('Export failed: ' + e.message); }
  },

  _showSessMenu(x, y, sess) {
    const menu = document.getElementById('sess-menu');
    const folders = this._folders || [];
    let moveItems = '';
    folders.forEach(f => {
      if (f.id === sess.folder_id) return;  // already in this folder
      moveItems += `<div class="menu-item" data-act="move:${f.id}">`
        + `&nbsp;&nbsp;${this._escapeHtml(f.name)}</div>`;
    });
    if (sess.folder_id != null) {
      moveItems += '<div class="menu-item" data-act="move:null">'
        + '&nbsp;&nbsp;(Ungrouped)</div>';
    }
    moveItems += '<div class="menu-item" data-act="move:new">'
      + '&nbsp;&nbsp;+ New folder…</div>';
    menu.innerHTML = `
      <div class="menu-item" data-act="rename">Rename...</div>
      <div class="menu-item" data-act="export">Export Markdown</div>
      <div class="menu-sep"></div>
      <div class="menu-item" data-act="movehdr"
           style="cursor:default;color:var(--text-muted);font-size:11px;">
        Move to:</div>
      ${moveItems}
      <div class="menu-sep"></div>
      <div class="menu-item danger" data-act="delete">Delete</div>`;
    menu.querySelectorAll('.menu-item').forEach(item => {
      const act = item.dataset.act;
      if (act === 'movehdr') return;  // header, not clickable
      item.onclick = async () => {
        menu.style.display = 'none';
        if (act === 'rename') this.renameSession(sess);
        else if (act === 'export') this.exportSession(sess);
        else if (act === 'delete') this.deleteSession(sess);
        else if (act && act.startsWith('move:')) {
          const target = act.slice(5);
          if (target === 'new') {
            const name = prompt('New folder name:');
            if (!name || !name.trim()) return;
            const f = await this._createFolder(name.trim());
            if (f) await this.moveSessionToFolder(sess.id, f.id);
          } else if (target === 'null') {
            await this.moveSessionToFolder(sess.id, null);
          } else {
            await this.moveSessionToFolder(sess.id, parseInt(target, 10));
          }
        }
      };
    });
    menu.style.display = 'block';
    const rect = menu.getBoundingClientRect();
    const px = Math.min(x, window.innerWidth - rect.width - 8);
    const py = Math.min(y, window.innerHeight - rect.height - 8);
    menu.style.left = px + 'px';
    menu.style.top = py + 'px';
    const dismiss = (ev) => {
      if (!menu.contains(ev.target)) {
        menu.style.display = 'none';
        document.removeEventListener('click', dismiss);
      }
    };
    setTimeout(() => document.addEventListener('click', dismiss), 0);
  },

  _showFolderMenu(x, y, folder) {
    const menu = document.getElementById('sess-menu');
    menu.innerHTML = `
      <div class="menu-item" data-act="rename">Rename...</div>
      <div class="menu-sep"></div>
      <div class="menu-item danger" data-act="delete">Delete folder</div>`;
    menu.querySelectorAll('.menu-item').forEach(item => {
      item.onclick = () => {
        menu.style.display = 'none';
        const act = item.dataset.act;
        if (act === 'rename') this.renameFolder(folder);
        else if (act === 'delete') this.deleteFolder(folder);
      };
    });
    menu.style.display = 'block';
    const rect = menu.getBoundingClientRect();
    const px = Math.min(x, window.innerWidth - rect.width - 8);
    const py = Math.min(y, window.innerHeight - rect.height - 8);
    menu.style.left = px + 'px';
    menu.style.top = py + 'px';
    const dismiss = (ev) => {
      if (!menu.contains(ev.target)) {
        menu.style.display = 'none';
        document.removeEventListener('click', dismiss);
      }
    };
    setTimeout(() => document.addEventListener('click', dismiss), 0);
  },

  async _createFolder(name) {
    try {
      const r = await this._fetchAuth('/api/folders', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({name}),
      });
      const data = await r.json();
      if (!r.ok) {
        alert(data.error || `Server error (${r.status})`);
        return null;
      }
      return data;
    } catch(e) { alert('Create folder failed: ' + e.message); return null; }
  },

  async newFolder() {
    const name = prompt('New folder name:');
    if (!name || !name.trim()) return;
    const f = await this._createFolder(name.trim());
    if (f) this.loadSessions();
  },

  async renameFolder(folder) {
    const name = prompt('Rename folder:', folder.name);
    if (name === null) return;
    const t = name.trim();
    if (!t || t === folder.name) return;
    try {
      const r = await this._fetchAuth(`/api/folders/${folder.id}`, {
        method: 'PATCH',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({name: t}),
      });
      const data = await r.json();
      if (!r.ok) { alert(data.error || `Server error (${r.status})`); return; }
      this.loadSessions();
    } catch(e) { alert('Rename failed: ' + e.message); }
  },

  async deleteFolder(folder) {
    if (!confirm(
      `Delete folder "${folder.name}"?\n\n` +
      `Sessions inside (${folder.session_count || 0}) will become Ungrouped — ` +
      `they are NOT deleted.`
    )) return;
    try {
      const r = await this._fetchAuth(`/api/folders/${folder.id}`, {
        method: 'DELETE',
      });
      if (!r.ok) {
        const data = await r.json().catch(() => ({}));
        alert(data.error || `Server error (${r.status})`);
        return;
      }
      this.loadSessions();
    } catch(e) { alert('Delete failed: ' + e.message); }
  },

  async moveSessionToFolder(sid, folderId) {
    try {
      const r = await this._fetchAuth(`/api/sessions/${sid}/folder`, {
        method: 'PATCH',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({folder_id: folderId}),
      });
      if (!r.ok) {
        const data = await r.json().catch(() => ({}));
        alert(data.error || `Server error (${r.status})`);
        return;
      }
      this.loadSessions();
    } catch(e) { alert('Move failed: ' + e.message); }
  },

  async renameSession(sess) {
    const title = prompt('Rename session:', sess.title || '');
    if (title === null) return;
    const t = title.trim();
    if (!t) return;
    try {
      const r = await this._fetchAuth(`/api/sessions/${sess.id}`, {
        method: 'PATCH',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({title: t}),
      });
      if (r.ok) this.loadSessions();
    } catch(e) { console.error('rename:', e); }
  },

  async deleteSession(sess) {
    if (!confirm(`Delete session "${sess.title || sess.id}"?\n\n` +
                 `This removes ${sess.message_count || 0} messages permanently.`)) {
      return;
    }
    try {
      const r = await this._fetchAuth(`/api/sessions/${sess.id}`, {
        method: 'DELETE',
      });
      if (r.ok) {
        if (this.sessionId === sess.id) {
          this._disconnectWS();
          this.sessionId = null;
          this._clearChat();
          this._showWelcome();
        }
        this.loadSessions();
      }
    } catch(e) { console.error('delete:', e); }
  },

  exportSession(sess) {
    window.location.href = `/api/sessions/${sess.id}/export`;
  },

  async newSession() {
    this._disconnectWS();
    this._clearChat();
    try {
      const r = await this._fetchAuth('/api/prompt', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({prompt: '', session_id: ''}),
      });
      const data = await r.json();
      if (r.ok && data.session_id) {
        this.sessionId = data.session_id;
        // If the user is "in" a folder, drop the new session into it.
        const fid = this._getActiveFolderId();
        if (fid) {
          try {
            await this._fetchAuth(
              `/api/sessions/${data.session_id}/folder`, {
                method: 'PATCH',
                headers: {'Content-Type':'application/json'},
                body: JSON.stringify({folder_id: fid}),
              });
          } catch(e) { /* non-fatal — session still exists, just ungrouped */ }
        }
        this._connectWS(this.sessionId);
        this._showWelcome();
      }
    } catch(e) { console.error('newSession:', e); }
    this.loadSessions();
  },

  async switchSession(sid) {
    if (sid === this.sessionId) return;
    this._disconnectWS();
    this.sessionId = sid;
    // Sync active folder to the session's folder so subsequent + New stays
    // in the user's current context (ChatGPT-style follow-the-breadcrumb).
    const s = (this._sessions || []).find(x => x.id === sid);
    if (s) {
      const fid = s.folder_id || null;
      if (fid !== this._getActiveFolderId()) {
        this._activeFolderId = fid;
        if (fid) localStorage.setItem('cc-active-folder', String(fid));
        else localStorage.removeItem('cc-active-folder');
        this._updateTopbarFolder();
      }
    }
    this._clearChat();
    try {
      const r = await this._fetchAuth(`/api/sessions/${sid}`);
      const data = await r.json();
      (data.messages || []).forEach(m => {
        if (m.role === 'user') this._addUserBubble(m.content);
        else if (m.role === 'assistant') {
          this._addAssistantBubble(m.content);
          if (m.tool_calls) m.tool_calls.forEach(tc => {
            this._addToolCard(tc.name, tc.inputs, tc.status, tc.result);
          });
        }
      });
    } catch(e) { console.error('switchSession:', e); }
    this._connectWS(sid);
    this.loadSessions();
  },
});
