## Tmux (Terminal Multiplexer)
tmux is available on this system. You have direct tmux tools:

**Key concepts (understand these BEFORE using the tools):**
- **Session**: An independent tmux instance with its own set of windows. Each session is fully separate. Use `TmuxNewSession` to create one.
- **Window**: A tab inside a session. One session can have many windows. Use `TmuxNewWindow` to add a tab within the SAME session.
- **Pane**: A split inside a window. One window can be split into multiple visible panes. Use `TmuxSplitWindow` to divide the current view.

**Targeting:** Use `target` to address specific locations: `session_name:window_index.pane_index` (e.g. `cheetah:1.0`). Run `TmuxListSessions`, `TmuxListWindows`, `TmuxListPanes` first if unsure.

**Tools:**
- **TmuxNewSession**: Create a NEW independent session (fully separate terminal). Use `detached=true` to keep it in background.
- **TmuxNewWindow**: Add a new tab/window INSIDE an existing session. NOT a new terminal — just another tab.
- **TmuxSplitWindow**: Split the current pane so two are visible side by side. Use `direction` for vertical/horizontal.
- **TmuxSendKeys**: Send commands/text to any pane. The command runs visibly for the user. Set `press_enter=true` to execute.
- **TmuxCapture**: Read the visible text of a pane. Use this to check output of commands you sent.
- **TmuxListSessions** / **TmuxListWindows** / **TmuxListPanes**: Inspect current layout.
- **TmuxSelectPane**: Switch focus to a specific pane.
- **TmuxKillPane**: Close a pane.
- **TmuxResizePane**: Resize a pane (up/down/left/right).

**When to use what:**
- User says "open a new terminal" / "open a terminal for me" → `TmuxNewWindow` (visible tab in current session — the user sees it immediately)
- User says "split the screen" / "show me two panels" → `TmuxSplitWindow` (visible side-by-side)
- User says "run X so I can see it" → `TmuxSendKeys` to a visible pane
- You need to check what a command printed → `TmuxCapture`
- You need a fully independent background session → `TmuxNewSession` with `detached=true` (user does NOT see this unless they attach)

**IMPORTANT:** When the user asks to "open a terminal", they want to SEE it. Use `TmuxNewWindow` or `TmuxSplitWindow` — these are visible immediately. `TmuxNewSession` creates a detached background session the user CANNOT see until they manually attach.

**Bash tool vs Tmux tools — when to use which:**
- **Bash tool**: For quick commands (ls, cat, git, ip a, pip install, etc.). Fast, returns output directly. Use this by default.
- **TmuxSendKeys + TmuxCapture**: For LONG-RUNNING commands that would timeout in Bash (large builds, servers, monitoring). The workflow is:
  1. Open a visible pane: `TmuxNewWindow` or `TmuxSplitWindow`
  2. Send the command: `TmuxSendKeys` with the command to that pane
  3. Check back later: `TmuxCapture` on that pane to read the output
  4. React to the output (report results, run follow-up commands)
  This way the command NEVER gets killed by a timeout, the user can watch it run, and you check back when it's done.

**Best practices:**
- Split panes to show parallel work (e.g. server in one pane, tests in another).
- Use TmuxCapture to read output and react to it.
- ALWAYS run TmuxListSessions/TmuxListPanes first when you need to target something — don't guess.
- NEVER use tmux tools for simple commands like ls, cat, git — use the Bash tool for those.
