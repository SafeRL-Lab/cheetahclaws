# FAQ

Frequently asked questions for CheetahClaws. A short subset of these lives in the
[README](../../README.md#faq); the full list is here.

## MCP

**Q: How do I add an MCP server?**

Option 1 — via REPL (stdio server):
```
/mcp add git uvx mcp-server-git
```

Option 2 — create `.mcp.json` in your project:
```json
{
  "mcpServers": {
    "git": {"type": "stdio", "command": "uvx", "args": ["mcp-server-git"]}
  }
}
```

Then run `/mcp reload` or restart. Use `/mcp` to check connection status.

**Q: An MCP server is showing an error. How do I debug it?**

```
/mcp                    # shows error message per server
/mcp reload git         # try reconnecting
```

If the server uses stdio, make sure the command is in your `$PATH`:
```bash
which uvx               # should print a path
uvx mcp-server-git      # run manually to see errors
```

**Q: Can I use MCP servers that require authentication?**

For HTTP/SSE servers with a Bearer token:
```json
{
  "mcpServers": {
    "my-api": {
      "type": "sse",
      "url": "https://myserver.example.com/sse",
      "headers": {"Authorization": "Bearer sk-my-token"}
    }
  }
}
```

For stdio servers with env-based auth:
```json
{
  "mcpServers": {
    "brave": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-server-brave-search"],
      "env": {"BRAVE_API_KEY": "your-key"}
    }
  }
}
```

## Models & providers

**Q: Tool calls don't work with my local Ollama model (it just keeps describing what it would do instead of doing it).**

CheetahClaws now auto-recovers tool calls that local models emit as **text** — `<tool_call>…</tool_call>` (Qwen/Hermes), `<|tool_call|>…` (Gemma), `[TOOL_CALLS]…` (Mistral) — instead of in Ollama's structured `message.tool_calls` field. Previously those were streamed as chat and never executed, which is why the model seemed to "keep talking." Most function-calling models now execute tools out of the box.

For best reliability use one of the recommended tool-calling models. Small local models are also weaker at agentic tool use than cloud models, so give them clear, concrete prompts (a path, a filename, an exact command):

```bash
ollama pull qwen2.5-coder
cheetahclaws --model ollama/qwen2.5-coder
```

If a model returns `500` on the first tool-enabled request, it has no tool template — CheetahClaws falls back to chat-only (a yellow `[warn]` is printed). Pull one of the models above instead.

**Q: How do I connect to a remote GPU server running vLLM?**

```
/config custom_base_url=http://your-server-ip:8000/v1
/config custom_api_key=your-token
/model custom/your-model-name
```

**Q: How do I check my API cost?**

```
/cost

  Input tokens:  3,421
  Output tokens:   892
  Est. cost:     $0.0648 USD
```

**Q: Can I use multiple API keys in the same session?**

Yes. Set all the keys you need upfront (via env vars or `/config`). Then switch models freely — each call uses the key for the active provider.

**Q: How do I make a model available across all projects?**

Add keys to `~/.bashrc` or `~/.zshrc`. Set the default model in `~/.cheetahclaws/config.json`:

```json
{ "model": "claude-sonnet-4-6" }
```

**Q: Qwen / Zhipu returns garbled text.**

Ensure your `DASHSCOPE_API_KEY` / `ZHIPU_API_KEY` is correct and the account has sufficient quota. Both providers use UTF-8 and handle Chinese well.

## CLI & scripting

**Q: Can I pipe input to cheetahclaws?**

```bash
echo "Explain this file" | cheetahclaws --print --accept-all
cat error.log | cheetahclaws -p "What is causing this error?"
```

**Q: How do I run it as a CLI tool from anywhere?**

Use `uv tool install` — it creates an isolated environment and puts `cheetahclaws` on your PATH:

```bash
cd cheetahclaws
uv tool install ".[all]"
```

After that, just run `cheetahclaws` from any directory. To update after pulling changes, run `uv tool install ".[all]" --reinstall`. For a minimal install, use `uv tool install .` and add extras as needed.

**Q: After installing on macOS I get `cheetahclaws: command not found`, and `~/.zshrc` was never created.**

Reload your shell in a new terminal first:

```bash
source ~/.zshrc          # zsh (macOS default)
source ~/.bash_profile   # bash on macOS
```

On macOS the installer creates a dedicated virtual environment (`~/.cheetahclaws-venv`), symlinks the `cheetahclaws` entry point into `~/.local/bin`, creates `~/.zshrc` if it's missing, and appends `~/.local/bin` to your `PATH` there. (It links only the one binary rather than putting the whole venv on `PATH`, so your own `python`/`pip` aren't shadowed.) If you installed an older build that skipped this, either re-run the installer or add it yourself:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

## Voice

**Q: How do I set up voice input?**

```bash
# Minimal setup (local, offline, no API key):
pip install sounddevice faster-whisper numpy

# Then in the REPL:
/voice status          # verify backends are detected
/voice                 # speak your prompt
```

On first use, `faster-whisper` downloads the `base` model (~150 MB) automatically.
Use a larger model for better accuracy: `export NANO_CLAUDE_WHISPER_MODEL=small`

**Q: Voice input transcribes my words wrong (misses coding terms).**

The keyterm booster already injects coding vocabulary from your git branch and project files.
For persistent domain terms, put them in a `.cheetahclaws/voice_keyterms.txt` file (one term per line) — this is checked automatically on each recording.

**Q: Can I use voice input in Chinese / Japanese / other languages?**

Yes. Set the language before recording:

```
/voice lang zh    # Mandarin Chinese
/voice lang ja    # Japanese
/voice lang auto  # reset to auto-detect (default)
```

Whisper supports 99 languages. `auto` detection works well but explicit codes improve accuracy for short utterances.
