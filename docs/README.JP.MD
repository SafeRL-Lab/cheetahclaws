[English](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/README.md) | [中文](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/README.CN.MD)  |[한국어](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/README.KO.MD) | 日本語 | [Deutsch](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/README.DE.MD) | [Português](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/README.ES.MD)

<div align="center">
  <a href="[https://github.com/SafeRL-Lab/Robust-Gymnasium](https://github.com/SafeRL-Lab/nano-claude-code)">
    <img src="https://github.com/SafeRL-Lab/cheetahclaws/blob/main/docs/logo-5.png" alt="ロゴ" width="280"> 
  </a>

<h1 align="center" style="font-size: 30px;"><strong><em>Nano Claude Code</em></strong>: あらゆるモデルをサポートする、高速で使いやすい Claude Code の Python 再実装</h1>
<p align="center">
    <a href="https://github.com/chauncygu/collection-claude-code-source-code">最新の Claude Code ソース集</a>
    ·
    <a href="https://github.com/SafeRL-Lab/nano-claude-code/issues">Issue</a>
  ·
    <a href="https://deepwiki.com/SafeRL-Lab/nano-claude-code">概要紹介</a>
  </p>
</div>

<div align=center>
 <img src="https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/demo.gif" width="850"/> 
</div>

---

## 🔥🔥🔥 ニュース（太平洋時間）
- 2026年4月3日 18:00: **v3.03** — タスク管理システム（`task/` パッケージ）: `TaskCreate` / `TaskUpdate` / `TaskGet` / `TaskList`、連番ID、依存関係エッジ（blocks/blocked_by）、メタデータ、`.nano_claude/tasks.json` への永続化、スレッドセーフなストア、`/tasks` REPL コマンド、37 個の新規テスト（**約 9500 行** の Python）。
- 2026年4月3日 14:50: **v3.02** — プラグインシステム（`plugin/` パッケージ）: `/plugin` CLI による install/uninstall/enable/disable/update、推薦エンジン（キーワード + タグ一致）、マルチスコープ（user/project）、git ベースのマーケットプレイス。`AskUserQuestion` ツールで、タスク途中に番号付き選択肢や自由入力を使った対話的質問が可能（**約 8500 行** の Python）。
- 2026年4月3日 10:00: **v3.01** — MCP（Model Context Protocol）対応: `mcp/` パッケージ、stdio + SSE + HTTP トランスポート、自動ツール検出、`/mcp` コマンド、34 個の新規テスト（**約 7000 行** の Python）。
- 2026年4月2日 12:20: **v3.0** — マルチエージェント（`multi_agent/`）、メモリ（`memory/`）、スキル（`skill/`）に加え、組み込みスキル、引数置換、fork/inline 実行、AI メモリ検索、git worktree 分離、エージェントタイプ定義を追加（**約 5000 行** の Python）。
- 2026年4月2日 10:00: **v2.0** — コンテキスト圧縮、メモリ、サブエージェント、スキル、diff 表示、ツールプラグインシステム（**約 3400 行** の Python）。
- 2026年4月1日 13:47: VLLM 推論をサポート（**約 2000 行** の Python）。
- 2026年4月1日 11:30: より多くの**クローズドソース**／**オープンソース**モデルをサポート: Claude, GPT, Gemini, Kimi, Qwen, Zhipu, DeepSeek, さらに Ollama または任意の OpenAI 互換エンドポイント経由のローカル OSS モデル（**約 1700 行** の Python）。
- 2026年4月1日 09:50: より多くの**クローズドソース**モデルをサポート: Claude, GPT, Gemini（**約 1300 行** の Python）。
- 2026年4月1日 08:23: Nano Claude Code の初版を公開（**約 900 行** の Python）。

---

# Nano Claude Code

Nano Claude Code は、Claude, GPT, Gemini, Kimi, Qwen, Zhipu, DeepSeek、さらに Ollama や任意の OpenAI 互換エンドポイント経由のローカル OSS モデルなど、**あらゆるモデルをサポート**する、**軽量で使いやすい** Claude Code の Python 再実装です。

---

## 目次
- [なぜ Nano Claude Code なのか](#なぜ-nano-claude-code-なのか)
- [機能](#機能)
- [対応モデル](#対応モデル)
- [インストール](#インストール)
- [使い方: クローズド API モデル](#使い方-クローズド-api-モデル)
- [使い方: オープンソースモデル（ローカル）](#使い方-オープンソースモデルローカル)
- [モデル名の形式](#モデル名の形式)
- [CLI リファレンス](#cli-リファレンス)
- [スラッシュコマンド（REPL）](#スラッシュコマンドrepl)
- [API キーの設定](#api-キーの設定)
- [権限システム](#権限システム)
- [組み込みツール](#組み込みツール)
- [メモリ](#メモリ)
- [スキル](#スキル)
- [サブエージェント](#サブエージェント)
- [MCP](#mcp)
- [プラグインシステム](#プラグインシステム)
- [AskUserQuestion ツール](#askuserquestion-ツール)
- [タスク管理](#タスク管理)
- [コンテキスト圧縮](#コンテキスト圧縮)
- [Diff 表示](#diff-表示)
- [CLAUDE.md サポート](#claudemd-サポート)
- [セッション管理](#セッション管理)
- [プロジェクト構成](#プロジェクト構成)
- [FAQ](#faq)

## なぜ Nano Claude Code なのか

Claude Code は強力な本番品質の AI コーディングアシスタントですが、そのソースコードはコンパイル済みの 12MB TypeScript/Node.js バンドル（約 1300 ファイル、約 28.3 万行）です。Anthropic API に強く結びついており、変更しづらく、ローカルモデルや他のモデルで実行することは困難です。

**Nano Claude Code** は、同じコアループを、読みやすい Python 約 1 万行で再実装したものです。必要なものは残し、不要なものは削ぎ落としています。より詳細な比較は以下を参照してください（Nano Claude Code v3.03）: [英語版](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/comparison_claude_code_vs_nano_v3.03_en.md)、[中国語版](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/comparison_claude_code_vs_nano_v3.03_cn.md)

### ひと目でわかる比較

| 項目 | Claude Code (TypeScript) | Nano Claude Code (Python) |
|---|---|---|
| 言語 | TypeScript + React/Ink | Python 3.8+ |
| ソースファイル数 | 約 1,332 TS/TSX ファイル | 51 Python ファイル |
| コード行数 | 約 283K | 約 10.2K |
| 組み込みツール | 44+ | 21 |
| スラッシュコマンド | 88 | 17 |
| モデル提供元 | Anthropic のみ | 7+（Anthropic · OpenAI · Gemini · Kimi · Qwen · DeepSeek · Ollama · …） |
| ローカルモデル | なし | あり — Ollama, LM Studio, vLLM, OpenAI 互換エンドポイント |
| ビルド工程 | 必要（Bun + esbuild） | 不要 — `python nano_claude.py` を直接実行 |
| 実行時拡張性 | 閉じている（コンパイル時） | 開いている — `register_tool()`、Markdown スキル、git プラグイン |
| タスク依存グラフ | なし | あり — `task/` パッケージの `blocks` / `blocked_by` |

### Claude Code が優れている点

- **UI 品質** — ストリーミング表示、細かな diff 可視化、ダイアログシステムを備えた React/Ink コンポーネントツリー
- **ツールの幅** — `NotebookEdit`、`LSP Diagnostics`、`RemoteTrigger`、`EnterWorktree` などを含む 44 以上のツール
- **エンタープライズ機能** — MDM 管理設定、チーム権限同期、OAuth、キーチェーン保存、GrowthBook フラグ
- **AI 主導のメモリ抽出** — `extractMemories` により、明示的なツール呼び出しなしで会話から知識を抽出
- **本番向け信頼性** — 単一配布ファイル `cli.js`、包括的なテスト、バージョン固定リリース

### Nano Claude Code が優れている点

- **マルチプロバイダ** — `--model` または `/model` で Claude、GPT-4o、Gemini 2.5 Pro、DeepSeek、Qwen、ローカル Llama を即時切り替え
- **ローカルモデル対応** — Ollama、LM Studio、vLLM ベースのモデルで完全オフライン実行
- **読みやすいソース** — エージェントループ本体は 174 行（`agent.py`）。Python 開発者なら短時間で理解し、fork・拡張できる
- **ビルド不要** — `pip install -r requirements.txt` 後すぐ実行可能。変更はその場で反映
- **動的拡張性** — `register_tool(ToolDef(...))` によるランタイム登録、git URL からのスキル導入、MCP サーバ接続
- **タスク依存グラフ** — `TaskCreate` / `TaskUpdate` が `blocks` / `blocked_by` をサポートし、構造化された複数ステップ計画が可能
- **二層コンテキスト圧縮** — ルールベースの snip + AI 要約。`preserve_last_n_turns` で調整可能

### 主要な設計上の違い

**エージェントループ** — Nano は、型付きイベント（`TextChunk`, `ToolStart`, `ToolEnd`, `TurnDone`）を `yield` する Python ジェネレータを用います。ループ全体が 1 ファイルで見通せるため、フック、独自レンダラ、ロギングの追加が容易です。

**ツール登録** — すべてのツールは `ToolDef(name, schema, func, read_only, concurrent_safe)` という dataclass です。任意のモジュールが import 時に `register_tool()` を呼べます。MCP サーバ、プラグイン、スキルも同じ仕組みです。

**コンテキスト圧縮**

| | Claude Code | Nano Claude Code |
|---|---|---|
| 発火条件 | 正確なトークン数 | `len / 3.5` による推定、70% で発火 |
| レイヤー 1 | — | Snip: 古いツール出力を切り詰める（API コストなし） |
| レイヤー 2 | AI 要約 | 古いターンを AI が要約 |
| 制御 | システム管理 | `preserve_last_n_turns` パラメータ |

**メモリ** — Claude Code の `extractMemories` はモデルが自発的に事実を取り出す方式です。Nano の `memory/` はツール主導で、モデルが明示的に `MemorySave` を呼びます。より予測しやすく、監査しやすい設計です。

### Nano Claude Code を使うべき人

- コーディング支援に**ローカルモデルや Anthropic 以外のモデル**を使いたい開発者
- **エージェント型コーディングアシスタントの仕組み**を研究したい研究者
- 独自ツール、独自権限ポリシー、特殊なエージェント種別を追加できる**ハッカブルな基盤**が必要なチーム
- **Node.js のビルドチェーンなしで** Claude Code 的な生産性を得たい人

---

## 機能

| 機能 | 詳細 |
|---|---|
| マルチプロバイダ | Anthropic · OpenAI · Gemini · Kimi · Qwen · Zhipu · DeepSeek · Ollama · LM Studio · カスタムエンドポイント |
| 対話型 REPL | readline 履歴、Tab 補完付きスラッシュコマンド |
| エージェントループ | ストリーミング API + 自動ツール使用ループ |
| 21 個の組み込みツール | Read · Write · Edit · Bash · Glob · Grep · WebFetch · WebSearch · MemorySave · MemoryDelete · MemorySearch · MemoryList · Agent · SendMessage · CheckAgentResult · ListAgentTasks · ListAgentTypes · Skill · SkillList · AskUserQuestion · *(MCP + plugin ツールは起動時に自動追加)* |
| MCP 統合 | 任意の MCP サーバ（stdio/SSE/HTTP）に接続可能。ツールは自動登録され Claude から呼び出し可能 |
| プラグインシステム | git URL またはローカルパスから install/uninstall/enable/disable/update。user/project のマルチスコープ。推薦エンジン付き |
| AskUserQuestion | タスクの途中で、番号付き選択肢を含む確認質問をユーザーに対して行える |
| タスク管理 | TaskCreate/Update/Get/List、連番ID、依存関係、メタデータ、`.nano_claude/tasks.json` への保存、`/tasks` コマンド |
| Diff 表示 | Edit と Write に対して Git 風の赤/緑 diff を表示 |
| コンテキスト圧縮 | 長い会話をモデルの上限内に保つため自動圧縮 |
| 永続メモリ | 4 タイプの二重スコープメモリ（user + project）、AI 検索、古さ警告 |
| マルチエージェント | 型付きサブエージェント（coder/reviewer/researcher/...）、git worktree 分離、バックグラウンドモード |
| スキル | 組み込み `/commit`・`/review` と、引数置換や fork/inline 実行を備えた Markdown スキル |
| プラグインツール | `tool_registry.py` によるカスタムツール登録 |
| 権限システム | `auto` / `accept-all` / `manual` |
| 17 個のスラッシュコマンド | `/model` · `/config` · `/save` · `/cost` · `/memory` · `/skills` · `/agents` · … |
| コンテキスト注入 | `CLAUDE.md`、git status、cwd、永続メモリを自動ロード |
| セッション保存 | 会話を `~/.nano_claude/sessions/` に保存 / 読み込み |
| Extended Thinking | ON/OFF 切り替え（Claude 系モデルのみ） |
| コスト追跡 | トークン使用量 + 推定 USD コスト |
| 非対話モード | スクリプト / CI 用 `--print` フラグ |

---

## 対応モデル

### クローズドソース（API）

| Provider | Model | Context | 強み | API Key Env |
|---|---|---|---|---|
| **Anthropic** | `claude-opus-4-6` | 200k | 最も高性能。複雑な推論向け | `ANTHROPIC_API_KEY` |
| **Anthropic** | `claude-sonnet-4-6` | 200k | 速度と品質のバランス | `ANTHROPIC_API_KEY` |
| **Anthropic** | `claude-haiku-4-5-20251001` | 200k | 高速・低コスト | `ANTHROPIC_API_KEY` |
| **OpenAI** | `gpt-4o` | 128k | 強力なマルチモーダルとコーディング | `OPENAI_API_KEY` |
| **OpenAI** | `gpt-4o-mini` | 128k | 高速・低価格 | `OPENAI_API_KEY` |
| **OpenAI** | `o3-mini` | 200k | 強い推論性能 | `OPENAI_API_KEY` |
| **OpenAI** | `o1` | 200k | 高度な推論 | `OPENAI_API_KEY` |
| **Google** | `gemini-2.5-pro-preview-03-25` | 1M | 長文コンテキスト、マルチモーダル | `GEMINI_API_KEY` |
| **Google** | `gemini-2.0-flash` | 1M | 高速、大規模コンテキスト | `GEMINI_API_KEY` |
| **Google** | `gemini-1.5-pro` | 2M | 最大級のコンテキスト窓 | `GEMINI_API_KEY` |
| **Moonshot (Kimi)** | `moonshot-v1-8k` | 8k | 中国語・英語 | `MOONSHOT_API_KEY` |
| **Moonshot (Kimi)** | `moonshot-v1-32k` | 32k | 中国語・英語 | `MOONSHOT_API_KEY` |
| **Moonshot (Kimi)** | `moonshot-v1-128k` | 128k | 長文コンテキスト | `MOONSHOT_API_KEY` |
| **Alibaba (Qwen)** | `qwen-max` | 32k | Qwen 系で最高品質 | `DASHSCOPE_API_KEY` |
| **Alibaba (Qwen)** | `qwen-plus` | 128k | バランス型 | `DASHSCOPE_API_KEY` |
| **Alibaba (Qwen)** | `qwen-turbo` | 1M | 高速・低価格 | `DASHSCOPE_API_KEY` |
| **Alibaba (Qwen)** | `qwq-32b` | 32k | 強い推論性能 | `DASHSCOPE_API_KEY` |
| **Zhipu (GLM)** | `glm-4-plus` | 128k | GLM 系で最高品質 | `ZHIPU_API_KEY` |
| **Zhipu (GLM)** | `glm-4` | 128k | 汎用 | `ZHIPU_API_KEY` |
| **Zhipu (GLM)** | `glm-4-flash` | 128k | 無料ティアあり | `ZHIPU_API_KEY` |
| **DeepSeek** | `deepseek-chat` | 64k | コーディングに強い | `DEEPSEEK_API_KEY` |
| **DeepSeek** | `deepseek-reasoner` | 64k | chain-of-thought 推論 | `DEEPSEEK_API_KEY` |

### オープンソース（Ollama 経由のローカル）

| Model | Size | 強み | Pull Command |
|---|---|---|---|
| `llama3.3` | 70B | 汎用、強い推論 | `ollama pull llama3.3` |
| `llama3.2` | 3B / 11B | 軽量 | `ollama pull llama3.2` |
| `qwen2.5-coder` | 7B / 32B | **コーディングに最適** | `ollama pull qwen2.5-coder` |
| `qwen2.5` | 7B / 72B | 中国語・英語 | `ollama pull qwen2.5` |
| `deepseek-r1` | 7B–70B | 推論、数学 | `ollama pull deepseek-r1` |
| `deepseek-coder-v2` | 16B | コーディング | `ollama pull deepseek-coder-v2` |
| `mistral` | 7B | 高速・高効率 | `ollama pull mistral` |
| `mixtral` | 8x7B | 強力な MoE モデル | `ollama pull mixtral` |
| `phi4` | 14B | Microsoft、強い推論 | `ollama pull phi4` |
| `gemma3` | 4B / 12B / 27B | Google のオープンモデル | `ollama pull gemma3` |
| `codellama` | 7B / 34B | コード生成 | `ollama pull codellama` |

> **注:** ツール呼び出しには function calling をサポートするモデルが必要です。推奨ローカルモデル: `qwen2.5-coder`, `llama3.3`, `mistral`, `phi4`。

---

## インストール

```bash
git clone <repo-url>
cd nano_claude_code

pip install -r requirements.txt
# または手動:
pip install anthropic openai httpx rich
```

---

## 使い方: クローズド API モデル

### Anthropic Claude

API キーは [console.anthropic.com](https://console.anthropic.com) で取得します。

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...

python nano_claude.py
python nano_claude.py --model claude-sonnet-4-6
python nano_claude.py --model claude-haiku-4-5-20251001
python nano_claude.py --model claude-opus-4-6 --thinking --verbose
```

### OpenAI GPT

API キーは [platform.openai.com](https://platform.openai.com) で取得します。

```bash
export OPENAI_API_KEY=sk-...

python nano_claude.py --model gpt-4o
python nano_claude.py --model gpt-4o-mini
python nano_claude.py --model gpt-4.1-mini
python nano_claude.py --model o3-mini
```

### Google Gemini

API キーは [aistudio.google.com](https://aistudio.google.com) で取得します。

```bash
export GEMINI_API_KEY=AIza...

python nano_claude.py --model gemini/gemini-2.0-flash
python nano_claude.py --model gemini/gemini-1.5-pro
python nano_claude.py --model gemini/gemini-2.5-pro-preview-03-25
```

### Kimi (Moonshot AI)

```bash
export MOONSHOT_API_KEY=sk-...

python nano_claude.py --model kimi/moonshot-v1-32k
python nano_claude.py --model kimi/moonshot-v1-128k
```

### Qwen (Alibaba DashScope)

```bash
export DASHSCOPE_API_KEY=sk-...

python nano_claude.py --model qwen/Qwen3.5-Plus
python nano_claude.py --model qwen/Qwen3-MAX
python nano_claude.py --model qwen/Qwen3.5-Flash
```

### Zhipu GLM

```bash
export ZHIPU_API_KEY=...

python nano_claude.py --model zhipu/glm-4-plus
python nano_claude.py --model zhipu/glm-4-flash
```

### DeepSeek

```bash
export DEEPSEEK_API_KEY=sk-...

python nano_claude.py --model deepseek/deepseek-chat
python nano_claude.py --model deepseek/deepseek-reasoner
```

---

## 使い方: オープンソースモデル（ローカル）

### Option A — Ollama（推奨）

Ollama は追加設定なしでローカル実行できます。API キーは不要です。

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

ollama pull qwen2.5-coder
ollama pull qwen2.5-coder:32b
ollama pull llama3.3
ollama pull deepseek-r1
ollama serve

python nano_claude.py --model ollama/qwen2.5-coder
python nano_claude.py --model ollama/llama3.3
python nano_claude.py --model ollama/deepseek-r1
```

ローカルにあるモデル一覧:
```bash
ollama list
```

### Option B — LM Studio

LM Studio は、モデルのダウンロードと実行を GUI で行い、OpenAI 互換サーバも提供します。

```bash
python nano_claude.py --model lmstudio/<model-name>
python nano_claude.py --model lmstudio/phi-4-GGUF
python nano_claude.py --model lmstudio/qwen2.5-coder-7b
```

### Option C — vLLM / Self-Hosted OpenAI-Compatible Server

```bash
CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-Coder-7B-Instruct \
      --host 0.0.0.0 \
      --port 8000 \
      --enable-auto-tool-choice \
      --tool-call-parser hermes

export CUSTOM_BASE_URL=http://localhost:8000/v1
export CUSTOM_API_KEY=none
python nano_claude.py --model custom/Qwen/Qwen2.5-Coder-7B-Instruct
```

REPL 内では:

```text
/config custom_base_url=http://localhost:8000/v1
/config custom_api_key=token-abc123
/model custom/Qwen2.5-Coder-32B-Instruct
```

---

## モデル名の形式

以下の 3 つの形式をサポートします。

```bash
python nano_claude.py --model gpt-4o
python nano_claude.py --model gemini-2.0-flash
python nano_claude.py --model deepseek-chat

python nano_claude.py --model ollama/qwen2.5-coder
python nano_claude.py --model kimi/moonshot-v1-128k

python nano_claude.py --model kimi:moonshot-v1-32k
python nano_claude.py --model qwen:qwen-max
```

**自動判定ルール:**

| Model prefix | Detected provider |
|---|---|
| `claude-` | anthropic |
| `gpt-`, `o1`, `o3` | openai |
| `gemini-` | gemini |
| `moonshot-`, `kimi-` | kimi |
| `qwen`, `qwq-` | qwen |
| `glm-` | zhipu |
| `deepseek-` | deepseek |
| `llama`, `mistral`, `phi`, `gemma`, `mixtral`, `codellama` | ollama |

---

## CLI リファレンス

```text
python nano_claude.py [OPTIONS] [PROMPT]

Options:
  -p, --print          非対話で実行して終了
  -m, --model MODEL    モデル指定
  --accept-all         すべて自動承認
  --verbose            thinking とトークン数を表示
  --thinking           Extended Thinking を有効化
  --version            バージョン表示
  -h, --help           ヘルプ表示
```

**例:**

```bash
python nano_claude.py
python nano_claude.py --model gpt-4o
python nano_claude.py -m ollama/deepseek-r1:32b
python nano_claude.py --print "Write a Python fibonacci function"
python nano_claude.py --accept-all --print "Initialize a Python project with pyproject.toml"
python nano_claude.py --thinking --verbose
```

---

## スラッシュコマンド（REPL）

| コマンド | 説明 |
|---|---|
| `/help` | 全コマンドを表示 |
| `/clear` | 会話履歴をクリア |
| `/model` | 現在モデルと利用可能モデル一覧 |
| `/model <name>` | モデルを即時切替 |
| `/config` | 現在の設定を表示 |
| `/config key=value` | 設定を保存 |
| `/save` | セッション保存 |
| `/save <filename>` | 名前を付けて保存 |
| `/load` | 保存済みセッション一覧 |
| `/load <filename>` | セッション読み込み |
| `/history` | 会話履歴全体を表示 |
| `/context` | メッセージ数とトークン概算 |
| `/cost` | トークン使用量と推定コスト |
| `/verbose` | verbose の切替 |
| `/thinking` | Extended Thinking の切替 |
| `/permissions` | 権限モード表示 |
| `/permissions <mode>` | `auto` / `accept-all` / `manual` |
| `/cwd` | 現在ディレクトリ表示 |
| `/cwd <path>` | 作業ディレクトリ変更 |
| `/memory` | 永続メモリ一覧 |
| `/memory <query>` | メモリ検索 |
| `/skills` | スキル一覧 |
| `/agents` | サブエージェントの状態 |
| `/mcp` | MCP サーバとツール一覧 |
| `/mcp reload` | 全 MCP サーバ再接続 |
| `/mcp reload <name>` | 特定サーバ再接続 |
| `/mcp add <name> <cmd> [args]` | stdio MCP サーバ追加 |
| `/mcp remove <name>` | サーバ削除 |
| `/exit` / `/quit` | 終了 |

---

## API キーの設定

### 方法 1: 環境変数（推奨）

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=AIza...
export MOONSHOT_API_KEY=sk-...
export DASHSCOPE_API_KEY=sk-...
export ZHIPU_API_KEY=...
export DEEPSEEK_API_KEY=sk-...
```

### 方法 2: REPL 内で設定

```text
/config anthropic_api_key=sk-ant-...
/config openai_api_key=sk-...
/config gemini_api_key=AIza...
/config kimi_api_key=sk-...
/config qwen_api_key=sk-...
/config zhipu_api_key=...
/config deepseek_api_key=sk-...
```

### 方法 3: 設定ファイルを直接編集

```json
{
  "model": "qwen/qwen-max",
  "max_tokens": 8192,
  "permission_mode": "auto",
  "verbose": false,
  "thinking": false,
  "qwen_api_key": "sk-..."
}
```

---

## 権限システム

| モード | 挙動 |
|---|---|
| `auto`（既定） | 読み取り専用操作は常に許可。Bash 実行や書き込み前に確認 |
| `accept-all` | 一切確認しない |
| `manual` | 読み取りを含むすべての操作の前に確認 |

`auto` モードで自動承認される代表例: `ls`, `cat`, `head`, `tail`, `wc`, `pwd`, `echo`, `git status`, `git log`, `git diff`, `git show`, `find`, `grep`, `rg`, `python`, `node`, `pip show`, `npm list` など。

---

## 組み込みツール

### コアツール

| ツール | 説明 | 主な引数 |
|---|---|---|
| `Read` | 行番号付きでファイルを読む | `file_path`, `limit`, `offset` |
| `Write` | ファイル作成 / 上書き（diff 表示） | `file_path`, `content` |
| `Edit` | 文字列の正確な置換（diff 表示） | `file_path`, `old_string`, `new_string`, `replace_all` |
| `Bash` | シェルコマンド実行 | `command`, `timeout` |
| `Glob` | glob パターンでファイル検索 | `pattern`, `path` |
| `Grep` | 正規表現検索 | `pattern`, `path`, `glob`, `output_mode` |
| `WebFetch` | URL からテキスト取得 | `url`, `prompt` |
| `WebSearch` | DuckDuckGo による検索 | `query` |

### メモリツール

| ツール | 説明 | 主な引数 |
|---|---|---|
| `MemorySave` | 永続メモリを保存 / 更新 | `name`, `type`, `description`, `content`, `scope` |
| `MemoryDelete` | 名前で削除 | `name`, `scope` |
| `MemorySearch` | キーワードまたは AI で検索 | `query`, `scope`, `use_ai`, `max_results` |
| `MemoryList` | 全メモリ一覧 | `scope` |

### サブエージェントツール

| ツール | 説明 | 主な引数 |
|---|---|---|
| `Agent` | サブエージェント生成 | `prompt`, `subagent_type`, `isolation`, `name`, `model`, `wait` |
| `SendMessage` | バックグラウンドエージェントにメッセージ送信 | `name`, `message` |
| `CheckAgentResult` | 状態 / 結果確認 | `task_id` |
| `ListAgentTasks` | アクティブ / 完了済みタスク一覧 | — |
| `ListAgentTypes` | エージェントタイプ一覧 | — |

### スキルツール

| ツール | 説明 | 主な引数 |
|---|---|---|
| `Skill` | 名前でスキル呼び出し | `name`, `args` |
| `SkillList` | スキル一覧表示 | — |

### MCP ツール

MCP ツールは設定済みサーバから自動検出され、`mcp__<server>__<tool>` の名前で登録されます。

| 例 | 説明 |
|---|---|
| `mcp__git__git_status` | `git` サーバの `git_status` |
| `mcp__filesystem__read_file` | `filesystem` サーバの `read_file` |
| `mcp__myserver__my_action` | カスタムサーバのツール |

---

## メモリ

モデルは組み込みメモリシステムによって会話をまたいで情報を記憶できます。

- **ユーザースコープ**: `~/.nano_claude/memory/`
- **プロジェクトスコープ**: `.nano_claude/memory/`

`MEMORY.md` インデックス（最大 200 行 / 25KB）は保存・削除のたびに自動再生成され、システムプロンプトに注入されます。

**メモリタイプ:**
- `user`
- `feedback`
- `project`
- `reference`

**メモリファイル例:**
```markdown
---
name: coding style
description: Python formatting preferences
type: feedback
created: 2026-04-02
---
Prefer 4-space indentation and full type hints in all Python code.
```

---

## スキル

スキルは再利用可能なプロンプトテンプレートです。組み込みスキル:
- `/commit`
- `/review [PR]`

カスタムスキル例:

```bash
mkdir -p ~/.nano_claude/skills
```

```markdown
---
name: deploy
description: Deploy to an environment
triggers: [/deploy]
allowed-tools: [Bash, Read]
when_to_use: Use when the user wants to deploy a version to an environment.
argument-hint: [env] [version]
arguments: [env, version]
context: inline
---

Deploy $VERSION to the $ENV environment.
Full args: $ARGUMENTS
```

---

## サブエージェント

組み込みタイプ:
- `general-purpose`
- `coder`
- `reviewer`
- `researcher`
- `tester`

バックグラウンド実行、git worktree 分離、カスタムエージェント定義に対応しています。

---

## MCP

`.mcp.json` または `~/.nano_claude/mcp.json` で設定します。

```json
{
  "mcpServers": {
    "git": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-server-git"]
    }
  }
}
```

代表的な REPL コマンド:
```text
/mcp
/mcp reload
/mcp add git uvx mcp-server-git
/mcp remove git
```

---

## プラグインシステム

```bash
/plugin install my-plugin@https://github.com/user/my-plugin
/plugin install local-plugin@/path/to/local/plugin
/plugin enable my-plugin
/plugin disable my-plugin
/plugin update my-plugin
/plugin uninstall my-plugin
/plugin info my-plugin
/plugin recommend
/plugin recommend "docker database"
```

`plugin.json` 例:
```json
{
  "name": "my-plugin",
  "version": "0.1.0",
  "description": "Does something useful",
  "author": "you",
  "tags": ["git", "python"],
  "tools": ["tools"],
  "skills": ["skills/my.md"],
  "mcp_servers": {},
  "dependencies": ["httpx"]
}
```

---

## AskUserQuestion ツール

Claude は処理を一時停止し、ユーザーに質問してから続行できます。

```json
{
  "tool": "AskUserQuestion",
  "question": "Which database should I use?",
  "options": [
    {"label": "SQLite", "description": "Simple, file-based"},
    {"label": "PostgreSQL", "description": "Full-featured, requires server"}
  ],
  "allow_freetext": true
}
```

---

## タスク管理

利用可能なツール:
- `TaskCreate`
- `TaskUpdate`
- `TaskGet`
- `TaskList`

状態:
`pending` → `in_progress` → `completed` / `cancelled` / `deleted`

REPL:
```text
/tasks
/tasks create <subject>
/tasks start <id>
/tasks done <id>
/tasks cancel <id>
/tasks delete <id>
/tasks get <id>
/tasks clear
```

---

## コンテキスト圧縮

2 層構成:
1. **Snip** — 古いツール出力を切り詰める
2. **Auto-compact** — 上限の 70% を超えると古い会話を AI が要約

---

## Diff 表示

```diff
--- a/config.py
+++ b/config.py
@@ -12,7 +12,7 @@
     "model": "claude-opus-4-6",
-    "max_tokens": 8192,
+    "max_tokens": 16384,
```

---

## CLAUDE.md サポート

`CLAUDE.md` を置くと、コードベースに関する永続的なコンテキストをモデルへ注入できます。

```text
~/.claude/CLAUDE.md
/your/project/CLAUDE.md
```

---

## セッション管理

```bash
/save
/save debug_auth_bug

/load
/load debug_auth_bug
/load session_20260401_143022.json
```

---

## プロジェクト構成

```text
nano_claude_code/
├── nano_claude.py
├── agent.py
├── providers.py
├── tools.py
├── tool_registry.py
├── compaction.py
├── context.py
├── config.py
├── multi_agent/
├── memory/
├── skill/
├── mcp/
└── tests/
```

---

## FAQ

**Q: MCP サーバの追加方法は？**
```text
/mcp add git uvx mcp-server-git
```

**Q: MCP サーバのエラーをどう調べる？**
```text
/mcp
/mcp reload git
which uvx
uvx mcp-server-git
```

**Q: 認証付き MCP サーバは使える？**
はい。HTTP/SSE の `headers` や stdio の `env` を使って設定できます。

**Q: Ollama モデルで tool call が動かない**
function calling 対応モデルを使用してください。推奨: `qwen2.5-coder`, `llama3.3`, `mistral`, `phi4`

**Q: API コストの確認方法**
```text
/cost
```

**Q: 複数 API キーを同一セッションで使える？**
はい。環境変数または `/config` で先に設定しておけば、モデル切替時に自動で対応するキーが使われます。

**Q: どこからでも CLI のように使うには？**
```bash
alias nc='python /path/to/nano_claude_code/nano_claude.py'
pip install -e .
```
