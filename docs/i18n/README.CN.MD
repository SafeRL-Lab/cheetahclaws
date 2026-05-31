English | [中文](README.CN.MD) | [Français](README.FR.MD) | [한국어](README.KO.MD) | [日本語](README.JP.MD) | [Deutsch](README.DE.MD) | [Português](README.ES.MD)

> ⚠️ 此翻译可能已过时。最新、最准确的内容请以英文 [README](../../README.md) 为准。

<br>

<div align="center">
  <a href="[https://github.com/SafeRL-Lab/Robust-Gymnasium](https://github.com/SafeRL-Lab/cheetahclaws)">
    <img src="../media/logos/logo-5.png" alt="Logo" width="280"> 
  </a>

  
<h2 align="center" style="font-size: 30px;"><strong><em>CheetahClaws (代码猎豹)</em></strong>：一个快速、易用、原生基于 Python 的任意模型个人 AI 助手，灵感来自 OpenClaw 与 Claude Code，旨在为你 24/7 自主工作</h2>
<p align="center">
    <a href="https://github.com/chauncygu/collection-claude-code-source-code">Claude Code 最新源码集合</a>
    ·
    <a href="https://github.com/SafeRL-Lab/cheetahclaws/issues">Issue</a>
  ·
    <a href="https://deepwiki.com/SafeRL-Lab/cheetahclaws">简要介绍</a>
  
  </p>
</div>

 <div align=center>
 <img src="../media/demos/demo.gif" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline">任务执行</center>
 </div>
 
 
---

  <div align=center>
 <img src="../media/demos/brainstorm_demo.gif" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline">头脑风暴模式：多智能体头脑风暴</center>
 </div>



---

  <div align=center>
 <img src="../media/demos/proactive_demo.gif" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline">主动模式：自主 Agent</center>
 </div>

---

  <div align=center>
 <img src="../media/demos/ssj_demo.gif" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline">SSJ 开发者模式：强力菜单工作流</center>
 </div>

---

  <div align=center>
 <img src="../media/demos/telegram_demo.gif" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline">Telegram 桥接：用手机控制 cheetahclaws</center>
 </div>

---


 
## 🔥🔥🔥 新闻（太平洋时间）


- 2026 年 5 月 5 日：**Telegram 桥 —— 文件双向传输 + 权限提示真正可点击的按钮(修复 #84)**。issue #84 的两个抱怨同一根因:`bridges/telegram.py` 只有 `_tg_send`(走 `sendMessage` 的纯文本路径)。一,机器人输出"文件已发送"但聊天里没有附件 —— 因为没有 `sendDocument` multipart 上传、没有入站 `document` 处理、agent 也没有"发文件"的代码路径,`--accept-all` 跟这个无关;二,权限提示里的 `[y/N/a(ccept-all)]` 看着像按钮但不能点 —— 因为轮询只监听 `message` 更新,从来没渲染过 inline keyboard。本次补丁:
  - **`_tg_send_document`** —— 手工拼装 `multipart/form-data`(`urllib` 的 JSON 路径无法承载二进制),49 MB 上限(Telegram 硬限 50 MB,留 1 MB 余量给编码开销)。六类失败模式都把具体原因回传到聊天:文件不存在、`stat` 失败、空文件、超大、网络异常、API `ok: false`(描述原文转发)。
  - **入站 `document` 处理** —— 通过 `getFile` 下载,把文件名清洗为 `[A-Za-z0-9._-]_` 防路径注入,Docker 场景写入 `/workspace`,否则写入 `tempfile.gettempdir()`,把保存路径回执到聊天,并向 agent 提交带路径的提示("I just uploaded a file at <path>. Please review it."—— 用户带 caption 时由 caption 覆盖)。
  - **`!sendfile <absolute_path>`** —— 用户显式按需发送,后台守护线程执行,不阻塞轮询循环。会去除路径两端的反引号/引号。
  - **`Write` 工具自动回传** —— `_bg_runner._on_tool_start` 记录 `Write` 调用的 `file_path`,`_on_tool_end` 在写入成功后自动把文件作为 document 发回(FIFO 配对,确保并发写入不串)。结果以 `Error:` / `Denied:` 开头时跳过。每个 turn 用 `_sent_files: set[str]` 去重,agent 重写同一文件不会重复发送。
  - **跨渠道的权限提示交互(`[approve][reject]` 现在每个端都是可点 / 可选的)** —— `ask_input_interactive(prompt, config, options=[(label, value), …])` 是新约定,`ask_permission_interactive` 传 `[("✅ Approve", "y"), ("❌ Reject", "n"), ("✅✅ Accept all", "a")]`,所有渠道都会渲染交互式选择器,UX 与媒介对齐:
    - **Telegram** —— 真正的 `inline_keyboard` 按钮。`callback_data` 是 `cc:<prompt_id>:<value>`,`prompt_id` 是 8 位短随机串;`_tg_poll_loop` 的 `allowed_updates` 扩到 `["message", "callback_query"]`,新增的 `_handle_callback_query(token, chat_id, cb, session_ctx)` 负责鉴权(chat_id 匹配)、`answerCallbackQuery` 清掉点击转圈、`prompt_id` 校验(过期 prompt 上的旧按钮点击被静默丢弃,两个连续的权限提示不会串)、`editMessageText` 在原消息后追加 `✓ Selected: <value>` 留下视觉痕迹,最后触发 `tg_input_event` 解锁阻塞的 agent 线程。Markdown 解析失败退到无 `parse_mode` 的同样按钮,再失败退到纯文本 —— prompt body 里嵌入的 menu block 让该路径仍然可用。
    - **Slack / WeChat** —— 在消息正文里渲染编号菜单(Slack 头是 `❓ Input Required`,WeChat 是 `❓ 需要输入`)。每行格式 `[1] ✅ Approve  (reply 1 or y)`,用户回复数字、规范字母、或者 label 词(`approve` / `reject` / `accept` / `all`),三种形式服务端都归一化为规范值。
    - **终端** —— 同样的编号菜单在输入光标上方打印,同样支持数字 / 字母 / label 词三种回复。
    - **Web (chat API)** —— 不动,沿用浏览器既有的 approval UI。
    跨渠道的归一化由 `tools/interaction.py` 三个纯函数完成:`_format_menu_block(options)`(渲染编号菜单文本)、`_build_value_map(options)`(数字 + 规范值 + label 词的查找表,冲突时 first-write-wins)、`_resolve_choice(raw, value_map)`(去空白 / 大小写不敏感 / 未知输入直通,这样 callers 用 `options=` 同时也想接收自由文本时仍然 work)。完全向后兼容:既有所有 `ask_input_interactive` 调用方(`/checkpoint`、`/session`、`/agent`、`/config`、`/voice` 等)都没传 `options=`,行为零变化。`RuntimeContext` 新字段:`tg_callback_prompt_id: str`、`tg_callback_message_id: int`。
  - **测试** —— 新增 49 个 pytest 用例。`tests/test_telegram_bridge.py`(27 个):`urllib.request.urlopen` 与 `_tg_api` 全程 mock,`threading.Thread` 替换为同步 stub,端到端由工作线程跑 `ask_input_interactive(options=…)`、主线程通过 `_handle_callback_query` 模拟点击、断言工作线程返回点击值。覆盖文本切片+Markdown 回退、multipart body 字段断言、`_tg_send_document` 六条失败路径、`_bg_runner` Write 自动发送四种变体、`_tg_send_keyboard` 四条路径、`_handle_callback_query` 五条路径、端到端点击两种值。`tests/test_options_menu.py`(22 个):三个纯函数(emoji-stripped label tokens、大小写不敏感、空白处理、防御性非字符串、冲突 first-write-wins)+ Slack(4 种回复)/ WeChat(2 种回复)/ 终端(数字 / label / 规范值 / 不传 options 的回归)端到端。完整套件 **718 passed in 43s**,既有 669 个测试零回归。

- 2026 年 4 月 7 日（**v3.05.53**）：**Telegram 图片/语音支持、Bash 超时时进程树终止、Windows shell 提示、worker 修复**
  - **Telegram 图片视觉支持** —— 向 Telegram 桥发送图片后，CheetahClaws 会使用当前激活的视觉模型（GPT-4o、Gemini 2.0 Flash、Claude 等）对图片进行描述。机器人会下载最高分辨率版本，将其编码为 Base64，并通过与 `/img` 相同的 `_pending_image` 路径进行处理。图片标题文本（或默认的 “describe this image” 提示）会与图片一起转发。
  - **Telegram 语音/音频 STT** —— 向 Telegram 桥发送语音消息或音频文件时，CheetahClaws 会自动转写。OGG 语音会通过 `ffmpeg` 转为 PCM 并传给本地 Whisper 后端；如果 `ffmpeg` 不可用，则回退到 OpenAI Whisper API。转写结果会先回显到聊天中，再作为查询提交。
  - **Bash 超时时终止整个进程树** —— 当 `Bash` 命令超时时，CheetahClaws 现在会终止整个子进程树，而不是只杀掉 shell 本身。在 Unix 上，`os.killpg` 会向进程组发送 `SIGKILL`；在 Windows 上，`taskkill /F /T` 会终止所有子进程。Agent 启动的 GUI 应用（如 PyQt 游戏）在超时后也不会再留下僵尸进程。内部实现为保证线程安全，使用 `start_new_session=True` 代替了 `preexec_fn=os.setsid`。
  - **Worker 默认运行所有待处理任务** —— `/worker` 之前每个会话只处理 1 个任务（一个 bug）。现在默认会处理所有待完成任务。`--workers N` 标志仍然可以在需要时限制批量大小。
  - **系统提示中加入 Windows shell 提示** —— 非 Claude 模型现在会在系统提示中获得一份 Windows 专用 shell 速查表（如 `type` vs `cat`、`dir /s /b` vs `find`、`del` vs `rm` 等），从而无需手动指导也能在 Windows 上生成正确命令。
  - **Bash 超时提示** —— `Bash` 工具说明现在建议模型对较慢命令（`npm install`、`npx`、`pip install`、构建等）使用 `timeout=120–300`，从而减少包安装类操作上误触发的 30 秒超时。
  - **Bug 修复：后台事件提示显示真实 cwd** —— 后台事件完成后打印的黄色重新提示之前被硬编码为 `[claude-code-local]`；现在会显示真实的工作目录名（`[{cwd.name}]`），与主 REPL 提示保持一致。

- 2026 年 4 月 6 日（**v3.05.53**）：**Telegram 交互菜单、`/img` 别名、`/voice device`、OpenAI/Gemini 视觉支持**
  - **修复 Telegram 交互菜单** —— 带交互输入的斜杠命令（如 `/ollama`、`/permission`、`/checkpoint`）之前会阻塞 Telegram 轮询循环，导致无法回复菜单提示。现在斜杠命令会像普通查询一样在守护线程中运行，从而保持轮询循环空闲。所有交互菜单（`ask_input_interactive`）都可以在 Telegram 上正常工作。
  - **`/img` 别名** —— `/img` 现在是 `/image` 的别名，便于更快的剪贴板图片工作流。
  - **`/voice device`** —— 新子命令，用于列出所有可用输入麦克风并以交互方式选择一个。选中的设备索引会持久化到会话配置中，并显示在 `/voice status` 中。对有多个音频接口的系统（如 USB 耳机 + 内置麦克风）很有用。
  - **OpenAI / Gemini 模型的视觉支持** —— `/img`（以及 `/image`）现在除了已有的 Ollama 原生格式外，也会将图片以 OpenAI multipart `image_url` 格式发送给云端视觉模型（GPT-4o、Gemini 2.0 Flash 等）。无需改配置——系统会根据当前 provider 自动选择正确格式。
  - **Bug 修复：线程竞态条件** —— `_in_telegram_turn` 现在通过每个斜杠执行线程的 `threading.local()` 跟踪，而不是共享配置键，从而消除了当常规消息在交互式斜杠命令等待输入时到达所导致的竞态问题。

- 2026 年 4 月 6 日（**v3.05.52**）：**Checkpoint 系统、计划模式、compact 与实用命令、支持 MiniMax 模型、修复 Telegram bug** 
  - **Checkpoint 系统**（`checkpoint/` 包）：每轮后自动快照对话状态和文件变更。`/checkpoint` 列出所有快照；`/checkpoint <id>` 可将文件与对话历史同时回滚到任意之前状态；`/checkpoint clear` 删除当前会话的所有快照。`/rewind` 是别名。使用 100 个快照的滑动窗口；会话启动时捕获初始快照。节流策略：若无任何变化则跳过。文件备份使用 copy-on-write；快照捕获的是编辑后的状态。
  - **计划模式**：`/plan <desc>` 进入只读分析模式 —— Claude 只能读取代码库，并只允许写入专用计划文件（`.nano_claude/plans/<session_id>.md`）。其他所有写操作都会被静默阻止并显示友好提示。`/plan` 显示当前计划；`/plan done` 退出计划模式并恢复原权限；`/plan status` 显示计划模式是否激活。两个新的 Agent 工具 —— `EnterPlanMode` 与 `ExitPlanMode` —— 允许 Claude 在复杂多文件任务中自主进入和退出计划模式；在所有权限模式下都会自动批准。
  - **`/compact [focus]`**：可随时手动触发对话压缩。可选的 focus 字符串可引导 LLM 总结器保留哪些上下文。自动压缩与手动压缩都会在压缩后恢复计划文件上下文。
  - **实用命令**：`/init` 会在当前目录创建 `CLAUDE.md` 模板；`/export [filename]` 可将对话导出为 Markdown（默认）或 JSON；`/copy` 可将上一条 assistant 回复复制到剪贴板（Windows/macOS/Linux）；`/status` 显示版本、模型、provider、权限、session ID、token 使用量与上下文占比；`/doctor` 用于诊断安装健康状况（Python 版本、git、API key 与实时连接测试、可选依赖、CLAUDE.md 是否存在、checkpoint 磁盘占用、权限模式）。

- 2026 年 4 月 6 日（**v3.05.51**）：**项目从 Nano Claude Code 重命名为 CheetahClaws**
  - 项目已从 **Nano Claude Code** 更名为 **CheetahClaws** —— 一个更有辨识度的名字，更能体现这个工具的气质：一个锐利、敏捷的编程助手。CheetahClaws 中的 `Cl` 也暗含了对 Claude 的致意。
  - CLI 命令：`nano_claude` → `cheetahclaws`
  - PyPI 包：`nano-claude-code` → `cheetahclaws`
  - 配置目录：`~/.nano_claude/` → `~/.clawnest/` → `~/.cheetahclaws/`
  - 主入口：`nano_claude.py` → `cheetahclaws.py`
  - 所有文档、GitHub URL 和内部引用均已同步更新。
  - README 中新增 **CheetahClaws vs OpenClaw** 对比章节。

更多更新见 [这里](../news.md)


---

# CheetahClaws

CheetahClaws：**一个轻量级**、**易于使用**、支持**任意模型**的 Claude Code Python 重实现，例如 Claude、GPT、Gemini、Kimi、Qwen、Zhipu、DeepSeek、MiniMax，以及通过 Ollama 或任意 OpenAI 兼容端点接入的本地开源模型。

---

## 目录
  * [为什么选择 CheetahClaws](#为什么选择-cheetahclaws)
  * [CheetahClaws vs OpenClaw](#cheetahclaws-vs-openclaw)
  * [功能特性](#功能特性)
  * [支持的模型](#支持的模型)
  * [安装](#安装)
  * [使用：闭源 API 模型](#使用闭源-api-模型)
  * [使用：开源模型（本地）](#使用开源模型本地)
  * [模型名格式](#模型名格式)
  * [CLI 参考](#cli-参考)
  * [斜杠命令（REPL）](#斜杠命令repl)
  * [配置 API Key](#配置-api-key)
  * [权限系统](#权限系统)
  * [内置工具](#内置工具)
  * [记忆](#记忆)
  * [Skills](#skills)
  * [子 Agent](#子-agent)
  * [MCP（模型上下文协议）](#mcp模型上下文协议)
  * [插件系统](#插件系统)
  * [AskUserQuestion 工具](#askuserquestion-工具)
  * [任务管理](#任务管理)
  * [语音输入](#语音输入)
  * [头脑风暴](#头脑风暴)
  * [SSJ 开发者模式](#ssj-开发者模式)
  * [Telegram 桥接](#telegram-桥接)
  * [主动后台监控](#主动后台监控)
  * [Checkpoint 系统](#checkpoint-系统)
  * [计划模式](#计划模式)
  * [上下文压缩](#上下文压缩)
  * [差异视图](#差异视图)
  * [CLAUDE.md 支持](#claudemd-支持)
  * [会话管理](#会话管理)
  * [云同步（GitHub Gist）](#云同步github-gist)
  * [项目结构](#项目结构)
  * [FAQ](#faq)




## 为什么选择 CheetahClaws

Claude Code 是一个强大、面向生产环境的 AI 编程助手 —— 但它的源码是一个编译后的 12 MB TypeScript/Node.js bundle（约 1,300 个文件，约 283K 行代码）。它与 Anthropic API 紧密耦合，难以修改，也无法运行在本地模型或其他替代模型之上。

**CheetahClaws** 用约 10K 行可读的 Python 代码重实现了相同的核心循环，保留你需要的部分，去掉你不需要的部分。更详细分析见此（CheetahClaws v3.03），[英文版](../comparison_claude_code_vs_nano_v3.03_en.md) 与 [中文版](../comparison_claude_code_vs_nano_v3.03_cn.md)

### 快速概览

| 维度 | Claude Code（TypeScript） | CheetahClaws（Python） |
|-----------|--------------------------|---------------------------|
| 语言 | TypeScript + React/Ink | Python 3.8+ |
| 源文件数 | ~1,332 个 TS/TSX 文件 | 51 个 Python 文件 |
| 代码行数 | ~283K | ~12K |
| 内置工具 | 44+ | 27 |
| 斜杠命令 | 88 | 36 |
| 语音输入 | 专有 Anthropic WebSocket（需要 OAuth） | 本地 Whisper / OpenAI API —— 可离线使用，无需订阅 |
| 模型提供方 | 仅 Anthropic | 8+（Anthropic · OpenAI · Gemini · Kimi · Qwen · DeepSeek · MiniMax · Ollama · …） |
| 本地模型 | 不支持 | 支持 —— Ollama、LM Studio、vLLM、任何 OpenAI-compatible endpoint |
| 是否需要构建步骤 | 是（Bun + esbuild） | 否 —— 可直接 `python cheetahclaws.py` 运行（或安装后使用 `cheetahclaws`） |
| 运行时可扩展性 | 封闭（编译时） | 开放 —— 可在运行时 `register_tool()`，支持 Markdown skills、git 插件 |
| 任务依赖图 | 否 | 是 —— `task/` 包支持 `blocks` / `blocked_by` 边 |

### Claude Code 更强的地方

- **UI 质量** —— React/Ink 组件树、流式渲染、细粒度 diff 可视化、对话框系统。
- **工具广度** —— 44 个工具，包括 `RemoteTrigger`、`EnterWorktree` 以及更多深度 UI 集成工具。
- **企业功能** —— MDM 管理配置、团队权限同步、OAuth、keychain 存储、GrowthBook feature flags。
- **AI 驱动的记忆抽取** —— `extractMemories` 服务可主动从对话中提取知识，而不需要显式工具调用。
- **生产级可靠性** —— 单一可分发的 `cli.js`、全面测试覆盖、版本锁定发布。

### CheetahClaws 更强的地方

- **多 Provider** —— 用 `--model` 或 `/model` 即可在 Claude、GPT-4o、Gemini 2.5 Pro、DeepSeek、Qwen、MiniMax 或本地 Llama 模型之间切换 —— 无需重新编译。
- **本地模型支持** —— 可通过 Ollama、LM Studio 或任意 vLLM 托管模型实现完全离线运行。
- **源码可读** —— 完整 agent 循环只有 174 行（`agent.py`）。任何 Python 开发者都可以在几分钟内读懂、fork 和扩展。
- **零构建** —— `pip install -r requirements.txt` 后即可运行。修改立即生效。
- **动态扩展** —— 可通过 `register_tool(ToolDef(...))` 在运行时注册新工具、从 git URL 安装 skill 包，或接入任意 MCP server。
- **任务依赖图** —— `TaskCreate` / `TaskUpdate` 支持 `blocks` / `blocked_by` 边，用于结构化多步规划（Claude Code 不支持）。
- **双层上下文压缩** —— 基于规则的裁剪 + AI 总结，可通过 `preserve_last_n_turns` 配置。
- **Notebook 编辑** —— `NotebookEdit` 可直接操作 `.ipynb` JSON（替换/插入/删除 cell），无需 kernel。
- **无需 LSP server 的诊断** —— `GetDiagnostics` 将 pyright → mypy → flake8 → py_compile 串联用于 Python，并对其他语言使用 tsc/shellcheck，零配置。
- **离线语音输入** —— `/voice` 通过 `sounddevice`/`arecord`/SoX 录音，使用本地 `faster-whisper` 转写（无需 API key、无需订阅），并自动提交。来自 git branch 和项目文件的关键词会提升编码术语识别准确率。
- **云端会话同步** —— `/cloudsave` 可将对话备份到私有 GitHub Gist，无需额外依赖；可在任意机器上通过 `/cloudsave load <id>` 恢复。
- **SSJ 开发者模式** —— `/ssj` 打开一个持久化强力菜单，包含 10 个工作流快捷入口：Brainstorm → TODO → Worker 管道、专家辩论、代码审查、README 生成、commit 助手等。菜单在各动作间保持打开；支持 `/command` 透传。
- **Telegram Bot 桥** —— `/telegram <token> <chat_id>` 可将 cheetahclaws 变成 Telegram 机器人：从手机接收用户消息、运行模型、返回响应。斜杠命令可透传，并带 typing indicator。
- **Worker 命令** —— `/worker` 会自动读取 `brainstorm_outputs/todo_list.txt` 中的待办任务并执行，完成后将其标记为 done，也支持通过编号选择任务（如 `1,4,6`）。
- **强制退出** —— 2 秒内按 3 次 Ctrl+C 会立即触发 `os._exit(1)`，可强制中断卡住的 I/O。
- **主动后台监控** —— `/proactive 5m` 会启动一个 sentinel 守护线程，在一段时间无交互后自动唤醒 agent，从而支持连续监控循环、定时检查、交易机器人等，无需用户再发提示。
- **Rich 实时流式渲染** —— 安装 `rich` 后，回复会以实时更新的 Markdown 形式原地流式显示，不会重复打印原始文本，并能整洁地交织 tool 调用输出。
- **原生 Ollama reasoning** —— 本地 reasoning 模型（deepseek-r1、qwen3、gemma4）可通过 `ThinkingChunk` 事件将 `<think>` token 直接流式输出到终端；通过 `/verbose` 与 `/thinking` 启用。
- **原生 Ollama vision** —— `/image [prompt]` 可从剪贴板捕获图片，并通过 Ollama 原生图像 API 发给本地视觉模型（llava、gemma4、llama3.2-vision）。无需云端。
- **可靠的多行粘贴** —— Bracketed Paste Mode（`ESC[?2004h`）会将任意粘贴文本 —— 代码块、多段提示、长 diff —— 收集为单次 turn，零延迟且无空行伪影。
- **丰富的 Tab 补全** —— `/` 后按 Tab 可显示全部命令及一行说明和子命令提示；子命令 Tab 补全支持 `/mcp`、`/plugin`、`/tasks`、`/cloudsave` 等。
- **Checkpoint 与回滚** —— `/checkpoint` 可列出所有自动快照（对话 + 文件状态）；`/checkpoint <id>` 可将文件与历史一起回滚到任意更早时间点。
- **计划模式** —— `/plan <desc>`（或 `EnterPlanMode` 工具）会让 Claude 进入结构化只读分析阶段；只有计划文件可写。Claude 写出详细计划后，使用 `/plan done` 恢复完整写权限并开始实现。

---

## CheetahClaws vs OpenClaw

[OpenClaw](https://github.com/openclaw/openclaw) 是另一个流行的开源 AI 助手，基于 TypeScript/Node.js。两个项目的**主要目标不同** —— 下面是它们的对比。

### 快速概览

| 维度 | OpenClaw（TypeScript） | CheetahClaws（Python） |
|-----------|----------------------|---------------------|
| 语言 | TypeScript + Node.js | Python 3.8+ |
| 源文件数 | ~10,349 个 TS/JS 文件 | 51 个 Python 文件 |
| 代码行数 | ~245K | ~12K |
| 主要定位 | 跨消息渠道的个人生活助手 | AI **编程** 助手 / 开发者工具 |
| 架构 | 常驻 Gateway 守护进程 + 配套应用 | 零安装终端 REPL |
| 消息渠道 | 20+（WhatsApp · Telegram · Slack · Discord · Signal · iMessage · Matrix · WeChat · …） | 终端 + 可选 Telegram 桥 |
| 模型提供方 | 多个（偏云端） | 7+，且完整支持本地模型（Ollama · vLLM · LM Studio · …） |
| 本地 / 离线模型 | 有限 | 完整支持 —— Ollama、vLLM、任意 OpenAI-compatible endpoint |
| 语音 | 唤醒词 · PTT · Talk Mode（macOS/iOS/Android） | 离线 Whisper STT（本地，无 API key） |
| 代码编辑工具 | 浏览器控制、Canvas 工作区 | Read · Write · Edit · Bash · Glob · Grep · NotebookEdit · GetDiagnostics |
| 是否需要构建步骤 | 是（`pnpm install` + daemon setup） | 否 —— `pip install` 后直接运行 |
| 移动端伴侣 | macOS 菜单栏 + iOS/Android 应用 | — |
| 实时 Canvas / UI | 支持（A2UI agent-driven visual workspace） | — |
| MCP 支持 | — | 支持（stdio/SSE/HTTP） |
| 运行时扩展性 | Skills 平台（bundled/managed/workspace） | 运行时 `register_tool()`、MCP、git 插件、Markdown skills |
| 可 hack 性 | 代码库很大（245K 行），不易修改 | ~12K 行 —— 完整 agent 循环一眼可见 |

### OpenClaw 更强的地方

- **全渠道收件箱** —— 可连接 20+ 消息平台（WhatsApp、Signal、iMessage、Discord、Teams、Matrix、WeChat…）；用户可在自己原本就在用的平台上直接交互。
- **常驻后台守护进程** —— Gateway 作为后台服务运行（launchd/systemd）；日常使用不需要打开终端。
- **移动优先** —— macOS 菜单栏、iOS 语音唤醒 / Talk Mode、Android 摄像头/录屏 —— 更像原生应用，而不是 CLI 工具。
- **实时 Canvas** —— 浏览器中的 agent-driven visual workspace；支持 A2UI push/eval/snapshot。
- **浏览器自动化** —— 带专用 Chrome/Chromium 配置，可执行 snapshot、actions、upload 工具。
- **生产级可靠性** —— 版本化 npm 发布、完善 CI、onboarding wizard、`openclaw doctor` 诊断。

### CheetahClaws 更强的地方

- **编程工具集** —— Read/Write/Edit/Bash/Glob/Grep/NotebookEdit/GetDiagnostics 专为软件开发而设计；CheetahClaws 能理解 diff、文件树和代码结构。
- **真正的本地模型支持** —— 完整支持 Ollama/vLLM/LM Studio，包含流式输出、tool-calling 与 vision —— 无需云端。
- **8+ 模型提供方** —— 只需一个 `--model` 参数即可在 Claude、GPT-4o、Gemini、DeepSeek、Qwen、MiniMax 以及本地模型间切换。
- **几分钟内即可上手 hack** —— 12K 行可读 Python；完整 agent 循环位于 `agent.py`；无需重建即可通过运行时 `register_tool()` 扩展。
- **零配置** —— `pip install cheetahclaws` 后直接运行 `cheetahclaws`；无需 daemon、无需设备配对、无需 onboarding wizard。
- **MCP 支持** —— 可连接任意 MCP server（stdio/SSE/HTTP）；工具会自动注册。
- **SSJ 开发者模式** —— `/ssj` 强力菜单可在持久交互会话中串联 Brainstorm → TODO → Worker → Debate，自动化整个开发流程。
- **离线语音** —— `/voice` 使用 `faster-whisper` 本地转写；无需订阅、无需 OAuth、无网络也可用。
- **会话云同步** —— `/cloudsave` 可将完整对话备份到私有 GitHub Gist，且无需任何额外依赖。

### 何时该选哪个

| 如果你想要… | 选择 |
|---|---|
| 一个可以通过 WhatsApp/Signal/Discord 与之对话的个人助手 | **OpenClaw** |
| 一个在终端中的 AI 编程助手 | **CheetahClaws** |
| 完整离线 / 本地模型支持 | **CheetahClaws** |
| 适合移动端、始终在线的体验 | **OpenClaw** |
| 能在一个下午读懂并改动源码 | **CheetahClaws** |
| 浏览器自动化与可视化 Canvas | **OpenClaw** |
| 无需重建即可切换多种 LLM provider | **CheetahClaws** |

---

### 核心设计差异

**Agent 循环** —— CheetahClaws 使用一个 Python generator，`yield` 出带类型的事件（`TextChunk`、`ToolStart`、`ToolEnd`、`TurnDone`）。整个循环都在一个文件中可见，便于添加 hooks、自定义渲染器或日志。

**工具注册** —— 每个工具都是一个 `ToolDef(name, schema, func, read_only, concurrent_safe)` dataclass。任意模块都可以在导入时调用 `register_tool()`；MCP servers、plugins 和 skills 都使用相同机制。

**上下文压缩**

| | Claude Code | CheetahClaws |
|-|-------------|-----------------|
| 触发方式 | 精确 token 计数 | `len / 3.5` 估算，在 70% 时触发 |
| 第 1 层 | — | Snip：截断旧工具输出（无 API 成本） |
| 第 2 层 | AI 总结 | 对较早 turn 进行 AI 总结 |
| 控制方式 | 系统管理 | `preserve_last_n_turns` 参数 |

**记忆** —— Claude Code 的 `extractMemories` 服务会让模型主动浮现事实；CheetahClaws 的 `memory/` 包是工具驱动的：模型必须显式调用 `MemorySave`，这更可预测、更可审计。每条记忆都带有 `confidence`、`source`、`last_used_at`、`conflict_group` 元数据；搜索时会按 confidence × recency 重新排序；`/memory consolidate` 则提供手动整合流程，不会在后台静默篡改记忆。

### 谁适合使用 CheetahClaws

- 想把**本地模型或非 Anthropic 模型**用作编程助手的开发者。
- 研究**agentic coding assistant 如何工作**的研究者 —— 整个系统一屏就能看完。
- 需要一个**可 hack 的基线系统**来加入专有工具、自定义权限策略或专门化 agent 类型的团队。
- 想要 Claude Code 风格生产力、但**不想碰 Node.js 构建链**的任何人。

---

## 功能特性

| 功能 | 说明 |
|---|---|
| 多 Provider | Anthropic · OpenAI · Gemini · Kimi · Qwen · Zhipu · DeepSeek · MiniMax · Ollama · LM Studio · 自定义端点 |
| 交互式 REPL | readline 历史、Tab 补全斜杠命令（带描述与子命令提示）；Bracketed Paste Mode 支持可靠多行粘贴 |
| Agent 循环 | 流式 API + 自动工具调用循环 |
| 27 个内置工具 | Read · Write · Edit · Bash · Glob · Grep · WebFetch · WebSearch · **NotebookEdit** · **GetDiagnostics** · MemorySave · MemoryDelete · MemorySearch · MemoryList · Agent · SendMessage · CheckAgentResult · ListAgentTasks · ListAgentTypes · Skill · SkillList · AskUserQuestion · TaskCreate/Update/Get/List · **SleepTimer** · **EnterPlanMode** · **ExitPlanMode** · *(MCP + plugin 工具启动时自动加入)* |
| MCP 集成 | 可连接任意 MCP server（stdio/SSE/HTTP），工具自动注册并可由 Claude 调用 |
| 插件系统 | 可从 git URL 或本地路径安装/卸载/启用/禁用/更新插件；支持用户级/项目级多作用域与推荐引擎 |
| AskUserQuestion | Claude 可在执行中途暂停并向用户提问，支持可选编号选项 |
| 任务管理 | TaskCreate/Update/Get/List 工具；顺序 ID；依赖边；元数据；持久化到 `.cheetahclaws/tasks.json`；`/tasks` REPL 命令 |
| 差异视图 | 对 Edit 和 Write 显示 git 风格红/绿 diff |
| 上下文压缩 | 自动压缩长对话，以保持在模型上下文限制内 |
| 持久记忆 | 双作用域记忆（用户 + 项目），支持 4 种类型、confidence/source 元数据、冲突检测、基于 recency 的搜索、`last_used_at` 跟踪，以及用于自动提取的 `/memory consolidate` |
| 多智能体 | 可创建带类型的子 agent（coder/reviewer/researcher/…）、支持 git worktree 隔离与后台模式 |
| Skills | 内置 `/commit` · `/review`，并支持自定义 markdown skills、参数替换与 fork/inline 执行 |
| Plugin 工具 | 通过 `tool_registry.py` 注册自定义工具 |
| 权限系统 | `auto` / `accept-all` / `manual` / `plan` 模式 |
| Checkpoints | 每轮自动快照对话 + 文件状态；`/checkpoint` 列表、`/checkpoint <id>` 回滚；`/rewind` 别名；100 快照滑动窗口 |
| 计划模式 | `/plan <desc>` 进入只读分析模式；Claude 仅写计划文件；`EnterPlanMode` / `ExitPlanMode` agent 工具支持自主规划 |
| 37 个斜杠命令 | `/model` · `/config` · `/save` · `/cost` · `/memory` · `/skills` · `/agents` · `/voice` · `/proactive` · `/checkpoint` · `/plan` · `/compact` · `/status` · `/doctor` · `/theme` · … |
| 控制台主题 | `/theme` 列出 15 套精选配色（default · dracula · nord · gruvbox · solarized · tokyo-night · catppuccin · matrix · synthwave · midnight · ocean · monokai · cheetah · mono · none），每行显示该主题自身的 `info / ok / warn / err` 4 色实时预览。`/theme <name>` 应用并写入 config，下次启动自动恢复，同时驱动 Rich Markdown 代码块的语法高亮风格。 |
| 语音输入 | 录音 → 转写 → 自动提交。后端：`sounddevice` / `arecord` / SoX + `faster-whisper` / `openai-whisper` / OpenAI API。可完全离线。 |
| Brainstorm | `/brainstorm [topic]` 会根据主题生成 N 个专家 persona（2–100，默认 5，交互选择），运行迭代式辩论，将结果保存到 `brainstorm_outputs/`，并综合成 Master Plan + 自动生成 `brainstorm_outputs/todo_list.txt`。 |
| SSJ 开发者模式 | `/ssj` 打开持久交互强力菜单，提供 10 个快捷入口：Brainstorm、TODO 查看、Worker、Expert Debate、Propose、Review、Readme、Commit、Scan、Promote。动作后菜单持续保留；支持 `/command` 透传。辩论中显示逐轮动画 spinner，并将结果保存在被讨论文件旁边。 |
| Worker | `/worker [task#s]` 读取 `brainstorm_outputs/todo_list.txt`，用专门的模型提示逐个实现待办任务，并将它们标记为完成（`- [x]`）。支持任务选择（`/worker 1,4,6`）、自定义路径（`--path`）和 worker 数量上限（`--workers`）。还能检测误传的 brainstorm `.md` 路径并自动重定向。 |
| Telegram 桥接 | `/telegram <token> <chat_id>` 启动 bot 桥：接收 Telegram 消息、运行模型、回复 —— 全都可从手机完成。支持 typing indicator、斜杠命令透传（包括交互菜单），若已配置还会在启动时自动运行。 |
| 视觉输入 | `/image`（或 `/img`）从剪贴板获取图片并发给任意支持视觉的模型 —— Ollama（`llava`、`gemma4`、`llama3.2-vision`）使用原生格式，云模型（GPT-4o、Gemini 2.0 Flash、…）使用 OpenAI `image_url` multipart 格式。需要 `pip install cheetahclaws[vision]`；Linux 还需要 `xclip`。 |
| 主动监控 | `/proactive [duration]` 启动后台 sentinel daemon；在无交互一段时间后自动唤醒 agent，实现无需用户提示的持续监控循环 |
| 强制退出 | 2 秒内按 3 次 Ctrl+C 会触发 `os._exit(1)` —— 无论 I/O 是否阻塞都立即结束进程 |
| Rich Live 流式输出 | 安装 `rich` 后，回复会以实时更新的 Markdown 原地显示。在 SSH 会话中会自动关闭以避免重复输出；可用 `/config rich_live=false` 覆盖。 |
| 上下文注入 | 自动加载 `CLAUDE.md`、git 状态、cwd、持久记忆 |
| 会话持久化 | 退出时自动保存到 `daily/YYYY-MM-DD/`（每日上限）+ `history.json`（主历史，所有会话）+ `session_latest.json`（`/resume`）；会话包含 `session_id` 与 `saved_at` 元数据；`/load` 按日期分组 |
| 云同步 | `/cloudsave` 将会话同步到私有 GitHub Gist；退出时自动同步；可通过 Gist ID 从云端加载。无需新依赖（标准库 `urllib`）。 |
| Extended Thinking | Claude 模型可切换开启/关闭；本地 Ollama reasoning 模型（deepseek-r1、qwen3、gemma4）支持原生 `<think>` 流输出 |
| 成本跟踪 | Token 使用量 + 预估美元成本 |
| 非交互模式 | `--print` 标志便于脚本 / CI 使用 |

---

## 支持的模型

### 闭源模型（API）

| Provider | 模型 | 上下文 | 优势 | API Key 环境变量 |
|---|---|---|---|---|
| **Anthropic** | `claude-opus-4-6` | 200k | 最强能力，适合复杂推理 | `ANTHROPIC_API_KEY` |
| **Anthropic** | `claude-sonnet-4-6` | 200k | 速度与质量平衡 | `ANTHROPIC_API_KEY` |
| **Anthropic** | `claude-haiku-4-5-20251001` | 200k | 快速、性价比高 | `ANTHROPIC_API_KEY` |
| **OpenAI** | `gpt-4o` | 128k | 强多模态与编码能力 | `OPENAI_API_KEY` |
| **OpenAI** | `gpt-4o-mini` | 128k | 快、便宜 | `OPENAI_API_KEY` |
| **OpenAI** | `o3-mini` | 200k | 强推理 | `OPENAI_API_KEY` |
| **OpenAI** | `o1` | 200k | 高级推理 | `OPENAI_API_KEY` |
| **Google** | `gemini-2.5-pro-preview-03-25` | 1M | 长上下文、多模态 | `GEMINI_API_KEY` |
| **Google** | `gemini-2.0-flash` | 1M | 快速、大上下文 | `GEMINI_API_KEY` |
| **Google** | `gemini-1.5-pro` | 2M | 最大上下文窗口 | `GEMINI_API_KEY` |
| **Moonshot (Kimi)** | `moonshot-v1-8k` | 8k | 中英双语 | `MOONSHOT_API_KEY` |
| **Moonshot (Kimi)** | `moonshot-v1-32k` | 32k | 中英双语 | `MOONSHOT_API_KEY` |
| **Moonshot (Kimi)** | `moonshot-v1-128k` | 128k | 长上下文 | `MOONSHOT_API_KEY` |
| **Alibaba (Qwen)** | `qwen-max` | 32k | 最佳 Qwen 质量 | `DASHSCOPE_API_KEY` |
| **Alibaba (Qwen)** | `qwen-plus` | 128k | 平衡型 | `DASHSCOPE_API_KEY` |
| **Alibaba (Qwen)** | `qwen-turbo` | 1M | 快、便宜 | `DASHSCOPE_API_KEY` |
| **Alibaba (Qwen)** | `qwq-32b` | 32k | 强推理 | `DASHSCOPE_API_KEY` |
| **Zhipu (GLM)** | `glm-4-plus` | 128k | 最佳 GLM 质量 | `ZHIPU_API_KEY` |
| **Zhipu (GLM)** | `glm-4` | 128k | 通用型 | `ZHIPU_API_KEY` |
| **Zhipu (GLM)** | `glm-4-flash` | 128k | 有免费层 | `ZHIPU_API_KEY` |
| **DeepSeek** | `deepseek-chat` | 64k | 强编码能力 | `DEEPSEEK_API_KEY` |
| **DeepSeek** | `deepseek-reasoner` | 64k | 链式推理 | `DEEPSEEK_API_KEY` |
| **MiniMax** | `MiniMax-Text-01` | 1M | 长上下文、强推理 | `MINIMAX_API_KEY` |
| **MiniMax** | `MiniMax-VL-01` | 1M | 视觉 + 语言 | `MINIMAX_API_KEY` |
| **MiniMax** | `abab6.5s-chat` | 256k | 快速、性价比高 | `MINIMAX_API_KEY` |
| **MiniMax** | `abab6.5-chat` | 256k | 质量平衡 | `MINIMAX_API_KEY` |

### 开源模型（通过 Ollama 本地运行）

| 模型 | 尺寸 | 优势 | 拉取命令 |
|---|---|---|---|
| `llama3.3` | 70B | 通用型，推理强 | `ollama pull llama3.3` |
| `llama3.2` | 3B / 11B | 轻量 | `ollama pull llama3.2` |
| `qwen2.5-coder` | 7B / 32B | **最适合编码任务** | `ollama pull qwen2.5-coder` |
| `qwen2.5` | 7B / 72B | 中英双语 | `ollama pull qwen2.5` |
| `deepseek-r1` | 7B–70B | 推理、数学 | `ollama pull deepseek-r1` |
| `deepseek-coder-v2` | 16B | 编码 | `ollama pull deepseek-coder-v2` |
| `mistral` | 7B | 快速、高效 | `ollama pull mistral` |
| `mixtral` | 8x7B | 强大的 MoE 模型 | `ollama pull mixtral` |
| `phi4` | 14B | Microsoft 模型，推理强 | `ollama pull phi4` |
| `gemma3` | 4B / 12B / 27B | Google 开源模型 | `ollama pull gemma3` |
| `codellama` | 7B / 34B | 代码生成 | `ollama pull codellama` |
| `llava` | 7B / 13B | **视觉** —— 图像理解 | `ollama pull llava` |
| `llama3.2-vision` | 11B | **视觉** —— 多模态推理 | `ollama pull llama3.2-vision` |

> **注意：** 工具调用需要模型支持 function calling。推荐本地模型：`qwen2.5-coder`、`llama3.3`、`mistral`、`phi4`。

> **Reasoning 模型：** `deepseek-r1`、`qwen3` 和 `gemma4` 会流式输出原生 `<think>` 块。通过 `/verbose` 与 `/thinking` 可在终端中查看。注意：在接收大型系统提示（例如 cheetahclaws 的 25 个工具 schema）时，模型可能会抑制其 thinking 阶段以避免破坏预期 JSON 格式 —— 这是模型行为，不是 bug。

---

## 安装

### 推荐：使用 `uv` 安装为全局命令

[uv](https://docs.astral.sh/uv/) 会将 `cheetahclaws` 安装到隔离环境中，并加入你的 PATH，这样你可以在任意目录下运行：

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆并安装
git clone https://github.com/SafeRL-Lab/cheetahclaws
cd cheetahclaws
uv tool install .
```

之后，`cheetahclaws` 将作为全局命令可用：

```bash
cheetahclaws                        # 启动 REPL
cheetahclaws --model gpt-4o         # 选择模型
cheetahclaws -p "explain this"      # 非交互模式
```

在拉取新代码后更新：

```bash
uv tool install . --reinstall
```

卸载：

```bash
uv tool uninstall cheetahclaws
```

### 备选方案：直接从仓库运行

```bash
git clone https://github.com/SafeRL-Lab/cheetahclaws
cd cheetahclaws

pip install -r requirements.txt
# 或手动安装（sounddevice 为可选，仅 /voice 需要）：
pip install anthropic openai httpx rich
pip install sounddevice  # 可选：语音输入

python cheetahclaws.py
```

---

## 使用：闭源 API 模型

### Anthropic Claude

在 [console.anthropic.com](https://console.anthropic.com) 获取 API key。

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...

# 默认模型（claude-opus-4-6）
cheetahclaws

# 指定模型
cheetahclaws --model claude-sonnet-4-6
cheetahclaws --model claude-haiku-4-5-20251001

# 启用 Extended Thinking
cheetahclaws --model claude-opus-4-6 --thinking --verbose
```

### OpenAI GPT

在 [platform.openai.com](https://platform.openai.com) 获取 API key。

```bash
export OPENAI_API_KEY=sk-...

cheetahclaws --model gpt-4o
cheetahclaws --model gpt-4o-mini
cheetahclaws --model gpt-4.1-mini
cheetahclaws --model o3-mini
```

### Google Gemini

在 [aistudio.google.com](https://aistudio.google.com) 获取 API key。

```bash
export GEMINI_API_KEY=AIza...

cheetahclaws --model gemini/gemini-2.0-flash
cheetahclaws --model gemini/gemini-1.5-pro
cheetahclaws --model gemini/gemini-2.5-pro-preview-03-25
```

### Kimi（Moonshot AI）

在 [platform.moonshot.cn](https://platform.moonshot.cn) 获取 API key。

```bash
export MOONSHOT_API_KEY=sk-...

cheetahclaws --model kimi/moonshot-v1-32k
cheetahclaws --model kimi/moonshot-v1-128k
```

### Qwen（Alibaba DashScope）

在 [dashscope.aliyun.com](https://dashscope.aliyun.com) 获取 API key。

```bash
export DASHSCOPE_API_KEY=sk-...

cheetahclaws --model qwen/Qwen3.5-Plus
cheetahclaws --model qwen/Qwen3-MAX
cheetahclaws --model qwen/Qwen3.5-Flash
```

### Zhipu GLM

在 [open.bigmodel.cn](https://open.bigmodel.cn) 获取 API key。

```bash
export ZHIPU_API_KEY=...

cheetahclaws --model zhipu/glm-4-plus
cheetahclaws --model zhipu/glm-4-flash   # 免费层
```

### DeepSeek

在 [platform.deepseek.com](https://platform.deepseek.com) 获取 API key。

```bash
export DEEPSEEK_API_KEY=sk-...

cheetahclaws --model deepseek/deepseek-chat
cheetahclaws --model deepseek/deepseek-reasoner
```

### MiniMax

在 [platform.minimaxi.chat](https://platform.minimaxi.chat) 获取 API key。

```bash
export MINIMAX_API_KEY=...

cheetahclaws --model minimax/MiniMax-Text-01
cheetahclaws --model minimax/MiniMax-VL-01
cheetahclaws --model minimax/abab6.5s-chat
```

---

## 使用：开源模型（本地）

### 方案 A —— Ollama（推荐）

Ollama 可在本地零配置运行模型。无需 API key。

**第 1 步：安装 Ollama**

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# 或从 https://ollama.com/download 下载
```

**第 2 步：拉取模型**

```bash
# 最适合编码（推荐）
ollama pull qwen2.5-coder          # 4.7 GB (7B)
ollama pull qwen2.5-coder:32b      # 19 GB (32B)

# 通用模型
ollama pull llama3.3               # 42 GB (70B)
ollama pull llama3.2               # 2.0 GB (3B)

# 推理模型
ollama pull deepseek-r1            # 4.7 GB (7B)
ollama pull deepseek-r1:32b        # 19 GB (32B)

# 其他
ollama pull phi4                   # 9.1 GB (14B)
ollama pull mistral                # 4.1 GB (7B)
```

**第 3 步：启动 Ollama 服务**（在 macOS 上通常会自动运行；Linux 上手动启动）

```bash
ollama serve     # 启动在 http://localhost:11434
```

**第 4 步：运行 cheetahclaws**

```bash
cheetahclaws --model ollama/qwen2.5-coder
cheetahclaws --model ollama/llama3.3
cheetahclaws --model ollama/deepseek-r1
```

或者

```bash
python cheetahclaws.py --model ollama/qwen2.5-coder
python cheetahclaws.py --model ollama/llama3.3
python cheetahclaws.py --model ollama/deepseek-r1
python cheetahclaws.py --model ollama/qwen3.5:35b
```

**查看你本地可用的模型：**

```bash
ollama list
```

然后从列表中任选模型：

```bash
cheetahclaws --model ollama/<model-name>
```

---

### 方案 B —— LM Studio

LM Studio 提供图形界面用于下载和运行模型，并带有内置的 OpenAI-compatible server。

**第 1 步：** 下载并安装 [LM Studio](https://lmstudio.ai)。

**第 2 步：** 在 LM Studio 中搜索并下载模型（GGUF 格式）。

**第 3 步：** 进入 **Local Server** 标签页 → 点击 **Start Server**（默认端口：1234）。

**第 4 步：**

```bash
cheetahclaws --model lmstudio/<model-name>
# 例如：
cheetahclaws --model lmstudio/phi-4-GGUF
cheetahclaws --model lmstudio/qwen2.5-coder-7b
```

模型名应与 LM Studio 服务器状态栏中显示的一致。

---

### 方案 C —— vLLM / 自托管 OpenAI-Compatible Server

对于暴露 OpenAI-compatible API 的自托管推理服务（vLLM、TGI、llama.cpp server 等）：

方案 C 快速开始：
第 1 步：启动 vllm：
 ```
CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-Coder-7B-Instruct \
      --host 0.0.0.0 \
      --port 8000 \
      --enable-auto-tool-choice \
      --tool-call-parser hermes
```


 第 2 步：启动 cheetahclaws：
```
  export CUSTOM_BASE_URL=http://localhost:8000/v1
  export CUSTOM_API_KEY=none
  cheetahclaws --model custom/Qwen/Qwen2.5-Coder-7B-Instruct
```


```bash
# 示例：vLLM 提供 Qwen2.5-Coder-32B 服务
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --port 8000

# 然后运行 cheetahclaws 并指向你的服务：
cheetahclaws
```

在 REPL 中：

```
/config custom_base_url=http://localhost:8000/v1
/config custom_api_key=token-abc123    # 如果没有认证可跳过
/model custom/Qwen2.5-Coder-32B-Instruct
```

或通过环境变量设置：

```bash
export CUSTOM_BASE_URL=http://localhost:8000/v1
export CUSTOM_API_KEY=token-abc123

cheetahclaws --model custom/Qwen2.5-Coder-32B-Instruct
```

对于远程 GPU 服务器：

```bash
/config custom_base_url=http://192.168.1.100:8000/v1
/model custom/your-model-name
```

---

## 模型名格式

支持三种等价格式：

```bash
# 1. 通过前缀自动识别（适用于常见模型）
cheetahclaws --model gpt-4o
cheetahclaws --model gemini-2.0-flash
cheetahclaws --model deepseek-chat

# 2. 使用斜杠的显式 provider 前缀
cheetahclaws --model ollama/qwen2.5-coder
cheetahclaws --model kimi/moonshot-v1-128k

# 3. 使用冒号的显式 provider 前缀（也支持）
cheetahclaws --model kimi:moonshot-v1-32k
cheetahclaws --model qwen:qwen-max
```

**自动识别规则：**

| 模型前缀 | 识别出的 provider |
|---|---|
| `claude-` | anthropic |
| `gpt-`, `o1`, `o3` | openai |
| `gemini-` | gemini |
| `moonshot-`, `kimi-` | kimi |
| `qwen`, `qwq-` | qwen |
| `glm-` | zhipu |
| `deepseek-` | deepseek |
| `MiniMax-`, `minimax-`, `abab` | minimax |
| `llama`, `mistral`, `phi`, `gemma`, `mixtral`, `codellama` | ollama |

---

## CLI 参考

```
cheetahclaws [OPTIONS] [PROMPT]
# 或：python cheetahclaws.py [OPTIONS] [PROMPT]

Options:
  -p, --print          非交互模式：运行 prompt 后退出
  -m, --model MODEL    覆盖模型（如 gpt-4o、ollama/llama3.3）
  --accept-all         自动批准所有操作（无权限提示）
  --verbose            显示 thinking 块和每轮 token 计数
  --thinking           启用 Extended Thinking（仅 Claude）
  --version            打印版本并退出
  -h, --help           显示帮助
```

**示例：**

```bash
# 使用默认模型启动交互式 REPL
cheetahclaws

# 启动时切换模型
cheetahclaws --model gpt-4o
cheetahclaws -m ollama/deepseek-r1:32b

# 非交互 / 脚本模式
cheetahclaws --print "Write a Python fibonacci function"
cheetahclaws -p "Explain the Rust borrow checker in 3 sentences" -m gemini/gemini-2.0-flash

# CI / 自动化（无权限提示）
cheetahclaws --accept-all --print "Initialize a Python project with pyproject.toml"

# 调试模式（查看 tokens + thinking）
cheetahclaws --thinking --verbose
```

---

## 斜杠命令（REPL）

输入 `/` 后按 **Tab** 可查看带描述的全部命令。继续输入可过滤，再按 Tab 自动补全。输入命令名后再按 **Tab** 可查看子命令（例如 `/plugin ` → `install`、`uninstall`、`enable`、…）。

| 命令 | 说明 |
|---|---|
| `/help` | 显示所有命令 |
| `/clear` | 清空对话历史 |
| `/model` | 显示当前模型并列出所有可用模型 |
| `/model <name>` | 切换模型（立即生效） |
| `/config` | 显示当前所有配置值 |
| `/config key=value` | 设置配置值（持久化到磁盘） |
| `/save` | 保存会话（按时间戳自动命名） |
| `/save <filename>` | 保存为指定文件名 |
| `/load` | 交互式按日期分组列表；输入编号、`1,2,3` 以合并，或 `H` 查看完整历史 |
| `/load <filename>` | 按文件名加载保存的会话 |
| `/resume` | 恢复最后自动保存的会话（`mr_sessions/session_latest.json`） |
| `/resume <filename>` | 从 `mr_sessions/`（或绝对路径）加载指定文件 |
| `/history` | 打印完整对话历史 |
| `/context` | 显示消息数量和 token 估计 |
| `/cost` | 显示 token 使用量和预估美元成本 |
| `/verbose` | 切换 verbose 模式（tokens + thinking） |
| `/thinking` | 切换 Extended Thinking（仅 Claude） |
| `/permissions` | 显示当前权限模式 |
| `/permissions <mode>` | 设置权限模式：`auto` / `accept-all` / `manual` |
| `/cwd` | 显示当前工作目录 |
| `/cwd <path>` | 更改当前工作目录 |
| `/memory` | 列出全部持久记忆 |
| `/memory <query>` | 按关键词搜索记忆（按 confidence × recency 排序） |
| `/memory consolidate` | 从当前会话中 AI 提取最多 3 条长期洞见 |
| `/skills` | 列出可用 skills |
| `/agents` | 显示子 agent 任务状态 |
| `/mcp` | 列出已配置的 MCP server 及其工具 |
| `/mcp reload` | 重新连接所有 MCP server 并刷新工具 |
| `/mcp reload <name>` | 重新连接单个 server |
| `/mcp add <name> <cmd> [args]` | 向用户配置中添加一个 stdio MCP server |
| `/mcp remove <name>` | 从用户配置中移除一个 server |
| `/voice` | 录音、用 Whisper 转写、自动作为 prompt 提交 |
| `/voice status` | 显示录音和 STT 后端可用性，以及当前麦克风 |
| `/voice lang <code>` | 设置 STT 语言（如 `zh`、`en`、`ja`；`auto` 自动识别） |
| `/voice device` | 列出可用输入麦克风并以交互方式选择 |
| `/image [prompt]` | 获取剪贴板图片并发送给视觉模型，可附加提示 |
| `/img [prompt]` | `/image` 的别名 |
| `/proactive` | 显示当前主动轮询状态（开/关及间隔） |
| `/proactive <duration>` | 启用后台 sentinel 轮询（如 `5m`、`30s`、`1h`） |
| `/proactive off` | 关闭后台轮询 |
| `/cloudsave setup <token>` | 配置 GitHub Personal Access Token 用于 Gist 同步 |
| `/cloudsave` | 将当前会话上传到私有 GitHub Gist |
| `/cloudsave push [desc]` | 以可选描述上传 |
| `/cloudsave auto on\|off` | 切换 `/exit` 时是否自动上传 |
| `/cloudsave list` | 列出你的 cheetahclaws Gists |
| `/cloudsave load <gist_id>` | 下载并恢复某个 Gist 中的会话 |
| `/brainstorm` | 运行多 persona AI 头脑风暴；会提示 agent 数量（2–100，默认 5） |
| `/brainstorm <topic>` | 围绕特定主题头脑风暴；会提示 agent 数量 |
| `/ssj` | 打开 SSJ 开发者模式 —— 交互式强力菜单，含 10 个工作流快捷入口 |
| `/worker` | 自动实现 `brainstorm_outputs/todo_list.txt` 中的所有待办任务 |
| `/worker <n,m,…>` | 按编号实现指定待办任务（如 `/worker 1,4,6`） |
| `/worker --path <file>` | 使用自定义 todo 文件路径而不是默认路径 |
| `/worker --workers <n>` | 限制每次运行最多处理 N 个任务（如 `/worker --workers 3`） |
| `/telegram <token> <chat_id>` | 配置并启动 Telegram bot 桥 |
| `/telegram` | 使用已保存的 token + chat_id 启动桥 |
| `/telegram stop` | 停止桥 |
| `/telegram status` | 显示桥是否正在运行，以及配置的 chat_id |
| `/checkpoint` | 列出当前会话的所有 checkpoints（快照） |
| `/checkpoint <id>` | 回滚到指定 checkpoint —— 恢复文件与对话到该快照 |
| `/checkpoint clear` | 删除当前会话的全部 checkpoints |
| `/rewind` | `/checkpoint` 的别名 |
| `/plan <description>` | 进入计划模式：只读分析，仅允许写计划文件 |
| `/plan` | 显示当前计划文件内容 |
| `/plan done` | 退出计划模式并恢复原始权限 |
| `/plan status` | 显示计划模式是否激活 |
| `/compact` | 手动压缩对话（与自动压缩相同，但由用户触发） |
| `/compact <focus>` | 带聚焦指令进行压缩（例如 `/compact keep the auth refactor context`） |
| `/init` | 在当前工作目录创建一个 `CLAUDE.md` 模板 |
| `/export` | 将对话导出为 Markdown 到 `.nano_claude/exports/` |
| `/export <filename>` | 导出为 Markdown 或 JSON（由 `.json` 后缀决定） |
| `/copy` | 将上一条 assistant 回复复制到剪贴板 |
| `/status` | 显示版本、模型、provider、权限、session ID、token 使用量和上下文占比 |
| `/doctor` | 诊断安装状态：Python、git、API key、可选依赖、CLAUDE.md、checkpoint 磁盘占用 |
| `/theme` | 列出 15 套控制台配色，每行用自身颜色显示 `info / ok / warn / err` 实时预览（当前主题标 `*`） |
| `/theme <name>` | 应用并持久化主题（如 `dracula`、`nord`、`gruvbox`、`none`），同时驱动 Rich Markdown 代码块风格 |
| `/exit` / `/quit` | 退出 |

**会话内切换模型：**

```
[myproject] ❯ /model
  Current model: claude-opus-4-6  (provider: anthropic)

  Available models by provider:
    anthropic     claude-opus-4-6, claude-sonnet-4-6, ...
    openai        gpt-4o, gpt-4o-mini, o3-mini, ...
    ollama        llama3.3, llama3.2, phi4, mistral, ...
    ...

[myproject] ❯ /model gpt-4o
  Model set to gpt-4o  (provider: openai)

[myproject] ❯ /model ollama/qwen2.5-coder
  Model set to ollama/qwen2.5-coder  (provider: ollama)
```

---

## 配置 API Key

### 方法 1：环境变量（推荐）

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=AIza...
export MOONSHOT_API_KEY=sk-...       # Kimi
export DASHSCOPE_API_KEY=sk-...      # Qwen
export ZHIPU_API_KEY=...             # Zhipu GLM
export DEEPSEEK_API_KEY=sk-...       # DeepSeek
export MINIMAX_API_KEY=...           # MiniMax
```

### 方法 2：在 REPL 中设置（会持久化）

```
/config anthropic_api_key=sk-ant-...
/config openai_api_key=sk-...
/config gemini_api_key=AIza...
/config kimi_api_key=sk-...
/config qwen_api_key=sk-...
/config zhipu_api_key=...
/config deepseek_api_key=sk-...
/config minimax_api_key=...
```

这些 key 会保存到 `~/.cheetahclaws/config.json`，下次启动时自动加载。

### 方法 3：直接编辑配置文件

```json
// ~/.cheetahclaws/config.json
{
  "model": "qwen/qwen-max",
  "max_tokens": 8192,
  "permission_mode": "auto",
  "verbose": false,
  "thinking": false,
  "qwen_api_key": "sk-...",
  "kimi_api_key": "sk-...",
  "deepseek_api_key": "sk-...",
  "minimax_api_key": "..."
}
```

---

## 权限系统

| 模式 | 行为 |
|---|---|
| `auto`（默认） | 只读操作始终允许。执行 Bash 命令和文件写入前会提示。 |
| `accept-all` | 永不提示。所有操作自动执行。 |
| `manual` | 每一个操作都提示确认，包括读取。 |
| `plan` | 只读分析模式。只有计划文件（`.nano_claude/plans/`）可写。通过 `/plan <desc>` 或 `EnterPlanMode` 工具进入。 |

**出现提示时：**

```
  Allow: Run: git commit -am "fix bug"  [y/N/a(ccept-all)]
```

- `y` —— 批准本次操作
- `n` 或直接回车 —— 拒绝
- `a` —— 批准并在本次会话余下时间切换到 `accept-all`

**在 `auto` 模式下始终自动批准的命令：**
`ls`、`cat`、`head`、`tail`、`wc`、`pwd`、`echo`、`git status`、`git log`、`git diff`、`git show`、`find`、`grep`、`rg`、`python`、`node`、`pip show`、`npm list` 以及其他只读 shell 命令。

---

## 内置工具

### 核心工具

| 工具 | 说明 | 关键参数 |
|---|---|---|
| `Read` | 带行号读取文件 | `file_path`, `limit`, `offset` |
| `Write` | 创建或覆盖文件（显示 diff） | `file_path`, `content` |
| `Edit` | 精确字符串替换（显示 diff） | `file_path`, `old_string`, `new_string`, `replace_all` |
| `Bash` | 执行 shell 命令 | `command`, `timeout`（默认 30s） |
| `Glob` | 通过 glob 模式查找文件 | `pattern`（如 `**/*.py`）, `path` |
| `Grep` | 在文件中进行正则搜索（若可用则使用 ripgrep） | `pattern`, `path`, `glob`, `output_mode` |
| `WebFetch` | 抓取 URL 并提取文本 | `url`, `prompt` |
| `WebSearch` | 通过 DuckDuckGo 搜索网页 | `query` |

### Notebook 与诊断工具

| 工具 | 说明 | 关键参数 |
|---|---|---|
| `NotebookEdit` | 编辑 Jupyter notebook（`.ipynb`）中的 cell | `notebook_path`, `new_source`, `cell_id`, `cell_type`, `edit_mode`（`replace`/`insert`/`delete`） |
| `GetDiagnostics` | 获取源文件的 LSP 风格诊断信息（Python 使用 pyright/mypy/flake8；JS/TS 使用 tsc/eslint；shell 使用 shellcheck） | `file_path`, `language`（可选覆盖） |

### 记忆工具

| 工具 | 说明 | 关键参数 |
|---|---|---|
| `MemorySave` | 保存或更新持久记忆 | `name`, `type`, `description`, `content`, `scope` |
| `MemoryDelete` | 按名称删除记忆 | `name`, `scope` |
| `MemorySearch` | 按关键词（或 AI 排序）搜索记忆 | `query`, `scope`, `use_ai`, `max_results` |
| `MemoryList` | 列出全部记忆及其年龄和元数据 | `scope` |

### 子 Agent 工具

| 工具 | 说明 | 关键参数 |
|---|---|---|
| `Agent` | 为任务启动一个子 agent | `prompt`, `subagent_type`, `isolation`, `name`, `model`, `wait` |
| `SendMessage` | 向命名的后台 agent 发送消息 | `name`, `message` |
| `CheckAgentResult` | 检查后台 agent 的状态/结果 | `task_id` |
| `ListAgentTasks` | 列出所有活跃和已完成的 agent 任务 | — |
| `ListAgentTypes` | 列出可用的 agent 类型定义 | — |

### 后台与自主工具

| 工具 | 说明 | 关键参数 |
|---|---|---|
| `SleepTimer` | 安排一个静默后台计时器；触发时注入自动唤醒提示，使 agent 能继续监控或恢复延后任务 | `seconds` |

### Skill 工具

| 工具 | 说明 | 关键参数 |
|---|---|---|
| `Skill` | 在对话中调用指定 skill | `name`, `args` |
| `SkillList` | 列出所有可用 skill 及其触发器和元数据 | — |

### MCP 工具

MCP 工具会自动从已配置 server 中发现，并以 `mcp__<server>__<tool>` 的形式注册。Claude 可以像调用内置工具一样调用它们。

| 示例工具名 | 来源 |
|---|---|
| `mcp__git__git_status` | `git` server 的 `git_status` 工具 |
| `mcp__filesystem__read_file` | `filesystem` server 的 `read_file` 工具 |
| `mcp__myserver__my_action` | 你配置的自定义 server |

> **添加自定义工具：** 如何注册你自己的工具，请查看 [Architecture Guide](docs/architecture.md#tool-registry)。

---

## 记忆

模型可以通过内置记忆系统跨对话记住信息。

### 存储位置

记忆以独立 markdown 文件形式保存在两个作用域中：

| 作用域 | 路径 | 可见性 |
|---|---|---|
| **用户级**（默认） | `~/.cheetahclaws/memory/` | 跨所有项目共享 |
| **项目级** | 当前目录中的 `.cheetahclaws/memory/` | 仅当前仓库可见 |

系统会在每次保存或删除时自动重建一个 `MEMORY.md` 索引（≤ 200 行 / 25 KB），并将其注入系统提示，使模型始终能看到记忆总览。

### 记忆类型

| 类型 | 用途 |
|---|---|
| `user` | 你的身份、偏好、背景 |
| `feedback` | 你希望模型如何表现（纠正与正向确认） |
| `project` | 不在 git 历史中的进行中工作、deadline、决策 |
| `reference` | 外部系统链接（Linear、Grafana、Slack 等） |

### 记忆文件格式

每条记忆都是一个带 YAML frontmatter 的 markdown 文件：

```markdown
---
name: coding_style
description: Python formatting preferences
type: feedback
created: 2026-04-02
confidence: 0.95
source: user
last_used_at: 2026-04-05
conflict_group: coding_style
---
Prefer 4-space indentation and full type hints in all Python code.
**Why:** user explicitly stated this preference.
**How to apply:** apply to every Python file written or edited.
```

**元数据字段**（新增 —— 自动管理）：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `confidence` | `1.0` | 可靠性分数，0–1。显式用户声明 = 1.0；推断偏好 ≈ 0.8；自动整合 ≈ 0.8 |
| `source` | `user` | 来源：`user` / `model` / `tool` / `consolidator` |
| `last_used_at` | — | 每次被 MemorySearch 返回时自动更新 |
| `conflict_group` | — | 用于分组相关记忆（如 `writing_style`）以进行冲突跟踪 |

### 冲突检测

当 `MemorySave` 以相同名称但不同内容保存时，系统会在覆盖前报告冲突：

```
Memory saved: 'writing_style' [feedback/user]
⚠ Replaced conflicting memory (was user-sourced, 100% confidence, written 2026-04-01).
  Old content: Prefer formal, academic style...
```

### 排序检索

`MemorySearch` 会按照 **confidence × recency**（30 天指数衰减）排序，而不是按纯关键词顺序。长时间未使用的记忆会逐渐降低优先级。每次搜索命中也会更新 `last_used_at`，因此高频使用的记忆会保持突出。

```
You: /memory python
  [feedback/user] coding_style [conf:95% src:user]
    Python formatting preferences
    Prefer 4-space indentation and full type hints...
```

### `/memory consolidate` —— 自动提取长期洞见

在一次有意义的会话后，运行：

```
[myproject] ❯ /memory consolidate
  Analyzing session for long-term memories…
  ✓ Consolidated 2 memory/memories: user_prefers_direct_answers, avoid_trailing_summaries
```

该命令会把压缩后的会话转录发送给模型，要求它识别最多 **3** 条值得长期保留的洞见（用户偏好、反馈修正、项目决策）。提取出的记忆会以 `confidence: 0.80` 和 `source: consolidator` 保存 —— 它们**绝不会覆盖**已有且置信度更高的记忆。

适合运行 `/memory consolidate` 的时候：
- 当你连续多次纠正模型行为之后
- 当你在会话中分享了项目背景或关键决策之后
- 当你完成一项任务并形成了清晰的规划选择之后

### 示例交互

```
You: Remember that I prefer 4-space indentation and type hints.
AI: [calls MemorySave] Memory saved: 'coding_style' [feedback/user]

You: /memory
  1 memory/memories:
  [feedback  |user   ] coding_style.md
    Python formatting preferences

You: /memory python
  Found 1 relevant memory for 'python':
  [feedback/user] coding_style
    Prefer 4-space indentation and full type hints in all Python code.

You: /memory consolidate
  ✓ Consolidated 1 memory: user_prefers_verbose_commit_messages
```

**过期警告：** 超过 1 天的记忆会显示 `⚠ stale` 提示 —— 关于 file:line 引用或代码状态的断言可能已过时，使用前请核验。

**AI 排序搜索：** `MemorySearch(query="...", use_ai=true)` 会先让模型对候选结果按相关性排序，再应用 confidence × recency 重排。

---

## Skills

Skills 是可复用的提示模板，用于给模型提供专业化能力。系统开箱即带两个内置 skill —— 无需配置。

**内置 skills：**

| 触发器 | 说明 |
|---|---|
| `/commit` | 审查已暂存改动并生成结构良好的 git commit |
| `/review [PR]` | 对代码或 PR diff 进行结构化审查 |

**快速开始 —— 自定义 skill：**

```bash
mkdir -p ~/.cheetahclaws/skills
```

创建 `~/.cheetahclaws/skills/deploy.md`：

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

现在这样使用：

```
You: /deploy staging 2.1.0
AI: [deploys version 2.1.0 to staging]
```

**参数替换：**
- `$ARGUMENTS` —— 完整原始参数字符串
- `$ARG_NAME` —— 按命名参数位置替换（第一个词 → 第一个参数名）
- 缺失参数会替换为空字符串

**执行模式：**
- `context: inline`（默认）—— 在当前对话历史中运行
- `context: fork` —— 以隔离子 agent 方式运行，使用独立历史；支持 `model` 覆盖

**优先级**（越高者生效）：项目级 > 用户级 > 内置

**列出 skills：** `/skills` —— 显示触发器、参数提示、来源与 `when_to_use`

**Skill 搜索路径：**

```
./.cheetahclaws/skills/     # 项目级（覆盖用户级）
~/.cheetahclaws/skills/     # 用户级
```

---

## 子 Agent

模型可以启动独立子 agent 并行处理任务。

**内置专门化 agent 类型：**

| 类型 | 擅长 |
|---|---|
| `general-purpose` | 研究、探索、多步任务 |
| `coder` | 编写、阅读与修改代码 |
| `reviewer` | 安全性、正确性与代码质量分析 |
| `researcher` | Web 搜索与文档检索 |
| `tester` | 编写并运行测试 |

**基本用法：**
```
You: Search this codebase for all TODO comments and summarize them.
AI: [calls Agent(prompt="...", subagent_type="researcher")]
    Sub-agent reads files, greps for TODOs...
    Result: Found 12 TODOs across 5 files...
```

**后台模式** —— 启动后不等待，稍后再收集结果：
```
AI: [calls Agent(prompt="run all tests", name="test-runner", wait=false)]
AI: [continues other work...]
AI: [calls CheckAgentResult / SendMessage to follow up]
```

**Git worktree 隔离** —— agent 在隔离分支中工作，无冲突：
```
Agent(prompt="refactor auth module", isolation="worktree")
```
若无改动，worktree 会自动清理；否则会报告分支名。

**自定义 agent 类型** —— 创建 `~/.cheetahclaws/agents/myagent.md`：
```markdown
---
name: myagent
description: Specialized for X
model: claude-haiku-4-5-20251001
tools: [Read, Grep, Bash]
---
Extra system prompt for this agent type.
```

**列出正在运行的 agents：** `/agents`

子 agent 拥有独立的对话历史、共享文件系统，并限制为最多 3 层嵌套。

---

## MCP（模型上下文协议）

MCP 让你可以连接任意外部工具 server —— 本地子进程或远程 HTTP —— Claude 会自动使用这些工具。这与 Claude Code 用于扩展能力的协议相同。

### 支持的传输方式

| 传输 | 配置 `type` | 说明 |
|---|---|---|
| **stdio** | `"stdio"` | 启动本地子进程（最常见） |
| **SSE** | `"sse"` | HTTP Server-Sent Events 流 |
| **HTTP** | `"http"` | 可流式的 HTTP POST（较新的 server） |

### 配置方式

在项目目录中放置 `.mcp.json` 文件，**或者** 编辑用户级 `~/.cheetahclaws/mcp.json`。

```json
{
  "mcpServers": {
    "git": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-server-git"]
    },
    "filesystem": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-server-filesystem", "/tmp"]
    },
    "my-remote": {
      "type": "sse",
      "url": "http://localhost:8080/sse",
      "headers": {"Authorization": "Bearer my-token"}
    }
  }
}
```

配置优先级：`.mcp.json`（项目级）会按 server 名覆盖 `~/.cheetahclaws/mcp.json`（用户级）。

### 快速开始

```bash
# 安装一个常见的 MCP server
pip install uv        # uv 自带 uvx
uvx mcp-server-git --help   # 验证可用

# 通过 REPL 加入用户配置
/mcp add git uvx mcp-server-git

# 或在项目目录创建 .mcp.json，然后：
/mcp reload
```

### REPL 命令

```
/mcp                          # 列出 servers + 它们的 tools + 连接状态
/mcp reload                   # 重新连接所有 servers，并刷新工具列表
/mcp reload git               # 重新连接单个 server
/mcp add myserver uvx mcp-server-x   # 添加 stdio server
/mcp remove myserver          # 从用户配置移除
```

### Claude 如何使用 MCP 工具

连接成功后，Claude 可以直接调用 MCP 工具：

```
You: What files changed in the last git commit?
AI: [calls mcp__git__git_diff_staged()]
    → shows diff output from the git MCP server
```

工具名遵循 `mcp__<server_name>__<tool_name>` 模式。所有非字母数字或 `_` 的字符都会自动替换为 `_`。

### 常见 MCP servers

| Server | 安装 | 提供 |
|---|---|---|
| `mcp-server-git` | `uvx mcp-server-git` | git 操作（status、diff、log、commit） |
| `mcp-server-filesystem` | `uvx mcp-server-filesystem <path>` | 文件读/写/列出 |
| `mcp-server-fetch` | `uvx mcp-server-fetch` | HTTP 抓取工具 |
| `mcp-server-postgres` | `uvx mcp-server-postgres <conn-str>` | PostgreSQL 查询 |
| `mcp-server-sqlite` | `uvx mcp-server-sqlite --db-path x.db` | SQLite 查询 |
| `mcp-server-brave-search` | `uvx mcp-server-brave-search` | Brave 网页搜索 |

> 完整 registry 见 [modelcontextprotocol.io/servers](https://modelcontextprotocol.io/servers)

---

## 插件系统

`plugin/` 包让你可以通过 git 仓库或本地目录，为 cheetahclaws 扩展更多工具、skills 与 MCP servers。

### 安装插件

```bash
/plugin install my-plugin@https://github.com/user/my-plugin
/plugin install local-plugin@/path/to/local/plugin
```

### 管理插件

```bash
/plugin                   # 列出已安装插件
/plugin enable my-plugin  # 启用一个已禁用插件
/plugin disable my-plugin # 禁用但不卸载
/plugin disable-all       # 禁用所有插件
/plugin update my-plugin  # 从 git 拉取最新版本
/plugin uninstall my-plugin
/plugin info my-plugin    # 显示 manifest 详情
```

### 插件推荐引擎

```bash
/plugin recommend                    # 根据项目文件自动检测推荐
/plugin recommend "docker database"  # 按关键词上下文推荐
```

推荐引擎会将你的上下文与一个策划好的 marketplace 匹配（git-tools、python-linter、docker-tools、sql-tools、test-runner、diagram-tools、aws-tools、web-scraper），使用标签与关键词打分。

### 插件清单（plugin.json）

```json
{
  "name": "my-plugin",
  "version": "0.1.0",
  "description": "Does something useful",
  "author": "you",
  "tags": ["git", "python"],
  "tools": ["tools"],        // 导出 TOOL_DEFS 的 Python 模块
  "skills": ["skills/my.md"],
  "mcp_servers": {},
  "dependencies": ["httpx"]  // pip 包
}
```

也可以在 `PLUGIN.md` 中使用 YAML frontmatter。

### 作用域

| 作用域 | 位置 | 配置 |
|-------|----------|--------|
| user（默认） | `~/.cheetahclaws/plugins/` | `~/.cheetahclaws/plugins.json` |
| project | `.cheetahclaws/plugins/` | `.cheetahclaws/plugins.json` |

使用 `--project` 标志：`/plugin install name@url --project`

---

## AskUserQuestion 工具

Claude 可以在任务执行中途暂停，并先向你提出一个问题，再继续执行。

**Claude 调用示例：**
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

**你在终端中会看到：**
```
❓ Question from assistant:
   Which database should I use?

  [1] SQLite — Simple, file-based
  [2] PostgreSQL — Full-featured, requires server
  [0] Type a custom answer

Your choice (number or text):
```

- 可通过编号选择，也可直接输入自由文本
- Claude 收到你的回答后会继续任务
- 5 分钟超时（若无应答则返回 `"(no answer — timeout)"`）

---

## 任务管理

`task/` 包为 Claude（以及你自己）提供了结构化任务列表，用于在一个会话中跟踪多步工作。

### Claude 可用的工具

| 工具 | 参数 | 功能 |
|------|-----------|--------------|
| `TaskCreate` | `subject`, `description`, `active_form?`, `metadata?` | 创建任务；返回 `#id created: subject` |
| `TaskUpdate` | `task_id`, `subject?`, `description?`, `status?`, `owner?`, `add_blocks?`, `add_blocked_by?`, `metadata?` | 更新任意字段；`status='deleted'` 会移除该任务 |
| `TaskGet` | `task_id` | 返回单个任务的完整详情 |
| `TaskList` | _(none)_ | 列出所有任务及其状态图标和未解除的 blocker |

**合法状态：** `pending` → `in_progress` → `completed` / `cancelled` / `deleted`

### 依赖边

```
TaskUpdate(task_id="3", add_blocked_by=["1","2"])
# 任务 3 现在被任务 1 和 2 阻塞。
# 反向边会自动设置：任务 1 和 2 的 "blocks" 列表中会加入任务 3。
```

已完成任务会被视为已解除，因此 `TaskList` 不再将其作为依赖阻塞项显示。

### 持久化

任务每次变更后都会保存到当前工作目录的 `.cheetahclaws/tasks.json`，并在首次访问时自动重新加载。

### REPL 命令

```
/tasks                    列出全部任务
/tasks create <subject>   快速创建任务
/tasks start <id>         标记为 in_progress
/tasks done <id>          标记为 completed
/tasks cancel <id>        标记为 cancelled
/tasks delete <id>        删除任务
/tasks get <id>           显示完整详情
/tasks clear              删除所有任务
```

### Claude 的典型工作流

```
User: implement the login feature

Claude:
  TaskCreate(subject="Design auth schema", description="JWT vs session")  → #1
  TaskCreate(subject="Write login endpoint", description="POST /auth/login") → #2
  TaskCreate(subject="Write tests", description="Unit + integration") → #3
  TaskUpdate(task_id="2", add_blocked_by=["1"])
  TaskUpdate(task_id="3", add_blocked_by=["2"])

  TaskUpdate(task_id="1", status="in_progress", active_form="Designing schema")
  ... (does the work) ...
  TaskUpdate(task_id="1", status="completed")
  TaskList()  → task 2 is now unblocked
  ...
```

---

## 语音输入

CheetahClaws v3.05 增加了完整离线的 voice-to-prompt 管道。你说出请求，它会转写并像你手动输入一样提交。

### 快速开始

```bash
# 1. 安装一个录音后端（任选其一）
pip install sounddevice        # 推荐：跨平台，无需额外二进制
# sudo apt install alsa-utils  # Linux arecord 备选
# sudo apt install sox         # SoX rec 备选

# 2. 安装本地 STT 后端（推荐 —— 离线可用，无 API key）
pip install faster-whisper numpy

# 3. 启动 CheetahClaws 并开始说话
cheetahclaws
[myproject] ❯ /voice
  🎙  Listening… (speak now, auto-stops on silence, Ctrl+C to cancel)
  🎙  ████
✓  Transcribed: "fix the authentication bug in user.py"
[auto-submitting…]
```

### STT 后端（按顺序尝试）

| 后端 | 安装 | 说明 |
|---|---|---|
| `faster-whisper` | `pip install faster-whisper` | **推荐** —— 本地、离线、最快，可选 GPU |
| `openai-whisper` | `pip install openai-whisper` | 本地、离线、原版 OpenAI 模型 |
| OpenAI Whisper API | 设置 `OPENAI_API_KEY` | 云端，需要网络和 API key |

可通过 `NANO_CLAUDE_WHISPER_MODEL` 覆盖 Whisper 模型大小（默认：`base`）：

```bash
export NANO_CLAUDE_WHISPER_MODEL=small   # 更高精度，更慢
export NANO_CLAUDE_WHISPER_MODEL=tiny    # 最快、最轻量
```

### 录音后端（按顺序尝试）

| 后端 | 安装 | 说明 |
|---|---|---|
| `sounddevice` | `pip install sounddevice` | **推荐** —— 跨平台、Python 原生 |
| `arecord` | `sudo apt install alsa-utils` | Linux ALSA，无需 pip |
| `sox rec` | `sudo apt install sox` / `brew install sox` | 内置静音检测 |

### 关键词增强

在每次录音前，CheetahClaws 会从以下来源提取编程词汇：
- **Git 分支名**（例如 `feat/voice-input` → “feat”, “voice”, “input”）
- **项目根目录名**（例如 “cheetahclaws”）
- **最近源码文件名干**（例如 `authentication_handler.py` → “authentication”, “handler”）
- **全局编码术语**：`MCP`、`grep`、`TypeScript`、`OAuth`、`regex`、`gRPC`、…

这些词会作为 Whisper 的 `initial_prompt` 传入，从而提高 STT 对编码术语拼写的偏好。

### 命令

| 命令 | 说明 |
|---|---|
| `/voice` | 录音并自动将转写文本作为下一条 prompt 提交 |
| `/voice status` | 显示可用的录音和 STT 后端，以及当前麦克风 |
| `/voice lang <code>` | 设置转写语言（`en`、`zh`、`ja`、`de`、`fr`、…，默认：`auto`） |
| `/voice device` | 列出所有可用输入麦克风并交互选择；对当前会话持久生效 |

### 选择麦克风

在有多个音频接口的系统（USB 耳机、虚拟设备等）上，你可以精确选择输入设备：

```
[myproject] ❯ /voice device
  🎙  Available input devices:
    0. Built-in Microphone
    1. USB Headset (USB Audio)  ◀  (currently selected)
    2. Virtual Input (BlackHole)
  Select device # (Enter to cancel): 1
✓  Microphone set to: [1] USB Headset (USB Audio)
```

所选设备会显示在 `/voice status` 中，并用于后续所有录音，直到你修改它或重启程序。

### 与 Claude Code 的对比

| | Claude Code | CheetahClaws v3.05 |
|---|---|---|
| STT 服务 | Anthropic 私有 WebSocket（`voice_stream`） | `faster-whisper` / `openai-whisper` / OpenAI API |
| 是否要求 Anthropic OAuth | 是 | **否** |
| 是否可离线 | 否 | **可**（使用本地 Whisper） |
| 关键词提示 | Deepgram `keyterms` 参数 | Whisper `initial_prompt`（git + 文件 + 词汇） |
| 语言支持 | 服务器白名单语言代码 | Whisper 支持的任意语言 |

---

## 头脑风暴

`/brainstorm` 会围绕你的项目运行结构化多 persona AI 辩论，并最终综合出可执行计划。

### 工作方式

1. **上下文快照** —— 读取当前工作目录中的 `README.md`、`CLAUDE.md` 与根目录文件列表。
2. **Agent 数量** —— 系统会提示你选择 agent 数量（2–100，默认 5）。直接回车使用默认值。
3. **动态 persona 生成** —— 模型会根据你的主题生成 N 个合适的专家角色。软件主题会生成架构师、工程师；地缘政治会生成分析师、外交专家、经济学家；商业主题会生成战略与市场专家。若生成失败，则回退到内置技术 persona。
4. **Agents 按顺序辩论**，每个都基于前一个的回复继续推进。
5. **输出保存** 到当前目录的 `brainstorm_outputs/brainstorm_YYYYMMDD_HHMMSS.md`。
6. **综合** —— 主 agent 读取保存的文件，并产出按优先级排序的 Master Plan。

**按主题自动生成 persona 的示例：**

| 主题 | 示例 Persona |
|---|---|
| 软件架构 | 🏗️ 架构师 · 💡 产品创新者 · 🛡️ 安全工程师 · 🔧 代码质量负责人 · ⚡ 性能专家 |
| 美伊地缘政治 | 🌍 地缘分析师 · ⚖️ 国际法专家 · 💰 能源经济学家 · 🎖️ 军事战略家 · 🕊️ 冲突调停者 |
| 商业战略 | 📈 市场战略专家 · 💼 运营负责人 · 🔍 竞品情报专家 · 💡 创新总监 · 📊 财务分析师 |

### 用法

```
[myproject] ❯ /brainstorm
  How many agents? (2-100, default 5) > 5

[myproject] ❯ /brainstorm improve plugin architecture
  How many agents? (2-100, default 5) > 3

[myproject] ❯ /brainstorm US-Iran geopolitics
  How many agents? (2-100, default 5) > 7
```

### 输出示例

```
[myproject] ❯ /brainstorm medical research funding
  How many agents? (2-100, default 5) > 3
Generating 3 topic-appropriate expert personas...
Starting 3-Agent Brainstorming Session on: medical research funding
Generating diverse perspectives...
🩺 Clinical Trials Director is thinking...
  └─ Perspective captured.
⚖️ Medical Ethics Committee Member is thinking...
  └─ Perspective captured.
💰 Health Economics Policy Analyst is thinking...
  └─ Perspective captured.
✓  Brainstorming complete! Results saved to brainstorm_outputs/brainstorm_20260405_224117.md

   ── Analysis from Main Agent ──
[synthesized Master Plan streams here…]
```

### 说明

- Brainstorm 会使用**当前选中的模型**（用 `/model` 查看）。使用能力较强的模型（Claude Sonnet/Opus、GPT-4o 或较大的本地模型）效果最佳。
- 当 agent 数较多（20+）时，根据模型速度不同，整个会话可能持续数分钟。
- 安装 `faker`（`pip install faker`）可获得随机 persona 名称；若缺失则退回内置名称。
- 输出文件会持续积累在 `brainstorm_outputs/` 中 —— 从 v3.05.5 起该目录已被加入 `.gitignore`。
- 如果在 SSH 中输出看起来混乱（重复行），请运行 `/config rich_live=false` 关闭 Rich Live 流式渲染。

---

## SSJ 开发者模式

`/ssj` 会打开一个持久化交互强力菜单 —— 统一入口覆盖最常见的开发工作流，这样你不再需要记命令名。

<div align=center>
<img src="../media/demos/ssj_demo.gif" width="850"/>
</div>

### 菜单项

| # | 名称 | 功能 |
|---|------|--------------|
| 1 | 💡 Brainstorm | 多 persona AI 辩论 → Master Plan → 自动生成 `brainstorm_outputs/todo_list.txt` |
| 2 | 📋 Show TODO | 查看 `brainstorm_outputs/todo_list.txt`，带 ✓/○ 状态与待办编号 |
| 3 | 👷 Worker | 自动实现待办任务（全部或按编号选择） |
| 4 | 🧠 Debate | 选择一个文件并指定 agent 数 —— 专家小组按轮辩论设计；结果保存在文件旁边 |
| 5 | ✨ Propose | 选择一个文件 —— AI 提出具体改进方案及代码 |
| 6 | 🔎 Review | 选择一个文件 —— 结构化代码审查，并按维度 1–10 打分 |
| 7 | 📘 Readme | 选择一个文件 —— 自动生成专业 README |
| 8 | 💬 Commit | 分析 git diff 并建议规范化 commit message |
| 9 | 🧪 Scan | 汇总所有 staged/unstaged 改动并建议下一步 |
| 10 | 📝 Promote | 读取最新 brainstorm 输出 → 转换为 `todo_list.txt` 任务 |
| 0 | 🚪 Exit | 返回主 REPL |

### 用法

```
[myproject] ❯ /ssj

╭─ SSJ Developer Mode ⚡ ─────────────────────────
│
│   1.  💡  Brainstorm — Multi-persona AI debate
│   2.  📋  Show TODO  — View todo_list.txt
│   3.  👷  Worker     — Auto-implement pending tasks
│   4.  🧠  Debate     — Expert debate on a file
│   5.  ✨  Propose    — AI improvement for a file
│   6.  🔎  Review     — Quick file analysis
│   7.  📘  Readme     — Auto-generate README.md
│   8.  💬  Commit     — AI-suggested commit message
│   9.  🧪  Scan       — Analyze git diff
│  10.  📝  Promote    — Idea to tasks
│   0.  🚪  Exit SSJ Mode
│
╰──────────────────────────────────────────────

  ⚡ SSJ » 1
  Topic (Enter for general): cheetahclaws plugin system

  # → Brainstorm 启动，结果保存到 brainstorm_outputs/，并生成 todo_list.txt
  # → 每个操作后菜单会自动重新打开

  ⚡ SSJ » 2
  # → 显示来自 brainstorm_outputs/todo_list.txt 的编号待办任务

  ⚡ SSJ » 3
  Task # (Enter for all, or e.g. 1,4,6): 2
  # → Worker 实现任务 #2 并将其标记为 done
```

### 斜杠命令透传

在 `⚡ SSJ »` 提示符下输入任意 `/command`，会直接透传给 REPL：

```
  ⚡ SSJ » /model gpt-4o
  # → 切换模型，然后重新打开 SSJ 菜单

  ⚡ SSJ » /exit
  # → 立即退出 cheetahclaws
```

### Worker 命令

`/worker`（也可通过 SSJ 菜单第 3 项进入）会读取 `brainstorm_outputs/todo_list.txt` 并自动实现每个待办任务：

```
[myproject] ❯ /worker
  ✓ Worker starting — 3 task(s) to implement
    1. ○ Add animated brainstorm spinner
    2. ○ Implement Telegram typing indicator
    3. ○ Write SSJ demo GIF for README

  ── Worker (1/3): Add animated brainstorm spinner ──
  [model reads code, implements the change, marks task done]

[myproject] ❯ /worker 2,3
  # 仅实现任务 2 和 3

[myproject] ❯ /worker --path docs/tasks.md
  # 使用自定义 todo 文件

[myproject] ❯ /worker --workers 2
  # 本次仅处理前 2 个待办任务
```

**智能路径检测** —— 若你误把 brainstorm 输出文件（`.md`）传给 Worker，它会识别这一情况，并提示将其重定向到同目录下对应的 `todo_list.txt`。如果该文件还不存在，它会先建议从 brainstorm 输出中生成 `todo_list.txt`（SSJ Promote），然后自动运行 Worker。

### Debate 命令

SSJ 菜单第 4 项会对任意文件运行结构化多轮专家辩论：

```
  ⚡ SSJ » 4

  Files in brainstorm_outputs/:
    1. brainstorm_20260406_143022.md
    2. cheetahclaws.py

  File to debate #: 2
  Number of debate agents (Enter for 2): 3
  ℹ Debate result will be saved to: cheetahclaws_debate_143055.md

⚔️  Assembling expert panel...
  Expert 1: 🏗️ Architecture Lead — focus: system design & modularity
  Expert 2: 🔐 Security Engineer — focus: attack surface & input validation
  Expert 3: ⚡ Performance Specialist — focus: latency & memory usage

⚔️  Round 1/5 — Expert 1 thinking...
  [Architecture Lead gives opening argument...]

💬  Round 1/5 — Expert 2 formulating...
  [Security Engineer responds...]
  ...

📜  Drafting final consensus...
  [model writes consensus + saves transcript]
✓ Debate complete. Saved to cheetahclaws_debate_143055.md
```

- Agent 数可配置（最少 2，默认 2）。总轮数设为 `agents × 2 − 1`，以形成完整开场-回应-收束结构。
- 动画 spinner 会显示当前轮次和专家（`⚔️ Round 2/3 — Expert 1 thinking...`），一旦该专家开始输出即停止转动。
- 完整辩论记录与排序后的共识会保存到 **与被辩论文件同目录下** 的 `<filename>_debate_HHMMSS.md`。

---

## Telegram 桥接

`/telegram` 会将 cheetahclaws 变成一个 Telegram bot —— 从手机接收消息、以完整工具能力运行模型并自动回复。

<div align=center>
<img src="../media/demos/telegram_demo.gif" width="850"/>
</div>

### 设置（一次性）

1. 在 Telegram 中打开 [@BotFather](https://t.me/BotFather) → `/newbot` → 复制 token。
2. 给你的新 bot 发任意消息（例如 “hi”），然后在浏览器中打开下面这个 URL —— 将 `<TOKEN>` 替换为你的真实 token：

```
https://api.telegram.org/bot<TOKEN>/getUpdates
```

返回结果是 JSON。找到其中的 `"chat"` → `"id"` —— 这个数字就是你的 chat ID：

```json
{
  "ok": true,
  "result": [
    {
      "update_id": 100000001,
      "message": {
        "from": { "id": 987654321, "first_name": "Zhang" },
        "chat": {
          "id": 987654321,
          "type": "private"
        },
        "text": "hi"
      }
    }
  ]
}
```

> **提示：** 如果 `result` 为空，回到 Telegram 再给 bot 发一条消息，然后刷新这个 URL。

3. 配置 cheetahclaws（以下使用上面的示例值）：

```
[myproject] ❯ /telegram 7812345678:AAFxyz123abcDEF456ghiJKL789 987654321
  ✓ Telegram config saved.
  ✓ Connected to @your_bot_name. Starting bridge...
  ✓ Telegram bridge active. Chat ID: 987654321
  ℹ Send messages to your bot — they'll be processed here.
  ℹ Stop with /telegram stop or send /stop in Telegram.
```

token 和 chat_id 会保存到 `~/.cheetahclaws/config.json`。下次启动时，如果已配置，它会**自动启动** —— 启动画面中会显示 `flags: [telegram]`。

### 工作方式

```
Phone (Telegram)                  cheetahclaws terminal
──────────────────                ──────────────────────────
"List Python files"      →        📩 Telegram: List Python files
                                  [typing indicator sent...]
                                  ⚙ Glob(**/*.py) → 5 files
                                  ⚙ response assembled
                          ←       "agent.py, tools.py, ..."
```

- **Typing indicator** 会在模型处理过程中每 4 秒发送一次，因此聊天体验更自然。
- **未授权发送者** 会收到 `⛔ Unauthorized.`，其消息将被丢弃。
- **斜杠命令透传**：你可以从 Telegram 发送 `/cost`、`/model gpt-4o`、`/clear` 等，它们会直接在 cheetahclaws 中执行。
- **Telegram 上的交互菜单**：带交互提示的命令（如 `/ollama` 模型选择器、`/permission`、`/checkpoint` 恢复）现在会在后台线程中运行，因此轮询循环始终保持可用。菜单选项会作为 Telegram 消息发出，你的下一条回复将被用作选择。
- **从 Telegram 发送 `/stop` 或 `/off`** 会优雅地停止桥接。

### 命令

| 命令 | 说明 |
|---|---|
| `/telegram <token> <chat_id>` | 配置 token + chat_id，并启动桥 |
| `/telegram` | 使用已保存配置启动桥 |
| `/telegram status` | 显示运行状态与 chat_id |
| `/telegram stop` | 停止桥 |

### 自动启动

如果 `~/.cheetahclaws/config.json` 中同时设置了 `telegram_token` 和 `telegram_chat_id`，那么每次启动 cheetahclaws 时桥都会自动启动：

```
╭─ CheetahClaws ────────────────────────────────╮
│  Model:       claude-opus-4-6
│  Permissions: auto   flags: [telegram]
│  Type /help for commands, Ctrl+C to cancel        │
╰───────────────────────────────────────────────────╯
✓ Telegram bridge started (auto). Bot: @your_bot_name
```

---

## 主动后台监控

CheetahClaws v3.05.2 增加了一个 **sentinel 守护线程**，它会在设定的无交互时长之后自动唤醒 agent —— 无需用户再次提示。这支持持续日志监控、行情轮询、定时代码检查等场景。

### 快速开始

```
[myproject] ❯ /proactive 5m
Proactive background polling: ON  (triggering every 300s of inactivity)

[myproject] ❯ keep monitoring the build log and alert me if errors appear

╭─ Claude ● ─────────────────────────
│ Understood. I'll check the build log each time I wake up.

[Background Event Triggered]
╭─ Claude ● ─────────────────────────
│ ⚙ Bash(tail -50 build.log)
│ ✓ → Build failed: ImportError in auth.py line 42
│ **Action needed:** fix the import before the next CI run.
```

### 命令

| 命令 | 说明 |
|---|---|
| `/proactive` | 显示当前状态（开/关及间隔） |
| `/proactive 5m` | 启用 —— 每 5 分钟无交互触发一次 |
| `/proactive 30s` | 启用 —— 每 30 秒无交互触发一次 |
| `/proactive 1h` | 启用 —— 每小时触发一次 |
| `/proactive off` | 关闭 sentinel 轮询 |

时间后缀：`s` = 秒，`m` = 分钟，`h` = 小时。纯整数默认按秒处理。

### 工作机制

- REPL 启动时就会有一个后台 daemon 线程启动（默认暂停）。
- 该线程每秒检查一次距离上次用户或 agent 交互的时间。
- 一旦达到无交互阈值，它就会以唤醒提示调用 agent。
- 主 agent 循环使用的 `threading.Lock` 可确保唤醒事件永远不会打断活跃会话 —— 它们会在当前 turn 完成后排队触发。
- 监视器中的异常会通过 `traceback` 记录，因此故障可见且便于调试。

### 与 SleepTimer 互补

| | `SleepTimer` | `/proactive` |
|---|---|---|
| 谁触发 | Agent | 用户 |
| 触发条件 | 从现在起固定延迟后 | N 秒无交互后 |
| 适用场景 | “10 分钟后再检查一次” | “只要我不说话就持续监控” |

---

## Checkpoint 系统

CheetahClaws 会在每一轮后自动快照你的对话与任何被编辑的文件，因此你始终可以回到更早的状态。

### 工作方式

- **自动快照** —— 每轮结束后，checkpoint 系统会保存当前对话消息、token 计数，以及该轮中所有被写入或编辑文件的 copy-on-write 备份。
- **100 快照滑动窗口** —— 当达到上限时，较旧快照会被自动淘汰。
- **节流** —— 如果自上次快照以来没有任何变化（无新消息、无文件编辑），则会跳过本次快照。
- **初始快照** —— 会在会话开始时创建，因此你总能回滚到一个干净的起始状态。
- **存储位置** —— `~/.nano_claude/checkpoints/<session_id>/`（快照元数据 + 备份文件）。

### 命令

| 命令 | 说明 |
|---|---|
| `/checkpoint` | 列出当前会话所有快照 |
| `/checkpoint <id>` | 回滚：将文件恢复到快照 `<id>` 时状态，并将对话裁剪到该点 |
| `/checkpoint clear` | 删除当前会话的全部快照 |
| `/rewind` | `/checkpoint` 的别名 |

### 示例

```
[myproject] ❯ /checkpoint
  Checkpoints (4 total):
  #1  [turn 0] 14:02:11  "(initial state)"           0 files
  #2  [turn 1] 14:03:45  "Create app.py"              1 file
  #3  [turn 2] 14:05:12  "Add error handling"         1 file
  #4  [turn 3] 14:06:30  "Explain the code"           1 file

[myproject] ❯ /checkpoint 2
  Rewound to checkpoint #2 (turn 1)
  Restored: app.py
  Conversation trimmed to 2 messages.
```

---

## 计划模式

计划模式是一种处理复杂多文件任务的结构化工作流：Claude 先在只读阶段分析代码库并写出明确计划，然后由用户审阅后再开始实现。

### 工作方式

在计划模式下：
- **只允许读取**（`Read`、`Glob`、`Grep`、`WebFetch`、`WebSearch`、安全的 `Bash` 命令）。
- **除专用计划文件外，所有写操作都会被阻止**，计划文件位置为 `.nano_claude/plans/<session_id>.md`。
- 被阻止的写操作会显示友好提示，而不是进入确认弹窗。
- 系统提示中会附加计划模式说明。
- 在上下文压缩后，计划文件上下文会自动恢复。

### 斜杠命令工作流

```
[myproject] ❯ /plan add WebSocket support
  Plan mode activated.
  Plan file: .nano_claude/plans/a3f9c1b2.md
  Reads allowed. All other writes blocked (except plan file).

[myproject] ❯ <describe your task>
  [Claude reads files, builds understanding, writes plan to plan file]

[myproject] ❯ /plan
  # Plan: Add WebSocket support

  ## Phase 1: Create ws_handler.py
  ## Phase 2: Modify server.py to mount the handler
  ## Phase 3: Add tests

[myproject] ❯ /plan done
  Plan mode exited. Permission mode restored to: auto
  Review the plan above and start implementing when ready.

[myproject] ❯ /plan status
  Plan mode: INACTIVE  (permission mode: auto)
```

### Agent 工具工作流（自主）

Claude 可以通过 `EnterPlanMode` 与 `ExitPlanMode` 工具自主进入和退出计划模式 —— 这两个工具在所有权限模式下都会自动批准：

```
User: Refactor the authentication module

Claude: [calls EnterPlanMode(task_description="Refactor auth module")]
  → reads auth.py, users.py, tests/test_auth.py ...
  → writes plan to .nano_claude/plans/...
  [calls ExitPlanMode()]
  → "Here is my plan. Please review and approve before I begin."

User: Looks good, go ahead.
Claude: [implements the plan]
```

### 命令

| 命令 | 说明 |
|---|---|
| `/plan <description>` | 带任务描述进入计划模式 |
| `/plan` | 打印当前计划文件内容 |
| `/plan done` | 退出计划模式并恢复之前权限 |
| `/plan status` | 显示计划模式是否激活 |

---

## 上下文压缩

长对话会被自动压缩，以保持在模型上下文窗口限制内。

**两层机制：**

1. **Snip** —— 若干轮之前的旧工具输出（文件读取、bash 结果）会被截断。快速、无 API 成本。
2. **自动压缩** —— 当 token 使用量超过上下文上限的 70% 时，较早消息会被模型总结成简洁回顾。

这一切都是透明发生的。你无需手动操作。

**手动压缩** —— 你也可以随时用 `/compact` 触发压缩。可选 focus 字符串可告诉总结器优先保留哪些上下文：

```
[myproject] ❯ /compact
  Compacted: ~12400 → ~3200 tokens (~9200 saved)

[myproject] ❯ /compact keep the WebSocket implementation details
  Compacted: ~11800 → ~3100 tokens (~8700 saved)
```

若计划模式处于激活状态，任何压缩后计划文件上下文都会自动恢复。

---

## 差异视图

当模型编辑或覆盖文件时，你会看到 git 风格的 diff：

```diff
  Changes applied to config.py:

--- a/config.py
+++ b/config.py
@@ -12,7 +12,7 @@
     "model": "claude-opus-4-6",
-    "max_tokens": 8192,
+    "max_tokens": 16384,
     "permission_mode": "auto",
```

绿色行 = 新增，红色行 = 删除。新建文件则会显示摘要。

---

## CLAUDE.md 支持

在你的项目中放置一个 `CLAUDE.md` 文件，即可为模型提供关于代码库的持久上下文。CheetahClaws 会自动发现并注入它到系统提示中。

```
~/.claude/CLAUDE.md          # 全局 —— 适用于所有项目
/your/project/CLAUDE.md      # 项目级 —— 从 cwd 向上查找
```

**示例 `CLAUDE.md`：**

```markdown
# Project: FastAPI Backend

## Stack
- Python 3.12, FastAPI, PostgreSQL, SQLAlchemy 2.0, Alembic
- Tests: pytest, coverage target 90%

## Conventions
- Format with black, lint with ruff
- Full type annotations required
- New endpoints must have corresponding tests

## Important Notes
- Never hard-code credentials — use environment variables
- Do not modify existing Alembic migration files
- The `staging` branch deploys automatically to staging on push
```

---

## 会话管理

### 存储布局

每次退出时会自动保存到三个位置：

```
~/.cheetahclaws/sessions/
├── history.json                          ← 主历史：所有会话（有上限）
├── mr_sessions/
│   └── session_latest.json              ← 始终是最近一次会话（/resume）
└── daily/
    ├── 2026-04-05/
    │   ├── session_110523_a3f9.json     ← 按天存储的文件，保留较新的
    │   └── session_143022_b7c1.json
    └── 2026-04-04/
        └── session_183100_3b4c.json
```

每个会话文件都包含元数据：

```json
{
  "session_id": "a3f9c1b2",
  "saved_at": "2026-04-05 11:05:23",
  "turn_count": 8,
  "messages": [...]
}
```

### 退出时自动保存

每次退出 —— 通过 `/exit`、`/quit`、`Ctrl+C` 或 `Ctrl+D` —— 会话都会自动保存：

```
✓ Session saved → /home/.../.cheetahclaws/sessions/mr_sessions/session_latest.json
✓              → /home/.../.cheetahclaws/sessions/daily/2026-04-05/session_110523_a3f9.json  (id: a3f9c1b2)
✓   history.json: 12 sessions / 87 total turns
```

### 快速恢复

要继续上次会话：

```bash
cheetahclaws
[myproject] ❯ /resume
✓  Session loaded from …/mr_sessions/session_latest.json (42 messages)
```

恢复指定文件：

```bash
/resume session_latest.json          # 从 mr_sessions/ 加载
/resume /absolute/path/to/file.json  # 从绝对路径加载
```

### 手动保存 / 加载

```bash
/save                          # 使用自动命名保存（session_TIMESTAMP_ID.json）
/save debug_auth_bug           # 按指定文件名保存到 ~/.cheetahclaws/sessions/

/load                          # 交互式按日期分组列表
/load debug_auth_bug           # 按文件名加载
```

**`/load` 的交互式列表：**

```
  ── 2026-04-05 ──
  [ 1] 11:05:23  id:a3f9c1b2  turns:8   session_110523_a3f9.json
  [ 2] 09:22:01  id:7e2d4f91  turns:3   session_092201_7e2d.json

  ── 2026-04-04 ──
  [ 3] 22:18:00  id:3b4c5d6e  turns:15  session_221800_3b4c.json

  ── Complete History ──
  [ H] Load ALL history  (3 sessions / 26 total turns)  /home/.../.cheetahclaws/sessions/history.json

  Enter number(s) (e.g. 1 or 1,2,3), H for full history, or Enter to cancel >
```

- 输入单个编号以加载一个会话
- 输入逗号分隔的编号（如 `1,3`）以按顺序合并多个会话
- 输入 `H` 加载完整历史 —— 会在确认前显示消息数和 token 估计

### 可配置上限

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `session_daily_limit` | `5` | `daily/` 中每天保留的最大会话文件数 |
| `session_history_limit` | `100` | `history.json` 中保留的最大会话数 |

```bash
/config session_daily_limit=10
/config session_history_limit=200
```

### history.json —— 全部对话历史

`history.json` 会将每个会话累积到一起，因此可以搜索完整对话历史或分析使用模式：

```json
{
  "total_turns": 150,
  "sessions": [
    {"session_id": "a3f9c1b2", "saved_at": "2026-04-05 11:05:23", "turn_count": 8, "messages": [...]},
    {"session_id": "7e2d4f91", "saved_at": "2026-04-05 09:22:01", "turn_count": 3, "messages": [...]}
  ]
}
```

---

## 云同步（GitHub Gist）

CheetahClaws v3.05.3 增加了通过 **GitHub Gist** 进行会话云备份的可选功能。会话以私有 Gist（JSON）形式保存，可在 GitHub UI 中浏览。无需额外依赖 —— 使用 Python 标准库 `urllib`。

### 设置（一次性）

1. 进入 [github.com/settings/tokens](https://github.com/settings/tokens) → **Generate new token (classic)**
2. 勾选 **`gist`** 权限
3. 复制 token 并运行：

```
[myproject] ❯ /cloudsave setup ghp_xxxxxxxxxxxxxxxxxxxx
✓ GitHub token saved (logged in as: Chauncygu). Cloud sync is ready.
```

### 上传会话

```
[myproject] ❯ /cloudsave
Uploading session to GitHub Gist…
✓ Session uploaded → https://gist.github.com/abc123def456
```

可添加可选描述：

```
[myproject] ❯ /cloudsave push auth refactor debug session
```

### 退出时自动同步

```
[myproject] ❯ /cloudsave auto on
✓ Auto cloud-sync ON — session will be uploaded to Gist on /exit.
```

从此以后，每次 `/exit` 或 `/quit` 都会在关闭前自动上传会话。

### 浏览与恢复

```
[myproject] ❯ /cloudsave list
  Found 3 session(s):
  abc123de…  2026-04-05 11:02  auth refactor debug session
  7f9e12ab…  2026-04-04 22:18  proactive monitoring test
  3b4c5d6e…  2026-04-04 18:31

[myproject] ❯ /cloudsave load abc123de...full-gist-id...
✓ Session loaded from Gist (42 messages).
```

### 命令参考

| 命令 | 说明 |
|---|---|
| `/cloudsave setup <token>` | 保存 GitHub token（需要 `gist` 权限） |
| `/cloudsave` | 将当前会话上传到新的或已有 Gist |
| `/cloudsave push [desc]` | 带可选描述上传 |
| `/cloudsave auto on\|off` | 切换退出时是否自动上传 |
| `/cloudsave list` | 列出所有 cheetahclaws Gists |
| `/cloudsave load <gist_id>` | 下载并恢复一个会话 |

---

## 项目结构

```
cheetahclaws/
├── cheetahclaws.py        # 入口：REPL + slash commands + diff 渲染 + Rich Live 流式输出 + proactive sentinel daemon + SSJ 模式 + Telegram 桥 + Worker 命令
├── agent.py              # Agent 循环：流式、工具调度、压缩
├── providers.py          # 多 provider：Anthropic、OpenAI-compatible streaming
├── tools.py              # 核心工具（Read/Write/Edit/Bash/Glob/Grep/Web/NotebookEdit/GetDiagnostics）+ 注册器接线
├── tool_registry.py      # 工具插件注册表：注册、查找、执行
├── compaction.py         # 上下文压缩：snip + 自动总结
├── context.py            # 系统提示构造：CLAUDE.md + git + memory
├── config.py             # 配置加载/保存/默认值；DAILY_DIR、SESSION_HIST_FILE 路径
├── cloudsave.py          # GitHub Gist 云同步（上传/下载/列出会话）
│
├── multi_agent/          # 多智能体包
│   ├── __init__.py       # 重新导出
│   ├── subagent.py       # AgentDefinition、SubAgentManager、worktree 帮助函数
│   └── tools.py          # Agent、SendMessage、CheckAgentResult、ListAgentTasks、ListAgentTypes
├── subagent.py           # 向后兼容 shim → multi_agent/
│
├── memory/               # 记忆包
│   ├── __init__.py       # 重新导出
│   ├── types.py          # MEMORY_TYPES 和格式说明
│   ├── store.py          # save/load/delete/search、MEMORY.md 索引重建
│   ├── scan.py           # MemoryHeader、年龄/新鲜度帮助函数
│   ├── context.py        # get_memory_context()、截断、AI 搜索
│   └── tools.py          # MemorySave、MemoryDelete、MemorySearch、MemoryList
├── memory.py             # 向后兼容 shim → memory/
│
├── skill/                # Skill 包
│   ├── __init__.py       # 重新导出；导入 builtin 以注册内置 skills
│   ├── loader.py         # SkillDef、parse、load_skills、find_skill、substitute_arguments
│   ├── builtin.py        # 内置 skills：/commit、/review
│   ├── executor.py       # execute_skill()：inline 或 forked sub-agent
│   └── tools.py          # Skill、SkillList
├── skills.py             # 向后兼容 shim → skill/
│
├── mcp/                  # MCP（模型上下文协议）包
│   ├── __init__.py       # 重新导出
│   ├── types.py          # MCPServerConfig、MCPTool、MCPServerState、JSON-RPC 帮助函数
│   ├── client.py         # StdioTransport、HttpTransport、MCPClient、MCPManager
│   ├── config.py         # 加载 .mcp.json（项目）+ ~/.cheetahclaws/mcp.json（用户）
│   └── tools.py          # 自动发现并将 MCP 工具注册到 tool_registry
│
├── voice/                # 语音输入包（v3.05）
│   ├── __init__.py       # 公共 API：check_voice_deps、voice_input
│   ├── recorder.py       # 音频捕获：sounddevice → arecord → sox rec
│   ├── stt.py            # STT：faster-whisper → openai-whisper → OpenAI API
│   └── keyterms.py       # 来自 git branch + 项目文件的编码词汇
│
├── checkpoint/           # Checkpoint 系统（v3.05.6）
│   ├── __init__.py       # 公共 API 导出
│   ├── types.py          # FileBackup + Snapshot dataclasses；MAX_SNAPSHOTS = 100
│   ├── store.py          # 文件级备份、快照持久化、回滚、清理
│   └── hooks.py          # Write/Edit/NotebookEdit 拦截 —— 修改前备份文件
│
└── tests/                # 263+ 单元测试
    ├── test_mcp.py
    ├── test_memory.py
    ├── test_skills.py
    ├── test_subagent.py
    ├── test_tool_registry.py
    ├── test_compaction.py
    ├── test_diff_view.py
    ├── test_voice.py         # 29 个语音测试（无需硬件）
    ├── test_checkpoint.py    # 24 个 checkpoint 单元测试
    ├── e2e_checkpoint.py     # 10 步 checkpoint 生命周期测试
    ├── e2e_plan_mode.py      # 10 步计划模式权限测试
    ├── e2e_plan_tools.py     # 8 步 EnterPlanMode/ExitPlanMode 工具测试
    ├── e2e_compact.py        # 9 步压缩测试
    └── e2e_commands.py       # 9 步 /init /export /copy /status 测试
```

> **给开发者：** 每个功能包（`multi_agent/`、`memory/`、`skill/`、`mcp/`、`voice/`、`checkpoint/`）都是自包含的。只需在任意被 `tools.py` 导入的模块中调用 `register_tool(ToolDef(...))`，即可添加自定义工具。

---

## FAQ

**Q：如何添加一个 MCP server？**

方式 1 —— 通过 REPL（stdio server）：
```
/mcp add git uvx mcp-server-git
```

方式 2 —— 在项目中创建 `.mcp.json`：
```json
{
  "mcpServers": {
    "git": {"type": "stdio", "command": "uvx", "args": ["mcp-server-git"]}
  }
}
```

然后运行 `/mcp reload` 或重启程序。使用 `/mcp` 可查看连接状态。

**Q：某个 MCP server 显示错误，如何调试？**

```
/mcp                    # 显示每个 server 的错误信息
/mcp reload git         # 尝试重新连接
```

如果 server 使用 stdio，请确保命令在你的 `$PATH` 中：
```bash
which uvx               # 应输出一个路径
uvx mcp-server-git      # 手动运行查看错误
```

**Q：能否使用需要认证的 MCP server？**

对于带 Bearer token 的 HTTP/SSE server：
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

对于使用环境变量认证的 stdio server：
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

**Q：我的本地 Ollama 模型无法进行工具调用。**

并非所有模型都支持 function calling。请使用推荐的支持工具调用模型：`qwen2.5-coder`、`llama3.3`、`mistral` 或 `phi4`。

```bash
ollama pull qwen2.5-coder
cheetahclaws --model ollama/qwen2.5-coder
```

**Q：如何连接到运行 vLLM 的远程 GPU server？**

```
/config custom_base_url=http://your-server-ip:8000/v1
/config custom_api_key=your-token
/model custom/your-model-name
```

**Q：如何查看 API 成本？**

```
/cost

  Input tokens:  3,421
  Output tokens:   892
  Est. cost:     $0.0648 USD
```

**Q：能否在同一会话中使用多个 API key？**

可以。你只需事先设置好所有需要的 key（通过环境变量或 `/config`）。然后可以自由切换模型 —— 每次调用都会自动使用当前 provider 对应的 key。

**Q：如何让某个模型在所有项目中可用？**

将 key 加入 `~/.bashrc` 或 `~/.zshrc`。在 `~/.cheetahclaws/config.json` 中设置默认模型：

```json
{ "model": "claude-sonnet-4-6" }
```

**Q：Qwen / Zhipu 返回乱码。**

请确认你的 `DASHSCOPE_API_KEY` / `ZHIPU_API_KEY` 正确，且账户有足够额度。两家 provider 都使用 UTF-8，并且对中文支持良好。

**Q：可以把输入通过管道传给 cheetahclaws 吗？**

```bash
echo "Explain this file" | cheetahclaws --print --accept-all
cat error.log | cheetahclaws -p "What is causing this error?"
```

**Q：如何把它当作任意目录都可用的 CLI 工具？**

使用 `uv tool install` —— 它会创建隔离环境并把 `cheetahclaws` 放入 PATH：

```bash
cd cheetahclaws
uv tool install .
```

之后你就可以在任意目录运行 `cheetahclaws`。拉取更新后，运行 `uv tool install . --reinstall` 进行更新。

**Q：如何设置语音输入？**

```bash
# 最小配置（本地、离线、无需 API key）：
pip install sounddevice faster-whisper numpy

# 然后在 REPL 中：
/voice status          # 检查后端是否被识别
/voice                 # 开始说话
```

首次使用时，`faster-whisper` 会自动下载 `base` 模型（约 150 MB）。
如需更高精度模型：`export NANO_CLAUDE_WHISPER_MODEL=small`

**Q：语音输入经常转错我的话（尤其漏掉编码术语）。**

关键词增强器已经会自动注入来自 git 分支和项目文件的编码词汇。
如果有固定领域术语，可把它们写入 `.cheetahclaws/voice_keyterms.txt`（每行一个词）—— 每次录音都会自动读取它。

**Q：语音输入支持中文 / 日语 / 其他语言吗？**

支持。录音前设置语言即可：

```
/voice lang zh    # 普通话中文
/voice lang ja    # 日语
/voice lang auto  # 恢复自动识别（默认）
```

Whisper 支持 99 种语言。`auto` 对大多数情况都有效，但对较短语音，显式指定语言代码会更准确。
