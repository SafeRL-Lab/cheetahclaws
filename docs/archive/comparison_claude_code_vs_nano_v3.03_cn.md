# Claude Code vs CheetahClaws v3.03：对比分析

## 整体概览

| 维度 | Claude Code (TypeScript) | CheetahClaws (Python) |
|------|--------------------------|---------------------------|
| **语言** | TypeScript + React/Ink | Python 3.8+ |
| **文件数量** | 1,332 个 TS/TSX 文件 | 51 个 Python 文件 |
| **代码规模** | ~283K 行 | ~10.2K 行 |
| **工具数量** | 44+ 工具 | 21 工具 |
| **Slash 命令** | 88 个 | 17 个 |
| **模型支持** | 仅 Anthropic | 7+ 提供商 |

---

## 架构对比

### Agent Loop（核心循环）

**Claude Code** — `QueryEngine.ts` (~47KB)
- 复杂的消息流管理，支持 extended thinking
- 多层权限检查系统（MDM、团队管理、per-tool gates）
- 内置上下文压缩调度器

**CheetahClaws** — `agent.py` (174 行)
- 生成器模式（`yield TextChunk | ToolStart | ToolEnd`）
- 清晰的事件流，代码极易阅读
- 合作式取消支持（sub-agents 用）

### 工具系统

**Claude Code**
- 每个工具独立目录，有完整 schema、权限说明、测试
- 工具之间有复杂依赖关系（EnterWorktree → 隔离环境 → ExitWorktree）
- `SyntheticOutputTool`、`RemoteTriggerTool` 等高级工具

**CheetahClaws**
- `ToolDef` 注册表模式，任何模块可 `register_tool()` 动态注入
- `read_only` / `concurrent_safe` 标志驱动自动权限决策
- 输出截断（32KB 上限）防止上下文膨胀

### UI 渲染

**Claude Code**：React + Ink，完整组件树，有 diff 可视化、对话框、进度条、流式渲染

**CheetahClaws**：`rich` 库，Markdown 语法高亮，简单但够用

---

## 优劣势分析

### Claude Code 优势
1. **工程完整性** — 88 个 slash 命令，44 工具，企业级权限管理（MDM/团队配置）
2. **UI 体验** — React/Ink 组件树，流式渲染质量高，diff 可视化精细
3. **服务层厚度** — LSP 集成、Analytics/Telemetry、OAuth、GrowthBook 特性开关
4. **会话内存** — AI 驱动的记忆提取（`extractMemories` 服务）
5. **生产可靠性** — 单一打包 `cli.js`（12MB），测试覆盖全面，版本冻结

### Claude Code 劣势
1. **单一提供商** — 仅 Anthropic API，无法接 OpenAI/Gemini/本地模型
2. **代码可读性差** — 1,332 文件 + decompiled 代码，逻辑分散，难以修改
3. **构建依赖重** — Bun + esbuild + TypeScript，改动需要完整构建链
4. **封闭性** — 功能开关（`internal-only modules`）在编译时死代码消除，外部无法扩展

### CheetahClaws 优势
1. **多提供商** — Anthropic/OpenAI/Gemini/Kimi/Qwen/DeepSeek/Ollama 自动识别
2. **极度可读** — 51 文件，10K 行，架构一眼看穿，适合学习和研究
3. **零构建** — 纯 Python，`pip install` 即用，改完即生效
4. **动态扩展** — `register_tool()` 运行时注入，Plugin 系统支持 git URL 安装
5. **Markdown Skills** — `~/.cheetahclaws/skills/*.md` 可自定义技能，无需改代码
6. **任务依赖图** — `task/store.py` 有 `blocks/blocked_by` 依赖追踪（Claude Code 没有）

### CheetahClaws 劣势
1. **UI 功能薄** — 无 diff 可视化、无对话框系统、无进度条组件
2. **工具覆盖少** — 缺少 `WebSearch`（需要 API key）、`NotebookEdit`、`LSP Diagnostics`
3. **安全体系弱** — 无 MDM 管控、无团队权限同步、无 keychain 集成
4. **性能** — Python 启动慢，大文件处理比 Node.js 慢
5. **Sub-agent 不成熟** — `subagent.py` 仍有 stub，多智能体协作不如原版稳定

---

## 关键设计差异

### 上下文压缩策略

| | Claude Code | CheetahClaws |
|-|-------------|-----------------|
| 触发条件 | 精确 token 计数 | `len/3.5` 估算触发 70% |
| 压缩层 | 单层 AI 摘要 | 双层：Snip（规则）+ AI 摘要 |
| 保留策略 | 系统级调度 | `preserve_last_n_turns` 参数 |

### 内存系统

Claude Code 的 `extractMemories` 是 AI 主动提取对话中的知识点；Nano 的 `memory/` 是工具驱动的显式存储（`MemorySave` 工具调用），更可控但需要模型主动使用。

### 模块结构

**Claude Code**（服务层）：
```
services/
├── api/             # Claude API、OAuth、analytics
├── mcp/             # Model Context Protocol 客户端
├── plugins/         # 插件安装/执行
├── SessionMemory/   # 持久化会话状态
├── compact/         # 上下文压缩
├── lsp/             # Language Server Protocol
├── extractMemories/ # AI 驱动的记忆提取
└── MagicDocs/       # 文档生成
```

**CheetahClaws**（扁平包结构）：
```
mcp/          # MCP 客户端（stdio/SSE/HTTP）
memory/       # 双作用域持久化内存
task/         # 含依赖图的任务管理
skill/        # Markdown 技能加载器
multi_agent/  # 子智能体生命周期 + git worktrees
plugin/       # 插件加载/推荐系统
```

---

## 支持的模型

**Claude Code**：仅 Anthropic（`claude-opus-4-6`、`claude-sonnet-4-6`、`claude-haiku-4-5`）

**CheetahClaws**：

| 提供商 | 支持模型 |
|--------|----------|
| Anthropic | claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5 |
| OpenAI | gpt-4o, gpt-4o-mini, o1, o3-mini |
| Google | gemini-2.5-pro, gemini-2.0-flash, gemini-1.5-pro |
| 月之暗面 (Kimi) | kimi-latest |
| 阿里云 (Qwen) | qwen-max, qwen-plus |
| 智谱 (GLM) | glm-4 |
| DeepSeek | deepseek-chat, deepseek-reasoner |
| Ollama（本地） | llama3.3, qwen2.5-coder, deepseek-r1, mistral, phi4, codellama |

---

## 总结

CheetahClaws 是对 Claude Code 核心理念的**极简复现**，在多提供商支持和代码可读性上超越了原版，非常适合作为研究基础或个人工具。Claude Code 则是完整的工程产品，在 UI、安全、企业功能上不可比拟。

两者之间最值得填补的 gap：

| 缺口 | 优先级 | 备注 |
|------|--------|------|
| WebSearch 工具 | 高 | 需要搜索 API 后端 |
| Diff 可视化渲染 | 中 | Rich 库有 diff 支持 |
| Sub-agent 稳定性 | 中 | `subagent.py` stub 需完善 |
| LSP 诊断集成 | 低 | 需要各语言 language server |
| Keychain 集成 | 低 | Python 有 `keyring` 库可用 |
