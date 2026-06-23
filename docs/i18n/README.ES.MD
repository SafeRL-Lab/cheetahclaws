Español | [中文](README.CN.MD) | [Français](README.FR.MD) | [한국어](README.KO.MD) | [日本語](README.JP.MD) | [Deutsch](README.DE.MD) | [Português](README.ES.MD)

> ⚠️ Esta traducción puede estar desactualizada. Para la versión más reciente y precisa, consulta el [README](../../README.md) en inglés.

<br>

<div align="center">
  <a href="[https://github.com/SafeRL-Lab/Robust-Gymnasium](https://github.com/SafeRL-Lab/cheetahclaws)">
    <img src="../media/logos/logo-5.png" alt="Logo" width="280"> 
  </a>


<h2 align="center" style="font-size: 30px;"><strong><em>CheetahClaws (Nano Claude Code)</em></strong>: Un asistente personal de IA nativo en Python, rápido y fácil de usar, para cualquier modelo, inspirado en OpenClaw y Claude Code, diseñado para trabajar por ti de forma autónoma 24/7</h2>
<p align="center">
    <a href="https://github.com/chauncygu/collection-claude-code-source-code">La fuente más reciente de Claude Code</a>
    ·
    <a href="https://github.com/SafeRL-Lab/cheetahclaws/issues">Issues</a>
  ·
    <a href="https://deepwiki.com/SafeRL-Lab/cheetahclaws">Introducción breve</a>

  </p>
</div>

 <div align=center>
 <img src="../media/demos/demo.gif" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline">Ejecución de tareas</center>
 </div>


---

  <div align=center>
 <img src="../media/demos/brainstorm_demo.gif" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline">Modo Brainstorm: lluvia de ideas multiagente</center>
 </div>



---

  <div align=center>
 <img src="../media/demos/proactive_demo.gif" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline">Modo proactivo: agente autónomo</center>
 </div>

---

  <div align=center>
 <img src="../media/demos/ssj_demo.gif" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline">Modo de desarrollador SSJ: flujo de trabajo con menú avanzado</center>
 </div>

---

  <div align=center>
 <img src="../media/demos/telegram_demo.gif" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline">Puente de Telegram: controla cheetahclaws desde tu teléfono</center>
 </div>

---



## 🔥🔥🔥 Novedades (hora del Pacífico)


- 07 de abril de 2026 (**v3.5.53**): **soporte de fotos/voz en Telegram, eliminación del árbol de procesos al expirar Bash, sugerencias de shell para Windows, corrección del worker**
  - **Visión con fotos de Telegram** — envía una foto al puente de Telegram y CheetahClaws la describirá usando el modelo de visión activo (GPT-4o, Gemini 2.0 Flash, Claude, etc.). El bot descarga la versión de mayor resolución, la codifica en Base64 y la enruta por la misma ruta `_pending_image` que `/img`. El texto del pie de foto (o un prompt por defecto, “describe esta imagen”) se reenvía junto con la imagen.
  - **STT para voz/audio de Telegram** — envía un mensaje de voz o un archivo de audio al puente de Telegram y CheetahClaws lo transcribirá automáticamente. Las notas de voz en OGG se convierten a PCM mediante `ffmpeg` y se pasan al backend local de Whisper; si `ffmpeg` no está disponible, se usa como alternativa la API Whisper de OpenAI. La transcripción se devuelve al chat antes de enviarse como consulta.
  - **Eliminación del árbol de procesos al expirar Bash** — cuando un comando `Bash` excede el tiempo límite, CheetahClaws ahora elimina todo el árbol de procesos hijo en lugar de solo la shell. En Unix, `os.killpg` envía `SIGKILL` al grupo de procesos; en Windows, `taskkill /F /T` termina todos los procesos hijo. Las aplicaciones GUI (por ejemplo, juegos PyQt lanzados por el agente) ya no dejan procesos zombis tras un timeout. La implementación interna usa `start_new_session=True` en lugar de `preexec_fn=os.setsid` por seguridad en entornos con hilos.
  - **El worker ejecuta todas las tareas pendientes por defecto** — antes, `/worker` procesaba solo 1 tarea por sesión (un bug). Ahora ejecuta todas las tareas pendientes por defecto. La bandera `--workers N` sigue limitando el tamaño del lote cuando hace falta.
  - **Sugerencias de shell de Windows en el prompt del sistema** — los modelos no-Claude ahora reciben una chuleta específica de Windows en el prompt del sistema (`type` vs `cat`, `dir /s /b` vs `find`, `del` vs `rm`, etc.) para que el agente genere comandos correctos en Windows sin guía manual.
  - **Sugerencias para timeouts de Bash** — la descripción de la herramienta `Bash` ahora aconseja al modelo usar `timeout=120–300` para comandos lentos (`npm install`, `npx`, `pip install`, compilaciones), reduciendo falsos timeouts de 30 segundos en operaciones de paquetes.
  - **Corrección: el prompt de eventos en segundo plano muestra el cwd real** — el re-prompt amarillo que aparecía tras finalizar un evento en segundo plano estaba fijado a `[claude-code-local]`; ahora muestra el nombre real del directorio de trabajo (`[{cwd.name}]`), consistente con el prompt principal del REPL.

- 06 de abril de 2026 (**v3.5.53**): **menús interactivos de Telegram, alias `/img`, `/voice device`, soporte de visión para OpenAI/Gemini**
  - **Menús interactivos de Telegram corregidos** — los comandos slash con entrada interactiva (por ejemplo, `/ollama`, `/permission`, `/checkpoint`) bloqueaban el bucle de sondeo de Telegram, impidiendo responder a los menús. Ahora los comandos slash se ejecutan en un hilo daemon (igual que las consultas normales), manteniendo libre el bucle de sondeo. Todos los menús interactivos (`ask_input_interactive`) funcionan correctamente sobre Telegram.
  - **Alias `/img`** — `/img` ahora es un alias de `/image`, para flujos de trabajo más rápidos con imágenes del portapapeles.
  - **`/voice device`** — nuevo subcomando para listar todos los micrófonos de entrada disponibles y seleccionar uno de forma interactiva. El índice del dispositivo elegido se guarda en la configuración de la sesión y se muestra en `/voice status`. Útil en sistemas con múltiples interfaces de audio (por ejemplo, auriculares USB + micrófono integrado).
  - **Soporte de visión para modelos OpenAI / Gemini** — `/img` (y `/image`) ahora envía imágenes en el formato OpenAI multipart `image_url` a modelos de visión en la nube (GPT-4o, Gemini 2.0 Flash, etc.), además del formato nativo existente de Ollama. No se necesita cambiar la configuración: el formato correcto se selecciona automáticamente según el proveedor activo.
  - **Corrección: condición de carrera de hilos** — `_in_telegram_turn` ahora se rastrea mediante `threading.local()` por hilo ejecutor de slash, en lugar de una clave compartida de configuración, eliminando una condición de carrera que podía corromper la bandera cuando llegaba un mensaje normal mientras un comando slash interactivo esperaba entrada.

- 06 de abril de 2026 (**v3.5.52**): **sistema de checkpoints, modo plan, compactación y utilidades, soporte para MiniMax, correcciones de Telegram**
  - **Sistema de checkpoints** (`checkpoint/`): instantáneas automáticas del estado de la conversación y de los cambios en archivos tras cada turno. `/checkpoint` lista todas las instantáneas; `/checkpoint <id>` revierte tanto archivos como historial de conversación a cualquier estado anterior; `/checkpoint clear` elimina todas las instantáneas de la sesión. `/rewind` es un alias. Ventana deslizante de 100 instantáneas; se captura una instantánea inicial al comienzo de la sesión. Limitación: se omite cuando nada ha cambiado. Las copias de seguridad de archivos usan copy-on-write; las instantáneas capturan el estado posterior a la edición.
  - **Modo plan**: `/plan <desc>` entra en un modo de análisis de solo lectura — Claude solo puede leer el código y escribir en un archivo de plan dedicado (`.nano_claude/plans/<session_id>.md`). Todas las demás escrituras se bloquean silenciosamente con un mensaje útil. `/plan` muestra el plan actual; `/plan done` sale del modo plan y restaura los permisos originales; `/plan status` informa si el modo plan está activo. Dos nuevas herramientas del agente — `EnterPlanMode` y `ExitPlanMode` — permiten a Claude entrar y salir autónomamente del modo plan para tareas complejas de múltiples archivos; ambas se aprueban automáticamente en todos los modos de permiso.
  - **`/compact [focus]`**: activa manualmente la compactación de la conversación en cualquier momento. Una cadena de enfoque opcional guía al resumidor LLM sobre qué contexto preservar. Tanto la compactación automática como la manual restauran el contexto del archivo de plan después de compactar.
  - **Comandos utilitarios**: `/init` crea una plantilla `CLAUDE.md` en el directorio actual; `/export [filename]` exporta la conversación como Markdown (por defecto) o JSON; `/copy` copia al portapapeles la última respuesta del asistente (Windows/macOS/Linux); `/status` muestra versión, modelo, proveedor, permisos, ID de sesión, uso de tokens y porcentaje de contexto; `/doctor` diagnostica el estado de la instalación (versión de Python, git, clave API + prueba real de conectividad, dependencias opcionales, presencia de CLAUDE.md, uso de disco de checkpoints, modo de permisos).

- 06 de abril de 2026 (**v3.5.51**): **el proyecto cambió de nombre de Nano Claude Code a CheetahClaws**
  - El proyecto ha sido renombrado de **Nano Claude Code** a **CheetahClaws** — un nombre más distintivo que captura el espíritu de la herramienta: un asistente de programación ágil y afilado. El `Cl` de CheetahClaws es una sutil referencia a Claude.
  - Comando CLI: `nano_claude` → `cheetahclaws`
  - Paquete PyPI: `nano-claude-code` → `cheetahclaws`
  - Directorio de configuración: `~/.nano_claude/` → `~/.clawnest/` → `~/.cheetahclaws/`
  - Punto de entrada principal: `nano_claude.py` → `cheetahclaws.py`
  - Toda la documentación, URLs de GitHub y referencias internas se actualizaron en consecuencia.
  - Se añadió una sección de comparación **CheetahClaws vs OpenClaw** al README.

Para más novedades, consulta [aquí](../news.md)


---

# CheetahClaws

CheetahClaws: una reimplementación en Python de Claude Code, **ligera** y **fácil de usar**, que **admite cualquier modelo**, como Claude, GPT, Gemini, Kimi, Qwen, Zhipu, DeepSeek, MiniMax y modelos locales de código abierto mediante Ollama o cualquier endpoint compatible con OpenAI.

---

## Contenido
  * [Por qué CheetahClaws](#por-qué-cheetahclaws)
  * [CheetahClaws vs OpenClaw](#cheetahclaws-vs-openclaw)
  * [Características](#características)
  * [Modelos compatibles](#modelos-compatibles)
  * [Instalación](#instalación)
  * [Uso: modelos API de código cerrado](#uso-modelos-api-de-código-cerrado)
  * [Uso: modelos de código abierto (local)](#uso-modelos-de-código-abierto-local)
  * [Formato del nombre del modelo](#formato-del-nombre-del-modelo)
  * [Referencia de la CLI](#referencia-de-la-cli)
  * [Comandos slash (REPL)](#comandos-slash-repl)
  * [Configuración de claves API](#configuración-de-claves-api)
  * [Sistema de permisos](#sistema-de-permisos)
  * [Herramientas integradas](#herramientas-integradas)
  * [Memoria](#memoria)
  * [Skills](#skills)
  * [Subagentes](#subagentes)
  * [MCP (Model Context Protocol)](#mcp-model-context-protocol)
  * [Sistema de plugins](#sistema-de-plugins)
  * [Herramienta AskUserQuestion](#herramienta-askuserquestion)
  * [Gestión de tareas](#gestión-de-tareas)
  * [Entrada por voz](#entrada-por-voz)
  * [Brainstorm](#brainstorm)
  * [Modo de desarrollador SSJ](#modo-de-desarrollador-ssj)
  * [Puente de Telegram](#puente-de-telegram)
  * [Monitoreo proactivo en segundo plano](#monitoreo-proactivo-en-segundo-plano)
  * [Sistema de checkpoints](#sistema-de-checkpoints)
  * [Modo plan](#modo-plan)
  * [Compresión de contexto](#compresión-de-contexto)
  * [Vista diff](#vista-diff)
  * [Soporte para CLAUDE.md](#soporte-para-claudemd)
  * [Gestión de sesiones](#gestión-de-sesiones)
  * [Sincronización en la nube (GitHub Gist)](#sincronización-en-la-nube-github-gist)
  * [Estructura del proyecto](#estructura-del-proyecto)
  * [FAQ](#faq)




## Por qué CheetahClaws

Claude Code es un potente asistente de programación con IA listo para producción, pero su código fuente es un paquete compilado de TypeScript/Node.js de 12 MB (~1,300 archivos, ~283K líneas). Está fuertemente acoplado a la API de Anthropic, es difícil de modificar y es imposible ejecutarlo contra un modelo local o alternativo.

**CheetahClaws** reimplementa el mismo bucle central en ~10K líneas de Python legible, conservando todo lo que necesitas y eliminando lo que no. Consulta aquí un análisis más detallado (CheetahClaws v3.03), [versión en inglés](../comparison_claude_code_vs_nano_v3.03_en.md) y [versión en chino](../comparison_claude_code_vs_nano_v3.03_cn.md)

### De un vistazo

| Dimensión | Claude Code (TypeScript) | CheetahClaws (Python) |
|-----------|--------------------------|---------------------------|
| Lenguaje | TypeScript + React/Ink | Python 3.8+ |
| Archivos fuente | ~1,332 archivos TS/TSX | 51 archivos Python |
| Líneas de código | ~283K | ~12K |
| Herramientas integradas | 44+ | 27 |
| Comandos slash | 88 | 36 |
| Entrada por voz | WebSocket propietario de Anthropic (requiere OAuth) | Whisper local / API de OpenAI — funciona sin conexión, sin suscripción |
| Proveedores de modelos | Solo Anthropic | 8+ (Anthropic · OpenAI · Gemini · Kimi · Qwen · DeepSeek · MiniMax · Ollama · …) |
| Modelos locales | No | Sí — Ollama, LM Studio, vLLM, cualquier endpoint compatible con OpenAI |
| Requiere paso de compilación | Sí (Bun + esbuild) | No — ejecútalo directamente con `python cheetahclaws.py` (o instálalo para usar `cheetahclaws`) |
| Extensibilidad en tiempo de ejecución | Cerrada (tiempo de compilación) | Abierta — `register_tool()` en tiempo de ejecución, skills Markdown, plugins git |
| Grafo de dependencias de tareas | No | Sí — relaciones `blocks` / `blocked_by` en el paquete `task/` |

### Dónde gana Claude Code

- **Calidad de interfaz** — árbol de componentes React/Ink con renderizado streaming, visualización diff detallada y sistemas de diálogo.
- **Amplitud de herramientas** — 44 herramientas, incluidas `RemoteTrigger`, `EnterWorktree` y otras más integradas con la interfaz.
- **Funciones empresariales** — configuración gestionada por MDM, sincronización de permisos de equipo, OAuth, almacenamiento en keychain, flags de funciones con GrowthBook.
- **Extracción de memoria impulsada por IA** — el servicio `extractMemories` extrae conocimiento de las conversaciones sin llamadas explícitas a herramientas.
- **Fiabilidad de producción** — un único binario distribuible `cli.js`, cobertura completa de pruebas y versiones bloqueadas.

### Dónde gana CheetahClaws

- **Multiproveedor** — cambia entre Claude, GPT-4o, Gemini 2.5 Pro, DeepSeek, Qwen, MiniMax o un modelo local Llama con `--model` o `/model`, sin recompilar.
- **Soporte para modelos locales** — ejecútalo completamente sin conexión con Ollama, LM Studio o cualquier modelo servido por vLLM.
- **Código fuente legible** — el bucle completo del agente ocupa 174 líneas (`agent.py`). Cualquier desarrollador Python puede leerlo, bifurcarlo y ampliarlo en minutos.
- **Sin build** — `pip install -r requirements.txt` y ya está en marcha. Los cambios surten efecto inmediatamente.
- **Extensibilidad dinámica** — registra nuevas herramientas en tiempo de ejecución con `register_tool(ToolDef(...))`, instala paquetes de skills desde URLs git o conecta cualquier servidor MCP.
- **Grafo de dependencias de tareas** — `TaskCreate` / `TaskUpdate` admiten relaciones `blocks` / `blocked_by` para planificación estructurada de varios pasos (no disponible en Claude Code).
- **Compresión de contexto en dos capas** — recorte por reglas + resumido por IA, configurable mediante `preserve_last_n_turns`.
- **Edición de notebooks** — `NotebookEdit` manipula directamente el JSON de `.ipynb` (reemplazar/insertar/eliminar celdas) sin requerir kernel.
- **Diagnóstico sin servidor LSP** — `GetDiagnostics` encadena pyright → mypy → flake8 → py_compile para Python y tsc/shellcheck para otros lenguajes, sin configuración.
- **Entrada de voz offline** — `/voice` graba vía `sounddevice`/`arecord`/SoX, transcribe con `faster-whisper` local (sin clave API, sin suscripción) y envía automáticamente. Los términos clave tomados de tu rama git y archivos del proyecto mejoran la precisión en términos de programación.
- **Sincronización de sesiones en la nube** — `/cloudsave` respalda conversaciones en Gists privados de GitHub sin dependencias extra; restaura cualquier sesión pasada en cualquier máquina con `/cloudsave load <id>`.
- **Modo de desarrollador SSJ** — `/ssj` abre un menú persistente con 10 atajos de flujo de trabajo: Brainstorm → TODO → Worker, debate entre expertos, revisión de código, generación de README, ayuda para commits y más. Permanece abierto entre acciones; admite passthrough de `/command`.
- **Puente de bot de Telegram** — `/telegram <token> <chat_id>` convierte cheetahclaws en un bot de Telegram: recibe mensajes del usuario, ejecuta el modelo y devuelve respuestas, todo desde tu teléfono. Los comandos slash pasan a través, y un indicador de escritura hace que el chat se sienta vivo.
- **Comando Worker** — `/worker` implementa automáticamente las tareas pendientes de `brainstorm_outputs/todo_list.txt`, marca cada una como hecha al completarla y admite selección de tareas por número (por ejemplo, `1,4,6`).
- **Cierre forzado** — 3× Ctrl+C en 2 segundos activa `os._exit(1)` inmediatamente, desbloqueando cualquier E/S congelada.
- **Monitoreo proactivo en segundo plano** — `/proactive 5m` activa un daemon centinela que despierta al agente automáticamente tras un periodo de inactividad, permitiendo bucles continuos de monitoreo, comprobaciones programadas o bots de trading sin prompts del usuario.
- **Renderizado streaming Live con Rich** — cuando `rich` está instalado, las respuestas se muestran como Markdown que se actualiza en vivo (sin texto crudo duplicado), con intercalado limpio de llamadas a herramientas.
- **Razonamiento nativo de Ollama** — los modelos locales de razonamiento (deepseek-r1, qwen3, gemma4) transmiten sus tokens `<think>` directamente al terminal mediante eventos `ThinkingChunk`; actívalo con `/verbose` y `/thinking`.
- **Visión nativa de Ollama** — `/image [prompt]` captura la imagen del portapapeles y la envía a modelos locales de visión (llava, gemma4, llama3.2-vision) mediante la API nativa de imágenes de Ollama. No requiere nube.
- **Pegado multilínea fiable** — Bracketed Paste Mode (`ESC[?2004h`) agrupa cualquier texto pegado — bloques de código, prompts de varios párrafos, diffs largos — como un único turno, sin latencia y sin artefactos de líneas en blanco.
- **Autocompletado Tab enriquecido** — Tab tras `/` muestra todos los comandos con descripciones de una línea y sugerencias de subcomandos; el autocompletado de subcomandos funciona para `/mcp`, `/plugin`, `/tasks`, `/cloudsave` y más.
- **Checkpoint y rebobinado** — `/checkpoint` lista todas las instantáneas automáticas de conversación + estado de archivos; `/checkpoint <id>` revierte ambos al estado anterior.
- **Modo plan** — `/plan <desc>` (o la herramienta `EnterPlanMode`) pone a Claude en una fase estructurada de análisis de solo lectura; solo el archivo de plan es escribible. Claude escribe un plan detallado y luego `/plan done` restaura todos los permisos de escritura para la implementación.

---

## CheetahClaws vs OpenClaw

[OpenClaw](https://github.com/openclaw/openclaw) es otro asistente de IA de código abierto muy popular construido sobre TypeScript/Node.js. Los dos proyectos tienen **objetivos principales distintos**; así es como se comparan.

### De un vistazo

| Dimensión | OpenClaw (TypeScript) | CheetahClaws (Python) |
|-----------|----------------------|---------------------|
| Lenguaje | TypeScript + Node.js | Python 3.8+ |
| Archivos fuente | ~10,349 archivos TS/JS | 51 archivos Python |
| Líneas de código | ~245K | ~12K |
| Enfoque principal | Asistente personal para la vida diaria a través de canales de mensajería | Asistente de **programación** con IA / herramienta para desarrolladores |
| Arquitectura | Gateway daemon siempre activo + apps complementarias | REPL de terminal sin instalación compleja |
| Canales de mensajería | 20+ (WhatsApp · Telegram · Slack · Discord · Signal · iMessage · Matrix · WeChat · …) | Terminal + puente opcional de Telegram |
| Proveedores de modelos | Múltiples (prioridad nube) | 7+ con soporte local completo (Ollama · vLLM · LM Studio · …) |
| Modelos locales / offline | Limitado | Completo — Ollama, vLLM, cualquier endpoint compatible con OpenAI |
| Voz | Wake word · PTT · Talk Mode (macOS/iOS/Android) | Whisper STT offline (local, sin clave API) |
| Herramientas de edición de código | Control de navegador, espacio Canvas | Read · Write · Edit · Bash · Glob · Grep · NotebookEdit · GetDiagnostics |
| Requiere paso de compilación | Sí (`pnpm install` + configuración del daemon) | No — `pip install` y ejecutar |
| Compañero móvil | Barra de menús macOS + apps iOS/Android | — |
| Canvas / UI en vivo | Sí (espacio visual A2UI dirigido por agentes) | — |
| Soporte MCP | — | Sí (stdio/SSE/HTTP) |
| Extensibilidad en tiempo de ejecución | Plataforma de skills (integradas/gestionadas/workspace) | `register_tool()` en tiempo de ejecución, MCP, plugins git, skills Markdown |
| Facilidad para hackear | Código grande (245K líneas), más difícil de modificar | ~12K líneas — todo el bucle del agente visible en un archivo |

### Dónde gana OpenClaw

- **Bandeja de entrada omnicanal** — conecta con 20+ plataformas de mensajería (WhatsApp, Signal, iMessage, Discord, Teams, Matrix, WeChat…); los usuarios interactúan desde donde ya están.
- **Daemon siempre activo** — Gateway corre como servicio en segundo plano (launchd/systemd); no hace falta terminal para el uso diario.
- **Enfoque mobile-first** — barra de menús de macOS, Voice Wake / Talk Mode en iOS, grabación de cámara/pantalla en Android; se siente como una app nativa, no como una herramienta CLI.
- **Canvas en vivo** — espacio visual en el navegador dirigido por agentes; admite A2UI push/eval/snapshot.
- **Automatización del navegador** — perfil dedicado de Chrome/Chromium con herramientas de snapshot, acciones y carga.
- **Fiabilidad de producción** — versiones npm con control, CI completo, asistente de onboarding, diagnósticos `openclaw doctor`.

### Dónde gana CheetahClaws

- **Conjunto de herramientas de programación** — Read/Write/Edit/Bash/Glob/Grep/NotebookEdit/GetDiagnostics están diseñadas específicamente para desarrollo de software; CheetahClaws entiende diffs, árboles de archivos y estructura de código.
- **Soporte local real para modelos** — integración completa con Ollama/vLLM/LM Studio, incluyendo streaming, tool-calling y visión, sin nube.
- **8+ proveedores de modelos** — cambia entre Claude, GPT-4o, Gemini, DeepSeek, Qwen, MiniMax y modelos locales con una sola bandera `--model`.
- **Hackeable en minutos** — 12K líneas de Python legible; todo el bucle del agente está en `agent.py`; amplíalo con `register_tool()` en tiempo de ejecución sin recompilar.
- **Cero configuración** — `pip install cheetahclaws` y ejecuta `cheetahclaws`; sin daemon, sin emparejamiento, sin asistente de onboarding.
- **Soporte MCP** — conecta cualquier servidor MCP (stdio/SSE/HTTP); las herramientas se registran automáticamente.
- **Modo de desarrollador SSJ** — `/ssj` encadena Brainstorm → TODO → Worker → Debate en una sesión interactiva persistente; automatiza flujos completos de desarrollo.
- **Voz offline** — `/voice` transcribe localmente con `faster-whisper`; sin suscripción, sin OAuth, funciona sin internet.
- **Sincronización de sesiones en la nube** — `/cloudsave` respalda conversaciones completas en Gists privados de GitHub sin dependencias adicionales.

### Cuándo elegir cada uno

| Si quieres… | Usa |
|---|---|
| Un asistente personal al que puedas escribir por WhatsApp/Signal/Discord | **OpenClaw** |
| Un asistente de programación con IA en tu terminal | **CheetahClaws** |
| Soporte completo offline / para modelos locales | **CheetahClaws** |
| Una experiencia móvil, siempre activa | **OpenClaw** |
| Leer y modificar el código fuente en una tarde | **CheetahClaws** |
| Automatización del navegador y un Canvas visual | **OpenClaw** |
| Cambiar entre LLMs multiproveedor sin recompilar | **CheetahClaws** |

---

### Diferencias clave de diseño

**Bucle del agente** — CheetahClaws usa un generador Python que hace `yield` de eventos tipados (`TextChunk`, `ToolStart`, `ToolEnd`, `TurnDone`). Todo el bucle es visible en un único archivo, lo que facilita añadir hooks, renderizadores personalizados o logging.

**Registro de herramientas** — cada herramienta es una dataclass `ToolDef(name, schema, func, read_only, concurrent_safe)`. Cualquier módulo puede llamar a `register_tool()` al importarse; los servidores MCP, plugins y skills usan todos el mismo mecanismo.

**Compresión de contexto**

| | Claude Code | CheetahClaws |
|-|-------------|-----------------|
| Activación | Conteo exacto de tokens | estimación `len / 3.5`, se activa al 70 % |
| Capa 1 | — | Snip: trunca salidas antiguas de herramientas (sin coste API) |
| Capa 2 | Resumen por IA | Resumen por IA de turnos anteriores |
| Control | Gestionado por el sistema | parámetro `preserve_last_n_turns` |

**Memoria** — el servicio `extractMemories` de Claude Code hace que el modelo haga aflorar hechos de forma proactiva. El paquete `memory/` de CheetahClaws está impulsado por herramientas: el modelo llama explícitamente a `MemorySave`, lo que resulta más predecible y auditable. Cada memoria ahora incluye metadatos `confidence`, `source`, `last_used_at` y `conflict_group`; la búsqueda reordena por confianza × recencia; y `/memory consolidate` ofrece una consolidación manual sin modificar silenciosamente memorias en segundo plano.

### Quién debería usar CheetahClaws

- Desarrolladores que quieran **usar un modelo local o no-Anthropic** como asistente de programación.
- Investigadores que estudien **cómo funcionan los asistentes de programación agénticos** — todo el sistema cabe en una sola pantalla.
- Equipos que necesiten una **base hackeable** para añadir herramientas propietarias, políticas de permisos personalizadas o tipos de agente especializados.
- Cualquiera que quiera productividad estilo Claude Code **sin una cadena de build de Node.js**.

---

## Características

| Característica | Detalles |
|---|---|
| Multiproveedor | Anthropic · OpenAI · Gemini · Kimi · Qwen · Zhipu · DeepSeek · MiniMax · Ollama · LM Studio · endpoint personalizado |
| REPL interactivo | historial readline, autocompletado Tab de comandos slash con descripciones + sugerencias de subcomandos; Bracketed Paste Mode para pegado multilínea fiable |
| Bucle del agente | API de streaming + bucle automático de uso de herramientas |
| 27 herramientas integradas | Read · Write · Edit · Bash · Glob · Grep · WebFetch · WebSearch · **NotebookEdit** · **GetDiagnostics** · MemorySave · MemoryDelete · MemorySearch · MemoryList · Agent · SendMessage · CheckAgentResult · ListAgentTasks · ListAgentTypes · Skill · SkillList · AskUserQuestion · TaskCreate/Update/Get/List · **SleepTimer** · **EnterPlanMode** · **ExitPlanMode** · *(las herramientas MCP + plugin se añaden automáticamente al inicio)* |
| Integración MCP | Conecta cualquier servidor MCP (stdio/SSE/HTTP), las herramientas se registran automáticamente y Claude puede invocarlas |
| Sistema de plugins | Instala/desinstala/habilita/deshabilita/actualiza plugins desde URLs git o rutas locales; multi-ámbito (usuario/proyecto); motor de recomendaciones |
| AskUserQuestion | Claude puede pausar y hacer una pregunta aclaratoria al usuario a mitad de la tarea, con opciones numeradas opcionales |
| Gestión de tareas | Herramientas TaskCreate/Update/Get/List; IDs secuenciales; aristas de dependencia; metadatos; persistencia en `.cheetahclaws/tasks.json`; comando REPL `/tasks` |
| Vista diff | Visualización diff rojo/verde estilo git para Edit y Write |
| Compresión de contexto | Compacta automáticamente conversaciones largas para mantenerse dentro de los límites del modelo |
| Memoria persistente | Memoria de doble ámbito (usuario + proyecto) con 4 tipos, metadatos de confianza/fuente, detección de conflictos, búsqueda ponderada por recencia, seguimiento de `last_used_at` y `/memory consolidate` para extracción automática |
| Multiagente | Lanza subagentes tipados (coder/reviewer/researcher/…), aislamiento con git worktree, modo en segundo plano |
| Skills | `/commit` · `/review` integrados + skills Markdown personalizados con sustitución de argumentos y ejecución inline o fork |
| Herramientas de plugins | Registra herramientas personalizadas mediante `tool_registry.py` |
| Sistema de permisos | modos `auto` / `accept-all` / `manual` / `plan` |
| Checkpoints | Instantáneas automáticas de conversación + estado de archivos tras cada turno; `/checkpoint` para listar, `/checkpoint <id>` para rebobinar; alias `/rewind`; ventana deslizante de 100 instantáneas |
| Modo plan | `/plan <desc>` entra en modo de análisis de solo lectura; Claude solo escribe en el archivo de plan; herramientas de agente `EnterPlanMode` / `ExitPlanMode` para planificación autónoma |
| 37 comandos slash | `/model` · `/config` · `/save` · `/cost` · `/memory` · `/skills` · `/agents` · `/voice` · `/proactive` · `/checkpoint` · `/plan` · `/compact` · `/status` · `/doctor` · `/theme` · … |
| Temas de consola | `/theme` lista 15 paletas (default · dracula · nord · gruvbox · solarized · tokyo-night · catppuccin · matrix · synthwave · midnight · ocean · monokai · cheetah · mono · none); cada fila muestra una vista previa en vivo `info / ok / warn / err` con los colores del propio tema. `/theme <nombre>` aplica y persiste la elección — también define el estilo de bloque de código de Rich Markdown. |
| Entrada por voz | Grabar → transcribir → enviar automáticamente. Backends: `sounddevice` / `arecord` / SoX + `faster-whisper` / `openai-whisper` / API de OpenAI. Funciona completamente offline. |
| Brainstorm | `/brainstorm [tema]` genera N personas expertas adecuadas al tema (2–100, por defecto 5, elegidas de forma interactiva), ejecuta un debate iterativo, guarda resultados en `brainstorm_outputs/` y sintetiza un Master Plan + genera automáticamente `brainstorm_outputs/todo_list.txt`. |
| Modo de desarrollador SSJ | `/ssj` abre un menú persistente e interactivo con 10 atajos: Brainstorm, visor TODO, Worker, Debate entre expertos, Propose, Review, Readme, Commit, Scan, Promote. Permanece abierto entre acciones; admite passthrough de `/command`. El debate muestra un spinner animado por ronda y guarda el resultado junto al archivo debatido. |
| Worker | `/worker [task#s]` lee `brainstorm_outputs/todo_list.txt`, implementa cada tarea pendiente con un prompt dedicado del modelo y la marca como hecha (`- [x]`). Admite selección de tareas (`/worker 1,4,6`), ruta personalizada (`--path`) y límite de workers (`--workers`). Detecta y redirige rutas `.md` de brainstorm introducidas por error. |
| Puente de Telegram | `/telegram <token> <chat_id>` inicia un puente bot: recibe mensajes desde Telegram, ejecuta el modelo y responde, todo desde tu teléfono. Indicador de escritura, passthrough de comandos slash (incluidos menús interactivos) y autoarranque al inicio si está configurado. |
| Entrada de visión | `/image` (o `/img`) captura la imagen del portapapeles y la envía a cualquier modelo con capacidad de visión — Ollama (`llava`, `gemma4`, `llama3.2-vision`) mediante formato nativo, o modelos en la nube (GPT-4o, Gemini 2.0 Flash, …) mediante formato OpenAI `image_url` multipart. Requiere `pip install cheetahclaws[vision]`; en Linux también necesita `xclip`. |
| Monitoreo proactivo | `/proactive [duration]` inicia un daemon centinela en segundo plano; el agente despierta automáticamente tras inactividad, permitiendo bucles continuos de monitoreo sin prompts del usuario |
| Cierre forzado | 3× Ctrl+C en 2 segundos activa `os._exit(1)` — mata el proceso inmediatamente sin importar bloqueos de E/S |
| Streaming Live con Rich | Cuando `rich` está instalado, las respuestas se muestran como Markdown que se actualiza en vivo. Las respuestas largas que desbordarían la terminal siguen renderizándose en vivo pero solo muestran la última pantalla (una «ventana de cola» acotada, que tiene en cuenta la altura de la terminal y el ajuste de línea); la salida completa se confirma al terminar la respuesta, evitando los fotogramas duplicados u obsoletos que algunas terminales dejan. El streaming de texto plano se usa solo como respaldo. Se desactiva automáticamente en sesiones SSH; puedes forzarlo con `/config rich_live=false`. |
| Inyección de contexto | Carga automáticamente `CLAUDE.md`, estado git, cwd, memoria persistente |
| Persistencia de sesión | Autoguardado al salir en `daily/YYYY-MM-DD/` (límite por día) + `history.json` (maestro, todas las sesiones) + `session_latest.json` (`/resume`); las sesiones incluyen metadatos `session_id` y `saved_at`; `/load` agrupado por fecha |
| Sincronización en la nube | `/cloudsave` sincroniza sesiones con Gists privados de GitHub; auto-sync al salir; carga desde la nube por Gist ID. Sin nuevas dependencias (stdlib `urllib`). |
| Pensamiento extendido | Activar/desactivar para modelos Claude; streaming nativo de bloques `<think>` para modelos locales de razonamiento de Ollama (deepseek-r1, qwen3, gemma4) |
| Seguimiento de costes | Uso de tokens + coste estimado en USD |
| Modo no interactivo | bandera `--print` para scripting / CI |

---

## Modelos compatibles

### Código cerrado (API)

| Proveedor | Modelo | Contexto | Fortalezas | Variable de entorno de API key |
|---|---|---|---|---|
| **Anthropic** | `claude-opus-4-6` | 200k | Máxima capacidad, mejor para razonamiento complejo | `ANTHROPIC_API_KEY` |
| **Anthropic** | `claude-sonnet-4-6` | 200k | Equilibrio entre velocidad y calidad | `ANTHROPIC_API_KEY` |
| **Anthropic** | `claude-haiku-4-5-20251001` | 200k | Rápido y rentable | `ANTHROPIC_API_KEY` |
| **OpenAI** | `gpt-4o` | 128k | Fuerte en multimodal y código | `OPENAI_API_KEY` |
| **OpenAI** | `gpt-4o-mini` | 128k | Rápido y barato | `OPENAI_API_KEY` |
| **OpenAI** | `o3-mini` | 200k | Razonamiento sólido | `OPENAI_API_KEY` |
| **OpenAI** | `o1` | 200k | Razonamiento avanzado | `OPENAI_API_KEY` |
| **Google** | `gemini-2.5-pro-preview-03-25` | 1M | Contexto largo, multimodal | `GEMINI_API_KEY` |
| **Google** | `gemini-2.0-flash` | 1M | Rápido, gran contexto | `GEMINI_API_KEY` |
| **Google** | `gemini-1.5-pro` | 2M | Ventana de contexto más grande | `GEMINI_API_KEY` |
| **Moonshot (Kimi)** | `moonshot-v1-8k` | 8k | Chino e inglés | `MOONSHOT_API_KEY` |
| **Moonshot (Kimi)** | `moonshot-v1-32k` | 32k | Chino e inglés | `MOONSHOT_API_KEY` |
| **Moonshot (Kimi)** | `moonshot-v1-128k` | 128k | Contexto largo | `MOONSHOT_API_KEY` |
| **Alibaba (Qwen)** | `qwen-max` | 32k | Mejor calidad de Qwen | `DASHSCOPE_API_KEY` |
| **Alibaba (Qwen)** | `qwen-plus` | 128k | Equilibrado | `DASHSCOPE_API_KEY` |
| **Alibaba (Qwen)** | `qwen-turbo` | 1M | Rápido, barato | `DASHSCOPE_API_KEY` |
| **Alibaba (Qwen)** | `qwq-32b` | 32k | Razonamiento sólido | `DASHSCOPE_API_KEY` |
| **Zhipu (GLM)** | `glm-4-plus` | 128k | Mejor calidad de GLM | `ZHIPU_API_KEY` |
| **Zhipu (GLM)** | `glm-4` | 128k | Uso general | `ZHIPU_API_KEY` |
| **Zhipu (GLM)** | `glm-4-flash` | 128k | Nivel gratuito disponible | `ZHIPU_API_KEY` |
| **DeepSeek** | `deepseek-chat` | 64k | Fuerte para programación | `DEEPSEEK_API_KEY` |
| **DeepSeek** | `deepseek-reasoner` | 64k | Razonamiento con chain-of-thought | `DEEPSEEK_API_KEY` |
| **MiniMax** | `MiniMax-Text-01` | 1M | Contexto largo, razonamiento sólido | `MINIMAX_API_KEY` |
| **MiniMax** | `MiniMax-VL-01` | 1M | Visión + lenguaje | `MINIMAX_API_KEY` |
| **MiniMax** | `abab6.5s-chat` | 256k | Rápido y rentable | `MINIMAX_API_KEY` |
| **MiniMax** | `abab6.5-chat` | 256k | Calidad equilibrada | `MINIMAX_API_KEY` |

### Código abierto (local vía Ollama)

| Modelo | Tamaño | Fortalezas | Comando de descarga |
|---|---|---|---|
| `llama3.3` | 70B | Uso general, razonamiento fuerte | `ollama pull llama3.3` |
| `llama3.2` | 3B / 11B | Ligero | `ollama pull llama3.2` |
| `qwen2.5-coder` | 7B / 32B | **Mejor para tareas de programación** | `ollama pull qwen2.5-coder` |
| `qwen2.5` | 7B / 72B | Chino e inglés | `ollama pull qwen2.5` |
| `deepseek-r1` | 7B–70B | Razonamiento, matemáticas | `ollama pull deepseek-r1` |
| `deepseek-coder-v2` | 16B | Programación | `ollama pull deepseek-coder-v2` |
| `mistral` | 7B | Rápido, eficiente | `ollama pull mistral` |
| `mixtral` | 8x7B | Modelo MoE potente | `ollama pull mixtral` |
| `phi4` | 14B | Microsoft, buen razonamiento | `ollama pull phi4` |
| `gemma3` | 4B / 12B / 27B | Modelo abierto de Google | `ollama pull gemma3` |
| `codellama` | 7B / 34B | Generación de código | `ollama pull codellama` |
| `llava` | 7B / 13B | **Visión** — comprensión de imágenes | `ollama pull llava` |
| `llama3.2-vision` | 11B | **Visión** — razonamiento multimodal | `ollama pull llama3.2-vision` |

> **Nota:** el tool calling requiere un modelo compatible con function calling. Modelos locales recomendados: `qwen2.5-coder`, `llama3.3`, `mistral`, `phi4`.

> **Modelos de razonamiento:** `deepseek-r1`, `qwen3` y `gemma4` transmiten bloques `<think>` nativos. Actívalos con `/verbose` y `/thinking` para ver el razonamiento en la terminal. Nota: los modelos a los que se les proporciona un prompt de sistema grande (como los 25 esquemas de herramientas de cheetahclaws) pueden suprimir su fase de pensamiento para no romper el formato JSON esperado; esto es comportamiento del modelo, no un bug.

---

## Instalación

### Recomendado: instalar como comando global con `uv`

[uv](https://docs.astral.sh/uv/) instala `cheetahclaws` en un entorno aislado y lo coloca en tu PATH para que puedas ejecutarlo desde cualquier lugar:

```bash
# Instala uv (si aún no está instalado)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clona e instala
git clone https://github.com/SafeRL-Lab/cheetahclaws
cd cheetahclaws
uv tool install .
```

Después de eso, `cheetahclaws` estará disponible como comando global:

```bash
cheetahclaws                        # iniciar REPL
cheetahclaws --model gpt-4o         # elegir modelo
cheetahclaws -p "explain this"      # modo no interactivo
```

Para actualizar tras hacer pull de nuevo código:

```bash
uv tool install . --reinstall
```

Para desinstalar:

```bash
uv tool uninstall cheetahclaws
```

### Alternativa: ejecutar directamente desde el repositorio

```bash
git clone https://github.com/SafeRL-Lab/cheetahclaws
cd cheetahclaws

pip install -r requirements.txt
# o manualmente (sounddevice es opcional — solo se necesita para /voice):
pip install anthropic openai httpx rich
pip install sounddevice  # opcional: entrada por voz

python cheetahclaws.py
```

---

## Uso: modelos API de código cerrado

### Anthropic Claude

Obtén tu clave API en [console.anthropic.com](https://console.anthropic.com).

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...

# Modelo por defecto (claude-opus-4-6)
cheetahclaws

# Elegir un modelo específico
cheetahclaws --model claude-sonnet-4-6
cheetahclaws --model claude-haiku-4-5-20251001

# Activar Extended Thinking
cheetahclaws --model claude-opus-4-6 --thinking --verbose
```

### OpenAI GPT

Obtén tu clave API en [platform.openai.com](https://platform.openai.com).

```bash
export OPENAI_API_KEY=sk-...

cheetahclaws --model gpt-4o
cheetahclaws --model gpt-4o-mini
cheetahclaws --model gpt-4.1-mini
cheetahclaws --model o3-mini
```

### Google Gemini

Obtén tu clave API en [aistudio.google.com](https://aistudio.google.com).

```bash
export GEMINI_API_KEY=AIza...

cheetahclaws --model gemini/gemini-2.0-flash
cheetahclaws --model gemini/gemini-1.5-pro
cheetahclaws --model gemini/gemini-2.5-pro-preview-03-25
```

### Kimi (Moonshot AI)

Obtén tu clave API en [platform.moonshot.cn](https://platform.moonshot.cn).

```bash
export MOONSHOT_API_KEY=sk-...

cheetahclaws --model kimi/moonshot-v1-32k
cheetahclaws --model kimi/moonshot-v1-128k
```

### Qwen (Alibaba DashScope)

Obtén tu clave API en [dashscope.aliyun.com](https://dashscope.aliyun.com).

```bash
export DASHSCOPE_API_KEY=sk-...

cheetahclaws --model qwen/Qwen3.5-Plus
cheetahclaws --model qwen/Qwen3-MAX
cheetahclaws --model qwen/Qwen3.5-Flash
```

### Zhipu GLM

Obtén tu clave API en [open.bigmodel.cn](https://open.bigmodel.cn).

```bash
export ZHIPU_API_KEY=...

cheetahclaws --model zhipu/glm-4-plus
cheetahclaws --model zhipu/glm-4-flash   # nivel gratuito
```

### DeepSeek

Obtén tu clave API en [platform.deepseek.com](https://platform.deepseek.com).

```bash
export DEEPSEEK_API_KEY=sk-...

cheetahclaws --model deepseek/deepseek-chat
cheetahclaws --model deepseek/deepseek-reasoner
```

### MiniMax

Obtén tu clave API en [platform.minimaxi.chat](https://platform.minimaxi.chat).

```bash
export MINIMAX_API_KEY=...

cheetahclaws --model minimax/MiniMax-Text-01
cheetahclaws --model minimax/MiniMax-VL-01
cheetahclaws --model minimax/abab6.5s-chat
```

---

## Uso: modelos de código abierto (local)

### Opción A — Ollama (recomendado)

Ollama ejecuta modelos localmente sin configuración. No se requiere clave API.

**Paso 1: instala Ollama**

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# O descarga desde https://ollama.com/download
```

**Paso 2: descarga un modelo**

```bash
# Mejor para programación (recomendado)
ollama pull qwen2.5-coder          # 4.7 GB (7B)
ollama pull qwen2.5-coder:32b      # 19 GB (32B)

# Uso general
ollama pull llama3.3               # 42 GB (70B)
ollama pull llama3.2               # 2.0 GB (3B)

# Razonamiento
ollama pull deepseek-r1            # 4.7 GB (7B)
ollama pull deepseek-r1:32b        # 19 GB (32B)

# Otros
ollama pull phi4                   # 9.1 GB (14B)
ollama pull mistral                # 4.1 GB (7B)
```

**Paso 3: inicia el servidor de Ollama** (se ejecuta automáticamente en macOS; en Linux ejecútalo manualmente)

```bash
ollama serve     # inicia en http://localhost:11434
```

**Paso 4: ejecuta cheetahclaws**

```bash
cheetahclaws --model ollama/qwen2.5-coder
cheetahclaws --model ollama/llama3.3
cheetahclaws --model ollama/deepseek-r1
```

O

```bash
python cheetahclaws.py --model ollama/qwen2.5-coder
python cheetahclaws.py --model ollama/llama3.3
python cheetahclaws.py --model ollama/deepseek-r1
python cheetahclaws.py --model ollama/qwen3.5:35b
```

**Lista tus modelos locales disponibles:**

```bash
ollama list
```

Luego usa cualquier modelo de la lista:

```bash
cheetahclaws --model ollama/<nombre-del-modelo>
```

---

### Opción B — LM Studio

LM Studio ofrece una interfaz gráfica para descargar y ejecutar modelos, con un servidor compatible con OpenAI integrado.

**Paso 1:** descarga [LM Studio](https://lmstudio.ai) e instálalo.

**Paso 2:** busca y descarga un modelo dentro de LM Studio (formato GGUF).

**Paso 3:** ve a la pestaña **Local Server** → haz clic en **Start Server** (puerto por defecto: 1234).

**Paso 4:**

```bash
cheetahclaws --model lmstudio/<nombre-del-modelo>
# p. ej.:
cheetahclaws --model lmstudio/phi-4-GGUF
cheetahclaws --model lmstudio/qwen2.5-coder-7b
```

El nombre del modelo debe coincidir con el que LM Studio muestra en la barra de estado del servidor.

---

### Opción C — vLLM / servidor self-hosted compatible con OpenAI

Para servidores de inferencia autohospedados (vLLM, TGI, llama.cpp server, etc.) que exponen una API compatible con OpenAI:

Inicio rápido para la opción C:
Paso 1: inicia vllm:
 ```
CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-Coder-7B-Instruct \
      --host 0.0.0.0 \
      --port 8000 \
      --enable-auto-tool-choice \
      --tool-call-parser hermes
```


 Paso 2: inicia cheetahclaws：
```
  export CUSTOM_BASE_URL=http://localhost:8000/v1
  export CUSTOM_API_KEY=none
  cheetahclaws --model custom/Qwen/Qwen2.5-Coder-7B-Instruct
```


```bash
# Ejemplo: vLLM sirviendo Qwen2.5-Coder-32B
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --port 8000

# Luego ejecuta cheetahclaws apuntando a tu servidor:
cheetahclaws
```

Dentro del REPL:

```
/config custom_base_url=http://localhost:8000/v1
/config custom_api_key=token-abc123    # omite si no hay autenticación
/model custom/Qwen2.5-Coder-32B-Instruct
```

O configúralo mediante variables de entorno:

```bash
export CUSTOM_BASE_URL=http://localhost:8000/v1
export CUSTOM_API_KEY=token-abc123

cheetahclaws --model custom/Qwen2.5-Coder-32B-Instruct
```

Para un servidor GPU remoto:

```bash
/config custom_base_url=http://192.168.1.100:8000/v1
/model custom/your-model-name
```

---

## Formato del nombre del modelo

Se admiten tres formatos equivalentes:

```bash
# 1. Detección automática por prefijo (funciona para modelos conocidos)
cheetahclaws --model gpt-4o
cheetahclaws --model gemini-2.0-flash
cheetahclaws --model deepseek-chat

# 2. Prefijo explícito de proveedor con slash
cheetahclaws --model ollama/qwen2.5-coder
cheetahclaws --model kimi/moonshot-v1-128k

# 3. Prefijo explícito de proveedor con dos puntos (también funciona)
cheetahclaws --model kimi:moonshot-v1-32k
cheetahclaws --model qwen:qwen-max
```

**Reglas de detección automática:**

| Prefijo del modelo | Proveedor detectado |
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

## Referencia de la CLI

```
cheetahclaws [OPTIONS] [PROMPT]
# o: python cheetahclaws.py [OPTIONS] [PROMPT]

Opciones:
  -p, --print          No interactivo: ejecuta el prompt y sale
  -m, --model MODEL    Sobrescribe el modelo (p. ej. gpt-4o, ollama/llama3.3)
  --accept-all         Aprueba automáticamente todas las operaciones (sin prompts de permiso)
  --verbose            Muestra bloques de thinking y conteos de tokens por turno
  --thinking           Activa Extended Thinking (solo Claude)
  --version            Imprime la versión y sale
  -h, --help           Muestra la ayuda
```

**Ejemplos:**

```bash
# REPL interactivo con el modelo por defecto
cheetahclaws

# Cambiar de modelo al inicio
cheetahclaws --model gpt-4o
cheetahclaws -m ollama/deepseek-r1:32b

# No interactivo / scripting
cheetahclaws --print "Write a Python fibonacci function"
cheetahclaws -p "Explain the Rust borrow checker in 3 sentences" -m gemini/gemini-2.0-flash

# CI / automatización (sin prompts de permiso)
cheetahclaws --accept-all --print "Initialize a Python project with pyproject.toml"

# Modo depuración (ver tokens + thinking)
cheetahclaws --thinking --verbose
```

---

## Comandos slash (REPL)

Escribe `/` y pulsa **Tab** para ver todos los comandos con sus descripciones. Sigue escribiendo para filtrar y pulsa Tab de nuevo para autocompletar. Después del nombre de un comando, pulsa **Tab** otra vez para ver sus subcomandos (por ejemplo, `/plugin ` → `install`, `uninstall`, `enable`, …).

| Comando | Descripción |
|---|---|
| `/help` | Mostrar todos los comandos |
| `/clear` | Borrar el historial de la conversación |
| `/model` | Mostrar el modelo actual + listar todos los modelos disponibles |
| `/model <name>` | Cambiar de modelo (surte efecto inmediatamente) |
| `/config` | Mostrar todos los valores actuales de configuración |
| `/config key=value` | Establecer un valor de configuración (persistido en disco) |
| `/save` | Guardar la sesión (nombre automático con marca temporal) |
| `/save <filename>` | Guardar la sesión con un nombre concreto |
| `/load` | Lista interactiva agrupada por fecha; introduce un número, `1,2,3` para fusionar, o `H` para historial completo |
| `/load <filename>` | Cargar una sesión guardada por nombre de archivo |
| `/resume` | Restaurar la última sesión autoguardada (`mr_sessions/session_latest.json`) |
| `/resume <filename>` | Cargar un archivo específico de `mr_sessions/` (o una ruta absoluta) |
| `/history` | Imprimir el historial completo de la conversación |
| `/context` | Mostrar el número de mensajes y la estimación de tokens |
| `/cost` | Mostrar uso de tokens y coste estimado en USD |
| `/verbose` | Activar/desactivar modo verbose (tokens + thinking) |
| `/thinking` | Activar/desactivar Extended Thinking (solo Claude) |
| `/permissions` | Mostrar el modo actual de permisos |
| `/permissions <mode>` | Establecer modo de permisos: `auto` / `accept-all` / `manual` |
| `/cwd` | Mostrar el directorio de trabajo actual |
| `/cwd <path>` | Cambiar el directorio de trabajo |
| `/memory` | Listar todas las memorias persistentes |
| `/memory <query>` | Buscar memorias por palabra clave (ordenadas por confianza × recencia) |
| `/memory consolidate` | Extraer con IA hasta 3 aprendizajes de largo plazo de la sesión actual |
| `/skills` | Listar skills disponibles |
| `/agents` | Mostrar el estado de tareas de subagentes |
| `/mcp` | Listar servidores MCP configurados y sus herramientas |
| `/mcp reload` | Reconectar todos los servidores MCP y refrescar herramientas |
| `/mcp reload <name>` | Reconectar un solo servidor |
| `/mcp add <name> <cmd> [args]` | Añadir un servidor MCP stdio a la configuración del usuario |
| `/mcp remove <name>` | Eliminar un servidor de la configuración del usuario |
| `/voice` | Grabar voz, transcribir con Whisper y autoenviarla como prompt |
| `/voice status` | Mostrar la disponibilidad del backend de grabación y STT |
| `/voice lang <code>` | Establecer el idioma STT (por ejemplo, `zh`, `en`, `ja`; `auto` para detectar) |
| `/voice device` | Listar micrófonos de entrada disponibles y seleccionar uno interactivamente |
| `/image [prompt]` | Capturar la imagen del portapapeles y enviarla a un modelo de visión con un prompt opcional |
| `/img [prompt]` | Alias de `/image` |
| `/proactive` | Mostrar el estado actual del sondeo proactivo (ON/OFF e intervalo) |
| `/proactive <duration>` | Activar el sondeo centinela en segundo plano (por ejemplo, `5m`, `30s`, `1h`) |
| `/proactive off` | Desactivar el sondeo en segundo plano |
| `/cloudsave setup <token>` | Configurar un token personal de acceso de GitHub para sincronización con Gist |
| `/cloudsave` | Subir la sesión actual a un GitHub Gist privado |
| `/cloudsave push [desc]` | Subir con una descripción opcional |
| `/cloudsave auto on\|off` | Activar/desactivar subida automática al usar `/exit` |
| `/cloudsave list` | Listar tus Gists de cheetahclaws |
| `/cloudsave load <gist_id>` | Descargar y restaurar una sesión desde Gist |
| `/brainstorm` | Ejecutar una lluvia de ideas IA multi-persona; solicita el número de agentes (2–100, por defecto 5) |
| `/brainstorm <topic>` | Centrar la lluvia de ideas en un tema específico; solicita el número de agentes |
| `/ssj` | Abrir SSJ Developer Mode — menú interactivo avanzado con 10 atajos de flujo de trabajo |
| `/worker` | Implementar automáticamente todas las tareas pendientes de `brainstorm_outputs/todo_list.txt` |
| `/worker <n,m,…>` | Implementar tareas pendientes concretas por número (por ejemplo, `/worker 1,4,6`) |
| `/worker --path <file>` | Usar una ruta personalizada al archivo TODO en lugar de la predeterminada |
| `/worker --workers <n>` | Limitar el lote a N tareas por ejecución (por ejemplo, `/worker --workers 3`) |
| `/telegram <token> <chat_id>` | Configurar e iniciar el puente bot de Telegram |
| `/telegram` | Iniciar el puente usando el token + chat_id ya guardados |
| `/telegram stop` | Detener el puente de Telegram |
| `/telegram status` | Mostrar si el puente está activo y el chat_id configurado |
| `/checkpoint` | Listar todos los checkpoints (instantáneas) de la sesión actual |
| `/checkpoint <id>` | Rebobinar al checkpoint: restaura archivos y conversación a esa instantánea |
| `/checkpoint clear` | Eliminar todos los checkpoints de la sesión actual |
| `/rewind` | Alias de `/checkpoint` |
| `/plan <description>` | Entrar en modo plan: análisis de solo lectura, escritura solo en el archivo de plan |
| `/plan` | Mostrar el contenido actual del archivo de plan |
| `/plan done` | Salir del modo plan y restaurar los permisos originales |
| `/plan status` | Mostrar si el modo plan está activo |
| `/compact` | Compactar manualmente la conversación (igual que la compactación automática, pero activada por el usuario) |
| `/compact <focus>` | Compactar con instrucciones de enfoque (por ejemplo, `/compact keep the auth refactor context`) |
| `/init` | Crear una plantilla `CLAUDE.md` en el directorio de trabajo actual |
| `/export` | Exportar la conversación como archivo Markdown a `.nano_claude/exports/` |
| `/export <filename>` | Exportar como Markdown o JSON (detectado por extensión `.json`) |
| `/copy` | Copiar la última respuesta del asistente al portapapeles |
| `/status` | Mostrar versión, modelo, proveedor, permisos, ID de sesión, uso de tokens y % de contexto |
| `/doctor` | Diagnosticar el estado de la instalación: Python, git, clave API, dependencias opcionales, CLAUDE.md, uso de disco de checkpoints |
| `/theme` | Listar 15 paletas de consola, cada fila muestra una vista previa `info / ok / warn / err` con los colores del propio tema (la actual marcada con `*`) |
| `/theme <nombre>` | Aplicar y persistir un tema (p. ej. `dracula`, `nord`, `gruvbox`, `none`); también define el estilo de bloque de código de Rich Markdown |
| `/exit` / `/quit` | Salir |

**Cambiar de modelo dentro de una sesión:**

```
[myproject] ❯ /model
  Modelo actual: claude-opus-4-6  (proveedor: anthropic)

  Modelos disponibles por proveedor:
    anthropic     claude-opus-4-6, claude-sonnet-4-6, ...
    openai        gpt-4o, gpt-4o-mini, o3-mini, ...
    ollama        llama3.3, llama3.2, phi4, mistral, ...
    ...

[myproject] ❯ /model gpt-4o
  Modelo establecido en gpt-4o  (proveedor: openai)

[myproject] ❯ /model ollama/qwen2.5-coder
  Modelo establecido en ollama/qwen2.5-coder  (proveedor: ollama)
```

---

## Configuración de claves API

### Método 1: variables de entorno (recomendado)

```bash
# Añade a ~/.bashrc o ~/.zshrc
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=AIza...
export MOONSHOT_API_KEY=sk-...       # Kimi
export DASHSCOPE_API_KEY=sk-...      # Qwen
export ZHIPU_API_KEY=...             # Zhipu GLM
export DEEPSEEK_API_KEY=sk-...       # DeepSeek
export MINIMAX_API_KEY=...           # MiniMax
```

### Método 2: establecerlas dentro del REPL (persistente)

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

Las claves se guardan en `~/.cheetahclaws/config.json` y se cargan automáticamente en el siguiente inicio.

### Método 3: editar directamente el archivo de configuración

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

## Sistema de permisos

| Modo | Comportamiento |
|---|---|
| `auto` (por defecto) | Las operaciones de solo lectura están siempre permitidas. Pide confirmación antes de ejecutar Bash y escrituras en archivos. |
| `accept-all` | Nunca pregunta. Todas las operaciones se realizan automáticamente. |
| `manual` | Pide confirmación antes de cada operación, incluso lecturas. |
| `plan` | Modo de análisis de solo lectura. Solo el archivo de plan (`.nano_claude/plans/`) es escribible. Se entra mediante `/plan <desc>` o la herramienta `EnterPlanMode`. |

**Cuando aparece un prompt:**

```
  Permitir: Run: git commit -am "fix bug"  [y/N/a(ccept-all)]
```

- `y` — aprueba solo esta acción
- `n` o Enter — deniega
- `a` — aprueba y cambia a `accept-all` para el resto de la sesión

**Comandos siempre autoaprobados en modo `auto`:**
`ls`, `cat`, `head`, `tail`, `wc`, `pwd`, `echo`, `git status`, `git log`, `git diff`, `git show`, `find`, `grep`, `rg`, `python`, `node`, `pip show`, `npm list` y otros comandos shell de solo lectura.

---

## Herramientas integradas

### Herramientas principales

| Herramienta | Descripción | Parámetros clave |
|---|---|---|
| `Read` | Leer archivo con números de línea | `file_path`, `limit`, `offset` |
| `Write` | Crear o sobrescribir archivo (muestra diff) | `file_path`, `content` |
| `Edit` | Reemplazo exacto de cadenas (muestra diff) | `file_path`, `old_string`, `new_string`, `replace_all` |
| `Bash` | Ejecutar comando shell | `command`, `timeout` (30s por defecto) |
| `Glob` | Encontrar archivos por patrón glob | `pattern` (por ejemplo, `**/*.py`), `path` |
| `Grep` | Búsqueda regex en archivos (usa ripgrep si está disponible) | `pattern`, `path`, `glob`, `output_mode` |
| `WebFetch` | Obtener y extraer texto de una URL | `url`, `prompt` |
| `WebSearch` | Buscar en la web mediante DuckDuckGo | `query` |

### Herramientas de notebook y diagnóstico

| Herramienta | Descripción | Parámetros clave |
|---|---|---|
| `NotebookEdit` | Editar una celda de notebook Jupyter (`.ipynb`) | `notebook_path`, `new_source`, `cell_id`, `cell_type`, `edit_mode` (`replace`/`insert`/`delete`) |
| `GetDiagnostics` | Obtener diagnósticos estilo LSP para un archivo fuente (pyright/mypy/flake8 para Python; tsc/eslint para JS/TS; shellcheck para shell) | `file_path`, `language` (opcional) |

### Herramientas de memoria

| Herramienta | Descripción | Parámetros clave |
|---|---|---|
| `MemorySave` | Guardar o actualizar una memoria persistente | `name`, `type`, `description`, `content`, `scope` |
| `MemoryDelete` | Eliminar una memoria por nombre | `name`, `scope` |
| `MemorySearch` | Buscar memorias por palabra clave (o ranking por IA) | `query`, `scope`, `use_ai`, `max_results` |
| `MemoryList` | Listar todas las memorias con edad y metadatos | `scope` |

### Herramientas de subagentes

| Herramienta | Descripción | Parámetros clave |
|---|---|---|
| `Agent` | Lanzar un subagente para una tarea | `prompt`, `subagent_type`, `isolation`, `name`, `model`, `wait` |
| `SendMessage` | Enviar un mensaje a un agente en segundo plano con nombre | `name`, `message` |
| `CheckAgentResult` | Comprobar el estado/resultado de un agente en segundo plano | `task_id` |
| `ListAgentTasks` | Listar todas las tareas activas y finalizadas de agentes | — |
| `ListAgentTypes` | Listar las definiciones de tipos de agente disponibles | — |

### Herramientas de segundo plano y autonomía

| Herramienta | Descripción | Parámetros clave |
|---|---|---|
| `SleepTimer` | Programa un temporizador silencioso en segundo plano; al dispararse inyecta un prompt de reactivación automático para que el agente reanude el monitoreo o tareas diferidas | `seconds` |

### Herramientas de skills

| Herramienta | Descripción | Parámetros clave |
|---|---|---|
| `Skill` | Invocar una skill por nombre dentro de la conversación | `name`, `args` |
| `SkillList` | Listar todas las skills disponibles con disparadores y metadatos | — |

### Herramientas MCP

Las herramientas MCP se descubren automáticamente en los servidores configurados y se registran bajo el nombre `mcp__<server>__<tool>`. Claude puede utilizarlas exactamente igual que las herramientas integradas.

| Nombre de herramienta de ejemplo | De dónde viene |
|---|---|
| `mcp__git__git_status` | servidor `git`, herramienta `git_status` |
| `mcp__filesystem__read_file` | servidor `filesystem`, herramienta `read_file` |
| `mcp__myserver__my_action` | servidor personalizado que configuraste |

> **Añadir herramientas personalizadas:** consulta [Architecture Guide](docs/architecture.md#tool-registry) para ver cómo registrar tus propias herramientas.

---

## Memoria

El modelo puede recordar cosas entre conversaciones usando el sistema de memoria integrado.

### Almacenamiento

Las memorias se almacenan como archivos Markdown individuales en dos ámbitos:

| Ámbito | Ruta | Visibilidad |
|---|---|---|
| **Usuario** (por defecto) | `~/.cheetahclaws/memory/` | Compartido entre todos los proyectos |
| **Proyecto** | `.cheetahclaws/memory/` en cwd | Local al repositorio actual |

Un índice `MEMORY.md` (≤ 200 líneas / 25 KB) se reconstruye automáticamente en cada guardado o borrado y se inyecta en el prompt del sistema para que el modelo siempre tenga una visión general de lo recordado.

### Tipos de memoria

| Tipo | Úsalo para |
|---|---|
| `user` | Tu rol, preferencias, contexto |
| `feedback` | Cómo quieres que se comporte el modelo (correcciones Y confirmaciones) |
| `project` | Trabajo en curso, fechas límite, decisiones que no están en el historial git |
| `reference` | Enlaces a sistemas externos (Linear, Grafana, Slack, etc.) |

### Formato de archivo de memoria

Cada memoria es un archivo Markdown con frontmatter YAML:

```markdown
---
name: coding_style
description: Preferencias de formato para Python
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

**Campos de metadatos** (nuevos — gestionados automáticamente):

| Campo | Por defecto | Descripción |
|---|---|---|
| `confidence` | `1.0` | Puntuación de fiabilidad 0–1. Declaraciones explícitas del usuario = 1.0; preferencias inferidas ≈ 0.8; consolidación automática ≈ 0.8 |
| `source` | `user` | Origen: `user` / `model` / `tool` / `consolidator` |
| `last_used_at` | — | Se actualiza automáticamente cada vez que esta memoria aparece en MemorySearch |
| `conflict_group` | — | Agrupa memorias relacionadas (por ejemplo, `writing_style`) para seguimiento de conflictos |

### Detección de conflictos

Cuando `MemorySave` se llama con un nombre ya existente pero contenido distinto, el sistema informa del conflicto antes de sobrescribir:

```
Memoria guardada: 'writing_style' [feedback/user]
⚠ Replaced conflicting memory (was user-sourced, 100% confidence, written 2026-04-01).
  Old content: Prefer formal, academic style...
```

### Recuperación ordenada

`MemorySearch` clasifica los resultados por **confianza × recencia** (decaimiento exponencial de 30 días) en lugar de solo por coincidencia de palabras clave. Las memorias no usadas recientemente pierden prioridad. Cada resultado también actualiza `last_used_at`, de modo que las memorias frecuentemente consultadas siguen siendo prominentes.

```
Tú: /memory python
  [feedback/user] coding_style [conf:95% src:user]
    Preferencias de formato para Python
    Prefer 4-space indentation and full type hints...
```

### `/memory consolidate` — extracción automática de aprendizajes duraderos

Después de una sesión relevante, ejecuta:

```
[myproject] ❯ /memory consolidate
  Analizando la sesión para extraer memorias de largo plazo…
  ✓ Consolidated 2 memory/memories: user_prefers_direct_answers, avoid_trailing_summaries
```

El comando envía una transcripción condensada de la sesión al modelo y le pide identificar hasta **3** aprendizajes que merezcan conservarse a largo plazo (preferencias del usuario, correcciones de feedback, decisiones del proyecto). Las memorias extraídas se guardan con `confidence: 0.80` y `source: consolidator`; **nunca sobrescriben** una memoria existente que ya tenga mayor confianza.

Buenos momentos para ejecutar `/memory consolidate`:
- Después de corregir varias veces seguidas el comportamiento del modelo
- Después de una sesión en la que compartiste contexto o decisiones del proyecto
- Después de completar una tarea con elecciones de planificación claras

### Ejemplo de interacción

```
Tú: Recuerda que prefiero indentación de 4 espacios y type hints.
IA: [calls MemorySave] Memory saved: 'coding_style' [feedback/user]

Tú: /memory
  1 memory/memories:
  [feedback  |user   ] coding_style.md
    Preferencias de formato para Python

Tú: /memory python
  Found 1 relevant memory for 'python':
  [feedback/user] coding_style
    Prefer 4-space indentation and full type hints in all Python code.

Tú: /memory consolidate
  ✓ Consolidated 1 memory: user_prefers_verbose_commit_messages
```

**Advertencias de obsolescencia:** las memorias con más de 1 día muestran una advertencia `⚠ stale`; afirmaciones sobre citas de archivo:línea o estado del código pueden estar desactualizadas; verifica antes de actuar.

**Búsqueda ordenada por IA:** `MemorySearch(query="...", use_ai=true)` usa el modelo para clasificar candidatos por relevancia antes de aplicar el reordenamiento confianza × recencia.

---

## Skills

Las skills son plantillas de prompt reutilizables que otorgan al modelo capacidades especializadas. Dos skills integradas vienen listas desde el principio; no requieren configuración.

**Skills integradas:**

| Disparador | Descripción |
|---|---|
| `/commit` | Revisar cambios staged y crear un commit git bien estructurado |
| `/review [PR]` | Revisar código o diff de PR con feedback estructurado |

**Inicio rápido — skill personalizada:**

```bash
mkdir -p ~/.cheetahclaws/skills
```

Crea `~/.cheetahclaws/skills/deploy.md`:

```markdown
---
name: deploy
description: Desplegar a un entorno
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

Ahora úsala:

```
Tú: /deploy staging 2.1.0
IA: [deploys version 2.1.0 to staging]
```

**Sustitución de argumentos:**
- `$ARGUMENTS` — la cadena completa de argumentos
- `$ARG_NAME` — sustitución posicional por argumento con nombre (primera palabra → primer nombre)
- Los argumentos faltantes se sustituyen por cadenas vacías

**Modos de ejecución:**
- `context: inline` (por defecto) — se ejecuta dentro del historial actual de conversación
- `context: fork` — se ejecuta como subagente aislado con historial fresco; admite sobrescritura de `model`

**Prioridad** (la más alta gana): nivel proyecto > nivel usuario > integrada

**Listar skills:** `/skills` — muestra disparadores, sugerencia de argumentos, origen y `when_to_use`

**Rutas de búsqueda de skills:**

```
./.cheetahclaws/skills/     # nivel proyecto (sobrescribe nivel usuario)
~/.cheetahclaws/skills/     # nivel usuario
```

---

## Subagentes

El modelo puede lanzar subagentes independientes para manejar tareas en paralelo.

**Tipos de agente especializados** — integrados:

| Tipo | Optimizado para |
|---|---|
| `general-purpose` | Investigación, exploración, tareas de varios pasos |
| `coder` | Escribir, leer y modificar código |
| `reviewer` | Seguridad, corrección y calidad del código |
| `researcher` | Búsqueda web y consulta de documentación |
| `tester` | Escribir y ejecutar pruebas |

**Uso básico:**
```
Tú: Busca todos los comentarios TODO en este código y resúmelos.
IA: [calls Agent(prompt="...", subagent_type="researcher")]
    El subagente lee archivos, busca TODOs...
    Resultado: encontró 12 TODOs en 5 archivos...
```

**Modo en segundo plano** — lanzar sin esperar, recoger el resultado más tarde:
```
IA: [calls Agent(prompt="run all tests", name="test-runner", wait=false)]
IA: [continues other work...]
IA: [calls CheckAgentResult / SendMessage to follow up]
```

**Aislamiento con git worktree** — los agentes trabajan en una rama aislada sin conflictos:
```
Agent(prompt="refactor auth module", isolation="worktree")
```
El worktree se limpia automáticamente si no hubo cambios; si los hubo, se informa el nombre de la rama.

**Tipos de agente personalizados** — crea `~/.cheetahclaws/agents/myagent.md`:
```markdown
---
name: myagent
description: Especializado en X
model: claude-haiku-4-5-20251001
tools: [Read, Grep, Bash]
---
Prompt del sistema adicional para este tipo de agente.
```

**Listar agentes en ejecución:** `/agents`

Los subagentes tienen historial de conversación independiente, comparten el sistema de archivos y están limitados a 3 niveles de anidamiento.

---

## MCP (Model Context Protocol)

MCP te permite conectar cualquier servidor externo de herramientas, ya sea un subproceso local o un endpoint HTTP remoto, y Claude puede usar sus herramientas automáticamente. Este es el mismo protocolo que usa Claude Code para ampliar sus capacidades.

### Transportes compatibles

| Transporte | `type` en config | Descripción |
|---|---|---|
| **stdio** | `"stdio"` | Lanza un subproceso local (el caso más común) |
| **SSE** | `"sse"` | Flujo HTTP Server-Sent Events |
| **HTTP** | `"http"` | HTTP POST streamable (servidores más recientes) |

### Configuración

Coloca un archivo `.mcp.json` en el directorio del proyecto **o** edita `~/.cheetahclaws/mcp.json` para servidores globales del usuario.

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

Prioridad de configuración: `.mcp.json` (proyecto) sobrescribe `~/.cheetahclaws/mcp.json` (usuario) por nombre de servidor.

### Inicio rápido

```bash
# Instala un servidor MCP popular
pip install uv        # uv incluye uvx
uvx mcp-server-git --help   # verifica que funcione

# Añádelo a la configuración del usuario vía REPL
/mcp add git uvx mcp-server-git

# O crea .mcp.json en el directorio del proyecto, y luego:
/mcp reload
```

### Comandos REPL

```
/mcp                          # listar servidores + sus herramientas + estado de conexión
/mcp reload                   # reconectar todos los servidores, refrescar lista de herramientas
/mcp reload git               # reconectar un solo servidor
/mcp add myserver uvx mcp-server-x   # añadir servidor stdio
/mcp remove myserver          # eliminar de la configuración del usuario
```

### Cómo usa Claude las herramientas MCP

Una vez conectadas, Claude puede invocar herramientas MCP directamente:

```
Tú: ¿Qué archivos cambiaron en el último commit git?
IA: [calls mcp__git__git_diff_staged()]
    → muestra la salida diff del servidor MCP git
```

Los nombres de herramientas siguen el patrón `mcp__<server_name>__<tool_name>`. Todos los caracteres que no sean alfanuméricos o `_` se reemplazan automáticamente por `_`.

### Servidores MCP populares

| Servidor | Instalación | Proporciona |
|---|---|---|
| `mcp-server-git` | `uvx mcp-server-git` | operaciones git (status, diff, log, commit) |
| `mcp-server-filesystem` | `uvx mcp-server-filesystem <path>` | leer/escribir/listar archivos |
| `mcp-server-fetch` | `uvx mcp-server-fetch` | herramienta de fetch HTTP |
| `mcp-server-postgres` | `uvx mcp-server-postgres <conn-str>` | consultas PostgreSQL |
| `mcp-server-sqlite` | `uvx mcp-server-sqlite --db-path x.db` | consultas SQLite |
| `mcp-server-brave-search` | `uvx mcp-server-brave-search` | búsqueda web Brave |

> Consulta el registro completo en [modelcontextprotocol.io/servers](https://modelcontextprotocol.io/servers)

---

## Sistema de plugins

El paquete `plugin/` te permite ampliar cheetahclaws con herramientas, skills y servidores MCP adicionales desde repositorios git o directorios locales.

### Instalar un plugin

```bash
/plugin install my-plugin@https://github.com/user/my-plugin
/plugin install local-plugin@/path/to/local/plugin
```

### Gestionar plugins

```bash
/plugin                   # listar plugins instalados
/plugin enable my-plugin  # habilitar un plugin deshabilitado
/plugin disable my-plugin # deshabilitar sin desinstalar
/plugin disable-all       # deshabilitar todos los plugins
/plugin update my-plugin  # traer la última versión desde git
/plugin uninstall my-plugin
/plugin info my-plugin    # mostrar detalles del manifiesto
```

### Motor de recomendación de plugins

```bash
/plugin recommend                    # autodetectar a partir de los archivos del proyecto
/plugin recommend "docker database"  # recomendar por contexto de palabras clave
```

El motor compara tu contexto con un marketplace curado (git-tools, python-linter, docker-tools, sql-tools, test-runner, diagram-tools, aws-tools, web-scraper) usando puntuación por etiquetas y palabras clave.

### Manifiesto del plugin (plugin.json)

```json
{
  "name": "my-plugin",
  "version": "0.1.0",
  "description": "Hace algo útil",
  "author": "you",
  "tags": ["git", "python"],
  "tools": ["tools"],        // módulos Python que exportan TOOL_DEFS
  "skills": ["skills/my.md"],
  "mcp_servers": {},
  "dependencies": ["httpx"]  // paquetes pip
}
```

Alternativamente usa frontmatter YAML en `PLUGIN.md`.

### Ámbitos

| Ámbito | Ubicación | Configuración |
|-------|----------|--------|
| usuario (por defecto) | `~/.cheetahclaws/plugins/` | `~/.cheetahclaws/plugins.json` |
| proyecto | `.cheetahclaws/plugins/` | `.cheetahclaws/plugins.json` |

Usa la bandera `--project`: `/plugin install name@url --project`

---

## Herramienta AskUserQuestion

Claude puede pausar a mitad de tarea e interactuar contigo haciendo una pregunta antes de continuar.

**Ejemplo de invocación por Claude:**
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

**Lo que ves en la terminal:**
```
❓ Pregunta del asistente:
   Which database should I use?

  [1] SQLite — Simple, file-based
  [2] PostgreSQL — Full-featured, requires server
  [0] Escribir una respuesta personalizada

Tu elección (número o texto):
```

- Selecciona por número o escribe texto libre directamente
- Claude recibe tu respuesta y continúa la tarea
- Timeout de 5 minutos (devuelve "(no answer — timeout)" si no respondes)

---

## Gestión de tareas

El paquete `task/` ofrece a Claude (y a ti) una lista estructurada de tareas para seguir trabajos de varios pasos dentro de una sesión.

### Herramientas disponibles para Claude

| Herramienta | Parámetros | Qué hace |
|------|-----------|--------------|
| `TaskCreate` | `subject`, `description`, `active_form?`, `metadata?` | Crear una tarea; devuelve `#id created: subject` |
| `TaskUpdate` | `task_id`, `subject?`, `description?`, `status?`, `owner?`, `add_blocks?`, `add_blocked_by?`, `metadata?` | Actualizar cualquier campo; `status='deleted'` elimina la tarea |
| `TaskGet` | `task_id` | Devolver detalles completos de una tarea |
| `TaskList` | _(none)_ | Listar todas las tareas con iconos de estado y bloqueos pendientes |

**Estados válidos:** `pending` → `in_progress` → `completed` / `cancelled` / `deleted`

### Aristas de dependencia

```
TaskUpdate(task_id="3", add_blocked_by=["1","2"])
# La tarea 3 ahora está bloqueada por las tareas 1 y 2.
# Las aristas inversas se establecen automáticamente: las tareas 1 y 2 reciben la tarea 3 en su lista "blocks".
```

Las tareas completadas se consideran resueltas; `TaskList` oculta su efecto de bloqueo sobre las dependientes.

### Persistencia

Las tareas se guardan en `.cheetahclaws/tasks.json` en el directorio de trabajo actual después de cada mutación y se recargan al primer acceso.

### Comandos REPL

```
/tasks                    listar todas las tareas
/tasks create <subject>   crear rápidamente una tarea
/tasks start <id>         marcar como in_progress
/tasks done <id>          marcar como completed
/tasks cancel <id>        marcar como cancelled
/tasks delete <id>        eliminar una tarea
/tasks get <id>           mostrar detalles completos
/tasks clear              eliminar todas las tareas
```

### Flujo típico de Claude

```
Usuario: implementa la funcionalidad de login

Claude:
  TaskCreate(subject="Design auth schema", description="JWT vs session")  → #1
  TaskCreate(subject="Write login endpoint", description="POST /auth/login") → #2
  TaskCreate(subject="Write tests", description="Unit + integration") → #3
  TaskUpdate(task_id="2", add_blocked_by=["1"])
  TaskUpdate(task_id="3", add_blocked_by=["2"])

  TaskUpdate(task_id="1", status="in_progress", active_form="Designing schema")
  ... (hace el trabajo) ...
  TaskUpdate(task_id="1", status="completed")
  TaskList()  → la tarea 2 ya no está bloqueada
  ...
```

---

## Entrada por voz

CheetahClaws v3.05 añade una canalización completamente offline de voz a prompt. Habla tu solicitud y se transcribirá y enviará como si la hubieras escrito.

### Inicio rápido

```bash
# 1. Instala un backend de grabación (elige uno)
pip install sounddevice        # recomendado: multiplataforma, sin binario extra
# sudo apt install alsa-utils  # alternativa arecord en Linux
# sudo apt install sox         # alternativa SoX rec

# 2. Instala un backend STT local (recomendado — offline, sin API key)
pip install faster-whisper numpy

# 3. Inicia CheetahClaws y habla
cheetahclaws
[myproject] ❯ /voice
  🎙  Escuchando… (habla ahora, se detiene automáticamente con el silencio, Ctrl+C para cancelar)
  🎙  ████
✓  Transcribed: "fix the authentication bug in user.py"
[envío automático…]
```

### Backends STT (probados en este orden)

| Backend | Instalación | Notas |
|---|---|---|
| `faster-whisper` | `pip install faster-whisper` | **Recomendado** — local, offline, más rápido, GPU opcional |
| `openai-whisper` | `pip install openai-whisper` | Local, offline, modelo original de OpenAI |
| API Whisper de OpenAI | establece `OPENAI_API_KEY` | En la nube, requiere internet + clave API |

Sobrescribe el tamaño del modelo Whisper con `NANO_CLAUDE_WHISPER_MODEL` (por defecto: `base`):

```bash
export NANO_CLAUDE_WHISPER_MODEL=small   # mejor precisión, más lento
export NANO_CLAUDE_WHISPER_MODEL=tiny    # más rápido, más ligero
```

### Backends de grabación (probados en este orden)

| Backend | Instalación | Notas |
|---|---|---|
| `sounddevice` | `pip install sounddevice` | **Recomendado** — multiplataforma, nativo en Python |
| `arecord` | `sudo apt install alsa-utils` | Linux ALSA, no necesita pip |
| `sox rec` | `sudo apt install sox` / `brew install sox` | Detección de silencio integrada |

### Refuerzo de términos clave

Antes de cada grabación, CheetahClaws extrae vocabulario de programación de:
- **Rama git** (por ejemplo, `feat/voice-input` → "feat", "voice", "input")
- **Nombre de la raíz del proyecto** (por ejemplo, "cheetahclaws")
- **Nombres recientes de archivos fuente** (por ejemplo, `authentication_handler.py` → "authentication", "handler")
- **Términos globales de programación**: `MCP`, `grep`, `TypeScript`, `OAuth`, `regex`, `gRPC`, …

Estos términos se pasan como `initial_prompt` de Whisper para que el motor STT prefiera la ortografía correcta de términos de programación.

### Comandos

| Comando | Descripción |
|---|---|
| `/voice` | Grabar voz y enviar automáticamente la transcripción como tu siguiente prompt |
| `/voice status` | Mostrar qué backends de grabación y STT están disponibles, además del micrófono activo |
| `/voice lang <code>` | Establecer el idioma de transcripción (`en`, `zh`, `ja`, `de`, `fr`, …; por defecto `auto`) |
| `/voice device` | Listar todos los micrófonos de entrada y seleccionar uno interactivamente; se guarda para la sesión |

### Seleccionar un micrófono

En sistemas con varias interfaces de audio (auriculares USB, dispositivos virtuales, etc.) puedes elegir exactamente el dispositivo de entrada:

```
[myproject] ❯ /voice device
  🎙  Dispositivos de entrada disponibles:
    0. Micrófono integrado
    1. Auriculares USB (USB Audio)  ◀  (seleccionado actualmente)
    2. Entrada virtual (BlackHole)
  Selecciona el dispositivo # (Enter para cancelar): 1
✓  Micrófono establecido en: [1] Auriculares USB (USB Audio)
```

El dispositivo seleccionado se muestra en `/voice status` y se usa para todas las grabaciones posteriores hasta que lo cambies o reinicies.

### Comparación con Claude Code

| | Claude Code | CheetahClaws v3.05 |
|---|---|---|
| Servicio STT | WebSocket privado de Anthropic (`voice_stream`) | `faster-whisper` / `openai-whisper` / API de OpenAI |
| Requiere OAuth de Anthropic | Sí | **No** |
| Funciona offline | No | **Sí** (con Whisper local) |
| Sugerencias de términos clave | parámetro `keyterms` de Deepgram | `initial_prompt` de Whisper (git + archivos + vocabulario) |
| Soporte de idiomas | códigos permitidos por el servidor | cualquier idioma compatible con Whisper |

---

## Brainstorm

`/brainstorm` ejecuta un debate estructurado de varias personas de IA sobre tu proyecto y luego sintetiza todas las perspectivas en un plan accionable.

### Cómo funciona

1. **Instantánea de contexto** — lee `README.md`, `CLAUDE.md` y la lista de archivos raíz del directorio de trabajo actual.
2. **Número de agentes** — se te pide elegir cuántos agentes quieres (2–100, por defecto 5). Pulsa Enter para usar el valor predeterminado.
3. **Generación dinámica de personas** — el modelo genera N roles expertos adaptados a tu tema. Los temas de software obtienen arquitectos e ingenieros; la geopolítica obtiene analistas, diplomáticos y economistas; los temas de negocio obtienen estrategas y expertos de mercado. Si falla la generación, usa personas técnicas integradas.
4. **Los agentes debaten secuencialmente**, cada uno construyendo sobre las respuestas anteriores.
5. **Salida guardada** en `brainstorm_outputs/brainstorm_YYYYMMDD_HHMMSS.md` en el directorio actual.
6. **Síntesis** — el agente principal lee el archivo guardado y produce un Master Plan priorizado.

**Ejemplos de personas por tema:**

| Tema | Ejemplos de personas generadas |
|---|---|
| Arquitectura de software | 🏗️ Arquitecto · 💡 Innovador de producto · 🛡️ Ingeniero de seguridad · 🔧 Líder de calidad de código · ⚡ Especialista en rendimiento |
| Geopolítica EE. UU.-Irán | 🌍 Analista geopolítico · ⚖️ Experto en derecho internacional · 💰 Economista energético · 🎖️ Estratega militar · 🕊️ Mediador de conflictos |
| Estrategia empresarial | 📈 Estratega de mercado · 💼 Responsable de operaciones · 🔍 Inteligencia competitiva · 💡 Director de innovación · 📊 Analista financiero |

### Uso

```
[myproject] ❯ /brainstorm
  ¿Cuántos agentes? (2-100, por defecto 5) > 5

[myproject] ❯ /brainstorm improve plugin architecture
  ¿Cuántos agentes? (2-100, por defecto 5) > 3

[myproject] ❯ /brainstorm US-Iran geopolitics
  ¿Cuántos agentes? (2-100, por defecto 5) > 7
```

### Ejemplo de salida

```
[myproject] ❯ /brainstorm medical research funding
  ¿Cuántos agentes? (2-100, por defecto 5) > 3
Generando 3 personas expertas apropiadas para el tema...
Iniciando sesión Brainstorm de 3 agentes sobre: medical research funding
Generando perspectivas diversas...
🩺 Clinical Trials Director está pensando...
  └─ Perspectiva capturada.
⚖️ Medical Ethics Committee Member está pensando...
  └─ Perspectiva capturada.
💰 Health Economics Policy Analyst está pensando...
  └─ Perspectiva capturada.
✓  ¡Brainstorm completado! Resultados guardados en brainstorm_outputs/brainstorm_20260405_224117.md

   ── Análisis del agente principal ──
[aquí aparece el Master Plan sintetizado…]
```

### Notas

- Brainstorm usa el **modelo actualmente seleccionado** (`/model` para comprobarlo). Un modelo potente (Claude Sonnet/Opus, GPT-4o o un gran modelo local) da los mejores resultados.
- Con muchos agentes (20+) la sesión puede tardar varios minutos según la velocidad del modelo.
- Instala `faker` (`pip install faker`) para nombres de persona aleatorios; si no, usa nombres integrados.
- Los archivos de salida se acumulan en `brainstorm_outputs/`; desde v3.5.5 ya está añadido a `.gitignore`.
- Las respuestas largas siguen renderizándose en vivo pero solo muestran la última pantalla (una ventana de cola acotada) hasta que terminan, por lo que las líneas duplicadas/obsoletas se evitan automáticamente. Si la salida aún se ve corrupta en SSH (líneas repetidas), ejecuta `/config rich_live=false` para desactivar por completo Rich Live streaming.

---

## Modo de desarrollador SSJ

`/ssj` abre un menú persistente avanzado: un único punto de entrada para los flujos de trabajo de desarrollo más comunes, para que no tengas que recordar nombres de comandos.

<div align=center>
<img src="../media/demos/ssj_demo.gif" width="850"/>
</div>

### Opciones del menú

| # | Nombre | Qué hace |
|---|------|--------------|
| 1 | 💡 Brainstorm | Debate IA multi-persona → Master Plan → genera automáticamente `brainstorm_outputs/todo_list.txt` |
| 2 | 📋 Show TODO | Ver `brainstorm_outputs/todo_list.txt` con indicadores ✓/○ y números de tareas pendientes |
| 3 | 👷 Worker | Implementar automáticamente tareas pendientes (todas o seleccionadas por número) |
| 4 | 🧠 Debate | Elegir un archivo y el número de agentes — un panel de expertos debate el diseño ronda por ronda; el resultado se guarda junto al archivo |
| 5 | ✨ Propose | Elegir un archivo — la IA propone mejoras específicas con código |
| 6 | 🔎 Review | Elegir un archivo — revisión estructurada de código con puntuaciones de 1–10 por dimensión |
| 7 | 📘 Readme | Elegir un archivo — generar automáticamente un README profesional |
| 8 | 💬 Commit | Analizar el diff git y sugerir un mensaje de commit convencional |
| 9 | 🧪 Scan | Resumir todos los cambios staged/unstaged y sugerir siguientes pasos |
| 10 | 📝 Promote | Leer la última salida de brainstorm → convertir ideas en tareas `todo_list.txt` |
| 0 | 🚪 Exit | Volver al REPL principal |

### Uso

```
[myproject] ❯ /ssj

╭─ SSJ Developer Mode ⚡ ─────────────────────────
│
│   1.  💡  Brainstorm — Debate IA multi-persona
│   2.  📋  Show TODO  — Ver todo_list.txt
│   3.  👷  Worker     — Implementar automáticamente tareas pendientes
│   4.  🧠  Debate     — Debate experto sobre un archivo
│   5.  ✨  Propose    — Mejora IA de un archivo
│   6.  🔎  Review     — Análisis rápido de archivo
│   7.  📘  Readme     — Generar README.md automáticamente
│   8.  💬  Commit     — Mensaje de commit sugerido por IA
│   9.  🧪  Scan       — Analizar diff git
│  10.  📝  Promote    — De idea a tareas
│   0.  🚪  Salir del modo SSJ
│
╰──────────────────────────────────────────────

  ⚡ SSJ » 1
  Tema (Enter para general): cheetahclaws plugin system

  # → Brainstorm se ejecuta, guarda en brainstorm_outputs/, genera todo_list.txt
  # → El menú se vuelve a abrir automáticamente tras cada acción

  ⚡ SSJ » 2
  # → Muestra tareas pendientes numeradas de brainstorm_outputs/todo_list.txt

  ⚡ SSJ » 3
  Tarea # (Enter para todas, o p. ej. 1,4,6): 2
  # → Worker implementa la tarea #2 y la marca como hecha
```

### Passthrough de comandos slash

Cualquier `/command` escrito en el prompt `⚡ SSJ »` se reenvía al REPL:

```
  ⚡ SSJ » /model gpt-4o
  # → cambia de modelo y vuelve a abrir el menú SSJ

  ⚡ SSJ » /exit
  # → sale de cheetahclaws inmediatamente
```

### Comando Worker

`/worker` (también accesible como opción 3 de SSJ) lee `brainstorm_outputs/todo_list.txt` e implementa automáticamente cada tarea pendiente:

```
[myproject] ❯ /worker
  ✓ Worker iniciando — 3 tarea(s) por implementar
    1. ○ Añadir spinner animado de brainstorm
    2. ○ Implementar indicador de escritura en Telegram
    3. ○ Escribir GIF de demostración SSJ para el README

  ── Worker (1/3): Añadir spinner animado de brainstorm ──
  [el modelo lee el código, implementa el cambio y marca la tarea como hecha]

[myproject] ❯ /worker 2,3
  # Implementar solo las tareas 2 y 3

[myproject] ❯ /worker --path docs/tasks.md
  # Usar una ruta personalizada al archivo todo

[myproject] ❯ /worker --workers 2
  # Procesar solo las primeras 2 tareas pendientes en esta ejecución
```

**Detección inteligente de rutas** — si pasas por error un archivo de salida de brainstorm (`.md`), Worker lo detecta y ofrece redirigirte al `todo_list.txt` correspondiente en la misma carpeta. Si ese archivo aún no existe, te ofrece generarlo primero a partir de la salida de brainstorm (SSJ Promote) y después ejecutar Worker automáticamente.

### Comando Debate

La opción 4 de SSJ ejecuta un debate estructurado de varias rondas entre expertos sobre cualquier archivo:

```
  ⚡ SSJ » 4

  Archivos en brainstorm_outputs/:
    1. brainstorm_20260406_143022.md
    2. cheetahclaws.py

  Archivo a debatir #: 2
  Número de agentes de debate (Enter para 2): 3
  ℹ El resultado del debate se guardará en: cheetahclaws_debate_143055.md

⚔️  Reuniendo panel de expertos...
  Experto 1: 🏗️ Líder de arquitectura — foco: diseño de sistemas y modularidad
  Experto 2: 🔐 Ingeniero de seguridad — foco: superficie de ataque y validación de entradas
  Experto 3: ⚡ Especialista en rendimiento — foco: latencia y uso de memoria

⚔️  Ronda 1/5 — Experto 1 pensando...
  [Architecture Lead da su argumento inicial...]

💬  Ronda 1/5 — Experto 2 formulando...
  [Security Engineer responde...]
  ...

📜  Redactando consenso final...
  [el modelo escribe el consenso + guarda la transcripción]
✓ Debate completado. Guardado en cheetahclaws_debate_143055.md
```

- El número de agentes es configurable (mínimo 2, por defecto 2). Las rondas se fijan en `agents × 2 − 1` para una estructura completa de apertura-cierre.
- Un spinner animado muestra la ronda y el experto actual (`⚔️ Ronda 2/3 — Experto 1 pensando...`), deteniéndose justo cuando ese experto empieza a producir salida.
- La transcripción completa del debate y el consenso ordenado se guardan en `<filename>_debate_HHMMSS.md` **en el mismo directorio que el archivo debatido**.

---

## Puente de Telegram

`/telegram` convierte cheetahclaws en un bot de Telegram: recibe mensajes desde tu teléfono, ejecuta el modelo con acceso completo a herramientas y responde automáticamente.

<div align=center>
<img src="../media/demos/telegram_demo.gif" width="850"/>
</div>

### Configuración (una sola vez)

1. Abre [@BotFather](https://t.me/BotFather) en Telegram → `/newbot` → copia el token.
2. Envía cualquier mensaje a tu nuevo bot (por ejemplo, “hi”), luego abre la URL de abajo en tu navegador — reemplaza `<TOKEN>` por tu token real:

```
https://api.telegram.org/bot<TOKEN>/getUpdates
```

La respuesta es JSON. Busca `"chat"` → `"id"` — ese número es tu chat ID:

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

> **Consejo:** si `result` está vacío, vuelve a Telegram, envía otro mensaje a tu bot y actualiza la URL.

3. Configura cheetahclaws (ejemplo con los valores de arriba):

```
[myproject] ❯ /telegram 7812345678:AAFxyz123abcDEF456ghiJKL789 987654321
  ✓ Configuración de Telegram guardada.
  ✓ Conectado a @your_bot_name. Iniciando puente...
  ✓ Puente de Telegram activo. Chat ID: 987654321
  ℹ Envía mensajes a tu bot: se procesarán aquí.
  ℹ Deténlo con /telegram stop o enviando /stop en Telegram.
```

El token y `chat_id` se guardan en `~/.cheetahclaws/config.json`. En el siguiente arranque, el puente **se inicia automáticamente** si está configurado; el banner de inicio muestra `flags: [telegram]`.

### Cómo funciona

```
Teléfono (Telegram)               terminal cheetahclaws
──────────────────                ──────────────────────────
"List Python files"      →        📩 Telegram: List Python files
                                  [se envía indicador de escritura...]
                                  ⚙ Glob(**/*.py) → 5 archivos
                                  ⚙ respuesta ensamblada
                          ←       "agent.py, tools.py, ..."
```

- **Indicador de escritura** se envía cada 4 segundos mientras el modelo procesa, para que el chat se sienta receptivo.
- **Remitentes no autorizados** reciben `⛔ Unauthorized.` y sus mensajes se descartan.
- **Passthrough de comandos slash**: envía `/cost`, `/model gpt-4o`, `/clear`, etc. desde Telegram y se ejecutarán en cheetahclaws.
- **Menús interactivos a través de Telegram**: los comandos con prompts interactivos (por ejemplo, selector de modelo `/ollama`, `/permission`, restauración `/checkpoint`) ahora se ejecutan en un hilo en segundo plano para que el bucle de sondeo siga libre. Las opciones del menú se envían como un mensaje de Telegram y la siguiente respuesta que envíes se usa como selección.
- **`/stop` o `/off`** enviados desde Telegram detienen el puente limpiamente.

### Comandos

| Comando | Descripción |
|---|---|
| `/telegram <token> <chat_id>` | Configurar token + chat_id y luego iniciar el puente |
| `/telegram` | Iniciar el puente usando la configuración guardada |
| `/telegram status` | Mostrar el estado de ejecución y el chat_id |
| `/telegram stop` | Detener el puente |

### Autoarranque

Si tanto `telegram_token` como `telegram_chat_id` están establecidos en `~/.cheetahclaws/config.json`, el puente se inicia automáticamente cada vez que se arranca cheetahclaws:

```
╭─ CheetahClaws ────────────────────────────────╮
│  Modelo:      claude-opus-4-6
│  Permisos:    auto   flags: [telegram]
│  Escribe /help para ver comandos, Ctrl+C para cancelar        │
╰───────────────────────────────────────────────╯
✓ Puente de Telegram iniciado (auto). Bot: @your_bot_name
```

---

## Monitoreo proactivo en segundo plano

CheetahClaws v3.5.2 añade un **daemon centinela** que despierta automáticamente al agente tras un período configurable de inactividad, sin necesidad de prompt del usuario. Esto permite casos de uso como monitoreo continuo de logs, sondeo de scripts de mercado o comprobaciones de código programadas.

### Inicio rápido

```
[myproject] ❯ /proactive 5m
Sondeo proactivo en segundo plano: ON  (se activará cada 300s de inactividad)

[myproject] ❯ sigue monitoreando el log de compilación y avísame si aparecen errores

╭─ Claude ● ─────────────────────────
│ Entendido. Revisaré el log de compilación cada vez que despierte.

[Evento en segundo plano activado]
╭─ Claude ● ─────────────────────────
│ ⚙ Bash(tail -50 build.log)
│ ✓ → Falló la compilación: ImportError en auth.py línea 42
│ **Acción necesaria:** corrige la importación antes de la siguiente ejecución de CI.
```

### Comandos

| Comando | Descripción |
|---|---|
| `/proactive` | Mostrar el estado actual (ON/OFF e intervalo) |
| `/proactive 5m` | Activar — disparar cada 5 minutos de inactividad |
| `/proactive 30s` | Activar — disparar cada 30 segundos |
| `/proactive 1h` | Activar — disparar cada hora |
| `/proactive off` | Desactivar el sondeo centinela |

Sufijos de duración: `s` = segundos, `m` = minutos, `h` = horas. Un entero plano = segundos.

### Cómo funciona

- Un hilo daemon en segundo plano se inicia cuando arranca el REPL (pausado por defecto).
- El daemon comprueba cada segundo el tiempo transcurrido desde la última interacción del usuario o del agente.
- Cuando se alcanza el umbral de inactividad, llama al agente con un prompt de reactivación.
- El `threading.Lock` usado por el bucle principal del agente garantiza que los despertares nunca interrumpan una sesión activa: se encolan y se ejecutan después de que termine el turno actual.
- Las excepciones del watcher se registran con `traceback` para que los fallos sean visibles y depurables.

### Complementa a SleepTimer

| | `SleepTimer` | `/proactive` |
|---|---|---|
| Quién lo inicia | El agente | El usuario |
| Activación | Tras un retraso fijo desde ahora | Tras N segundos de inactividad |
| Caso de uso | "Vuelve a comprobar en 10 minutos" | "Sigue vigilando hasta que deje de escribir" |

---

## Sistema de checkpoints

CheetahClaws crea automáticamente instantáneas de tu conversación y de cualquier archivo editado tras cada turno, para que siempre puedas volver a un estado anterior.

### Cómo funciona

- **Instantánea automática** — tras cada turno, el sistema de checkpoints guarda los mensajes actuales de la conversación, los conteos de tokens y una copia de seguridad copy-on-write de cada archivo escrito o editado en ese turno.
- **Ventana deslizante de 100 instantáneas** — las más antiguas se expulsan automáticamente al alcanzar el límite.
- **Limitación inteligente** — si no cambió nada (sin nuevos mensajes, sin ediciones de archivos) desde la última instantánea, se omite.
- **Instantánea inicial** — se captura al iniciar la sesión, para que siempre puedas volver a un estado limpio.
- **Almacenamiento** — `~/.nano_claude/checkpoints/<session_id>/` (metadatos de instantáneas + archivos de backup).

### Comandos

| Comando | Descripción |
|---|---|
| `/checkpoint` | Listar todas las instantáneas de la sesión actual |
| `/checkpoint <id>` | Rebobinar: restaurar archivos al estado de la instantánea `<id>` y recortar la conversación hasta ese punto |
| `/checkpoint clear` | Eliminar todas las instantáneas de la sesión actual |
| `/rewind` | Alias de `/checkpoint` |

### Ejemplo

```
[myproject] ❯ /checkpoint
  Checkpoints (4 total):
  #1  [turn 0] 14:02:11  "(estado inicial)"           0 archivos
  #2  [turn 1] 14:03:45  "Crear app.py"               1 archivo
  #3  [turn 2] 14:05:12  "Añadir manejo de errores"   1 archivo
  #4  [turn 3] 14:06:30  "Explicar el código"         1 archivo

[myproject] ❯ /checkpoint 2
  Rebobinado al checkpoint #2 (turno 1)
  Restaurado: app.py
  Conversación recortada a 2 mensajes.
```

---

## Modo plan

El modo plan es un flujo estructurado para abordar tareas complejas y de múltiples archivos: primero Claude analiza el código en una fase de solo lectura y escribe un plan explícito; después el usuario lo aprueba antes de comenzar la implementación.

### Cómo funciona

En modo plan:
- **Solo se permiten lecturas** (`Read`, `Glob`, `Grep`, `WebFetch`, `WebSearch`, comandos `Bash` seguros).
- **Las escrituras se bloquean** en todas partes **excepto** en el archivo de plan dedicado (`.nano_claude/plans/<session_id>.md`).
- Los intentos de escritura bloqueados producen un mensaje útil en vez de un prompt de permiso.
- El prompt del sistema se amplía con instrucciones de modo plan.
- Tras la compactación, el contexto del archivo de plan se restaura automáticamente.

### Flujo con comandos slash

```
[myproject] ❯ /plan add WebSocket support
  Modo plan activado.
  Archivo de plan: .nano_claude/plans/a3f9c1b2.md
  Lecturas permitidas. Todas las demás escrituras bloqueadas (excepto el archivo de plan).

[myproject] ❯ <describe your task>
  [Claude lee archivos, construye contexto y escribe el plan en el archivo de plan]

[myproject] ❯ /plan
  # Plan: Añadir soporte WebSocket

  ## Fase 1: Crear ws_handler.py
  ## Fase 2: Modificar server.py para montar el handler
  ## Fase 3: Añadir pruebas

[myproject] ❯ /plan done
  Modo plan desactivado. Modo de permisos restaurado a: auto
  Revisa el plan anterior y empieza a implementarlo cuando quieras.

[myproject] ❯ /plan status
  Modo plan: INACTIVO  (modo de permisos: auto)
```

### Flujo con herramientas del agente (autónomo)

Claude puede entrar y salir del modo plan de forma autónoma usando las herramientas `EnterPlanMode` y `ExitPlanMode`; ambas se aprueban automáticamente en todos los modos de permiso:

```
Usuario: Refactoriza el módulo de autenticación

Claude: [calls EnterPlanMode(task_description="Refactor auth module")]
  → lee auth.py, users.py, tests/test_auth.py ...
  → escribe el plan en .nano_claude/plans/...
  [calls ExitPlanMode()]
  → "Aquí está mi plan. Revísalo y apruébalo antes de que empiece."

Usuario: Está bien, adelante.
Claude: [implementa el plan]
```

### Comandos

| Comando | Descripción |
|---|---|
| `/plan <description>` | Entrar en modo plan con una descripción de tarea |
| `/plan` | Imprimir el contenido actual del archivo de plan |
| `/plan done` | Salir del modo plan y restaurar los permisos previos |
| `/plan status` | Mostrar si el modo plan está activo |

---

## Compresión de contexto

Las conversaciones largas se comprimen automáticamente para mantenerse dentro de la ventana de contexto del modelo.

**Dos capas:**

1. **Snip** — las salidas antiguas de herramientas (lecturas de archivos, resultados Bash) se truncan después de unos turnos. Rápido, sin coste API.
2. **Compactación automática** — cuando el uso de tokens supera el 70% del límite de contexto, los mensajes más antiguos se resumen mediante el modelo en una recapitulación concisa.

Esto ocurre de forma transparente. No necesitas hacer nada.

**Compactación manual** — también puedes activarla en cualquier momento con `/compact`. Una cadena de enfoque opcional indica al resumidor qué contexto debe priorizar:

```
[myproject] ❯ /compact
  Compactado: ~12400 → ~3200 tokens (~9200 ahorrados)

[myproject] ❯ /compact keep the WebSocket implementation details
  Compactado: ~11800 → ~3100 tokens (~8700 ahorrados)
```

Si el modo plan está activo, el contexto del archivo de plan se restaura automáticamente tras cualquier compactación.

---

## Vista diff

Cuando el modelo edita o sobrescribe un archivo, ves un diff estilo git:

```diff
  Cambios aplicados a config.py:

--- a/config.py
+++ b/config.py
@@ -12,7 +12,7 @@
     "model": "claude-opus-4-6",
-    "max_tokens": 8192,
+    "max_tokens": 16384,
     "permission_mode": "auto",
```

Las líneas verdes = añadidas; las rojas = eliminadas. Las creaciones de archivos nuevos muestran un resumen.

---

## Soporte para CLAUDE.md

Coloca un archivo `CLAUDE.md` en tu proyecto para dar al modelo contexto persistente sobre tu base de código. CheetahClaws lo encuentra automáticamente y lo inyecta en el prompt del sistema.

```
~/.claude/CLAUDE.md          # Global — se aplica a todos los proyectos
/your/project/CLAUDE.md      # Nivel proyecto — se encuentra subiendo desde cwd
```

**Ejemplo de `CLAUDE.md`:**

```markdown
# Project: FastAPI Backend

## Stack
- Python 3.12, FastAPI, PostgreSQL, SQLAlchemy 2.0, Alembic
- Tests: pytest, objetivo de cobertura 90%

## Convenciones
- Formatear con black, lint con ruff
- Type annotations completas obligatorias
- Los nuevos endpoints deben tener pruebas correspondientes

## Notas importantes
- Nunca hardcodees credenciales — usa variables de entorno
- No modifiques archivos de migración Alembic existentes
- La rama `staging` despliega automáticamente a staging al hacer push
```

---

## Gestión de sesiones

### Estructura de almacenamiento

Cada salida guarda automáticamente en tres lugares:

```
~/.cheetahclaws/sessions/
├── history.json                          ← maestro: todas las sesiones (capado)
├── mr_sessions/
│   └── session_latest.json              ← siempre la más reciente (/resume)
└── daily/
    ├── 2026-04-05/
    │   ├── session_110523_a3f9.json     ← archivos por día, se conservan los más nuevos
    │   └── session_143022_b7c1.json
    └── 2026-04-04/
        └── session_183100_3b4c.json
```

Cada archivo de sesión incluye metadatos:

```json
{
  "session_id": "a3f9c1b2",
  "saved_at": "2026-04-05 11:05:23",
  "turn_count": 8,
  "messages": [...]
}
```

### Autoguardado al salir

Cada vez que sales — mediante `/exit`, `/quit`, `Ctrl+C` o `Ctrl+D` — la sesión se guarda automáticamente:

```
✓ Sesión guardada → /home/.../.cheetahclaws/sessions/mr_sessions/session_latest.json
✓              → /home/.../.cheetahclaws/sessions/daily/2026-04-05/session_110523_a3f9.json  (id: a3f9c1b2)
✓   history.json: 12 sesiones / 87 turnos totales
```

### Reanudar rápidamente

Para continuar donde lo dejaste:

```bash
cheetahclaws
[myproject] ❯ /resume
✓  Sesión cargada desde …/mr_sessions/session_latest.json (42 mensajes)
```

Reanudar un archivo específico:

```bash
/resume session_latest.json          # carga desde mr_sessions/
/resume /absolute/path/to/file.json  # carga desde una ruta absoluta
```

### Guardado / carga manual

```bash
/save                          # guardar con nombre automático (session_TIMESTAMP_ID.json)
/save debug_auth_bug           # guardado con nombre personalizado en ~/.cheetahclaws/sessions/

/load                          # lista interactiva agrupada por fecha
/load debug_auth_bug           # cargar por nombre de archivo
```

**Lista interactiva de `/load`:**

```
  ── 2026-04-05 ──
  [ 1] 11:05:23  id:a3f9c1b2  turns:8   session_110523_a3f9.json
  [ 2] 09:22:01  id:7e2d4f91  turns:3   session_092201_7e2d.json

  ── 2026-04-04 ──
  [ 3] 22:18:00  id:3b4c5d6e  turns:15  session_221800_3b4c.json

  ── Historial completo ──
  [ H] Cargar TODO el historial  (3 sesiones / 26 turnos totales)  /home/.../.cheetahclaws/sessions/history.json

  Introduce número(s) (por ejemplo 1 o 1,2,3), H para historial completo, o Enter para cancelar >
```

- Introduce un solo número para cargar una sesión
- Introduce números separados por comas (por ejemplo, `1,3`) para fusionar varias sesiones en orden
- Introduce `H` para cargar el historial completo: muestra el número de mensajes y la estimación de tokens antes de confirmar

### Límites configurables

| Clave de configuración | Valor por defecto | Descripción |
|---|---|---|
| `session_daily_limit` | `5` | Máximo de archivos de sesión conservados por día en `daily/` |
| `session_history_limit` | `100` | Máximo de sesiones conservadas en `history.json` |

```bash
/config session_daily_limit=10
/config session_history_limit=200
```

### history.json — historial completo de conversaciones

`history.json` acumula todas las sesiones en un único lugar, lo que permite buscar en todo tu historial de conversaciones o analizar patrones de uso:

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

## Sincronización en la nube (GitHub Gist)

CheetahClaws v3.5.3 añade respaldo opcional en la nube de sesiones de conversación mediante **GitHub Gist**. Las sesiones se almacenan como Gists privados (JSON), navegables en la interfaz de GitHub. Sin dependencias adicionales: usa `urllib` de la stdlib de Python.

### Configuración (una vez)

1. Ve a [github.com/settings/tokens](https://github.com/settings/tokens) → **Generate new token (classic)**
2. Activa el permiso **`gist`**
3. Copia el token y ejecuta:

```
[myproject] ❯ /cloudsave setup ghp_xxxxxxxxxxxxxxxxxxxx
✓ Token de GitHub guardado (sesión iniciada como: Chauncygu). La sincronización en la nube está lista.
```

### Subir una sesión

```
[myproject] ❯ /cloudsave
Subiendo sesión a GitHub Gist…
✓ Sesión subida → https://gist.github.com/abc123def456
```

Añade una descripción opcional:

```
[myproject] ❯ /cloudsave push auth refactor debug session
```

### Sincronización automática al salir

```
[myproject] ❯ /cloudsave auto on
✓ Auto cloud-sync ON — la sesión se subirá a Gist al usar /exit.
```

A partir de ese momento, cada `/exit` o `/quit` sube automáticamente la sesión antes de cerrarse.

### Navegar y restaurar

```
[myproject] ❯ /cloudsave list
  Se encontraron 3 sesión(es):
  abc123de…  2026-04-05 11:02  auth refactor debug session
  7f9e12ab…  2026-04-04 22:18  proactive monitoring test
  3b4c5d6e…  2026-04-04 18:31

[myproject] ❯ /cloudsave load abc123de...full-gist-id...
✓ Sesión cargada desde Gist (42 mensajes).
```

### Referencia de comandos

| Comando | Descripción |
|---|---|
| `/cloudsave setup <token>` | Guardar token GitHub (requiere permiso `gist`) |
| `/cloudsave` | Subir la sesión actual a un Gist nuevo o existente |
| `/cloudsave push [desc]` | Subir con descripción opcional |
| `/cloudsave auto on\|off` | Activar/desactivar la subida automática al salir |
| `/cloudsave list` | Listar todos tus Gists de cheetahclaws |
| `/cloudsave load <gist_id>` | Descargar y restaurar una sesión |

---

## Estructura del proyecto

```text
cheetahclaws/
├── cheetahclaws.py        # Punto de entrada: REPL + comandos slash + renderizado diff + Rich Live streaming + daemon centinela proactivo + modo SSJ + puente de Telegram + comando Worker
├── agent.py               # Bucle del agente: streaming, despacho de herramientas, compactación
├── providers.py           # Multiproveedor: Anthropic, streaming OpenAI-compatible
├── tools.py               # Herramientas principales (Read/Write/Edit/Bash/Glob/Grep/Web/NotebookEdit/GetDiagnostics) + cableado del registro
├── tool_registry.py       # Registro de plugins de herramientas: registrar, buscar, ejecutar
├── compaction.py          # Compresión de contexto: snip + auto-resumen
├── context.py             # Constructor del prompt del sistema: CLAUDE.md + git + memoria
├── config.py              # Carga/guardado/valores por defecto de config; rutas DAILY_DIR, SESSION_HIST_FILE
├── cloudsave.py           # Sincronización en la nube con GitHub Gist (subir/descargar/listar sesiones)
│
├── multi_agent/           # Paquete multiagente
│   ├── __init__.py        # Reexportaciones
│   ├── subagent.py        # AgentDefinition, SubAgentManager, helpers de worktree
│   └── tools.py           # Agent, SendMessage, CheckAgentResult, ListAgentTasks, ListAgentTypes
├── subagent.py            # Shim de compatibilidad → multi_agent/
│
├── memory/                # Paquete de memoria
│   ├── __init__.py        # Reexportaciones
│   ├── types.py           # MEMORY_TYPES y guía de formato
│   ├── store.py           # guardar/cargar/eliminar/buscar, reconstrucción del índice MEMORY.md
│   ├── scan.py            # MemoryHeader, helpers de edad/frescura
│   ├── context.py         # get_memory_context(), truncado, búsqueda IA
│   └── tools.py           # MemorySave, MemoryDelete, MemorySearch, MemoryList
├── memory.py              # Shim de compatibilidad → memory/
│
├── skill/                 # Paquete de skills
│   ├── __init__.py        # Reexportaciones; importa builtin para registrar las integradas
│   ├── loader.py          # SkillDef, parse, load_skills, find_skill, substitute_arguments
│   ├── builtin.py         # Skills integradas: /commit, /review
│   ├── executor.py        # execute_skill(): inline o subagente fork
│   └── tools.py           # Skill, SkillList
├── skills.py              # Shim de compatibilidad → skill/
│
├── mcp/                   # Paquete MCP (Model Context Protocol)
│   ├── __init__.py        # Reexportaciones
│   ├── types.py           # MCPServerConfig, MCPTool, MCPServerState, helpers JSON-RPC
│   ├── client.py          # StdioTransport, HttpTransport, MCPClient, MCPManager
│   ├── config.py          # Carga .mcp.json (proyecto) + ~/.cheetahclaws/mcp.json (usuario)
│   └── tools.py           # Descubrimiento automático + registro de herramientas MCP en tool_registry
│
├── voice/                 # Paquete de entrada por voz (v3.05)
│   ├── __init__.py        # API pública: check_voice_deps, voice_input
│   ├── recorder.py        # Captura de audio: sounddevice → arecord → sox rec
│   ├── stt.py             # STT: faster-whisper → openai-whisper → API de OpenAI
│   └── keyterms.py        # Vocabulario técnico a partir de rama git + archivos del proyecto
│
├── checkpoint/            # Sistema de checkpoints (v3.5.6)
│   ├── __init__.py        # Exportaciones de API pública
│   ├── types.py           # Dataclasses FileBackup + Snapshot; MAX_SNAPSHOTS = 100
│   ├── store.py           # Backup a nivel de archivo, persistencia de snapshots, rewind, limpieza
│   └── hooks.py           # Intercepción de Write/Edit/NotebookEdit — respalda archivos antes de modificarlos
│
└── tests/                 # 263+ pruebas unitarias
    ├── test_mcp.py
    ├── test_memory.py
    ├── test_skills.py
    ├── test_subagent.py
    ├── test_tool_registry.py
    ├── test_compaction.py
    ├── test_diff_view.py
    ├── test_voice.py         # 29 pruebas de voz (sin hardware requerido)
    ├── test_checkpoint.py    # 24 pruebas unitarias de checkpoint
    ├── e2e_checkpoint.py     # prueba de ciclo de vida de checkpoint en 10 pasos
    ├── e2e_plan_mode.py      # prueba de permisos de modo plan en 10 pasos
    ├── e2e_plan_tools.py     # prueba de herramientas EnterPlanMode/ExitPlanMode en 8 pasos
    ├── e2e_compact.py        # prueba de compactación en 9 pasos
    └── e2e_commands.py       # prueba de /init /export /copy /status en 9 pasos
```

> **Para desarrolladores:** cada paquete de funcionalidad (`multi_agent/`, `memory/`, `skill/`, `mcp/`, `voice/`, `checkpoint/`) es autocontenido. Añade herramientas personalizadas llamando a `register_tool(ToolDef(...))` desde cualquier módulo importado por `tools.py`.

---

## FAQ

**P: ¿Cómo añado un servidor MCP?**

Opción 1 — vía REPL (servidor stdio):
```
/mcp add git uvx mcp-server-git
```

Opción 2 — crea `.mcp.json` en tu proyecto:
```json
{
  "mcpServers": {
    "git": {"type": "stdio", "command": "uvx", "args": ["mcp-server-git"]}
  }
}
```

Luego ejecuta `/mcp reload` o reinicia. Usa `/mcp` para comprobar el estado de conexión.

**P: Un servidor MCP muestra un error. ¿Cómo lo depuro?**

```
/mcp                    # muestra el mensaje de error por servidor
/mcp reload git         # intenta reconectar
```

Si el servidor usa stdio, asegúrate de que el comando está en tu `$PATH`:
```bash
which uvx               # debería imprimir una ruta
uvx mcp-server-git      # ejecútalo manualmente para ver errores
```

**P: ¿Puedo usar servidores MCP que requieran autenticación?**

Para servidores HTTP/SSE con token Bearer:
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

Para servidores stdio con autenticación por variables de entorno:
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

**P: Las llamadas a herramientas no funcionan con mi modelo local de Ollama.**

No todos los modelos admiten function calling. Usa uno de los modelos recomendados para tool calling: `qwen2.5-coder`, `llama3.3`, `mistral` o `phi4`.

```bash
ollama pull qwen2.5-coder
cheetahclaws --model ollama/qwen2.5-coder
```

**P: ¿Cómo me conecto a un servidor GPU remoto que ejecuta vLLM?**

```
/config custom_base_url=http://your-server-ip:8000/v1
/config custom_api_key=your-token
/model custom/your-model-name
```

**P: ¿Cómo consulto mi coste de API?**

```
/cost

  Input tokens:  3,421
  Output tokens:   892
  Coste estimado: $0.0648 USD
```

**P: ¿Puedo usar varias API keys en la misma sesión?**

Sí. Configura por adelantado todas las claves que necesites (mediante variables de entorno o `/config`). Luego cambia libremente de modelo: cada llamada usa la clave del proveedor activo.

**P: ¿Cómo hago que un modelo esté disponible en todos los proyectos?**

Añade las claves a `~/.bashrc` o `~/.zshrc`. Configura el modelo por defecto en `~/.cheetahclaws/config.json`:

```json
{ "model": "claude-sonnet-4-6" }
```

**P: Qwen / Zhipu devuelve texto corrupto.**

Asegúrate de que tu `DASHSCOPE_API_KEY` / `ZHIPU_API_KEY` es correcto y de que la cuenta tiene cuota suficiente. Ambos proveedores usan UTF-8 y manejan bien el chino.

**P: ¿Puedo pasar entrada por pipe a cheetahclaws?**

```bash
echo "Explain this file" | cheetahclaws --print --accept-all
cat error.log | cheetahclaws -p "What is causing this error?"
```

**P: ¿Cómo lo ejecuto como herramienta CLI desde cualquier lugar?**

Usa `uv tool install`: crea un entorno aislado y coloca `cheetahclaws` en tu PATH:

```bash
cd cheetahclaws
uv tool install .
```

Después de eso, simplemente ejecuta `cheetahclaws` desde cualquier directorio. Para actualizar tras hacer pull, ejecuta `uv tool install . --reinstall`.

**P: ¿Cómo configuro la entrada por voz?**

```bash
# Configuración mínima (local, offline, sin API key):
pip install sounddevice faster-whisper numpy

# Luego en el REPL:
/voice status          # verifica que los backends se detectan
/voice                 # habla tu prompt
```

En el primer uso, `faster-whisper` descarga automáticamente el modelo `base` (~150 MB).
Usa un modelo más grande para mejorar la precisión: `export NANO_CLAUDE_WHISPER_MODEL=small`

**P: La entrada de voz transcribe mal mis palabras (omite términos de programación).**

El reforzador de términos clave ya inyecta vocabulario técnico a partir de tu rama git y archivos del proyecto.
Para términos persistentes de dominio, colócalos en un archivo `.cheetahclaws/voice_keyterms.txt` (uno por línea); se comprueba automáticamente en cada grabación.

**P: ¿Puedo usar entrada por voz en chino / japonés / otros idiomas?**

Sí. Establece el idioma antes de grabar:

```
/voice lang zh    # chino mandarín
/voice lang ja    # japonés
/voice lang auto  # volver a autodetección (por defecto)
```

Whisper admite 99 idiomas. La detección automática funciona bien, pero indicar explícitamente el código mejora la precisión en frases cortas.
