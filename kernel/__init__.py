"""kernel — Phase-1 agent OS kernel for cheetahclaws.

Implements RFC 0003 (AgentProcess + EventLog WAL). Opt-in only: this
package is imported and activated exclusively from
``daemon/cli.py`` when ``cheetahclaws serve --enable-kernel`` is
passed. With the flag absent, this package is never touched and the
daemon behaves byte-for-byte identically to the pre-RFC build.

Public surface:

    register_with_daemon(daemon_state, db_path, *, recovery="suspend",
                         publish_to_bus=True) -> KernelStore

    KernelStore         — high-level CRUD + state-machine wrapper
    AgentProcess        — frozen dataclass for one agent row
    AgentState          — string enum of legal states
    Event               — frozen dataclass for one event row

    KernelError, UnknownPid, IllegalTransition, InvalidPayload,
    SchemaMismatch
"""
from __future__ import annotations

KERNEL_VERSION = "0.2.0"
# Re-exported from schema.py so this file doesn't drift from the actual
# DDL. Bump in schema.py only.
from .schema import EXPECTED_SCHEMA_VERSION as SCHEMA_VERSION  # noqa: E402

from .errors import (
    KernelError,
    UnknownPid,
    IllegalTransition,
    InvalidPayload,
    SchemaMismatch,
    # Phase 2 — capability
    CapabilityDerivationError,
    CapabilityExists,
    CapabilityUnknownPid,
    CapabilityInvalidGrant,
    # Phase 2 — ledger
    LedgerExists,
    LedgerUnknownDim,
    LedgerInvalidAmount,
    LedgerInvalidRefund,
    LedgerInvalidWarnAt,
    # Phase 3 — scheduler
    SchedIllegalTransition,
    SchedUnknownId,
    SchedInvalidPayload,
    # Phase 3 — mailbox
    MailboxNotFound,
    MailboxAlreadyExists,
    MailboxFull,
    MailboxInvalidPayload,
    MailboxSubscriptionMissing,
    # Phase 3 — registry
    RegistryNotFound,
    RegistryNameExists,
    RegistryInvalidName,
    # Phase 4 — agentfs
    FsNotFound,
    FsAlreadyExists,
    FsInvalidPath,
    FsReadOnly,
    FsQuotaExceeded,
    # Runner (RFC 0016)
    RunnerIllegalState,
    RunnerHandshakeFailed,
    RunnerUnknownPid,
    RunnerIpcTimeout,
)
from .process import AgentProcess, AgentState, ALLOWED_TRANSITIONS
from .store import KernelStore, Event
from .integration import register_with_daemon
from .sandbox import (
    SandboxPolicy,
    SandboxResult,
    SandboxPolicyError,
    SandboxNotAvailable,
    SANDBOX_OFF,
    SANDBOX_DEFAULT,
    SANDBOX_STRICT,
    detect_isolation_tools,
    apply_rlimits_in_child,
    wrap_with_bubblewrap,
    run_sandboxed,
)
from .capability import (
    Capability,
    CapabilityStore,
    FsGrant,
    host_matches_glob,
    fs_grant_matches,
)
from .ledger import (
    Ledger,
    LedgerEntry,
    LedgerStore,
    ChargeResult,
    CheckResult,
    STD_DIMS,
)
from .scheduler import (
    SchedulerStore,
    ScheduleSpec,
    ReadyEntry,
    STD_TRIGGERS,
    EXIT_KINDS as SCHED_EXIT_KINDS,
)
from .mailbox import (
    Mailbox,
    Message,
    Subscription,
    MailboxStore,
)
from .registry import (
    RegistryEntry,
    RegistryStore,
)
from .agentfs import (
    FsObject,
    AgentFSStore,
    DEFAULT_MAX_OBJECT_BYTES,
)
from .observability import (
    ObservabilityStore,
    TRACE_DEPTH_MAX,
)
from .contract import (
    STABLE_METHODS,
    EXPERIMENTAL_METHODS,
    DEPRECATED_METHODS,
    ALL_KNOWN_METHODS,
    RFCS_IMPLEMENTED,
    verify_contract,
)
from .chaos import ChaosMonkey
from .runner import (
    RunnerSupervisor,
    RunnerHandle,
    RunnerExitInfo,
    JsonLineChannel,
    IpcReadTimeout,
)
from .worker import WorkerLoop
from .api import Kernel
from .orchestrator import (
    DialogueOrchestrator,
    DialogueTurnFailed,
    DialogueTurnTimeout,
    DialogueQuotaBreached,
)
from .tools import (
    Tool,
    ToolContext,
    ToolRegistry,
    ToolError,
    ToolNotFound,
    ToolDenied,
    ToolFsDenied,
    ToolNetDenied,
    ToolInvalidArgs,
    ToolFailed,
    dispatch_tool_call,
    register_builtin_tools,
    EXEC_TOOL,
    register_exec_tool,
    FETCH_TOOL,
    register_fetch_tool,
)
from .bridge_mirror import (
    BridgeKind,
    BridgeMessage,
    BridgeMirror,
    OutboundReceiver,
    inbound_topic,
    outbound_topic,
)

__all__ = [
    "KERNEL_VERSION",
    "SCHEMA_VERSION",
    "register_with_daemon",
    "KernelStore",
    "AgentProcess",
    "AgentState",
    "Event",
    "ALLOWED_TRANSITIONS",
    "KernelError",
    "UnknownPid",
    "IllegalTransition",
    "InvalidPayload",
    "SchemaMismatch",
    # Sandbox (RFC 0008)
    "SandboxPolicy",
    "SandboxResult",
    "SandboxPolicyError",
    "SandboxNotAvailable",
    "SANDBOX_OFF",
    "SANDBOX_DEFAULT",
    "SANDBOX_STRICT",
    "detect_isolation_tools",
    "apply_rlimits_in_child",
    "wrap_with_bubblewrap",
    "run_sandboxed",
    # Capability (RFC 0005)
    "Capability",
    "CapabilityStore",
    "FsGrant",
    "host_matches_glob",
    "fs_grant_matches",
    "CapabilityDerivationError",
    "CapabilityExists",
    "CapabilityUnknownPid",
    "CapabilityInvalidGrant",
    # ResourceLedger (RFC 0006)
    "Ledger",
    "LedgerEntry",
    "LedgerStore",
    "ChargeResult",
    "CheckResult",
    "STD_DIMS",
    "LedgerExists",
    "LedgerUnknownDim",
    "LedgerInvalidAmount",
    "LedgerInvalidRefund",
    "LedgerInvalidWarnAt",
    # Scheduler (RFC 0007)
    "SchedulerStore",
    "ScheduleSpec",
    "ReadyEntry",
    "STD_TRIGGERS",
    "SCHED_EXIT_KINDS",
    "SchedIllegalTransition",
    "SchedUnknownId",
    "SchedInvalidPayload",
    # Mailbox (RFC 0009)
    "Mailbox",
    "Message",
    "Subscription",
    "MailboxStore",
    "MailboxNotFound",
    "MailboxAlreadyExists",
    "MailboxFull",
    "MailboxInvalidPayload",
    "MailboxSubscriptionMissing",
    # Registry (RFC 0010)
    "RegistryEntry",
    "RegistryStore",
    "RegistryNotFound",
    "RegistryNameExists",
    "RegistryInvalidName",
    # AgentFS (RFC 0011)
    "FsObject",
    "AgentFSStore",
    "DEFAULT_MAX_OBJECT_BYTES",
    "FsNotFound",
    "FsAlreadyExists",
    "FsInvalidPath",
    "FsReadOnly",
    "FsQuotaExceeded",
    # Observability (RFC 0012)
    "ObservabilityStore",
    "TRACE_DEPTH_MAX",
    "ChaosMonkey",
    # API stability (RFC 0013)
    "STABLE_METHODS",
    "EXPERIMENTAL_METHODS",
    "DEPRECATED_METHODS",
    "ALL_KNOWN_METHODS",
    "RFCS_IMPLEMENTED",
    "verify_contract",
    # Runner (RFC 0016)
    "RunnerSupervisor",
    "RunnerHandle",
    "RunnerExitInfo",
    "JsonLineChannel",
    "IpcReadTimeout",
    "RunnerIllegalState",
    "RunnerHandshakeFailed",
    "RunnerUnknownPid",
    "RunnerIpcTimeout",
    # Worker (RFC 0017)
    "WorkerLoop",
    # Facade
    "Kernel",
    # Bridge mirror (RFC 0018)
    "BridgeKind",
    "BridgeMessage",
    "BridgeMirror",
    "OutboundReceiver",
    "inbound_topic",
    "outbound_topic",
    # Dialogue orchestrator (RFC 0020)
    "DialogueOrchestrator",
    "DialogueTurnFailed",
    "DialogueTurnTimeout",
    "DialogueQuotaBreached",
    # Tool dispatch (RFC 0021)
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolError",
    "ToolNotFound",
    "ToolDenied",
    "ToolFsDenied",
    "ToolNetDenied",
    "ToolInvalidArgs",
    "ToolFailed",
    "dispatch_tool_call",
    "register_builtin_tools",
    "EXEC_TOOL",
    "register_exec_tool",
    "FETCH_TOOL",
    "register_fetch_tool",
]
