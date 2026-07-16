"""errors.py — Kernel exception hierarchy.

These exceptions are raised by ``KernelStore`` and translated to
JSON-RPC error codes by ``kernel.methods``. Application code should
import the specific subclass it cares about; ``KernelError`` is the
common root for blanket ``except`` clauses.

Error codes:

    Process / event-log (RFC 0003 §3.5):
        -32101  KERNEL_NOT_ENABLED
        -32102  KERNEL_UNKNOWN_PID
        -32103  KERNEL_ILLEGAL_TRANSITION
        -32104  KERNEL_INVALID_PAYLOAD
        -32105  KERNEL_SCHEMA_MISMATCH

    Capability (RFC 0005 §5):
        -32111  KERNEL_CAP_DERIVATION_INVALID
        -32112  KERNEL_CAP_ALREADY_EXISTS
        -32113  KERNEL_CAP_UNKNOWN_PID
        -32114  KERNEL_CAP_INVALID_GRANT

    ResourceLedger (RFC 0006 §4):
        -32121  KERNEL_LEDGER_UNKNOWN_DIM
        -32122  KERNEL_LEDGER_ALREADY_EXISTS
        -32123  KERNEL_LEDGER_INVALID_AMOUNT
        -32124  KERNEL_LEDGER_INVALID_REFUND
        -32125  KERNEL_LEDGER_INVALID_WARN_AT
"""
from __future__ import annotations

# JSON-RPC application error codes — kept here so methods.py and tests
# import the same constants.
KERNEL_NOT_ENABLED        = -32101
KERNEL_UNKNOWN_PID        = -32102
KERNEL_ILLEGAL_TRANSITION = -32103
KERNEL_INVALID_PAYLOAD    = -32104
KERNEL_SCHEMA_MISMATCH    = -32105

# Capability (RFC 0005)
KERNEL_CAP_DERIVATION_INVALID = -32111
KERNEL_CAP_ALREADY_EXISTS     = -32112
KERNEL_CAP_UNKNOWN_PID        = -32113
KERNEL_CAP_INVALID_GRANT      = -32114

# ResourceLedger (RFC 0006)
KERNEL_LEDGER_UNKNOWN_DIM     = -32121
KERNEL_LEDGER_ALREADY_EXISTS  = -32122
KERNEL_LEDGER_INVALID_AMOUNT  = -32123
KERNEL_LEDGER_INVALID_REFUND  = -32124
KERNEL_LEDGER_INVALID_WARN_AT = -32125

# Scheduler (RFC 0007)
KERNEL_SCHED_ILLEGAL_TRANSITION  = -32131
KERNEL_SCHED_UNKNOWN_ID          = -32132
KERNEL_SCHED_INVALID_PAYLOAD     = -32133
KERNEL_SCHED_ADMISSION_DENIED    = -32134
KERNEL_SCHED_INVALID_STATE_FILTER = -32135

# Mailbox (RFC 0009)
KERNEL_MBOX_NOT_FOUND             = -32141
KERNEL_MBOX_ALREADY_EXISTS        = -32142
KERNEL_MBOX_FULL                  = -32143
KERNEL_MBOX_INVALID_PAYLOAD       = -32144
KERNEL_MBOX_SUBSCRIPTION_MISSING  = -32145

# Registry (RFC 0010)
KERNEL_REGISTRY_NOT_FOUND     = -32151
KERNEL_REGISTRY_NAME_EXISTS   = -32152
KERNEL_REGISTRY_INVALID_NAME  = -32153

# AgentFS (RFC 0011)
KERNEL_FS_NOT_FOUND       = -32161
KERNEL_FS_ALREADY_EXISTS  = -32162
KERNEL_FS_INVALID_PATH    = -32163
KERNEL_FS_READ_ONLY       = -32164
KERNEL_FS_QUOTA_EXCEEDED  = -32165

# Runner (RFC 0016)
KERNEL_RUNNER_ILLEGAL_STATE   = -32171
KERNEL_RUNNER_HANDSHAKE_FAILED = -32172
KERNEL_RUNNER_UNKNOWN_PID     = -32173
KERNEL_RUNNER_IPC_TIMEOUT     = -32174


class KernelError(Exception):
    """Root for all kernel-raised errors."""
    code: int = -32100  # generic kernel error if no subclass overrides

    def to_rpc_data(self) -> dict:
        return {}


class UnknownPid(KernelError):
    code = KERNEL_UNKNOWN_PID

    def __init__(self, pid: int) -> None:
        super().__init__(f"unknown pid: {pid}")
        self.pid = pid

    def to_rpc_data(self) -> dict:
        return {"pid": self.pid}


class IllegalTransition(KernelError):
    code = KERNEL_ILLEGAL_TRANSITION

    def __init__(self, prev_state: str, target_state: str) -> None:
        super().__init__(
            f"illegal transition: {prev_state} -> {target_state}"
        )
        self.prev_state = prev_state
        self.target_state = target_state

    def to_rpc_data(self) -> dict:
        return {
            "prev_state": self.prev_state,
            "target_state": self.target_state,
        }


class InvalidPayload(KernelError):
    code = KERNEL_INVALID_PAYLOAD

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field

    def to_rpc_data(self) -> dict:
        return {"field": self.field} if self.field else {}


class SchemaMismatch(KernelError):
    code = KERNEL_SCHEMA_MISMATCH

    def __init__(self, expected: int, found: int | None) -> None:
        super().__init__(
            f"kernel.db schema_version mismatch: code expects {expected}, "
            f"db has {found!r}"
        )
        self.expected = expected
        self.found = found

    def to_rpc_data(self) -> dict:
        return {"expected": self.expected, "found": self.found}


# ── Capability errors (RFC 0005) ───────────────────────────────────────────


class CapabilityDerivationError(KernelError):
    code = KERNEL_CAP_DERIVATION_INVALID

    def __init__(self, reason: str, *, field: str | None = None) -> None:
        super().__init__(reason)
        self.field = field

    def to_rpc_data(self) -> dict:
        return {"field": self.field} if self.field else {}


class CapabilityExists(KernelError):
    code = KERNEL_CAP_ALREADY_EXISTS

    def __init__(self, pid: int) -> None:
        super().__init__(f"capability already exists for pid {pid}")
        self.pid = pid

    def to_rpc_data(self) -> dict:
        return {"pid": self.pid}


class CapabilityUnknownPid(KernelError):
    code = KERNEL_CAP_UNKNOWN_PID

    def __init__(self, pid: int) -> None:
        super().__init__(f"no capability row for pid {pid}")
        self.pid = pid

    def to_rpc_data(self) -> dict:
        return {"pid": self.pid}


class CapabilityInvalidGrant(KernelError):
    code = KERNEL_CAP_INVALID_GRANT

    def __init__(self, reason: str, *, field: str | None = None) -> None:
        super().__init__(reason)
        self.field = field

    def to_rpc_data(self) -> dict:
        return {"field": self.field} if self.field else {}


# ── Ledger errors (RFC 0006) ───────────────────────────────────────────────


class LedgerUnknownDim(KernelError):
    code = KERNEL_LEDGER_UNKNOWN_DIM

    def __init__(self, pid: int, dim: str) -> None:
        super().__init__(f"no ledger row for pid={pid} dim={dim!r}")
        self.pid = pid
        self.dim = dim

    def to_rpc_data(self) -> dict:
        return {"pid": self.pid, "dim": self.dim}


class LedgerExists(KernelError):
    code = KERNEL_LEDGER_ALREADY_EXISTS

    def __init__(self, pid: int, dim: str) -> None:
        super().__init__(f"ledger row already exists for pid={pid} dim={dim!r}")
        self.pid = pid
        self.dim = dim

    def to_rpc_data(self) -> dict:
        return {"pid": self.pid, "dim": self.dim}


class LedgerInvalidAmount(KernelError):
    code = KERNEL_LEDGER_INVALID_AMOUNT

    def __init__(self, amount) -> None:
        super().__init__(f"amount must be a non-negative int, got {amount!r}")
        self.amount = amount

    def to_rpc_data(self) -> dict:
        return {"amount": str(self.amount)}


class LedgerInvalidRefund(KernelError):
    code = KERNEL_LEDGER_INVALID_REFUND

    def __init__(self, pid: int, dim: str, used: int, refund: int) -> None:
        super().__init__(
            f"refund {refund} for pid={pid} dim={dim!r} would push used={used} "
            "below zero"
        )
        self.pid = pid
        self.dim = dim
        self.used = used
        self.refund = refund

    def to_rpc_data(self) -> dict:
        return {"pid": self.pid, "dim": self.dim,
                "used": self.used, "refund": self.refund}


class LedgerInvalidWarnAt(KernelError):
    code = KERNEL_LEDGER_INVALID_WARN_AT

    def __init__(self, value) -> None:
        super().__init__(f"warn_at must be in [0, 1], got {value!r}")
        self.value = value

    def to_rpc_data(self) -> dict:
        return {"value": self.value}


# ── Scheduler errors (RFC 0007) ────────────────────────────────────────────


class SchedIllegalTransition(KernelError):
    code = KERNEL_SCHED_ILLEGAL_TRANSITION

    def __init__(self, prev_state: str, op: str) -> None:
        super().__init__(
            f"illegal scheduler transition: cannot {op} entry in state "
            f"{prev_state!r}"
        )
        self.prev_state = prev_state
        self.op = op

    def to_rpc_data(self) -> dict:
        return {"prev_state": self.prev_state, "op": self.op}


class SchedUnknownId(KernelError):
    code = KERNEL_SCHED_UNKNOWN_ID

    def __init__(self, sched_id: int) -> None:
        super().__init__(f"unknown sched_id: {sched_id}")
        self.sched_id = sched_id

    def to_rpc_data(self) -> dict:
        return {"sched_id": self.sched_id}


class SchedInvalidPayload(KernelError):
    code = KERNEL_SCHED_INVALID_PAYLOAD

    def __init__(self, reason: str, *, field: str | None = None) -> None:
        super().__init__(reason)
        self.field = field

    def to_rpc_data(self) -> dict:
        return {"field": self.field} if self.field else {}


# ── Mailbox errors (RFC 0009) ──────────────────────────────────────────────


class MailboxNotFound(KernelError):
    code = KERNEL_MBOX_NOT_FOUND

    def __init__(self, pid: int) -> None:
        super().__init__(f"no mailbox for pid {pid}")
        self.pid = pid

    def to_rpc_data(self) -> dict:
        return {"pid": self.pid}


class MailboxAlreadyExists(KernelError):
    code = KERNEL_MBOX_ALREADY_EXISTS

    def __init__(self, pid: int) -> None:
        super().__init__(f"mailbox already exists for pid {pid}")
        self.pid = pid

    def to_rpc_data(self) -> dict:
        return {"pid": self.pid}


class MailboxFull(KernelError):
    code = KERNEL_MBOX_FULL

    def __init__(self, pid: int, queue_size: int) -> None:
        super().__init__(f"mailbox full: pid={pid} queue_size={queue_size}")
        self.pid = pid
        self.queue_size = queue_size

    def to_rpc_data(self) -> dict:
        return {"pid": self.pid, "queue_size": self.queue_size}


class MailboxInvalidPayload(KernelError):
    code = KERNEL_MBOX_INVALID_PAYLOAD

    def __init__(self, reason: str, *, field: str | None = None) -> None:
        super().__init__(reason)
        self.field = field

    def to_rpc_data(self) -> dict:
        return {"field": self.field} if self.field else {}


class MailboxSubscriptionMissing(KernelError):
    code = KERNEL_MBOX_SUBSCRIPTION_MISSING

    def __init__(self, pid: int, topic: str) -> None:
        super().__init__(f"no subscription: pid={pid} topic={topic!r}")
        self.pid = pid
        self.topic = topic

    def to_rpc_data(self) -> dict:
        return {"pid": self.pid, "topic": self.topic}


# ── Registry errors (RFC 0010) ─────────────────────────────────────────────


class RegistryNotFound(KernelError):
    code = KERNEL_REGISTRY_NOT_FOUND

    def __init__(self, name: str) -> None:
        super().__init__(f"no registry entry for name {name!r}")
        self.name = name

    def to_rpc_data(self) -> dict:
        return {"name": self.name}


class RegistryNameExists(KernelError):
    code = KERNEL_REGISTRY_NAME_EXISTS

    def __init__(self, name: str) -> None:
        super().__init__(f"name already registered: {name!r}")
        self.name = name

    def to_rpc_data(self) -> dict:
        return {"name": self.name}


class RegistryInvalidName(KernelError):
    code = KERNEL_REGISTRY_INVALID_NAME

    def __init__(self, reason: str, *, name: str | None = None) -> None:
        super().__init__(reason)
        self.name = name

    def to_rpc_data(self) -> dict:
        return {"name": self.name} if self.name else {}


# ── AgentFS errors (RFC 0011) ──────────────────────────────────────────────


class FsNotFound(KernelError):
    code = KERNEL_FS_NOT_FOUND

    def __init__(self, path: str) -> None:
        super().__init__(f"agentfs path not found: {path!r}")
        self.path = path

    def to_rpc_data(self) -> dict:
        return {"path": self.path}


class FsAlreadyExists(KernelError):
    code = KERNEL_FS_ALREADY_EXISTS

    def __init__(self, path: str) -> None:
        super().__init__(f"agentfs path already exists: {path!r}")
        self.path = path

    def to_rpc_data(self) -> dict:
        return {"path": self.path}


class FsInvalidPath(KernelError):
    code = KERNEL_FS_INVALID_PATH

    def __init__(self, reason: str, *, path: str | None = None) -> None:
        super().__init__(reason)
        self.path = path

    def to_rpc_data(self) -> dict:
        return {"path": self.path} if self.path else {}


class FsReadOnly(KernelError):
    code = KERNEL_FS_READ_ONLY

    def __init__(self, path: str) -> None:
        super().__init__(f"agentfs path is read-only: {path!r}")
        self.path = path

    def to_rpc_data(self) -> dict:
        return {"path": self.path}


class FsQuotaExceeded(KernelError):
    code = KERNEL_FS_QUOTA_EXCEEDED

    def __init__(self, pid: int, dim: str, used: int, hard_limit: int) -> None:
        super().__init__(
            f"agentfs quota exceeded for pid={pid} dim={dim!r}: "
            f"used={used} > hard_limit={hard_limit}"
        )
        self.pid = pid
        self.dim = dim
        self.used = used
        self.hard_limit = hard_limit

    def to_rpc_data(self) -> dict:
        return {
            "pid": self.pid,
            "dim": self.dim,
            "used": self.used,
            "hard_limit": self.hard_limit,
        }


# ── Runner errors (RFC 0016) ───────────────────────────────────────────────


class RunnerIllegalState(KernelError):
    code = KERNEL_RUNNER_ILLEGAL_STATE

    def __init__(self, pid: int, state: str, expected: str) -> None:
        super().__init__(
            f"runner spawn requires agent state {expected}, got {state} "
            f"for pid={pid}"
        )
        self.pid = pid
        self.state = state
        self.expected = expected

    def to_rpc_data(self) -> dict:
        return {"pid": self.pid, "state": self.state,
                "expected": self.expected}


class RunnerHandshakeFailed(KernelError):
    code = KERNEL_RUNNER_HANDSHAKE_FAILED

    def __init__(self, pid: int, reason: str) -> None:
        super().__init__(
            f"runner handshake failed for pid={pid}: {reason}"
        )
        self.pid = pid
        self.reason = reason

    def to_rpc_data(self) -> dict:
        return {"pid": self.pid, "reason": self.reason}


class RunnerUnknownPid(KernelError):
    code = KERNEL_RUNNER_UNKNOWN_PID

    def __init__(self, pid: int) -> None:
        super().__init__(f"no live runner for pid {pid}")
        self.pid = pid

    def to_rpc_data(self) -> dict:
        return {"pid": self.pid}


class RunnerIpcTimeout(KernelError):
    code = KERNEL_RUNNER_IPC_TIMEOUT

    def __init__(self, pid: int, timeout_s: float) -> None:
        super().__init__(
            f"runner IPC timed out after {timeout_s}s for pid={pid}"
        )
        self.pid = pid
        self.timeout_s = timeout_s

    def to_rpc_data(self) -> dict:
        return {"pid": self.pid, "timeout_s": self.timeout_s}
