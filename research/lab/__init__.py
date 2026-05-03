"""research/lab — multi-agent research lab.

Topic-in, paper-out: a directed-graph state machine that drives a
collaboration of agents (PI, Questioner, Surveyor, Designer, Writer,
Reviewer × 3, Lay Reader) through reseach-question framing, literature
review, outline drafting, section drafting, and reviewer-author
iteration until convergence or budget exhaustion.

Public surface (what callers should import):

  from research.lab import (
      LabRun, LabState, Stage, RoleAssignment,
      orchestrator, storage, verifier, output,
  )

The :class:`LabRun` orchestration is the only thing slash commands and
the daemon care about; everything else is internal scaffolding.

This is "engine v0":
  * No experiment execution yet (Phase 2)
  * No LaTeX/PDF rendering yet (Phase 2)
  * No web UI / multi-tenant (Phase 3)
  * No GPU pool (Phase 4)
"""
from __future__ import annotations

from .storage import (
    LabStorage, RunRecord, StageRecord, MessageRecord, ExperimentRecord,
)
from .sandbox import (
    SandboxResult, run_python_in_sandbox, make_workspace,
    extract_python_block, format_result_for_prompt,
)
from .roles import (
    Role, ROLE_PI, ROLE_QUESTIONER, ROLE_SURVEYOR, ROLE_DESIGNER,
    ROLE_WRITER, ROLE_REVIEWER, ROLE_LAY_READER,
    ROLE_ENGINEER, ROLE_ANALYST,
    RoleAssignment, build_default_assignment, load_role_template,
)
from .convergence import ConvergenceConfig, ReviewerVerdict, decide_advance
from .verifier import (
    Citation, CitationVerification, VerifierResult,
    verify_citations, verify_one,
)
from .orchestrator import (
    Stage, LabRun, LabState, run_one_lab_session,
)
from .output import write_markdown_report, format_bibtex

__all__ = [
    "LabStorage", "RunRecord", "StageRecord", "MessageRecord",
    "Role", "ROLE_PI", "ROLE_QUESTIONER", "ROLE_SURVEYOR", "ROLE_DESIGNER",
    "ROLE_WRITER", "ROLE_REVIEWER", "ROLE_LAY_READER",
    "RoleAssignment", "build_default_assignment", "load_role_template",
    "ConvergenceConfig", "ReviewerVerdict", "decide_advance",
    "Citation", "CitationVerification", "VerifierResult",
    "verify_citations", "verify_one",
    "Stage", "LabRun", "LabState", "run_one_lab_session",
    "write_markdown_report", "format_bibtex",
]
