"""research/lab/roles.py — agent role definitions and prompt loading.

Each role is a (name, model, prompt_template) triple. We keep models
*intentionally* heterogeneous so reviewer-author debate doesn't
collapse into same-family rubber-stamping (RFC 0001 §2's argument
adapted to the multi-agent debate setting).

The default assignment uses three different model families when their
keys are available; falls back gracefully to the user's primary model
when not.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Role names (used as DB strings + module-level constants for refactor safety)
ROLE_PI         = "pi"
ROLE_QUESTIONER = "questioner"
ROLE_SURVEYOR   = "surveyor"
ROLE_DESIGNER   = "designer"
ROLE_ENGINEER   = "engineer"      # writes experiment code
ROLE_ANALYST    = "analyst"       # interprets experiment results, drafts plots
ROLE_WRITER     = "writer"
ROLE_REVIEWER   = "reviewer"      # logical role; instances tagged reviewer_1, reviewer_2, reviewer_3
ROLE_LAY_READER = "lay_reader"


_TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "agent_templates" / "lab"


@dataclass
class Role:
    name: str
    model: str
    template_filename: str
    description: str


@dataclass
class RoleAssignment:
    """Maps each role to a concrete (model, template) pair."""
    pi: Role
    questioner: Role
    surveyor: Role
    designer: Role
    engineer: Role
    analyst: Role
    writer: Role
    reviewers: list[Role]
    lay_reader: Role

    def all_roles(self) -> list[Role]:
        return [self.pi, self.questioner, self.surveyor, self.designer,
                self.engineer, self.analyst, self.writer,
                *self.reviewers, self.lay_reader]


# ── Model selection ───────────────────────────────────────────────────────


def _has_env(*keys: str) -> bool:
    return any(os.environ.get(k) for k in keys)


def _pick_first_available(candidates: list[tuple[str, list[str]]],
                          fallback: str) -> str:
    """Return the first model whose env-var requirement is satisfied."""
    for model, env_keys in candidates:
        if _has_env(*env_keys):
            return model
    return fallback


def _default_pi_model(config: dict) -> str:
    return _pick_first_available(
        [
            ("claude-opus-4-6",       ["ANTHROPIC_API_KEY"]),
            ("gpt-4o",                ["OPENAI_API_KEY"]),
            ("gemini/gemini-2.5-pro", ["GEMINI_API_KEY"]),
        ],
        fallback=config.get("model", "claude-sonnet-4-6"),
    )


def _default_writer_model(config: dict) -> str:
    return _pick_first_available(
        [
            ("claude-sonnet-4-6",     ["ANTHROPIC_API_KEY"]),
            ("gpt-4o",                ["OPENAI_API_KEY"]),
            ("gemini/gemini-2.5-pro", ["GEMINI_API_KEY"]),
        ],
        fallback=config.get("model", "claude-sonnet-4-6"),
    )


def _default_reviewer_models(config: dict) -> list[str]:
    """Three reviewers, prefer three different families to reduce same-source bias."""
    pool = [
        ("claude-sonnet-4-6",        ["ANTHROPIC_API_KEY"]),
        ("gpt-4o",                   ["OPENAI_API_KEY"]),
        ("gemini/gemini-2.5-pro",    ["GEMINI_API_KEY"]),
        ("deepseek/deepseek-chat",   ["DEEPSEEK_API_KEY"]),
        ("qwen/qwen-max",            ["DASHSCOPE_API_KEY"]),
    ]
    chosen: list[str] = []
    for model, env_keys in pool:
        if _has_env(*env_keys):
            chosen.append(model)
            if len(chosen) == 3:
                return chosen
    # Fallbacks: pad out with the user's primary model if fewer than 3 keys.
    primary = config.get("model", "claude-sonnet-4-6")
    while len(chosen) < 3:
        chosen.append(primary)
    return chosen


def _default_aux_model(config: dict) -> str:
    """Cheap model for surveyor / questioner / lay_reader (low-stakes role)."""
    try:
        from auxiliary import get_auxiliary_model
        return get_auxiliary_model(config)
    except Exception:
        return config.get("model", "claude-sonnet-4-6")


def build_default_assignment(config: dict,
                             *, override: Optional[dict] = None
                             ) -> RoleAssignment:
    """Build the default 9-role assignment (was 7 in Phase 1; +Engineer +Analyst)."""
    o = override or {}
    pi_model         = o.get("pi", _default_pi_model(config))
    questioner_model = o.get("questioner", _default_aux_model(config))
    surveyor_model   = o.get("surveyor", _default_aux_model(config))
    designer_model   = o.get("designer", _default_writer_model(config))
    engineer_model   = o.get("engineer", _default_writer_model(config))
    analyst_model    = o.get("analyst", _default_writer_model(config))
    writer_model     = o.get("writer", _default_writer_model(config))
    lay_model        = o.get("lay_reader", _default_aux_model(config))

    reviewer_defaults = _default_reviewer_models(config)
    reviewers = []
    for i, default_model in enumerate(reviewer_defaults, start=1):
        key = f"reviewer_{i}"
        reviewers.append(Role(
            name=f"reviewer_{i}",
            model=o.get(key, default_model),
            template_filename="reviewer.md",
            description=f"Independent reviewer #{i} ({o.get(key, default_model)})",
        ))

    return RoleAssignment(
        pi=Role(ROLE_PI, pi_model, "pi.md",
                "Principal investigator — sets direction, breaks ties"),
        questioner=Role(ROLE_QUESTIONER, questioner_model, "questioner.md",
                        "Translates a topic into narrowable research questions"),
        surveyor=Role(ROLE_SURVEYOR, surveyor_model, "surveyor.md",
                      "Runs literature search, identifies gaps"),
        designer=Role(ROLE_DESIGNER, designer_model, "designer.md",
                      "Designs methodology / analysis plan"),
        engineer=Role(ROLE_ENGINEER, engineer_model, "engineer.md",
                      "Writes runnable experiment code"),
        analyst=Role(ROLE_ANALYST, analyst_model, "analyst.md",
                     "Interprets experiment outputs and drafts the results section"),
        writer=Role(ROLE_WRITER, writer_model, "writer.md",
                    "Drafts paper sections + revises based on review"),
        reviewers=reviewers,
        lay_reader=Role(ROLE_LAY_READER, lay_model, "lay_reader.md",
                        "Outsider perspective — clarity / accessibility"),
    )


# ── Template loader ───────────────────────────────────────────────────────


def load_role_template(role: Role) -> str:
    """Read the markdown prompt template for ``role``.

    Templates live at ``agent_templates/lab/<filename>``. Missing file
    raises FileNotFoundError so callers fail loudly rather than running
    with a blank prompt.
    """
    path = _TEMPLATES_DIR / role.template_filename
    return path.read_text(encoding="utf-8")


def get_templates_dir() -> Path:
    return _TEMPLATES_DIR
