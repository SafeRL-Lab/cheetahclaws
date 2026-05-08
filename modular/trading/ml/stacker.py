"""
ml/stacker.py — LightGBM model that learns from the agent's own track record.

How it works:
  1. Pull all closed paper trades
  2. Build features (LLM signal, confidence, position size, sector, …)
  3. Label = "did this trade beat zero?" (will switch to vs-SPY when
     enough history is recorded)
  4. Train LightGBM with stratified k-fold CV; report AUC + accuracy
  5. Persist model to ~/.cheetahclaws/trading/ml/stacker.pkl
  6. .predict_proba(features) → probability the trade will be a hit

When integrated into analyze, the stacker is run as a post-filter:
the LLM may say BUY, but if the stacker says p(hit) < 0.4, downgrade
to HOLD with a note about the model disagreement.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# numpy / sklearn / lightgbm are deferred to call sites so that
# `pip install .` (no [trading] extra) ships a wheel where
# `import modular.trading.ml.stacker` doesn't blow up — see
# tests/test_packaging.py::test_required_module_imports (issue #97).

from . import features as feat_mod
from .features import FeatureRow


_DEFAULT_MODEL_DIR = Path.home() / ".cheetahclaws" / "trading" / "ml"
_DEFAULT_MODEL_PATH = _DEFAULT_MODEL_DIR / "stacker.pkl"


@dataclass
class TrainResult:
    n_samples:   int
    cv_auc_mean: float
    cv_auc_std:  float
    cv_acc_mean: float
    feature_importance: dict[str, float]
    notes:       list[str]
    model_path:  str


def _import_lightgbm():
    try:
        import lightgbm as lgb
        return lgb, "lightgbm"
    except ImportError:
        return None, ""


def _import_sklearn_fallback():
    """Fallback if lightgbm not installed."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier, "sklearn-gbc"
    except ImportError:
        return None, ""


def train(
    rows: list[FeatureRow],
    cols: list[str] | None = None,
    n_folds: int = 5,
    model_path: Path | str | None = None,
    min_samples: int = 30,
) -> TrainResult:
    """Train the stacker model. Returns TrainResult with CV scores."""
    cols = cols or feat_mod.feature_columns()
    notes: list[str] = []

    if len(rows) < min_samples:
        notes.append(f"Only {len(rows)} samples — need ≥ {min_samples}; "
                     "model not trained.")
        return TrainResult(len(rows), 0.0, 0.0, 0.0, {}, notes,
                           str(model_path or _DEFAULT_MODEL_PATH))

    # Heavy deps land here, *after* the diagnostic-only early returns.
    # Lets `train(too_few_rows)` work even on installs without [trading]
    # extras — useful for the "what would the stacker say?" UX path.
    import numpy as np

    X = np.array([r.features for r in rows], dtype=float)
    y = np.array([r.label for r in rows], dtype=int)

    if len(set(y.tolist())) < 2:
        notes.append("All labels identical (model would be trivial). "
                     "Need both winning and losing closed trades.")
        return TrainResult(len(rows), 0.0, 0.0, float((y == 1).mean()),
                           {}, notes, str(model_path or _DEFAULT_MODEL_PATH))

    # Cross-validated AUC
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score

    n_folds = max(2, min(n_folds, len(rows) // max(1, min(y.sum(), (y == 0).sum()))))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    aucs, accs = [], []
    lgb_mod, backend = _import_lightgbm()
    if lgb_mod is None:
        clf_cls, backend = _import_sklearn_fallback()
        if clf_cls is None:
            notes.append("Neither lightgbm nor sklearn available — cannot train.")
            return TrainResult(len(rows), 0.0, 0.0, 0.0, {}, notes,
                               str(model_path or _DEFAULT_MODEL_PATH))

    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        if backend == "lightgbm":
            clf = lgb_mod.LGBMClassifier(
                n_estimators=80, learning_rate=0.05,
                num_leaves=15, min_child_samples=3,
                random_state=42, verbose=-1,
            )
        else:
            clf = clf_cls(n_estimators=80, learning_rate=0.05, max_depth=3,
                          random_state=42)
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:, 1]
        try:
            aucs.append(roc_auc_score(yte, proba))
        except ValueError:
            aucs.append(0.5)
        accs.append(accuracy_score(yte, proba >= 0.5))

    # Train final model on full data and persist
    if backend == "lightgbm":
        final = lgb_mod.LGBMClassifier(
            n_estimators=120, learning_rate=0.05,
            num_leaves=15, min_child_samples=3,
            random_state=42, verbose=-1,
        )
    else:
        final = clf_cls(n_estimators=120, learning_rate=0.05, max_depth=3,
                        random_state=42)
    final.fit(X, y)

    # Feature importance — different APIs depending on backend
    importances: dict[str, float] = {}
    if hasattr(final, "feature_importances_"):
        for col, imp in zip(cols, final.feature_importances_):
            importances[col] = float(imp)

    target = Path(model_path or _DEFAULT_MODEL_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        pickle.dump({"model": final, "cols": cols, "backend": backend}, f)

    notes.append(f"Trained {backend} model on {len(rows)} samples, "
                 f"{n_folds}-fold CV.")

    return TrainResult(
        n_samples=len(rows),
        cv_auc_mean=float(np.mean(aucs)),
        cv_auc_std=float(np.std(aucs)),
        cv_acc_mean=float(np.mean(accs)),
        feature_importance={k: round(v, 3) for k, v in importances.items()},
        notes=notes,
        model_path=str(target),
    )


def predict_proba(features: list[float],
                  model_path: Path | str | None = None) -> dict[str, Any]:
    """Return probability of the trade being a hit, plus diagnostics.

    Returns {} when the model file doesn't exist (caller should
    fall back to "no override" behaviour).
    """
    path = Path(model_path or _DEFAULT_MODEL_PATH)
    if not path.exists():
        return {}
    # Heavy deps land *after* the no-model early return, so the
    # `predict_proba(features, model_path=missing)` UX path still works
    # on a lean install (the agent simply doesn't get the ML override).
    import numpy as np

    with open(path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    cols = bundle["cols"]

    arr = np.array([features], dtype=float)
    proba = float(model.predict_proba(arr)[0, 1])
    return {
        "proba_hit": round(proba, 4),
        "verdict":   ("Disagrees with bullish thesis" if proba < 0.4 else
                      "Confirms bullish thesis" if proba > 0.6 else
                      "Inconclusive"),
        "backend":   bundle.get("backend", "unknown"),
        "n_features": len(cols),
    }


def render_train_report(r: TrainResult) -> str:
    """Human-readable training report."""
    lines = ["# Stacker Training Report"]
    lines.append(f"- Samples: {r.n_samples}")
    if r.notes:
        for n in r.notes:
            lines.append(f"- {n}")
    if r.cv_auc_mean > 0:
        lines.append("")
        lines.append("## Cross-validated performance")
        lines.append(f"- AUC: **{r.cv_auc_mean:.3f} ± {r.cv_auc_std:.3f}**")
        lines.append(f"- Accuracy: **{r.cv_acc_mean:.3f}**")
        if r.cv_auc_mean < 0.55:
            lines.append("")
            lines.append("> AUC near 0.5 — model has no edge over coin-flip. "
                         "Either too few samples, no real signal in the agent's "
                         "track record, or features need work.")

    if r.feature_importance:
        lines.append("")
        lines.append("## Top features")
        for col, imp in sorted(r.feature_importance.items(),
                               key=lambda kv: -kv[1])[:10]:
            lines.append(f"- {col}: {imp:.3f}")
    lines.append("")
    lines.append(f"Model saved to: `{r.model_path}`")
    return "\n".join(lines)
