"""
Shared analysis library for comparing the HPO methods used by the Masked AutoRL-SOP
models:

  - Optuna-based models  -> SQLite .db studies (best-so-far, per-trial values, timings)
  - HyperTuningSK models -> hypertuningsk_results_<instance>.csv optimizer logs

It exposes model discovery, convergence-curve computation, a plotly figure styled like
the project's Viewer, aggregate summaries and helpers reused by the comparison notebook
and by generate_convergence_plot.py.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

SOP_INSTANCES = [
    "br17.10.sop", "br17.12.sop",
    "ESC07.sop", "ESC12.sop", "ESC25.sop", "ESC47.sop", "ESC63.sop", "ESC78.sop",
    "ft53.1.sop", "ft53.2.sop", "ft53.3.sop", "ft53.4.sop",
    "ft70.1.sop", "ft70.2.sop", "ft70.3.sop", "ft70.4.sop",
    "kro124p.1.sop", "kro124p.2.sop", "kro124p.3.sop",
    "p43.1.sop", "p43.2.sop", "p43.3.sop", "p43.4.sop",
    "prob.42.sop",
    "ry48p.1.sop", "ry48p.2.sop", "ry48p.3.sop", "ry48p.4.sop",
]

BEST_KNOWN_SOLUTIONS = {
    "br17.10": 55, "br17.12": 55,
    "ESC07": 2125, "ESC12": 1675, "ESC25": 1681, "ESC47": 1288, "ESC63": 62, "ESC78": 18230,
    "ft53.1": 7531, "ft53.2": 8026, "ft53.3": 10262, "ft53.4": 14425,
    "ft70.1": 39313, "ft70.2": 40101, "ft70.3": 42535, "ft70.4": 53530,
    "kro124p.1": 38762, "kro124p.2": 39841, "kro124p.3": 43904,
    "p43.1": 28140, "p43.2": 28480, "p43.3": 28835, "p43.4": 83005,
    "prob.42": 243,
    "ry48p.1": 15805, "ry48p.2": 16074, "ry48p.3": 19490, "ry48p.4": 31446,
}

# Model registry. Paths are relative to the repository root and follow the same order
# used by the interactive launcher.
MODELS = [
    {"label": "[MULTIVARIATE] Hyperband TPE", "hpo": "Hyperband+TPE-M", "core": "masking",
     "type": "optuna", "dir": "variants/[MULTIVARIATE] HYPERBAND Masked AutoRL-SOP/Results",
     "file": "optuna_results_TPE_{key}.db"},
    {"label": "[MULTIVARIATE] TPE", "hpo": "TPE-M", "core": "masking", "type": "optuna",
     "dir": "variants/[MULTIVARIATE] TPE Masked AutoRL-SOP/Results",
     "file": "optuna_results_{key}.db"},
    {"label": "[UNIVARIATE] Hyperband TPE", "hpo": "Hyperband+TPE-U", "core": "masking",
     "type": "optuna", "dir": "variants/[UNIVARIATE] HYPERBAND Masked AutoRL-SOP/Results",
     "file": "optuna_results_TPE_{key}.db"},
    {"label": "[UNIVARIATE] TPE", "hpo": "TPE-U", "core": "masking", "type": "optuna",
     "dir": "variants/[UNIVARIATE] TPE Masked AutoRL-SOP/Results",
     "file": "optuna_results_{key}.db"},
    {"label": "Hyperband GP", "hpo": "Hyperband+GP", "core": "masking", "type": "optuna",
     "dir": "variants/[MULTIVARIATE] HYPERBAND Masked AutoRL-SOP/Results",
     "file": "optuna_results_GP_{key}.db"},
    {"label": "Gaussian Process", "hpo": "GP", "core": "masking", "type": "optuna",
     "dir": "variants/GP Masked AutoRL-SOP/Results", "file": "optuna_results_{key}.db"},
    {"label": "HyperTuningSK", "hpo": "Scott-Knott", "core": "masking", "type": "hypertuningsk",
     "dir": "variants/NO_BAYESIAN Masked AutoRL-SOP/Results", "file": "hypertuningsk_results_{key}.csv"},
    {"label": "[MULTIVARIATE] NO_MASK", "hpo": "TPE-M", "core": "resampling",
     "type": "optuna", "dir": "variants/[MULTIVARIATE] NO_MASK Masked AutoRL-SOP/Results",
     "file": "optuna_results_{key}.db", "ablation": True},
    {"label": "[UNIVARIATE] NO_MASK", "hpo": "TPE-U", "core": "resampling",
     "type": "optuna", "dir": "variants/[UNIVARIATE] NO_MASK Masked AutoRL-SOP/Results",
     "file": "optuna_results_{key}.db", "ablation": True},
    {"label": "NO_MASK_NO_BAYESIAN (resampling, SK)", "hpo": "Scott-Knott", "core": "resampling",
     "type": "hypertuningsk", "dir": "variants/NO_MASK_NO_BAYESIAN Masked AutoRL-SOP/Results",
     "file": "hypertuningsk_results_{key}.csv", "ablation": True},
    {"label": "Random Search", "hpo": "Random", "core": "masking", "type": "optuna",
     "dir": "variants/RANDOM_SEARCH Masked AutoRL-SOP/Results",
     "file": "optuna_results_{key}.db"},
]

# One color per HPO method, kept consistent across every figure.
HPO_COLORS = {
    "Random": "#7f7f7f",
    "TPE-M": "#1f77b4",
    "TPE-U": "#6baed6",
    "GP": "#2ca02c",
    "Hyperband+TPE-M": "#ff7f0e",
    "Hyperband+TPE-U": "#fdbf6f",
    "Hyperband+GP": "#d62728",
    "Scott-Knott": "#9467bd",
}


def key_of(instance: str) -> str:
    """Return the result-file key for an SOP instance name."""
    return instance.replace(".sop", "")


def model_color(model: dict) -> str:
    """Return the consistent plot color assigned to an HPO method."""
    base = HPO_COLORS.get(model["hpo"], "#17becf")
    return base


def result_path(model: dict, instance: str, root: str = ".") -> str:
    """Resolve the stored result path for a model and SOP instance."""
    return os.path.join(root, model["dir"], model["file"].format(key=key_of(instance)))


def available_models(instance: str, root: str = ".", include_ablations: bool = True) -> list:
    """Registry entries whose result file exists for this instance."""
    out = []
    for m in MODELS:
        if m.get("ablation") and not include_ablations:
            continue
        if os.path.exists(result_path(m, instance, root)):
            out.append(m)
    return out


# Optuna
def _parse_dt(s):
    """Parse an Optuna SQLite timestamp when one is available."""
    if s is None:
        return None
    return datetime.fromisoformat(s)


def load_optuna_convergence(db_path: str) -> pd.DataFrame:
    """Per-trial table with the running best objective and wall-clock elapsed time.

    Pruned/failed trials keep no objective value (they do not improve best_so_far) but
    still consume time, so Hyperband's early stopping shows up on the time axis.
    """
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT t.number, t.state, t.datetime_start, t.datetime_complete, v.value "
            "FROM trials t LEFT JOIN trial_values v ON t.trial_id = v.trial_id "
            "ORDER BY t.number"
        ).fetchall()
    finally:
        con.close()
    df = pd.DataFrame(rows, columns=["number", "state", "dt_start", "dt_complete", "value"])
    if df.empty:
        return df
    t0 = _parse_dt(df["dt_start"].iloc[0])
    df["elapsed_s"] = [
        (_parse_dt(s) - t0).total_seconds() if s else np.nan for s in df["dt_complete"]
    ]
    best = np.inf
    best_so_far = []
    for state, value in zip(df["state"], df["value"]):
        if str(state) == "COMPLETE" and value is not None and not (isinstance(value, float) and np.isnan(value)):
            best = min(best, value)
        best_so_far.append(best if best != np.inf else np.nan)
    df["best_so_far"] = best_so_far
    return df


# HyperTuningSK
HPO_STAGES = ["HPO_epsilon", "HPO_alpha", "HPO_gamma"]


def load_hypertuningsk(csv_path: str):
    """Return (hpo_evals, full_log, meta).

    hpo_evals: the sequential tuning evaluations (epsilon then alpha then gamma phases),
               ordered, with a running best-so-far column.
    meta:      the final_grid / overall_best / best_observed values from the log tail.
    """
    full = pd.read_csv(csv_path)
    hpo = full[full["stage"].isin(HPO_STAGES)].copy().reset_index(drop=True)
    hpo["eval_order"] = np.arange(1, len(hpo) + 1)
    best = np.inf
    best_so_far = []
    for v in hpo["min_distance"]:
        if pd.notna(v):
            best = min(best, float(v))
        best_so_far.append(best if best != np.inf else np.nan)
    hpo["best_so_far"] = best_so_far

    def tail_value(stage):
        """Read the final numeric value associated with a log stage."""
        r = full[full["stage"] == stage]
        if len(r) and pd.notna(r["min_distance"].iloc[0]):
            return float(r["min_distance"].iloc[0])
        return None

    meta = {
        "final_grid": tail_value("final_grid"),
        "overall_best": tail_value("overall_best"),
        "best_observed": tail_value("best_observed"),
    }
    return hpo, full, meta


# Plotly style
def apply_latex_style(fig):
    """The Viewer's LaTeX-ish plotly theme, so every figure matches the paper."""
    fig.update_layout(
        template="simple_white",
        font=dict(family="Computer Modern, serif", size=14),
        title=dict(x=0.5, xanchor="center"),
        margin=dict(l=70, r=30, t=70, b=60),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", borderwidth=0),
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray", zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray", zeroline=False)
    return fig


def convergence_figure(instance, root=".", x_axis="trial", include_ablations=False,
                       models=None, log_y=False):
    """Multi-method convergence overlay for one instance (best-so-far vs trial or time)."""
    import plotly.graph_objects as go

    key = key_of(instance)
    if models is None:
        models = available_models(instance, root, include_ablations)
    fig = go.Figure()
    for m in models:
        path = result_path(m, instance, root)
        if not os.path.exists(path):
            continue
        if m["type"] == "optuna":
            df = load_optuna_convergence(path)
            if df.empty:
                continue
            x = (df["number"] + 1) if x_axis == "trial" else df["elapsed_s"]
            y = df["best_so_far"]
        else:
            if x_axis == "time":
                continue  # the HyperTuningSK log has no per-eval timing
            hpo, _, _ = load_hypertuningsk(path)
            x, y = hpo["eval_order"], hpo["best_so_far"]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=m["label"],
            line=dict(color=model_color(m), width=2.2,
                      dash="dot" if m.get("ablation") else "solid"),
        ))
    bks = BEST_KNOWN_SOLUTIONS.get(key)
    if bks:
        fig.add_hline(y=bks, line_dash="dash", line_color="black",
                      annotation_text=f"BKS = {bks}", annotation_position="top right")
    fig = apply_latex_style(fig)
    xtitle = "Trial" if x_axis == "trial" else "Wall-clock time (s)"
    fig.update_layout(title=f"Convergence — {key}", xaxis_title=xtitle,
                      yaxis_title="Best tour length so far")
    if log_y:
        fig.update_yaxes(type="log")
    return fig


# Aggregate
def summary_dataframe(root=".", instances=None, include_ablations=True) -> pd.DataFrame:
    """One row per (instance, model): best value, gap to BKS, #trials, runtime."""
    rows = []
    for inst in (instances or SOP_INSTANCES):
        key = key_of(inst)
        bks = BEST_KNOWN_SOLUTIONS.get(key)
        for m in MODELS:
            if m.get("ablation") and not include_ablations:
                continue
            path = result_path(m, inst, root)
            if not os.path.exists(path):
                continue
            if m["type"] == "optuna":
                df = load_optuna_convergence(path)
                if df.empty:
                    continue
                best = float(np.nanmin(df["best_so_far"]))
                n_trials = int(len(df))
                n_complete = int((df["state"].astype(str) == "COMPLETE").sum())
                runtime = float(np.nanmax(df["elapsed_s"]))
            else:
                hpo, _, meta = load_hypertuningsk(path)
                best = meta["overall_best"] if meta["overall_best"] is not None \
                    else float(np.nanmin(hpo["best_so_far"]))
                n_trials = int(len(hpo))
                n_complete = n_trials
                runtime = np.nan
            gap = 100.0 * (best - bks) / bks if bks else np.nan
            rows.append(dict(instance=key, model=m["label"], hpo=m["hpo"], core=m["core"],
                             best=best, bks=bks, gap_pct=gap, n_trials=n_trials,
                             n_complete=n_complete, runtime_s=runtime,
                             ablation=bool(m.get("ablation"))))
    return pd.DataFrame(rows)


def instances_with_enough_trials(root=".", min_trials=20,
                                 model_label="[MULTIVARIATE] TPE"):
    """Instances where the reference model ran enough trials for a meaningful curve."""
    ref = next(m for m in MODELS if m["label"] == model_label)
    out = []
    for inst in SOP_INSTANCES:
        path = result_path(ref, inst, root)
        if os.path.exists(path) and len(load_optuna_convergence(path)) >= min_trials:
            out.append(inst)
    return out


# Exploration
SEARCH_PARAMS = ["alpha", "gamma", "epsilon"]


def load_optuna_configs(db_path, ndigits=2):
    """Ordered list of (alpha, gamma, epsilon) tuples, one per trial (trial order).

    Values are rounded to the search-space step so that "the same configuration" means
    the same grid point. With a frozen seed, two identical tuples give identical results,
    so repeated tuples are wasted, non-exploring trials.
    """
    con = sqlite3.connect(db_path)
    try:
        prm = {}
        for tid, pn, pv in con.execute(
                "SELECT trial_id, param_name, param_value FROM trial_params"):
            prm.setdefault(tid, {})[pn] = round(float(pv), ndigits)
        order = [tid for (tid,) in con.execute("SELECT trial_id FROM trials ORDER BY number")]
    finally:
        con.close()
    return [tuple(prm[t].get(p) for p in SEARCH_PARAMS) for t in order if t in prm]


def exploration_stats(seq):
    """Diversity / repetition metrics for a sequence of configuration tuples."""
    n = len(seq)
    if n == 0:
        return {}
    seen = set()
    cum_unique = []
    for c in seq:
        seen.add(c)
        cum_unique.append(len(seen))
    runs = []
    cur = 1
    for i in range(1, n):
        if seq[i] == seq[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    unique = len(seen)
    return dict(n_trials=n, unique=unique, duplicates=n - unique,
                diversity_pct=100.0 * unique / n,
                max_consecutive=max(runs), mean_run=float(np.mean(runs)),
                repeat_events=sum(1 for r in runs if r >= 2),
                run_lengths=runs, cum_unique=cum_unique)


def diversity_table(root=".", instances=None, include_ablations=False) -> pd.DataFrame:
    """One row per (instance, Optuna model) with diversity / repetition metrics."""
    rows = []
    for inst in (instances or SOP_INSTANCES):
        key = key_of(inst)
        for m in MODELS:
            if m["type"] != "optuna" or (m.get("ablation") and not include_ablations):
                continue
            path = result_path(m, inst, root)
            if not os.path.exists(path):
                continue
            seq = load_optuna_configs(path)
            if len(seq) < 2:
                continue
            s = exploration_stats(seq)
            rows.append(dict(instance=key, model=m["label"], hpo=m["hpo"],
                             trials=s["n_trials"], unique=s["unique"],
                             diversity_pct=round(s["diversity_pct"], 1),
                             duplicates=s["duplicates"],
                             max_consecutive=s["max_consecutive"],
                             repeat_events=s["repeat_events"]))
    return pd.DataFrame(rows)


def exploration_figure(instance, root=".", include_ablations=False, models=None):
    """Cumulative distinct configurations vs trial (overlay methods), with y = x reference.

    The closer a curve stays to y = x, the more the sampler keeps exploring new points;
    flat segments are runs of repeated (wasted) configurations.
    """
    import plotly.graph_objects as go

    key = key_of(instance)
    if models is None:
        models = [m for m in available_models(instance, root, include_ablations)
                  if m["type"] == "optuna"]
    fig = go.Figure()
    nmax = 0
    for m in models:
        seq = load_optuna_configs(result_path(m, instance, root))
        if len(seq) < 2:
            continue
        s = exploration_stats(seq)
        nmax = max(nmax, s["n_trials"])
        fig.add_trace(go.Scatter(
            x=list(range(1, s["n_trials"] + 1)), y=s["cum_unique"], mode="lines",
            name=f"{m['label']} ({s['diversity_pct']:.0f}% unique)",
            line=dict(color=model_color(m), width=2.2,
                      dash="dot" if m.get("ablation") else "solid")))
    if nmax:
        fig.add_trace(go.Scatter(x=[1, nmax], y=[1, nmax], mode="lines",
                                 name="no repetition (y = x)",
                                 line=dict(color="black", width=1, dash="dash")))
    fig = apply_latex_style(fig)
    fig.update_layout(title=f"Search-space exploration — {key}",
                      xaxis_title="Trial", yaxis_title="Distinct configurations so far")
    return fig


def coverage_scatter_figure(instance, root=".", include_ablations=False, models=None,
                            pair=("alpha", "gamma")):
    """Where each method spends its trials: the `pair` plane of evaluated configurations,
    one panel per method. Marker size / color grow with how many times a configuration was
    evaluated, so a repeat-heavy sampler shows a few big red dots instead of broad coverage
    (the search-space "amplitude" seen next to the repetition).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from collections import Counter

    ix = {p: i for i, p in enumerate(SEARCH_PARAMS)}
    if models is None:
        models = [m for m in available_models(instance, root, include_ablations)
                  if m["type"] == "optuna"]
    fig = make_subplots(rows=1, cols=max(len(models), 1),
                        subplot_titles=[m["label"] for m in models] or ["(no data)"],
                        horizontal_spacing=0.12)
    for col, m in enumerate(models, 1):
        counts = Counter(load_optuna_configs(result_path(m, instance, root)))
        if not counts:
            continue
        cs = list(counts.values())
        fig.add_trace(go.Scatter(
            x=[c[ix[pair[0]]] for c in counts], y=[c[ix[pair[1]]] for c in counts],
            mode="markers",
            marker=dict(size=[6 + 7 * (c - 1) ** 0.5 for c in cs], color=cs,
                        colorscale="Reds", cmin=1, showscale=(col == len(models)),
                        colorbar=dict(title="times<br>evaluated"),
                        line=dict(width=0.6, color="gray")),
            text=[f"×{c}" for c in cs], showlegend=False), row=1, col=col)
        fig.update_xaxes(title_text=pair[0], range=[-0.03, 1.03], row=1, col=col)
        fig.update_yaxes(title_text=pair[1], range=[-0.03, 1.03], row=1, col=col)
    fig = apply_latex_style(fig)
    fig.update_layout(title=f"Search-space coverage — {key_of(instance)} "
                            "(marker size = times a configuration was evaluated)")
    return fig
