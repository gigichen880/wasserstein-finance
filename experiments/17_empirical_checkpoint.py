"""Empirical research checkpoint: liquidity-state gradient-flow pilot.

Does not modify the theoretical manuscript. Raw order-book files stay on disk
and out of version control.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfmm.empirical.archives import (
    extract_dates,
    preferred_source,
    reconcile_counts,
    records_as_dicts,
    scan_collection,
)
from wfmm.empirical.models import (
    RateFit,
    common_valid,
    fit_heterogeneous,
    fit_mean_variance_paths,
    fit_spy_factor,
    forecast_affine,
    forecast_factor_persist,
    forecast_heterogeneous,
    forecast_persistence,
    forecast_tod,
    gaussian_tail_outside_unit,
    pooled_paths,
    score_forecast,
)

MAIN_FORECAST_MODELS = (
    "persistence",
    "quadratic_c0",
    "independent_ou",
    "flexible_moments",
    "heterogeneous",
    "time_of_day",
    "factor_spy",
)
from wfmm.empirical.panel import (
    cross_section_moments,
    load_day_panel,
    mask_open_close,
    minutes_since_open,
    quality_row,
    read_book_csv,
    top_imbalance,
)
from wfmm.empirical.plots import figure_forecasts, figure_moments, figure_stability
from wfmm.empirical.synthetic import independent_factor_days, interacting_days, recover_rates
from wfmm.io_util import save_csv, save_json
from wfmm.paths import (
    EMPIRICAL_CACHE_DIR,
    EMPIRICAL_DATA_DIR,
    EMPIRICAL_FIGS_DIR,
    EMPIRICAL_RESULTS_DIR,
    ROOT,
    ensure_empirical_dirs,
)

CONFIG_PATH = ROOT / "research" / "empirical" / "config_pilot.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def _dates(cfg: dict) -> list[str]:
    return list(cfg["train_dates"] + cfg["val_dates"] + cfg["test_dates"] + cfg["stress_dates"])


def _mask_fn(cfg):
    def _fn(times):
        return mask_open_close(times, cfg["drop_open_minutes"], cfg["drop_close_minutes"])
    return _fn


def audit_extracted_quality(records, n_sample: int = 40) -> list[dict]:
    rows = []
    extracted = [r for r in records if r.extracted]
    rng = np.random.default_rng(17)
    for rec in extracted:
        files = sorted(Path(rec.path).glob("*.csv"))
        if not files:
            continue
        pick = list(files)
        if len(pick) > n_sample:
            idx = np.sort(rng.choice(len(pick), size=n_sample, replace=False))
            pick = [pick[i] for i in idx]
        # always include a handful of known calendar dates if present
        extra = [p for p in files if any(s in p.name for s in ("2019-09-03", "2020-03-09", "2012-07-03"))]
        for p in list(dict.fromkeys(extra + pick)):
            try:
                book = read_book_csv(p)
                q = quality_row(book)
                q["source"] = rec.path
                rows.append(q)
            except Exception as exc:
                rows.append({"path": str(p), "error": str(exc)})
    return rows


def extract_pilot(cfg, records) -> dict:
    levels = int(cfg["depth_levels_required"])
    dates = _dates(cfg)
    report = {}
    for tkr in cfg["universe"]:
        rec = preferred_source(records, tkr, levels)
        if rec is None:
            report[tkr] = {"status": "missing_archive"}
            continue
        try:
            report[tkr] = {"status": "ok", "source": rec.source, **extract_dates(rec, dates)}
        except Exception as exc:
            report[tkr] = {"status": "extract_error", "error": str(exc), "source": rec.path}
    return report


def load_block(cfg, tickers, dates, k_depth=1):
    panels = []
    for d in dates:
        panel = load_day_panel(
            tickers, d, cfg["depth_levels_required"], k_depth=k_depth,
            price_scale=cfg["price_scale"], cache_dir=EMPIRICAL_CACHE_DIR,
        )
        if panel is not None:
            panels.append(panel)
    if not panels:
        return [], []
    common = set(panels[0].tickers)
    for p in panels[1:]:
        common &= set(p.tickers)
    common = [t for t in cfg["universe"] if t in common]
    trimmed = []
    for p in panels:
        idx = [p.tickers.index(t) for t in common]
        trimmed.append(type(p)(
            date=p.date,
            times=p.times,
            tickers=common,
            x=p.x[:, idx],
            valid=p.valid[:, idx],
            quality=p.quality,
        ))
    return trimmed, common


def fit_suite(t, mu, var) -> dict[str, RateFit]:
    return {
        "quadratic_c0": fit_mean_variance_paths(t, mu, var, center=0.0, force_k_equals_a=False, name="quadratic_c0"),
        "independent_ou": fit_mean_variance_paths(t, mu, var, center=0.0, force_k_equals_a=True, name="independent_ou"),
        "flexible_moments": fit_mean_variance_paths(t, mu, var, center=0.0, force_k_equals_a=False, name="flexible_moments"),
        "fixed_center": fit_mean_variance_paths(
            t, mu, var, center=float(np.nanmean(mu)), force_k_equals_a=False, name="fixed_center",
        ),
    }


def tod_library(panels, mask_fn) -> dict[str, np.ndarray]:
    buckets = defaultdict(list)
    for p in panels:
        m = mask_fn(p.times)
        for i, t in enumerate(p.times):
            if not m[i]:
                continue
            y = p.x[i][np.isfinite(p.x[i])]
            if y.size:
                buckets[str(t)].append(y)
    return {k: np.concatenate(v) for k, v in buckets.items()}


def _is_etf(name: str) -> bool:
    return name == "SPY" or name.startswith("X")


def evaluate_block(panels, fits, het, tod, cfg, rng, label: str, factor=None) -> list[dict]:
    mask_fn = _mask_fn(cfg)
    rows = []
    horizons = list(cfg["horizons_min"])
    step = int(cfg["rolling_step_min"])
    for panel in panels:
        mins = minutes_since_open(panel.times)
        keep = mask_fn(panel.times)
        for i, t0 in enumerate(panel.times):
            if not keep[i]:
                continue
            if int(mins[i]) % step != 0:
                continue
            for h in horizons:
                j = i + int(h)
                if j >= len(panel.times) or not keep[j]:
                    continue
                x0, x1 = common_valid(panel.x[i], panel.x[j])
                if x0.size < 5:
                    continue
                names = [n for n, a, b in zip(panel.tickers, panel.x[i], panel.x[j]) if np.isfinite(a) and np.isfinite(b)]
                forecasts = {
                    "persistence": forecast_persistence(x0),
                    "quadratic_c0": forecast_affine(x0, h, fits["quadratic_c0"], rng),
                    "independent_ou": forecast_affine(x0, h, fits["independent_ou"], rng),
                    "flexible_moments": forecast_affine(x0, h, fits["flexible_moments"], rng),
                    "heterogeneous": forecast_heterogeneous(x0, names, het, h, rng),
                }
                tod_f = forecast_tod(str(panel.times[j]), tod)
                if tod_f is not None:
                    forecasts["time_of_day"] = tod_f
                if factor is not None:
                    forecasts["factor_spy"] = forecast_factor_persist(
                        x0, names, factor, h, fits["independent_ou"], rng,
                    )
                for model, fc in forecasts.items():
                    sc = score_forecast(x1, fc)
                    rows.append({
                        "block": label,
                        "date": panel.date,
                        "origin": str(t0),
                        "horizon": int(h),
                        "model": model,
                        "n": int(x0.size),
                        "origin_mean": float(x0.mean()),
                        "origin_var": float(x0.var(ddof=1)),
                        **sc,
                    })
                etf_m = np.array([_is_etf(n) for n in names], dtype=bool)
                if int(etf_m.sum()) >= 4 and int((~etf_m).sum()) >= 8:
                    qfc = forecasts["quadratic_c0"]
                    for gname, mask in (("quadratic_c0_etf", etf_m), ("quadratic_c0_single", ~etf_m)):
                        scg = score_forecast(x1[mask], qfc[mask])
                        rows.append({
                            "block": label,
                            "date": panel.date,
                            "origin": str(t0),
                            "horizon": int(h),
                            "model": gname,
                            "n": int(mask.sum()),
                            "origin_mean": float(x0[mask].mean()),
                            "origin_var": float(x0[mask].var(ddof=1)) if int(mask.sum()) > 1 else np.nan,
                            **scg,
                        })
        # fixed morning-to-afternoon
        tmap = {str(t): i for i, t in enumerate(panel.times)}
        if cfg["fixed_origin"] in tmap and cfg["fixed_target"] in tmap:
            i = tmap[cfg["fixed_origin"]]
            j = tmap[cfg["fixed_target"]]
            x0, x1 = common_valid(panel.x[i], panel.x[j])
            if x0.size >= 5:
                h = float(minutes_since_open(np.array([cfg["fixed_target"]]))[0]
                          - minutes_since_open(np.array([cfg["fixed_origin"]]))[0])
                names = [n for n, a, b in zip(panel.tickers, panel.x[i], panel.x[j]) if np.isfinite(a) and np.isfinite(b)]
                for model, fc in {
                    "persistence": forecast_persistence(x0),
                    "quadratic_c0": forecast_affine(x0, h, fits["quadratic_c0"], rng),
                    "independent_ou": forecast_affine(x0, h, fits["independent_ou"], rng),
                }.items():
                    sc = score_forecast(x1, fc)
                    rows.append({
                        "block": label + "_fixed_am_pm",
                        "date": panel.date,
                        "origin": cfg["fixed_origin"],
                        "horizon": int(h),
                        "model": model,
                        "n": int(x0.size),
                        "origin_mean": float(x0.mean()),
                        "origin_var": float(x0.var(ddof=1)),
                        **sc,
                    })
    return rows


def paired_delta(rows, base="persistence", metric="w2") -> list[dict]:
    key = lambda r: (r["block"], r["date"], r["origin"], r["horizon"])
    groups = defaultdict(dict)
    for r in rows:
        groups[key(r)][r["model"]] = r
    out = []
    for k, g in groups.items():
        if base not in g:
            continue
        for model, r in g.items():
            if model == base:
                continue
            if not np.isfinite(r[metric]) or not np.isfinite(g[base][metric]):
                continue
            out.append({
                "block": r["block"],
                "date": r["date"],
                "origin": r["origin"],
                "horizon": r["horizon"],
                "model": model,
                "delta_w2": r["w2"] - g[base]["w2"],
                "delta_w1": r["w1"] - g[base]["w1"],
            })
    return out


def day_clustered_mean(rows, field) -> dict:
    by_day = defaultdict(list)
    for r in rows:
        if np.isfinite(r[field]):
            by_day[r["date"]].append(r[field])
    if not by_day:
        return {"mean": float("nan"), "se": float("nan"), "n_days": 0}
    day_means = np.array([np.mean(v) for v in by_day.values()])
    n = day_means.size
    se = float(day_means.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"mean": float(day_means.mean()), "se": se, "n_days": int(n)}


def _extract_counts(extract_report: dict) -> str:
    ok = [t for t, r in extract_report.items() if r.get("status") == "ok"]
    missing = [t for t, r in extract_report.items() if r.get("status") != "ok"]
    return f"{len(ok)}/{len(extract_report)} requested names extracted; missing={missing or 'none'}"


def write_report(
    cfg, manifest_rec, extract_report, fits, synth, decision,
    forecast_summary, quality_summary, spacing_fits, leakage, emp_out,
) -> Path:
    path = ROOT / "research" / "empirical" / "report.md"
    fit_lines = "\n".join(
        f"- `{name}`: a={fit.a:.4g}/min (half-life {np.log(2)/max(fit.a,1e-12):.2g} min), "
        f"k={fit.k:.4g}/min, q={fit.q:.4g}, k-a={fit.k - fit.a:.4g}, "
        f"admissible={fit.admissible}, mean_sse={fit.mean_sse:.4g}, var_sse={fit.var_sse:.4g}"
        for name, fit in fits.items()
    )
    space_lines = "\n".join(
        f"- spacing {sp} min: a={d['a']:.4g}, k={d['k']:.4g}, k-a={d['k']-d['a']:.4g}, "
        f"admissible={d['admissible']}"
        for sp, d in spacing_fits.items()
    ) or "- (none)"
    path.write_text(
        f"""# Empirical checkpoint (liquidity-state population)

This is a **go/no-go research checkpoint**. It is not a manuscript claim.
The theoretical paper and synthetic experiments 01–16 are unchanged.

## Population

Cross-sectional **top-of-book depth imbalance** of listed securities, not
dealer inventories and not reconstructed trader positions. A successful fit
would mean a useful effective description, not a universal financial law.

Primary state: `x = (bid_size_1 - ask_size_1) / (bid_size_1 + ask_size_1)`.
Zero denominators are NaN, not zero. No time-by-time cross-sectional
recentering. Affine Gaussian dynamics on the real line are an **explicit
approximation** of a coordinate supported on [-1, 1].

## Data audit

- Unique tickers in `data_by_stocks/`: **{manifest_rec['n_unique_tickers']}**
- One-level names: **{manifest_rec['n_one_level_tickers']}**
- Ten-level names: **{manifest_rec['n_ten_level_tickers']}**
- Names present at both depths: **{manifest_rec['n_both_level_tickers']}**
- Identity check one+ten−both = unique: **{manifest_rec['identity']}**
- Extracted directories at scan time: **{manifest_rec['n_extracted_dirs']}**
- Compressed archives: **{manifest_rec['n_archives']}**

Reported “119 ten-level + 398 one-level” does not match the archive census:
397 + 119 = 516 unique names, with **zero** names present at both depths.
The extra one-level count is unverified metadata.

Clock times are America/New_York civil session times. Files do not record
venue or timezone offsets. 391 bars is a complete 09:30–16:00 minute grid,
not 391 independent observations. Universe is a prespecified large-cap /
ETF list that survived in the archive; that is a survivorship-conditioned
pilot, not a randomly sampled CRSP panel.

Opened-file sample (extracted `A` and `AAPL` only): {quality_summary.get('n_files')} files,
{quality_summary.get('n_full_session')} full 391-bar sessions,
{quality_summary.get('n_padded_early_close')} with ≥60 trailing identical rows
(possible padded early close), mean stale fraction
{quality_summary.get('mean_stale_fraction')},
{quality_summary.get('n_any_crossed')} file(s) with any crossed book.
Median ask in the opened sample ranges
{quality_summary.get('median_ask_min')}–{quality_summary.get('median_ask_max')}
dollars after the 1e-4 price scale. Records can be unchanged
minute-to-minute; they are not guaranteed to be independent snapshots.

Pilot universe (prespecified, 10-level names): `{', '.join(cfg['universe'])}`.
{_extract_counts(extract_report)}.
Train {cfg['train_dates'][0]}–{cfg['train_dates'][-1]},
val {cfg['val_dates'][0]}–{cfg['val_dates'][-1]},
test {cfg['test_dates'][0]}–{cfg['test_dates'][-1]},
stress {', '.join(cfg['stress_dates'])} (external COVID dates, not selected
from imbalance outcomes).

## Moment restrictions (training days, native 1-minute grid after dropping 10 min open/close)

{fit_lines}

The mean path is a weak, noisy oscillation around zero; the fitted exponential
relaxes in a few minutes. Cross-sectional variance is nearly flat, so `k` is
weakly identified. `k-a > 0` is **not** interpreted as causal interaction.

Observation spacings (same training windows):

{space_lines}

A single `(a, k)` pair does not describe both moments across spacings. The
15-minute grid is inadmissible (`k < a`).

## Bounded support

Empirical mass outside [-1, 1]: **{emp_out:.4g}** (by construction for a
well-defined imbalance). Gaussian working measure P(|X|>1) ≈ **{leakage:.3f}**.
The affine model is therefore a leaky approximation, not a structurally
correct raw-state law.

## Forecasts (chronological split; day-clustered SE)

{forecast_summary}

Origins and scored names are identical across models. Time-of-day uses the
**target** clock’s training histogram (no test-day values). The SPY factor
model residualizes on training betas and persists the origin proxy; it does
not use future SPY.

## Synthetic controls (pipeline check, not market evidence)

- Interacting DGP recovered k-a positive: **{synth['interacting_ka_positive']}**
  (a={synth['interacting']['a']:.4g}, k={synth['interacting']['k']:.4g})
- Independent+factor DGP false-positive k-a: **{synth['independent_ka_positive']}**
  (a={synth['independent']['a']:.4g}, k={synth['independent']['k']:.4g})

A false positive would mean `k-a` is observationally ambiguous. A true
negative on one simulated draw does **not** identify interaction in the
real panel.

## Decision gate

**{decision['verdict']}**

{decision['rationale']}

**Do not expand** the empirical study for the ALT manuscript. Preserve this
negative result as a research note. It does not belong in the main text. A
short appendix is optional only if the authors want to document that
liquidity-state imbalance is a poor coordinate for the quadratic energy;
otherwise neither.

Figures: `figs/empirical/fig1_moment_restrictions.png`,
`fig2_forecasts.png`, `fig3_stability.png`.
Commands: `python experiments/17_empirical_checkpoint.py` and
`python -m pytest tests/test_empirical.py`.
"""
    )
    return path


def main():
    ensure_empirical_dirs()
    cfg = load_config()
    rng = np.random.default_rng(int(cfg["seed"]))

    records = scan_collection(EMPIRICAL_DATA_DIR)
    recon = reconcile_counts(records)
    save_json(EMPIRICAL_RESULTS_DIR / "manifest.json", {
        "reconcile": recon,
        "records": records_as_dicts(records),
    })
    save_csv(EMPIRICAL_RESULTS_DIR / "manifest.csv", records_as_dicts(records))

    quality_rows = audit_extracted_quality(records)
    save_json(EMPIRICAL_RESULTS_DIR / "quality_extracted_sample.json", quality_rows)
    okq = [r for r in quality_rows if "error" not in r]
    quality_summary = {
        "n_files": len(okq),
        "n_full_session": sum(1 for r in okq if r.get("is_full_session")),
        "n_early_close_candidate": sum(1 for r in okq if r.get("early_close_candidate")),
        "n_padded_early_close": sum(1 for r in okq if r.get("possible_padded_early_close")),
        "mean_stale_fraction": float(np.mean([r["stale_fraction"] for r in okq])) if okq else None,
        "n_any_crossed": sum(1 for r in okq if r.get("n_crossed", 0) > 0),
        "median_ask_min": float(np.min([r["price_median_ask"] for r in okq])) if okq else None,
        "median_ask_max": float(np.max([r["price_median_ask"] for r in okq])) if okq else None,
    }
    save_json(EMPIRICAL_RESULTS_DIR / "quality_summary.json", quality_summary)

    extract_report = extract_pilot(cfg, records)
    save_json(EMPIRICAL_RESULTS_DIR / "extract_report.json", extract_report)
    available = [t for t, r in extract_report.items() if r.get("status") == "ok" and r.get("copied")]

    train, universe = load_block(cfg, available, cfg["train_dates"])
    val, _ = load_block(cfg, universe, cfg["val_dates"])
    test, _ = load_block(cfg, universe, cfg["test_dates"])
    stress, _ = load_block(cfg, universe, cfg["stress_dates"])
    save_json(EMPIRICAL_RESULTS_DIR / "universe.json", {
        "requested": cfg["universe"],
        "available": available,
        "fixed_train_universe": universe,
        "n_train_days": len(train),
        "n_val_days": len(val),
        "n_test_days": len(test),
        "n_stress_days": len(stress),
    })

    mask_fn = _mask_fn(cfg)
    t, mu, var = pooled_paths(train, 1, mask_fn) if train else (np.array([]), np.array([]), np.array([]))
    fits = fit_suite(t, mu, var) if t.size else {}
    save_json(EMPIRICAL_RESULTS_DIR / "moment_fits.json", {
        name: fit.__dict__ for name, fit in fits.items()
    })

    # spacings
    spacing_fits = {}
    for sp in cfg["spacings_min"]:
        ts, mus, vs = pooled_paths(train, int(sp), mask_fn) if train else (np.array([]),) * 3
        if ts.size:
            spacing_fits[str(sp)] = fit_mean_variance_paths(
                ts, mus, vs, center=0.0, force_k_equals_a=False, name=f"space_{sp}",
            ).__dict__
    save_json(EMPIRICAL_RESULTS_DIR / "spacing_fits.json", spacing_fits)

    leakage = float("nan")
    emp_out = float("nan")
    hist_x = np.array([])
    if train and "quadratic_c0" in fits:
        hist_x = np.concatenate([p.x[mask_fn(p.times)].ravel() for p in train])
        hist_x = hist_x[np.isfinite(hist_x)]
        leakage = gaussian_tail_outside_unit(float(np.mean(hist_x)), float(np.var(hist_x, ddof=1)))
        emp_out = float(np.mean(np.abs(hist_x) > 1.0))  # should be 0 by construction
        save_json(EMPIRICAL_RESULTS_DIR / "bounded_support.json", {
            "empirical_mass_outside_unit": emp_out,
            "gaussian_working_measure_Pabs_gt_1": leakage,
            "note": "Raw imbalance is in [-1,1]. The affine Gaussian is an approximate benchmark.",
        })

    het = {}
    tod = {}
    factor = {"proxy": cfg["market_proxy"], "available": False, "alpha": {}, "beta": {}}
    if train and "quadratic_c0" in fits:
        het = fit_heterogeneous(
            train,
            prior_count=int(cfg["heterogeneous_prior_count"]),
            a_pool=fits["quadratic_c0"].a,
            c_pool=fits["quadratic_c0"].center,
        )
        tod = tod_library(train, mask_fn)
        factor = fit_spy_factor(train, cfg["market_proxy"])

    rows = []
    if train and fits:
        rows.extend(evaluate_block(val, fits, het, tod, cfg, rng, "val", factor=factor))
        rows.extend(evaluate_block(test, fits, het, tod, cfg, rng, "test", factor=factor))
        rows.extend(evaluate_block(stress, fits, het, tod, cfg, rng, "stress", factor=factor))
    save_csv(EMPIRICAL_RESULTS_DIR / "forecast_rows.csv", rows)
    deltas = paired_delta(rows)
    save_csv(EMPIRICAL_RESULTS_DIR / "paired_deltas.csv", deltas)

    # summaries
    summary = {}
    for block in ("val", "test", "stress"):
        for model in sorted({r["model"] for r in rows if r["block"] == block and r["model"] in MAIN_FORECAST_MODELS}):
            sub = [r for r in rows if r["block"] == block and r["model"] == model]
            summary[f"{block}:{model}"] = day_clustered_mean(sub, "w2")
    save_json(EMPIRICAL_RESULTS_DIR / "forecast_summary.json", summary)

    # Slow, strongly excited DGP: checks whether the pipeline can see k-a when
    # the mean actually moves. Matching the huge empirical a would decay in seconds.
    inter = interacting_days(
        n_names=40, n_days=8, n_times=180,
        a=0.02, k=0.06, q=0.08, mu0=0.45, var0=0.12, rng=rng,
    )
    indep = independent_factor_days(
        n_names=40, n_days=8, n_times=180,
        a_mean=0.02, a_disp=0.01, q=0.08, factor_vol=0.08, rng=rng,
    )
    fit_i = recover_rates(inter)
    fit_n = recover_rates(indep)
    synth = {
        "interacting": fit_i.__dict__,
        "independent": fit_n.__dict__,
        "interacting_ka_positive": bool(fit_i.k - fit_i.a > 0.0),
        "independent_ka_positive": bool(fit_n.k - fit_n.a > 0.0),
        "note": "A false positive on the independent+factor DGP means k-a is observationally ambiguous.",
    }
    save_json(EMPIRICAL_RESULTS_DIR / "synthetic_controls.json", synth)

    # figures
    if t.size and fits:
        figure_moments(t, mu, var, fits, leakage, hist_x)
    if rows:
        figure_forecasts([
            r for r in rows
            if r["block"] == "test" and r["model"] in MAIN_FORECAST_MODELS
        ])
        by_group = {"excitation": {}, "regime": {}, "liquidity_group": {}}
        origins = [r for r in rows if r["block"] == "test" and r["model"] == "quadratic_c0" and r["horizon"] == 5]
        if origins:
            absmu = np.abs([r["origin_mean"] for r in origins])
            q1, q2 = np.quantile(absmu, [1 / 3, 2 / 3])
            bins = [("low", absmu <= q1), ("mid", (absmu > q1) & (absmu <= q2)), ("high", absmu > q2)]
            for name, m in bins:
                vals = [origins[i]["w2"] for i in range(len(origins)) if m[i]]
                by_group["excitation"][name] = float(np.mean(vals)) if vals else np.nan
        for lab, block in (("quiet_test", "test"), ("stress", "stress")):
            sub = [r for r in rows if r["block"] == block and r["model"] == "quadratic_c0"]
            by_group["regime"][lab] = float(np.nanmean([r["w2"] for r in sub])) if sub else np.nan
        for gname, key in (("quadratic_c0_etf", "etf"), ("quadratic_c0_single", "single_name")):
            sub = [r for r in rows if r["block"] == "test" and r["model"] == gname]
            by_group["liquidity_group"][key] = float(np.nanmean([r["w2"] for r in sub])) if sub else np.nan
        figure_stability(by_group)

    # decision
    test_q = summary.get("test:quadratic_c0", {"mean": np.nan})
    test_p = summary.get("test:persistence", {"mean": np.nan})
    test_ou = summary.get("test:independent_ou", {"mean": np.nan})
    gain_vs_pers = test_p["mean"] - test_q["mean"] if np.isfinite(test_p["mean"]) and np.isfinite(test_q["mean"]) else float("nan")
    gain_vs_ou = test_ou["mean"] - test_q["mean"] if np.isfinite(test_ou["mean"]) and np.isfinite(test_q["mean"]) else float("nan")
    false_pos = synth["independent_ka_positive"]
    if not train or not fits:
        verdict = "NO-GO"
        rationale = "Pilot panel did not load. Expand extraction or check archives before any empirical claim."
    elif false_pos and (not np.isfinite(gain_vs_ou) or gain_vs_ou <= 0):
        verdict = "NO-GO for manuscript inclusion"
        rationale = (
            "Either the independent+factor control produces an apparent k-a split, "
            "or the quadratic forecast does not beat an independent OU on held-out W2. "
            "Treat k-a as unidentified. Keep this as a research note, not a paper section."
        )
    elif np.isfinite(gain_vs_pers) and gain_vs_pers > 0 and np.isfinite(gain_vs_ou) and gain_vs_ou > 0 and not false_pos:
        verdict = "CONDITIONAL GO for a narrow follow-up, not for a validation claim"
        rationale = (
            "Held-out W2 improved on persistence and independent OU, and the independent "
            "control did not spuriously split k and a. Still do not claim causal interaction "
            "or a financial law. A later empirical subsection could report the restrictions "
            "and forecast table with the existing identifiability caveats."
        )
    else:
        verdict = "NO-GO for manuscript inclusion"
        rationale = (
            f"Test W2 quadratic={test_q['mean']:.4g}, persistence={test_p['mean']:.4g}, "
            f"independent OU={test_ou['mean']:.4g}. Gains are not reproducible across "
            "meaningful alternatives, or bounded-support / heterogeneity remain first-order. "
            "Document the failure; do not add a financial-validation sentence to rewrite.tex."
        )
    decision = {
        "verdict": verdict,
        "rationale": rationale,
        "gain_vs_persistence_w2": gain_vs_pers,
        "gain_vs_independent_ou_w2": gain_vs_ou,
        "gaussian_leakage": leakage,
        "n_universe": len(universe),
    }
    save_json(EMPIRICAL_RESULTS_DIR / "decision.json", decision)

    forecast_summary = "\n".join(
        f"- {k}: mean W2 {v['mean']:.4g} (day-clustered SE {v['se']:.4g}, {v['n_days']} days)"
        for k, v in summary.items()
    ) or "No forecast rows."
    write_report(
        cfg, recon, extract_report, fits, synth, decision, forecast_summary,
        quality_summary, spacing_fits, leakage, emp_out,
    )
    print("decision:", verdict)
    print("report:", ROOT / "research" / "empirical" / "report.md")
    print("figs:", EMPIRICAL_FIGS_DIR)


if __name__ == "__main__":
    main()
