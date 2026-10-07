"""Load one-minute books and construct liquidity-state coordinates."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from wfmm.empirical.archives import cached_csv, parse_csv_name

EXPECTED_SESSION = tuple(
    (datetime.strptime("09:30:00", "%H:%M:%S") + timedelta(minutes=i)).strftime("%H:%M:%S")
    for i in range(391)
)


def _to_float(value: str) -> float:
    value = str(value).strip()
    if value == "":
        return float("nan")
    return float(value)


def read_book_csv(path, price_scale: float = 1e-4) -> dict:
    path = Path(path)
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = [h.strip() for h in next(reader)]
        rows = list(reader)
    col = {name: i for i, name in enumerate(header)}
    if "time" not in col:
        raise ValueError(f"missing time column in {path}")
    n = len(rows)
    times = [rows[i][col["time"]].strip() for i in range(n)]
    n_levels = 0
    arrays: dict[str, np.ndarray] = {}
    while f"ask_{n_levels + 1}" in col:
        n_levels += 1
        for kind in ("ask", "bid"):
            px = f"{kind}_{n_levels}"
            sz = f"{kind}_size_{n_levels}"
            px_arr = np.array([_to_float(rows[i][col[px]]) for i in range(n)], dtype=float)
            sz_arr = np.array([_to_float(rows[i][col[sz]]) for i in range(n)], dtype=float)
            arrays[px] = px_arr * float(price_scale)
            arrays[sz] = sz_arr
    meta = parse_csv_name(path.name)
    return {
        "time": np.asarray(times),
        "arrays": arrays,
        "n_levels": n_levels,
        "n_duplicate_times": int(len(times) - len(set(times))),
        "ticker": meta["ticker"] if meta else None,
        "date": meta["date"] if meta else None,
        "path": str(path),
        "header": header,
    }


def session_flags(times) -> dict:
    times = [str(t) for t in times]
    expected = set(EXPECTED_SESSION)
    got = set(times)
    return {
        "n_bars": len(times),
        "n_unique": len(got),
        "expected_full_session": 391,
        "is_full_session": times == list(EXPECTED_SESSION),
        "n_missing_vs_full": len(expected - got),
        "n_extra_vs_full": len(got - expected),
        "first": times[0] if times else None,
        "last": times[-1] if times else None,
        "early_close_candidate": len(times) <= 220 and (times[-1] if times else "") <= "13:05:00",
        "monotonic": times == sorted(times),
        "timezone_note": (
            "Clock times are civil America/New_York session time, not UTC. "
            "DST is absorbed by the calendar date; files do not store offsets."
        ),
    }


def quality_row(book: dict) -> dict:
    ask = book["arrays"]["ask_1"]
    bid = book["arrays"]["bid_1"]
    asz = book["arrays"]["ask_size_1"]
    bsz = book["arrays"]["bid_size_1"]
    n = len(book["time"])
    stale = 0
    trailing = 0
    if n > 1:
        keys = sorted(book["arrays"])
        stacked = np.column_stack([book["arrays"][k] for k in keys])
        same = np.all(stacked[1:] == stacked[:-1], axis=1)
        stale = int(same.sum())
        for flag in same[::-1]:
            if flag:
                trailing += 1
            else:
                break
    return {
        "ticker": book.get("ticker"),
        "date": book.get("date"),
        "n_levels": int(book["n_levels"]),
        "n_rows": int(n),
        "n_duplicate_times": int(book["n_duplicate_times"]),
        "n_nonpositive_ask": int(np.nansum(ask <= 0)),
        "n_nonpositive_bid": int(np.nansum(bid <= 0)),
        "n_nonpositive_ask_size": int(np.nansum(asz <= 0)),
        "n_nonpositive_bid_size": int(np.nansum(bsz <= 0)),
        "n_crossed": int(np.nansum(bid > ask)),
        "n_locked": int(np.nansum(bid == ask)),
        "n_stale_identical_rows": stale,
        "stale_fraction": float(stale / max(n - 1, 1)),
        "n_trailing_stale": trailing,
        "possible_padded_early_close": bool(trailing >= 60),
        "price_median_ask": float(np.nanmedian(ask)),
        "schema": ",".join(book["header"]),
        **session_flags(book["time"]),
    }


def top_imbalance(bid_size: np.ndarray, ask_size: np.ndarray) -> np.ndarray:
    """Primary state. Zero denominators are NaN, not zero."""
    bid_size = np.asarray(bid_size, dtype=float)
    ask_size = np.asarray(ask_size, dtype=float)
    den = bid_size + ask_size
    out = np.full(den.shape, np.nan, dtype=float)
    ok = np.isfinite(den) & (den > 0.0) & np.isfinite(bid_size) & np.isfinite(ask_size)
    ok &= (bid_size >= 0.0) & (ask_size >= 0.0)
    out[ok] = (bid_size[ok] - ask_size[ok]) / den[ok]
    return out


def depth_imbalance(book: dict, k: int) -> np.ndarray:
    n = len(book["time"])
    bid = np.zeros(n, dtype=float)
    ask = np.zeros(n, dtype=float)
    for level in range(1, k + 1):
        bcol = f"bid_size_{level}"
        acol = f"ask_size_{level}"
        if bcol not in book["arrays"] or acol not in book["arrays"]:
            return np.full(n, np.nan)
        bid = bid + book["arrays"][bcol]
        ask = ask + book["arrays"][acol]
    return top_imbalance(bid, ask)


@dataclass
class DayPanel:
    date: str
    times: np.ndarray
    tickers: list[str]
    x: np.ndarray
    valid: np.ndarray
    quality: list[dict]


def load_day_panel(
    tickers: list[str],
    date: str,
    levels: int,
    *,
    k_depth: int = 1,
    price_scale: float = 1e-4,
    cache_dir=None,
) -> DayPanel | None:
    frames = []
    quality = []
    for tkr in tickers:
        path = cached_csv(tkr, date, levels, cache_dir)
        if not path.exists():
            continue
        book = read_book_csv(path, price_scale=price_scale)
        if k_depth <= 1:
            x = top_imbalance(book["arrays"]["bid_size_1"], book["arrays"]["ask_size_1"])
        else:
            x = depth_imbalance(book, k_depth)
        frames.append((tkr, book["time"], x))
        quality.append(quality_row(book))
    if not frames:
        return None
    time_sets = [set(t.tolist()) for _, t, _ in frames]
    common = sorted(set.intersection(*time_sets))
    if not common:
        return None
    tickers_ok = [tkr for tkr, _, _ in frames]
    x = np.full((len(common), len(tickers_ok)), np.nan)
    t_index = {t: i for i, t in enumerate(common)}
    for j, (_, times, xs) in enumerate(frames):
        for t, val in zip(times, xs):
            i = t_index.get(str(t))
            if i is not None:
                x[i, j] = val
    return DayPanel(
        date=date,
        times=np.asarray(common),
        tickers=tickers_ok,
        x=x,
        valid=np.isfinite(x),
        quality=quality,
    )


def minutes_since_open(times: np.ndarray) -> np.ndarray:
    out = np.empty(len(times), dtype=float)
    origin = datetime.strptime("09:30:00", "%H:%M:%S")
    for i, t in enumerate(times):
        out[i] = (datetime.strptime(str(t), "%H:%M:%S") - origin).total_seconds() / 60.0
    return out


def mask_open_close(times: np.ndarray, drop_open: int, drop_close: int) -> np.ndarray:
    mins = minutes_since_open(times)
    last = float(mins.max()) if mins.size else 390.0
    return (mins >= float(drop_open)) & (mins <= last - float(drop_close))


def cross_section_moments(x_row: np.ndarray) -> tuple[float, float, int]:
    y = np.asarray(x_row, dtype=float)
    y = y[np.isfinite(y)]
    n = int(y.size)
    if n < 2:
        return float("nan"), float("nan"), n
    return float(y.mean()), float(y.var(ddof=1)), n
