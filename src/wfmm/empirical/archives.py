"""Scan and extract the one-minute order-book archive collection.

Archive names are treated as unverified metadata until files are opened.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from wfmm.paths import EMPIRICAL_CACHE_DIR, EMPIRICAL_DATA_DIR

ARCHIVE_RE = re.compile(
    r"_data_dwn_(?P<id1>\d+)_(?P<id2>\d+)__"
    r"(?P<ticker>[A-Z0-9.]+)_"
    r"(?P<start>\d{4}-\d{2}-\d{2})_"
    r"(?P<end>\d{4}-\d{2}-\d{2})_"
    r"(?P<levels>\d+)_(?P<freq>\d+)"
)
CSV_RE = re.compile(
    r"(?P<ticker>[A-Z0-9.]+)_(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<t0>\d+)_(?P<t1>\d+)_(?P<freq>\d+)_(?P<levels>\d+)\.csv$"
)


@dataclass
class ArchiveRecord:
    ticker: str
    start: str
    end: str
    levels: int
    freq: int
    source: str
    path: str
    size_bytes: int
    extracted: bool
    n_csv: int | None = None
    csv_date_min: str | None = None
    csv_date_max: str | None = None
    schema: str | None = None
    parse_ok: bool = True


def parse_archive_name(name: str) -> dict | None:
    stem = name[:-3] if name.endswith(".7z") else name
    m = ARCHIVE_RE.fullmatch(stem)
    if not m:
        return None
    d = m.groupdict()
    d["levels"] = int(d["levels"])
    d["freq"] = int(d["freq"])
    return d


def parse_csv_name(name: str) -> dict | None:
    m = CSV_RE.fullmatch(Path(name).name)
    if not m:
        return None
    d = m.groupdict()
    d["levels"] = int(d["levels"])
    d["freq"] = int(d["freq"])
    return d


def _csv_dates(folder: Path) -> tuple[int, str | None, str | None]:
    dates = []
    n = 0
    for p in folder.glob("*.csv"):
        meta = parse_csv_name(p.name)
        if meta is None:
            continue
        n += 1
        dates.append(meta["date"])
    if not dates:
        return n, None, None
    return n, min(dates), max(dates)


def scan_collection(data_dir: Path | None = None) -> list[ArchiveRecord]:
    data_dir = Path(data_dir) if data_dir else EMPIRICAL_DATA_DIR
    records: list[ArchiveRecord] = []
    if not data_dir.exists():
        return records
    for path in sorted(data_dir.iterdir()):
        if path.name.startswith("."):
            continue
        meta = parse_archive_name(path.name)
        if meta is None:
            continue
        extracted = path.is_dir()
        n_csv = dmin = dmax = None
        if extracted:
            n_csv, dmin, dmax = _csv_dates(path)
        records.append(
            ArchiveRecord(
                ticker=meta["ticker"],
                start=meta["start"],
                end=meta["end"],
                levels=meta["levels"],
                freq=meta["freq"],
                source="dir" if extracted else "7z",
                path=str(path),
                size_bytes=int(path.stat().st_size),
                extracted=extracted,
                n_csv=n_csv,
                csv_date_min=dmin,
                csv_date_max=dmax,
                schema=None,
            )
        )
    return records


def reconcile_counts(records: list[ArchiveRecord]) -> dict:
    tickers = {r.ticker for r in records}
    lev1 = {r.ticker for r in records if r.levels == 1}
    lev10 = {r.ticker for r in records if r.levels == 10}
    both = lev1 & lev10
    return {
        "n_records": len(records),
        "n_unique_tickers": len(tickers),
        "n_one_level_tickers": len(lev1),
        "n_ten_level_tickers": len(lev10),
        "n_both_level_tickers": len(both),
        "n_extracted_dirs": sum(1 for r in records if r.extracted),
        "n_archives": sum(1 for r in records if not r.extracted),
        "one_plus_ten": len(lev1) + len(lev10),
        "identity": (
            len(lev1) + len(lev10) - len(both) == len(tickers)
        ),
        "both_tickers": sorted(both),
    }


def records_as_dicts(records: list[ArchiveRecord]) -> list[dict]:
    return [asdict(r) for r in records]


def preferred_source(records: list[ArchiveRecord], ticker: str, levels: int) -> ArchiveRecord | None:
    hits = [r for r in records if r.ticker == ticker and r.levels == levels]
    if not hits:
        return None
    hits.sort(key=lambda r: (not r.extracted, -r.size_bytes))
    return hits[0]


def cached_csv(ticker: str, date: str, levels: int, cache_dir: Path | None = None) -> Path:
    cache_dir = cache_dir or EMPIRICAL_CACHE_DIR
    return cache_dir / ticker / f"{ticker}_{date}_34200000_57600000_60_{levels}.csv"


def extract_dates(
    archive: ArchiveRecord,
    dates: list[str],
    cache_dir: Path | None = None,
) -> dict:
    """Extract selected calendar dates only. Does not unpack the full archive."""
    cache_dir = cache_dir or EMPIRICAL_CACHE_DIR
    out_dir = cache_dir / archive.ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(dates)
    copied = []
    missing = []
    src = Path(archive.path)
    if archive.extracted:
        by_date = {}
        for p in src.glob("*.csv"):
            meta = parse_csv_name(p.name)
            if meta and meta["date"] in wanted:
                by_date[meta["date"]] = p
        for d in dates:
            if d not in by_date:
                missing.append(d)
                continue
            dest = cached_csv(archive.ticker, d, archive.levels, cache_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.write_bytes(by_date[d].read_bytes())
            copied.append(d)
        return {"copied": copied, "missing": missing, "method": "copy"}

    try:
        import py7zr
    except ImportError as exc:
        raise ImportError("py7zr is required to extract .7z archives") from exc

    with py7zr.SevenZipFile(src, mode="r") as zf:
        names = zf.getnames()
        targets = []
        date_for = {}
        for name in names:
            meta = parse_csv_name(Path(name).name)
            if meta is None:
                continue
            if meta["date"] in wanted and meta["ticker"] == archive.ticker:
                targets.append(name)
                date_for[name] = meta["date"]
        found_dates = set(date_for.values())
        missing = [d for d in dates if d not in found_dates]
        if targets:
            zf.extract(path=out_dir, targets=targets)
            for name in targets:
                src_csv = out_dir / name
                if not src_csv.exists():
                    src_csv = out_dir / Path(name).name
                dest = cached_csv(archive.ticker, date_for[name], archive.levels, cache_dir)
                if src_csv.exists() and src_csv.resolve() != dest.resolve():
                    dest.write_bytes(src_csv.read_bytes())
                    if src_csv.parent != dest.parent:
                        src_csv.unlink(missing_ok=True)
                copied.append(date_for[name])
    return {"copied": sorted(set(copied)), "missing": missing, "method": "7z"}
