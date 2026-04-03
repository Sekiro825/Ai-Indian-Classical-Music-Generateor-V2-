#!/usr/bin/env python3
"""
Build a curated V2 training dataset from manifest_processed.json.

What it does:
1. Keeps only processed entries with valid MIDI + expression pairs.
2. Canonicalizes common raga aliases (e.g., harikambhoji/hari_kambhoji).
3. Drops likely non-musical tracks via filename keywords.
4. Drops ragas with too few source tracks.
5. Materializes curated files under data/midi_v2_curated/{tradition}/{raga}/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_DROP_KEYWORDS = [
    "intro",
    "introduction",
    "speech",
    "concept",
    "lyrics",
    "theka",
    "taal and sur",
]


RAGA_ALIASES = {
    "harikambhoji": "hari_kambhoji",
    "sri_ranjani": "sriranjani",
    "mian_malhar": "miyan_malhar",
    "raageshree": "rageshri",
    "aahir": "ahir_bhairon",
}


@dataclass
class Segment:
    midi_path: Path
    expr_path: Path
    tradition: str
    raga: str
    source_file: str


def canonicalize_raga(raga: str) -> str:
    r = (raga or "unknown").strip().lower().replace(" ", "_")
    r = "_".join([p for p in r.split("_") if p])
    return RAGA_ALIASES.get(r, r)


def should_drop_filename(path_str: str, keywords: List[str]) -> bool:
    name = Path(path_str).name.lower()
    return any(k in name for k in keywords)


def split_midi_paths(midi_output_path: str) -> List[Path]:
    if not midi_output_path:
        return []
    return [Path(p) for p in midi_output_path.split(";") if p.strip()]


def safe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "symlink":
        os.symlink(src.resolve(), dst)
        return

    # hardlink (default), fallback to copy if cross-filesystem
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_manifest(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_segments(
    manifest: List[dict],
    min_confidence: float,
    drop_keywords: List[str],
    min_duration_s: float,
) -> Tuple[List[Segment], Counter]:
    stats = Counter()
    segments: List[Segment] = []

    for entry in manifest:
        if not entry.get("processed", False):
            stats["drop_not_processed"] += 1
            continue

        raga = canonicalize_raga(entry.get("raga", "unknown"))
        if raga == "unknown":
            stats["drop_unknown_raga"] += 1
            continue

        conf = float(entry.get("raga_confidence", 0.0))
        if conf < min_confidence:
            stats["drop_low_confidence"] += 1
            continue

        if should_drop_filename(entry.get("file_path", ""), drop_keywords):
            stats["drop_non_musical_keyword"] += 1
            continue

        duration = float(entry.get("duration_seconds", 0.0))
        if duration > 0.0 and duration < min_duration_s:
            stats["drop_too_short"] += 1
            continue

        midi_paths = split_midi_paths(entry.get("midi_output_path", ""))
        if not midi_paths:
            stats["drop_missing_midi_path"] += 1
            continue

        tradition = (entry.get("tradition") or "unknown").strip().lower()
        source_file = entry.get("file_path", "")

        added_any = False
        for midi_path in midi_paths:
            if not midi_path.exists():
                stats["drop_missing_midi_file"] += 1
                continue

            expr_path = Path(str(midi_path).replace(".mid", ".expr.npy"))
            if not expr_path.exists():
                stats["drop_missing_expr_pair"] += 1
                continue

            segments.append(
                Segment(
                    midi_path=midi_path,
                    expr_path=expr_path,
                    tradition=tradition,
                    raga=raga,
                    source_file=source_file,
                )
            )
            added_any = True

        if added_any:
            stats["kept_entries"] += 1

    stats["candidate_segments"] = len(segments)
    return segments, stats


def filter_min_tracks(segments: List[Segment], min_tracks_per_raga: int) -> Tuple[List[Segment], Counter]:
    track_sets: Dict[str, set] = defaultdict(set)
    for s in segments:
        track_sets[s.raga].add(s.source_file)

    eligible_ragas = {r for r, tracks in track_sets.items() if len(tracks) >= min_tracks_per_raga}
    dropped_ragas = {r for r in track_sets.keys() if r not in eligible_ragas}

    filtered = [s for s in segments if s.raga in eligible_ragas]
    stats = Counter()
    stats["ragas_before_min_track_filter"] = len(track_sets)
    stats["ragas_after_min_track_filter"] = len(eligible_ragas)
    stats["ragas_dropped_min_track_filter"] = len(dropped_ragas)
    stats["segments_after_min_track_filter"] = len(filtered)
    return filtered, stats


def materialize_dataset(
    segments: List[Segment],
    output_dir: Path,
    link_mode: str,
    overwrite: bool,
) -> Counter:
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    seen_dest = set()

    for s in segments:
        out_dir = output_dir / s.tradition / s.raga
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = s.midi_path.stem
        if stem in seen_dest:
            suffix = hashlib.md5(str(s.midi_path).encode("utf-8")).hexdigest()[:8]
            stem = f"{stem}_{suffix}"
        seen_dest.add(stem)

        dst_midi = out_dir / f"{stem}.mid"
        dst_expr = out_dir / f"{stem}.expr.npy"
        safe_link_or_copy(s.midi_path, dst_midi, link_mode)
        safe_link_or_copy(s.expr_path, dst_expr, link_mode)
        stats["written_pairs"] += 1

    return stats


def write_report(
    report_path: Path,
    stats: Dict[str, int],
    segments: List[Segment],
) -> None:
    raga_segments = Counter(s.raga for s in segments)
    tradition_segments = Counter(s.tradition for s in segments)

    report = {
        "summary": stats,
        "raga_segment_counts": dict(raga_segments.most_common()),
        "tradition_segment_counts": dict(tradition_segments.most_common()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare curated dataset for V2 training")
    p.add_argument("--manifest", type=Path, default=Path("data/midi_v2/manifest_processed.json"))
    p.add_argument("--output_dir", type=Path, default=Path("data/midi_v2_curated"))
    p.add_argument("--report_path", type=Path, default=Path("data/midi_v2_curated_report.json"))
    p.add_argument("--min_confidence", type=float, default=0.8)
    p.add_argument("--min_tracks_per_raga", type=int, default=5)
    p.add_argument("--min_duration_s", type=float, default=8.0)
    p.add_argument("--link_mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--drop_keywords",
        nargs="*",
        default=DEFAULT_DROP_KEYWORDS,
        help="Lowercase keywords to exclude likely non-musical tracks.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)

    segments, s1 = build_segments(
        manifest=manifest,
        min_confidence=args.min_confidence,
        drop_keywords=[k.lower() for k in args.drop_keywords],
        min_duration_s=args.min_duration_s,
    )
    segments, s2 = filter_min_tracks(segments, args.min_tracks_per_raga)

    stats = Counter()
    stats["manifest_entries"] = len(manifest)
    stats.update(s1)
    stats.update(s2)

    if not args.dry_run:
        s3 = materialize_dataset(
            segments=segments,
            output_dir=args.output_dir,
            link_mode=args.link_mode,
            overwrite=args.overwrite,
        )
        stats.update(s3)

    write_report(args.report_path, dict(stats), segments)

    print("Curated dataset preparation complete.")
    for k in sorted(stats.keys()):
        print(f"{k}: {stats[k]}")
    print(f"report_path: {args.report_path}")
    if not args.dry_run:
        print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
