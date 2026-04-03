"""
Merge Legacy Dataset into V2 Structure

Takes the old flat data/midi/ + data/audio_features/ and:
1. Classifies each file by raga (from filename)
2. Copies the MIDI into data/midi_v2/{tradition}/{raga}/
3. Extracts .expr.npy from the matching WAV file
4. Everything ends up alongside the NEW_dataset! output

Usage:
    python scripts/merge_legacy_data.py
    python scripts/merge_legacy_data.py --midi_dir data/midi --wav_dir data/audio_features --output_dir data/midi_v2
"""

import os
import sys
import re
import shutil
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


# ============================================================
# Raga Classification from Filename
# ============================================================

# Melody ragas — extracted from filename patterns
RAGA_PATTERNS = {
    # Hindustani ragas (from "Raag X.mp3_basic_pitch.mid" files)
    'yaman': 'yaman',
    'bhairavi': 'bhairavi',
    'malkauns': 'malkauns',
    'bageshree': 'bageshree',
    'bhoopali': 'bhoopali',
    'bhoop': 'bhoopali',
    'asavari': 'asavari',
    'sarang': 'sarang',
    'darbari': 'darbari',
    'dkanada': 'darbari_kanada',
    'todi': 'todi',
    'bilaskhani': 'bilaskhani_todi',
    'bihag': 'bihag',
    'bhimpalasi': 'bhimpalasi',
    'khamaj': 'khamaj',
    'kedar': 'kedar',
    'jog': 'jog',
    'desh': 'desh',
    'bahar': 'bahar',
    'multani': 'multani',
    'kalavati': 'kalavati',
    'kirwani': 'kirwani',
    'marwa': 'marwa',
    'puriya': 'puriya',
    'chandrakauns': 'chandrakauns',
    'hameer': 'hameer',
    'bhatiyar': 'bhatiyar',
    'bibhas': 'bibhas',
    'dhani': 'dhani',
    'gavti': 'gavti',
    'hindol': 'hindol',
    'jogiya': 'jogiya',
    'khokar': 'khokar',
    'gaud malhar': 'gaud_malhar',
    'jait kalyan': 'jait_kalyan',
    'basanti kedar': 'basanti_kedar',
    'bhairav': 'bhairav',
    'ahir bhairon': 'ahir_bhairon',
    'aahir bhairon': 'ahir_bhairon',
    'dagori': 'dagori',
    'khat todi': 'khat_todi',
    'nat bhairon': 'nat_bhairon',
    'bairagi': 'bairagi',
    'abhogi': 'abhogi',
    'rageshri': 'rageshri',
    'maru bihag': 'maru_bihag',
    'piloo': 'piloo',
    'irani bhairavi': 'bhairavi',
}

# Taal patterns — these are rhythm-only, labelled as taal type
TAAL_PATTERNS = [
    'addhatrital', 'trital', 'dadra', 'deepchandi', 'ektal',
    'jhaptal', 'rupak', 'bhajani',
]


def classify_legacy_file(filename: str) -> tuple:
    """
    Classify a file from the old dataset.
    Returns (tradition, raga_or_taal, is_taal)
    """
    name_lower = filename.lower()
    stem = Path(filename).stem.lower()

    # Remove suffixes like "_basic_pitch"
    clean = stem.replace('_basic_pitch', '').replace('.mp3', '')

    # Check taal patterns first (rhythm data)
    for taal in TAAL_PATTERNS:
        if clean.startswith(taal):
            return ('hindustani', taal, True)

    # Check "Raag X" pattern
    raag_match = re.match(r'^raag?\s+(.+?)(?:\s*[-_].*)?$', clean, re.IGNORECASE)
    if raag_match:
        raag_name = raag_match.group(1).strip().lower()
        # Check against known mappings
        for pattern, raga in RAGA_PATTERNS.items():
            if pattern in raag_name:
                return ('hindustani', raga, False)
        # Use the extracted name directly
        return ('hindustani', raag_name.replace(' ', '_'), False)

    # Check against all known patterns
    for pattern, raga in sorted(RAGA_PATTERNS.items(), key=lambda x: -len(x[0])):
        if pattern in clean:
            return ('hindustani', raga, False)

    # Carnatic-looking files (numbered suffix from saraga collection)
    if re.match(r'^\d+_\d+_', clean):
        return ('carnatic', 'unknown_carnatic', False)

    # Mixed/unknown
    if 'mixed' in clean or 'indian classical' in clean:
        return ('hindustani', 'mixed', False)

    return ('unknown', 'unknown', False)


def extract_expression_from_wav(wav_path: str, output_path: str):
    """Extract f0, amplitude, voiced, spectral_centroid from WAV"""
    import librosa

    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    if len(y) < sr:  # skip < 1 second
        return False

    hop_length = 512

    # f0 via PYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        sr=sr, hop_length=hop_length
    )
    f0_safe = np.nan_to_num(f0, nan=1e-6)
    f0_log = np.log2(np.maximum(f0_safe, 1e-6) / 440.0)

    # RMS
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    centroid_norm = centroid / (sr / 2)

    # Align
    min_len = min(len(f0_log), len(rms), len(centroid_norm))
    voiced = voiced_probs[:min_len] if voiced_probs is not None else voiced_flag[:min_len].astype(float)

    features = np.stack([
        f0_log[:min_len],
        rms[:min_len],
        voiced,
        centroid_norm[:min_len],
    ], axis=-1).astype(np.float32)

    np.save(output_path, features)
    return True


def process_one_file(midi_path: str, wav_path: str, output_dir: str) -> dict:
    """Process one legacy file: copy MIDI + extract expression from WAV"""
    result = {'midi': midi_path, 'status': 'ok', 'error': ''}
    try:
        filename = Path(midi_path).name
        tradition, raga, is_taal = classify_legacy_file(filename)

        # Create output directory
        if is_taal:
            dest_dir = Path(output_dir) / tradition / f"taal_{raga}"
        else:
            dest_dir = Path(output_dir) / tradition / raga
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Clean filename
        clean_name = filename.replace('_basic_pitch', '').replace('.mp3', '')
        dest_midi = dest_dir / f"{clean_name}.mid"

        # Copy MIDI
        if not dest_midi.exists():
            shutil.copy2(midi_path, dest_midi)

        result['dest'] = str(dest_midi)
        result['raga'] = raga
        result['tradition'] = tradition

        # Extract expression from WAV if available
        if wav_path and Path(wav_path).exists():
            expr_path = dest_dir / f"{clean_name}.expr.npy"
            if not expr_path.exists():
                try:
                    extract_expression_from_wav(wav_path, str(expr_path))
                    result['expression'] = str(expr_path)
                except Exception as e:
                    result['expr_error'] = str(e)
        else:
            result['no_wav'] = True

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)

    return result


def find_matching_wav(midi_path: Path, wav_dir: Path) -> str:
    """Find the WAV file that matches a MIDI file"""
    stem = midi_path.stem
    # Remove _basic_pitch suffix
    wav_stem = stem.replace('_basic_pitch', '').replace('.mp3', '')

    # Try exact match
    for ext in ['.wav', '.WAV']:
        candidate = wav_dir / f"{wav_stem}{ext}"
        if candidate.exists():
            return str(candidate)

    # Try with .mp3 suffix removed
    clean = wav_stem.replace('.mp3', '')
    for ext in ['.wav', '.WAV']:
        candidate = wav_dir / f"{clean}{ext}"
        if candidate.exists():
            return str(candidate)

    return ""


def main():
    parser = argparse.ArgumentParser(description='Merge legacy data into V2 structure')
    parser.add_argument('--midi_dir', type=str, default='data/midi',
                        help='Directory with old MIDI files')
    parser.add_argument('--wav_dir', type=str, default='data/audio_features',
                        help='Directory with old WAV files')
    parser.add_argument('--output_dir', type=str, default='data/midi_v2',
                        help='Unified V2 output directory')
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of parallel workers for expression extraction')
    parser.add_argument('--dry-run', action='store_true',
                        help='Just show classification, don\'t copy anything')
    args = parser.parse_args()

    midi_dir = Path(args.midi_dir)
    wav_dir = Path(args.wav_dir)
    output_dir = Path(args.output_dir)

    if not midi_dir.exists():
        logger.error(f"MIDI directory not found: {midi_dir}")
        return

    # Collect all MIDI files
    midi_files = sorted(midi_dir.glob("*.mid"))
    logger.info(f"Found {len(midi_files)} MIDI files in {midi_dir}")

    # Classify and show summary
    classification = {}
    for f in midi_files:
        tradition, raga, is_taal = classify_legacy_file(f.name)
        key = f"{tradition}/{raga}"
        if key not in classification:
            classification[key] = []
        classification[key].append(f.name)

    logger.info(f"\n=== CLASSIFICATION SUMMARY ===")
    total_melody = 0
    total_taal = 0
    for key in sorted(classification.keys()):
        count = len(classification[key])
        is_taal = 'taal_' in key
        tag = "🥁" if is_taal else "🎵"
        logger.info(f"  {tag} {key}: {count} files")
        if is_taal:
            total_taal += count
        else:
            total_melody += count

    logger.info(f"\n  Total melody files: {total_melody}")
    logger.info(f"  Total taal files: {total_taal}")
    logger.info(f"  Unique ragas: {len([k for k in classification if 'taal_' not in k])}")

    if args.dry_run:
        logger.info("Dry run — not copying anything.")
        return

    # Process files
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for midi_file in midi_files:
        wav_path = find_matching_wav(midi_file, wav_dir)
        tasks.append((str(midi_file), wav_path, str(output_dir)))

    logger.info(f"\nProcessing {len(tasks)} files with {args.workers} workers...")
    wav_found = sum(1 for _, w, _ in tasks if w)
    wav_missing = len(tasks) - wav_found
    logger.info(f"  WAV matches found: {wav_found}")
    logger.info(f"  WAV missing (MIDI-only): {wav_missing}")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one_file, m, w, o): m
            for m, w, o in tasks
        }
        
        # Wrapped with tqdm for real-time visual progress
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Merging Legacy Data"):
            result = future.result()
            results.append(result)

    # Final summary
    ok = sum(1 for r in results if r['status'] == 'ok')
    errors = sum(1 for r in results if r['status'] == 'error')
    with_expr = sum(1 for r in results if 'expression' in r)

    logger.info(f"\n=== MERGE COMPLETE ===")
    logger.info(f"  Copied: {ok}/{len(tasks)}")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  With expression (.expr.npy): {with_expr}")

    # Verify output
    midi_count = len(list(output_dir.rglob("*.mid")))
    expr_count = len(list(output_dir.rglob("*.expr.npy")))
    ragas = set()
    for f in output_dir.rglob("*.mid"):
        parts = f.relative_to(output_dir).parts
        if len(parts) >= 2:
            ragas.add(parts[1])

    logger.info(f"\n=== UNIFIED data/midi_v2/ ===")
    logger.info(f"  Total MIDI: {midi_count}")
    logger.info(f"  Total .expr.npy: {expr_count}")
    logger.info(f"  Unique ragas/taals: {len(ragas)}")
    logger.info(f"  Ragas: {sorted(ragas)}")


if __name__ == "__main__":
    main()
