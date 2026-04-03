import os
import json
import shutil
from pathlib import Path

# Paths
DATASET_ROOT = Path(r"d:\MUSIC_MP\EXTRACTED\ALLDATA\dataset")
CLEAN_DATASET_ROOT = Path(r"d:\MUSIC_MP\EXTRACTED\ALLDATA\dataset_clean")

# Stats
stats = {
    "processed_folders": 0,
    "found_audio": 0,
    "missing_audio": 0,
    "ragas_found": set(),
}

def clean_dataset():
    if CLEAN_DATASET_ROOT.exists():
        shutil.rmtree(CLEAN_DATASET_ROOT)
    CLEAN_DATASET_ROOT.mkdir(exist_ok=True)

    print(f"Scanning {DATASET_ROOT}...")

    # Iterate through both Carnatic and Hindustani folders
    for tradition in ["carnatic", "hindustani"]:
        tradition_path = DATASET_ROOT / tradition
        if not tradition_path.exists():
            print(f"Skipping {tradition} (folder not found)")
            continue

        # Walk through artist/album folders
        for artist_folder in tradition_path.iterdir():
            if not artist_folder.is_dir():
                continue
            
            # Walk through track folders
            for track_folder in artist_folder.iterdir():
                if not track_folder.is_dir():
                    continue
                
                stats["processed_folders"] += 1
                process_track_folder(track_folder)

    print_summary()

def process_track_folder(folder_path):
    # Find JSON metadata file
    json_files = list(folder_path.glob("*.json"))
    if not json_files:
        return

    json_file = json_files[0]
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return

    # Extract Raga name
    try:
        if "raaga" in metadata and metadata["raaga"]:
             raga_name = metadata["raaga"][0]["common_name"]
        elif "raag" in metadata and metadata["raag"]: # Hindustani sometimes uses 'raag'
             raga_name = metadata["raag"][0]["common_name"]
        else:
            raga_name = "unknown"
    except (KeyError, IndexError):
        raga_name = "unknown"

    # Normalize raga name (lowercase, no spaces)
    raga_name = raga_name.lower().replace(" ", "_")
    stats["ragas_found"].add(raga_name)

    # Find Audio File in the SAME folder
    audio_files = list(folder_path.glob("*.mp3"))
    
    if not audio_files:
        stats["missing_audio"] += 1
        # print(f"Missing audio in: {folder_path.name}") # Uncomment for verbose
        return

    # Copy audio to clean dataset
    src_audio = audio_files[0]
    dest_folder = CLEAN_DATASET_ROOT / raga_name
    dest_folder.mkdir(exist_ok=True, parents=True)
    
    dest_filename = f"{raga_name}_{src_audio.name}"
    dest_path = dest_folder / dest_filename
    
    try:
        shutil.copy2(src_audio, dest_path)
        stats["found_audio"] += 1
        print(f"Sorted: {raga_name} -> {dest_filename}")
    except Exception as e:
        print(f"Error copying {src_audio}: {e}")

def print_summary():
    print("\n" + "="*30)
    print("DATASET CLEANING SUMMARY")
    print("="*30)
    print(f"Folders Scanned: {stats['processed_folders']}")
    print(f"Audio Files Found: {stats['found_audio']}")
    print(f"Audio Files MISSING: {stats['missing_audio']}")
    print(f"Unique Ragas: {len(stats['ragas_found'])}")
    print("="*30)
    
    if stats['found_audio'] == 0:
        print("\nCRITICAL: No audio files were found!")
        print("It seems you only downloaded the metadata (JSON/Text files).")
        print("Please check your download source and ensure you downloaded the AUDIO content, not just annotations.")

if __name__ == "__main__":
    clean_dataset()
