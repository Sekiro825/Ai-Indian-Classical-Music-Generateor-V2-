# How to Expand Your Dataset

Your model quality improves significantly with more training data. Here's how to get more MIDI files for ragas.

---

## Current Dataset
- **684 MIDI files**
- **19 ragas** with varying sample counts
- More data = better generation quality

---

## Free Sources for Indian Classical MIDI

### 1. Comp Music Datasets (Recommended)
The best academic source for Indian Classical Music:

- **Website**: https://compmusic.upf.edu/carnatic-audio-corpus
- **Contains**: Carnatic and Hindustani recordings
- **How to use**: 
  1. Request access (free for research)
  2. Use audio-to-MIDI conversion tools

### 2. Basic Pitch (Audio to MIDI)
Convert YouTube audio to MIDI:

```bash
pip install basic-pitch

# Convert an audio file to MIDI
basic-pitch output_folder input_audio.mp3
```

**YouTube Sources for Raga Audio**:
- Search "Raag Yaman flute" or "Raag Bhairavi sitar"
- Download audio using `yt-dlp`:
  ```bash
  pip install yt-dlp
  yt-dlp -x --audio-format mp3 "https://youtube.com/watch?v=VIDEO_ID"
  ```

### 3. FluidSynth MIDI Files
- **Website**: https://archive.org/search?query=indian+classical+midi
- Search for "raga midi" or "hindustani midi"

### 4. MIDI World
- **Website**: https://www.midiworld.com/
- Search for Indian music or world music sections

### 5. Lakh MIDI Dataset
- **Website**: https://colinraffel.com/projects/lmd/
- Contains 170,000+ MIDI files
- Filter for world music / Indian tags

---

## How to Convert Audio to MIDI

### Method 1: Basic Pitch (Best Quality)

```bash
# Install
pip install basic-pitch

# Convert single file
basic-pitch ./output_midi ./input_audio.mp3

# Convert folder of audio files
basic-pitch ./output_midi ./audio_folder/
```

### Method 2: Online Converters
- https://www.bearaudiotool.com/mp3-to-midi
- https://www.conversion-tool.com/audiotomidi

---

## Organizing New MIDI Files

Name your MIDI files with the raga name:

```
yaman_track1.mid
yaman_performance2.mid
bhairavi_alap1.mid
malkauns_concert.mid
```

The dataset will automatically extract the raga from the filename.

---

## Adding New Ragas

If you add a new raga, update `config/raga_metadata.json`:

```json
{
  "new_raga_name": {
    "moods": ["meditative", "peaceful"],
    "time_of_day": "evening",
    "tempo_range": [60, 100],
    "thaat": "kalyan",
    "description": "Description of the raga"
  }
}
```

---

## Recommended Dataset Sizes

| Dataset Size | Expected Quality |
|-------------|------------------|
| 500-1000 | Basic generation |
| 1000-2000 | **Good quality** (recommended) |
| 2000-5000 | Very good quality |
| 5000+ | Excellent quality |

---

## Quick Expansion Script

Create a script to download and convert YouTube audio:

```python
# expand_dataset.py
import subprocess
import os
from pathlib import Path

# List of YouTube URLs (raga performances)
YOUTUBE_URLS = [
    ("yaman", "https://youtube.com/watch?v=EXAMPLE1"),
    ("bhairavi", "https://youtube.com/watch?v=EXAMPLE2"),
    # Add more...
]

OUTPUT_DIR = Path("all_midi")
TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)

for raga, url in YOUTUBE_URLS:
    try:
        # Download audio
        audio_file = TEMP_DIR / f"{raga}_temp.mp3"
        subprocess.run([
            "yt-dlp", "-x", "--audio-format", "mp3",
            "-o", str(audio_file), url
        ], check=True)
        
        # Convert to MIDI
        subprocess.run([
            "basic-pitch", str(OUTPUT_DIR), str(audio_file)
        ], check=True)
        
        print(f"✅ Converted: {raga}")
    except Exception as e:
        print(f"❌ Failed: {raga} - {e}")
```

---

## Tips for Better Data

1. **Prefer solo instruments** - Easier to convert accurately
2. **Avoid vocals-heavy tracks** - MIDI conversion struggles
3. **Look for clear, studio recordings** - Noisy live recordings convert poorly
4. **Balance your dataset** - Try to have similar counts per raga

---

## Verify Your Dataset

After adding new files, verify:

```bash
# Count MIDI files per raga
python -c "
from pathlib import Path
from collections import Counter
from models.tokenizer import extract_raga_from_filename

files = list(Path('all_midi').glob('*.mid'))
ragas = [extract_raga_from_filename(f.name) for f in files]
counts = Counter(ragas)
for raga, count in sorted(counts.items(), key=lambda x: -x[1]):
    print(f'{raga}: {count} files')
print(f'\\nTotal: {len(files)} files')
"
```
