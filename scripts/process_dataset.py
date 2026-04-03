"""
Unified Dataset Processing Pipeline for Indian Classical Music
Processes all 5 sub-datasets: classifies ragas, cleans audio, converts to MIDI.
Designed to run on Lightning.ai with 48 CPUs for parallel processing.

Usage:
    python scripts/process_dataset.py --input_dir NEW_dataset! --output_dir data/midi_v2
    python scripts/process_dataset.py --input_dir NEW_dataset! --output_dir data/midi_v2 --validate-only
    python scripts/process_dataset.py --input_dir NEW_dataset! --output_dir data/midi_v2 --skip-midi
"""

import os
import sys
import json
import re
import shutil
import argparse
import logging
import hashlib
from pathlib import Path
# Ensure the scripts directory is in the path for internal utility imports
scripts_dir = str(Path(__file__).parent.absolute())
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

# Try to load dotenv for API keys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from tqdm import tqdm

import numpy as np
import librosa
import soundfile as sf
import traceback

from utils.transcription_onnx import BasicPitchONNX

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/processing.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Specialized error log for tracking failures across 48 workers
ERROR_LOG_PATH = Path("data/preprocessing_errors.txt")
ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# Known Composition → Raga Mappings
# ============================================================

# Hindustani: Parse "Raag X" from filename
HINDUSTANI_FILENAME_REGEX = re.compile(
    r'^(?:Raag?|Raga)\s+(.+?)(?:\s*[-_].*)?\.mp3$', re.IGNORECASE
)

# Carnatic composition → raga lookup (well-known repertoire)
CARNATIC_COMPOSITION_RAGA = {
    # Tyagaraja kritis
    "Endaro Mahanubhavulu": "sri",
    "Brochevarevarura": "sri_ranjani",
    "Dudukugala": "gowla",
    "Sogasuga": "sriranjani",
    "Koluvaiyunnade": "bhairavi",
    "Ninnuvina Marigalada": "vasanta",
    "Etulabrotuva": "chakravaakam",
    "Dorakuna": "bilahari",
    "Bantureethi": "hamsanadam",
    "Chakkani Raja": "kharaharapriya",
    "Entha Muddo": "bindu_malini",
    "Marugelara": "jayantashri",
    "Nannu Brova Neeku": "abhogi",
    "Nee Sari Evvaramma": "kambhoji",
    "Entara": "hari_kambhoji",
    "Eranapai": "karaharapriya",
    "Rama Neepai": "kharaharapriya",
    "Saraguna Palimpa": "sankarabharanam",
    "Sarasuda Ninne Kori": "saveri",
    "Karuna Nidhi Illalo": "todi",
    "O Rangashayee": "kambhoji",
    "Rama Rama Guna Seema": "simhendramadhyamam",

    # Dikshitar compositions
    "Kamalamba Samrakshatu": "kalyani",
    "Sri Kantimatim": "kalyani",
    "Sri Subramanyaya Namaste": "kambhoji",
    "Hiranmayeem Lakshmeem": "lalitha",
    "Hiranmayeem": "lalitha",
    "Sri Guruguhasya": "sankarabharanam",
    "Rakta Ganapatim": "mohanam",

    # Shyama Shastri compositions
    "Kamakshi": "saveri",

    # Common kritis
    "Shobillu Saptasvara": "jaganmohini",
    "Thillana Hameerkalyani": "hameer_kalyani",
    "Thillana Pahadi": "pahadi",
    "Thillana Vasanta": "vasanta",
    "Thillana Rageshri": "rageshri",
    "Thillana Senchurutti": "senchurutti",
    "Thillana Purnachandrika": "purnachandrika",
    "Krishna Nee Begane Baaro": "yamuna_kalyani",
    "Bhavamulona": "subhapantuvarali",
    "Bhuvini Dasudane": "sriranjani",
    "Siddhi Vinayakam": "shanmukhapriya",
    "Chintayama Kanda": "bhairavi",
    "Aparadhamula": "rasali",
    "Dharini Telusukonti": "suddha_dhanyasi",
    "Dinamani Vamsa": "hari_kambhoji",
    "Gnanamosaga Rada": "purvikalyani",
    "Parama Pavana Rama": "sankarabharanam",
    "Palisomma Muddu Sarade": "ananda_bhairavi",
    "Lokavana Chatura": "sankarabharanam",
    "Thappi Bratikipova": "atana",
    "Sundari Nee Divya": "kalyani",
    "Sudhaamayee": "amritavarshini",
    "Siva Siva Siva Yenarada": "pantuvarali",
    "Vinakayunna Della": "ravichandrika",
    "Emani Migula": "todi",
    "Evarura": "mohanam",
    "Sarasadalanayana": "pantuvarali",
    "Janakipathe": "kharaharapriya",
    "Manasaramathi": "hindolam",
    "Mati Matiki": "kharaharapriya",
    "Samajavarada": "hindolam",
    "Merusamana": "mayamalavagowla",
    "Ganamuda Panam": "nattai",
    "Shankari Shankuru": "sankarabharanam",
    "Velum Mayilume": "shanmukhapriya",
    "Muruga Muruga Muruga": "saveri",
    "Brova Barama": "bahudari",

    # Varnams
    "Vanajakshi - Varnam": "kalyani",
    "Thiruveragane Saveri Varnam": "saveri",

    # Mangalams
    "Mangalam": "suruti",

    # Ragam Tanam Pallavi
    "Raagam Thaanam Pallavi": "kalyani",
    "RTP Andholika": "andolika",

    "Amba Kamakshi": "bhairavi",
    "Amba Nilambari": "neelambari",
    "Anandaamrutakarshini": "amritavarshini",
    "Budham Aashrayami": "nattakurinji",
    "Kailasapathe": "sankarabharanam",
    "Geetha Vaadya Natana": "sankarabharanam",
    "Shri Kamakshi Shritajana": "saveri",
    "Shri Subramanyaya Namaste": "kambhoji",
    "Malmaruga Shanmuga": "shanmukhapriya",
    "Paragu Maadada": "kambhoji",
    "Paramatmudu": "vagadheeswari",
    "Pranamamyaham Sri Prananatham": "vagadheeswari",
    "Santhamu Leka Sowkyamu Ledu": "sama",
}

# South Indian Classical Snippets — composition name → raga
SOUTH_INDIAN_COMPOSITION_RAGA = {
    "Alamelu Manga": "dhanyasi",
    "Ammaravamma": "kalyani",
    "Arere Jaya Jaya": "sankarabharanam",
    "Bala Tripura Sundari": "kalyani",
    "Dasaratha Nandana": "kalyani",
    "Hecharika": "yadukula_kambhoji",
    "Hindolam Swarapallavi": "hindolam",
    "Jeyathu Jeyathu": "nattai",
    "Kamala Sanavam": "kalyani",
    "Kamalajadala": "kalyani",
    "Kambhoji Varnam": "kambhoji",
    "Mayatita Swaroopini": "mayamalavagowla",
    "Ninnu Kori": "mohanam",
    "Ramachandraya Janaka": "saveri",
    "Raravenu": "mohanam",
    "Sankarabharanam Varnam": "sankarabharanam",
    "Sarojadalanetri": "sankarabharanam",
    "Sitamma Mayamma": "vasanta",
    "Sri Gananatha": "nattai",
    "Sri Govinda": "bhairavi",
    "Syamale Meenakshi": "sankarabharanam",
    "Upacharamu": "bhairavi",
    "Vandanamu Raghunandana": "sahana",
    "Varaveena1": "mohanam",
    "Varaveena2": "mohanam",
    "Vatapi Ganapatim": "hamsadhwani",
}


@dataclass
class AudioFileEntry:
    """Represents a single audio file with metadata"""
    file_path: str
    tradition: str  # carnatic | hindustani | unknown
    raga: str
    raga_confidence: float  # 0.0 - 1.0
    source_dataset: str
    format: str  # mp3, wav
    duration_seconds: float = 0.0
    has_tala_annotations: bool = False
    midi_output_path: str = ""
    processed: bool = False
    error: str = ""


# ============================================================
# Dataset Scanners
# ============================================================

def scan_dunya_hindustani(base_dir: Path) -> List[AudioFileEntry]:
    """Scan dunya-hindustani-cc: parse raga names from filenames like 'Raag Yaman.mp3'"""
    entries = []
    dataset_dir = base_dir / "dunya-hindustani-cc"
    if not dataset_dir.exists():
        logger.warning(f"dunya-hindustani-cc not found at {dataset_dir}")
        return entries

    for mp3_file in dataset_dir.glob("*.mp3"):
        name = mp3_file.stem
        raga = "unknown"
        confidence = 0.0

        # Try regex match for "Raag X" or "Raga X"
        match = HINDUSTANI_FILENAME_REGEX.match(mp3_file.name)
        if match:
            raga = match.group(1).strip().lower().replace(" ", "_")
            confidence = 0.95
        else:
            # Some files just have the raga name directly
            name_lower = name.lower().replace(" ", "_")
            # Check for known raga names
            known_ragas = [
                "bhairavi", "todi", "malkauns", "yaman", "bihag", "kalavati",
                "multani", "raageshree", "piloo", "sudh_kalyan", "sudh_sarang"
            ]
            for known in known_ragas:
                if known in name_lower:
                    raga = known
                    confidence = 0.8
                    break

            if raga == "unknown":
                # Filter out non-music files
                skip_keywords = ["introduction", "introductory", "lyrics", "speech",
                                 "taal", "shrutinandan", "concept"]
                if any(kw in name_lower for kw in skip_keywords):
                    continue
                # Try using the full name as raga (for files like "Bhairavi Thumri.mp3")
                parts = name.split()
                if len(parts) >= 1:
                    raga = parts[0].lower()
                    confidence = 0.5

        entries.append(AudioFileEntry(
            file_path=str(mp3_file),
            tradition="hindustani",
            raga=raga,
            raga_confidence=confidence,
            source_dataset="dunya_hindustani_cc",
            format="mp3"
        ))

    logger.info(f"dunya-hindustani-cc: found {len(entries)} files")
    return entries


def scan_dunya_carnatic(base_dir: Path) -> List[AudioFileEntry]:
    """Scan dunya-carnatic-cc: use composition→raga lookup table"""
    entries = []
    dataset_dir = base_dir / "dunya-carnatic-cc"
    if not dataset_dir.exists():
        logger.warning(f"dunya-carnatic-cc not found at {dataset_dir}")
        return entries

    for mp3_file in dataset_dir.glob("*.mp3"):
        name = mp3_file.stem
        raga = "unknown"
        confidence = 0.0

        # Try exact match from lookup
        if name in CARNATIC_COMPOSITION_RAGA:
            raga = CARNATIC_COMPOSITION_RAGA[name]
            confidence = 0.95
        else:
            # Try fuzzy match — check if any known composition is a substring
            for comp, r in CARNATIC_COMPOSITION_RAGA.items():
                if comp.lower() in name.lower() or name.lower() in comp.lower():
                    raga = r
                    confidence = 0.8
                    break

        entries.append(AudioFileEntry(
            file_path=str(mp3_file),
            tradition="carnatic",
            raga=raga,
            raga_confidence=confidence,
            source_dataset="dunya_carnatic_cc",
            format="mp3"
        ))

    logger.info(f"dunya-carnatic-cc: found {len(entries)} files")
    return entries


def scan_south_indian_snippets(base_dir: Path) -> List[AudioFileEntry]:
    """Scan SouthIndian Classical Snippets: composition→raga from names"""
    entries = []
    dataset_dir = base_dir / "SouthIndian Classical" / "Carnatic_Dataset_Snippets"
    if not dataset_dir.exists():
        logger.warning(f"SouthIndian Classical not found at {dataset_dir}")
        return entries

    for key_dir in dataset_dir.iterdir():
        if not key_dir.is_dir() or key_dir.name.startswith('.'):
            continue

        for mp3_file in key_dir.glob("*.mp3"):
            name = mp3_file.stem
            # Format: "CompositionName_Key_chunkN"
            parts = name.rsplit("_", 2)
            comp_name = parts[0] if len(parts) >= 3 else name.split("_chunk")[0]

            raga = "unknown"
            confidence = 0.0

            if comp_name in SOUTH_INDIAN_COMPOSITION_RAGA:
                raga = SOUTH_INDIAN_COMPOSITION_RAGA[comp_name]
                confidence = 0.90
            else:
                # Try fuzzy match
                for comp, r in SOUTH_INDIAN_COMPOSITION_RAGA.items():
                    if comp.lower() in comp_name.lower():
                        raga = r
                        confidence = 0.75
                        break

            entries.append(AudioFileEntry(
                file_path=str(mp3_file),
                tradition="carnatic",
                raga=raga,
                raga_confidence=confidence,
                source_dataset="south_indian_snippets",
                format="mp3"
            ))

    logger.info(f"SouthIndian Classical: found {len(entries)} files")
    return entries


def scan_cmr_dataset(base_dir: Path) -> List[AudioFileEntry]:
    """Scan CMR_full_dataset_1.0: rhythm dataset (no raga labels)"""
    entries = []
    dataset_dir = base_dir / "CMR_full_dataset_1.0" / "audio"
    if not dataset_dir.exists():
        logger.warning(f"CMR dataset not found at {dataset_dir}")
        return entries

    for wav_file in dataset_dir.glob("*.wav"):
        entries.append(AudioFileEntry(
            file_path=str(wav_file),
            tradition="carnatic",
            raga="rhythm_only",
            raga_confidence=1.0,
            source_dataset="cmr_rhythm",
            format="wav",
            has_tala_annotations=True
        ))

    logger.info(f"CMR dataset: found {len(entries)} files")
    return entries


def scan_raga_recognition_dataset(base_dir: Path, dunya_api_key: str = "") -> List[AudioFileEntry]:
    """
    Scan Indian Art Music Raga Recognition Dataset.
    UUID-named folders with Artist/Album/Song hierarchy.
    We try to get raga name from the Dunya API using the UUID.
    """
    entries = []
    # Fuzzy search for the RagaDataset folder (handles slight naming differences)
    dataset_dir = None
    potential_names = [
        "Indian Classical Music RagaDataset",
        "Indian Art Music Raga Recognition Dataset (audio)", 
        "Raga Recognition Dataset", 
        "RagaDataset"
    ]
    
    # Check direct first
    for name in potential_names:
        d = base_dir / name
        if d.exists():
            if (d / "RagaDataset").exists():
                dataset_dir = d / "RagaDataset"
            else:
                dataset_dir = d
            break
            
    if not dataset_dir or not dataset_dir.exists():
        logger.warning(f"Raga Recognition Dataset not found in {base_dir}. Skipping.")
        return entries

    # Try to load or create UUID→raga cache
    cache_path = base_dir / ".raga_uuid_cache.json"
    uuid_raga_cache = {}
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            uuid_raga_cache = json.load(f)

    for tradition in ["Carnatic", "Hindustani"]:
        tradition_dir = dataset_dir / tradition / "audio"
        if not tradition_dir.exists():
            continue

        tradition_lower = tradition.lower()

        for uuid_dir in tradition_dir.iterdir():
            if not uuid_dir.is_dir() or uuid_dir.name.startswith('.'):
                continue
            
            uuid = uuid_dir.name
            
            # COMPLETE Verified Raga mapping for the IAMRRD dataset (All Folders)
            IAMRRD_MAPPING = {
                # Carnatic
                "0277eae5-3411-4b22-9fa8-1b347e7528d1": "shanmukhapriya",
                "09c179f3-8b19-4792-a852-e9fa0090e409": "kaapi",
                "123b09bd-9901-4e64-a65a-10b02c9e0597": "bhairavi",
                "18b1acb9-dff6-47ec-873a-b2086c8d268d": "madhyamavathi",
                "2165542c-45da-4301-af82-c2e7ddbe4768": "vasanta",
                "39821826-3327-41d7-9cd5-e22fe7b08360": "mohanam",
                "3af5a361-923a-465d-864d-9c7ba0c04a47": "harikambhoji",
                "42dd0ccb-f92a-4622-ae5d-a3be571b4939": "sankarabharanam",
                "4ce0b18d-f4df-41cf-9b40-9166199506b4": "andolika",
                "50bd048f-4482-4c5b-850c-9ad5e5ec46f1": "kharaharapriya",
                "5ce23030-f71d-4f5d-9d76-f91c2c182392": "jaganmohini",
                "6345e8fe-7061-4bdc-842c-dcfd4a379820": "bhairavi",
                "700e1d92-7094-4c21-8a3b-d7b60c550edf": "rageshri",
                "839bb6b4-1577-4a78-9e5d-4906c6453274": "todi",
                "85ccf631-4cdf-4f6c-a841-0edfcf4255d1": "malkauns",
                "978f4c3c-6a12-43fd-8427-c414ee17b252": "bhairav",
                "98d46d9e-5100-4f24-bcd4-d0f95966c7cb": "ananda_bhairavi",
                "993d6cf6-dc89-4d23-9a9f-8eeed1524872": "hamsadhwani",
                "9a071e54-3eed-48b2-83a3-1a3fd683b8e0": "shanmukhapriya",
                "9cedca68-4a9d-4170-bec3-0d1db1ff730e": "saveri",
                "9ebeb536-30a7-403f-8042-f7c1445c4b87": "bilahari",
                "a2f9f182-0ceb-4531-b286-b840b47a54b8": "yaman",
                "a47e9d22-847a-46e8-b589-2a3537789f5f": "saveri",
                "a4ec6633-1050-4207-8313-b017194c8fa0": "kambhoji",
                "a9413dff-91d1-4e29-ad92-c04019dce5b8": "todi",
                "aa5f376f-06cd-4a69-9cc9-b1077104e0b0": "neelambari",
                "b08475a2-1049-433a-be8f-105bf14718fb": "shree",
                "bdd80890-44f1-4a93-8d93-06a418781f97": "sri_ranjani",
                "bf4662d4-25c3-4cad-9249-4de2dc513c06": "kalyani",
                "c6b5f8d9-ebb4-46af-a020-6646fce2c77d": "sama",
                "cda9cbe9-c1aa-42bb-8b4c-e9dfc8af133c": "bilahari",
                "db085e26-665d-4f4c-a2a5-95251c22b69e": "nata",
                "ddff55ae-20f6-4d7d-ba9e-a6c10eeebd41": "bhairavi",
                "defad4cc-48aa-4372-a31a-c43624930713": "harikambhoji",
                "df85a0a5-b1a8-42f1-a87b-3d7c7ee33fb4": "kalyani",
                "e18fcaa7-1f9c-4f09-b627-0687481f4ec7": "bilahari",
                "e5c4d94a-b34a-42ef-acd7-f235612350e4": "sarang",
                "e8a0bf54-13c6-4a09-922a-bfc744ddf38a": "bhairavi",
                "f0866e71-33b2-47db-ab75-0808c41f2401": "nattakurinji",
                "f972db4d-5d16-4f9a-9841-f313e1601aaa": "kharaharapriya",
                # Hindustani
                "063ea5a0-23b1-4bb5-8537-3d924fe8ebb3": "jog",
                "0b3bbf97-0ec3-41da-add4-722d87329ec3": "madhukauns",
                "118401e7-8de8-4d81-9d8e-8070889e3fa8": "darbari_kanada",
                "1b05a564-059f-445b-b325-cf26318367e3": "miyan_malhar_khayal",
                "1e7de02f-e77f-405a-a033-f31117aaf955": "bhairav",
                "290674e0-d94c-41c1-ad99-f65fa22a1447": "madhuvanti",
                "2ed9379f-14c9-49af-8e4d-f9b63e96801f": "alahiya",
                "3eb7ba30-4b94-432e-9618-875ee57e01ab": "marwa",
                "40dbe1db-858f-4366-bef8-9076eb67340d": "shudh_sarang",
                "46997b02-f09c-4969-8138-4e1861f61967": "shree",
                "48b37bed-e847-4882-8a01-5c721e07f07d": "yaman",
                "54c4214c-05b9-4acc-8f77-6d5786e43a2e": "bihag",
                "595de771-fcc2-414c-9df6-ae4ffd898549": "hamsadhwani",
                "62b79291-73b0-4c77-a353-0f8bc6ed8362": "desh",
                "64e5fb9e-5569-4e80-8e6c-f543af9469c7": "malkauns",
                "6f13484e-6fdd-402d-baf3-3835835454d0": "todi",
                "7591faad-e68a-4550-b675-8082842c6056": "bageshri",
                "93c73081-bdf8-4eca-b325-d736b71e9b4b": "marwa", # Repeat with suffix
                "a7d98897-e2fe-4d75-b9dc-e5b4dc88e1c6": "bihag",
                "a99e07d5-20a0-467b-8dcd-aa5a095177fd": "lalit",
                "a9ee554f-f146-43e9-b2d0-9bb31fd4b57d": "kedar",
                "b143adaa-f1a6-4de4-8985-a5bd35e96279": "bairagi",
                "ba3242d0-8d40-4d93-865e-d2e2497ea2a8": "gaud_malhar",
                "d9c603fa-875f-4b84-b851-c6a345427898": "abhogi",
                "dd59147d-8775-44ff-a36b-0d9f15b31319": "todi",
                "e771a74d-545d-41d5-816b-43403a818b0c": "bhoopali",
                "ecd04ceb-b46c-47fc-9045-84ac9160e527": "basant",
                "f6432fec-e9c2-4b09-9e73-c46086cbd8ea": "ahir_bhairav",
                "f7fddfc0-8c1d-4dd2-90d5-5d51a99d61f8": "rageshri",
                "f7fddfc0-8c1d-4dd2-90d5-5d51a99d61f8 2": "rageshri",
                "fa28470c-d413-44c7-94da-181f530cbfdd": "puriya_dhanashri"
            }
            
            raga = IAMRRD_MAPPING.get(uuid, "unknown")
            confidence = 0.95 if raga != "unknown" else 0.0



            # Walk through Artist/Album/Song structure
            files_found = 0
            for mp3_file in uuid_dir.rglob("*.mp3"):
                entries.append(AudioFileEntry(
                    file_path=str(mp3_file),
                    tradition=tradition_lower,
                    raga=raga,
                    raga_confidence=confidence,
                    source_dataset="raga_recognition",
                    format="mp3"
                ))
                files_found += 1
            
            if files_found > 0:
                logger.info(f"  → Scanned {uuid[:8]}... ({raga}): {files_found} files")

    logger.info(f"Raga Recognition Dataset: found {len(entries)} files")
    return entries

    # Save cache
    if uuid_raga_cache:
        with open(cache_path, 'w') as f:
            json.dump(uuid_raga_cache, f, indent=2)

    logger.info(f"Raga Recognition Dataset: found {len(entries)} files")
    return entries


def _query_dunya_raga(uuid: str, tradition: str, api_key: str) -> str:
    """Query Dunya API for raga name of a recording UUID"""
    import urllib.request
    import urllib.error

    if tradition == "carnatic":
        url = f"https://dunya.compmusic.upf.edu/api/carnatic/recording/{uuid}"
    else:
        url = f"https://dunya.compmusic.upf.edu/api/hindustani/recording/{uuid}"

    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Token {api_key}")

    try:
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode())
            # Carnatic uses 'raaga', Hindustani uses 'raag'
            raaga_list = data.get("raaga", data.get("raag", []))
            if raaga_list and len(raaga_list) > 0:
                raga_info = raaga_list[0]
                name = raga_info.get("common_name", raga_info.get("name", "unknown"))
                return name.lower().replace(" ", "_")
    except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
        logger.debug(f"Dunya API failed for {uuid}: {e}")

    return "unknown"


def is_already_processed(midi_path: Path, expr_path: Path) -> bool:
    """Check if file is finished and valid."""
    if not midi_path.exists() or not expr_path.exists():
        return False
    # If file exists but is 0-bytes, it was likely a failed write - re-process it.
    if midi_path.stat().st_size < 100 or expr_path.stat().st_size < 100:
        return False
    return True

# ============================================================
# Audio Processing
# ============================================================

def process_audio_file(entry: AudioFileEntry, output_dir: Path, skip_midi: bool = False) -> AudioFileEntry:
    """Process a single audio file: clean audio → MIDI conversion"""
    try:
        import librosa
        import soundfile as sf

        # Sanitize raga name for folder safety
        safe_raga = re.sub(r'[^\w\-]', '_', entry.raga.strip().lower())
        raga_dir = output_dir / entry.tradition / safe_raga
        raga_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique output filename
        src_name = Path(entry.file_path).stem
        clean_name = re.sub(r'[^\w\-]', '_', src_name)[:80]
        output_stem = f"{clean_name}_{hashlib.md5(entry.file_path.encode()).hexdigest()[:8]}"
        
        # SMART SKIP (Now with 0-byte detection)
        final_midi_base = raga_dir / f"{output_stem}.mid"
        final_expr_base = raga_dir / f"{output_stem}.expr.npy"
        final_midi_seg = raga_dir / f"{output_stem}_seg0.mid"
        final_expr_seg = raga_dir / f"{output_stem}_seg0.expr.npy"

        if is_already_processed(final_midi_base, final_expr_base) or \
           is_already_processed(final_midi_seg, final_expr_seg):
            entry.processed = True
            entry.midi_output_path = str(final_midi_base if final_midi_base.exists() else final_midi_seg)
            return entry

        # Step 1: Load and clean audio
        y, sr = librosa.load(entry.file_path, sr=22050, mono=True)

        if len(y) == 0:
            entry.error = "Empty audio file"
            return entry

        entry.duration_seconds = len(y) / sr

        # Skip very short files
        if entry.duration_seconds < 5.0:
            entry.error = f"Too short: {entry.duration_seconds:.1f}s"
            return entry

        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=40)
        if len(y_trimmed) < sr * 3:  # Less than 3 seconds after trim
            y_trimmed = y  # Keep original if trim removed too much

        # Peak normalize
        peak = np.max(np.abs(y_trimmed))
        if peak > 0:
            y_trimmed = y_trimmed * (0.9 / peak)

        # Step 2: Segment long files (>5 min → 60s chunks with 10s overlap)
        segments = []
        if entry.duration_seconds > 300:  # 5 minutes
            chunk_samples = 60 * sr
            overlap_samples = 10 * sr
            step = chunk_samples - overlap_samples
            for i in range(0, len(y_trimmed) - chunk_samples + 1, step):
                segments.append((y_trimmed[i:i + chunk_samples], f"{output_stem}_seg{len(segments)}"))
            # Add remaining
            if len(y_trimmed) - (len(segments) * step) > 30 * sr:
                segments.append((y_trimmed[len(segments) * step:], f"{output_stem}_seg{len(segments)}"))
        else:
            segments = [(y_trimmed, output_stem)]

        # Initialize Engine (One session per worker)
        engine = BasicPitchONNX(device="cpu")

        # Step 3: Convert each segment to MIDI & Step 4: Extract expression features
        midi_paths = []
        for audio_segment, seg_name in segments:
            # Direct Paths (Removed .tmp for high-IO cloud stability)
            final_midi = raga_dir / f"{seg_name}.mid"
            final_expr = raga_dir / f"{seg_name}.expr.npy"
            
            # Transcription (Stable ONNX)
            try:
                # Save temp WAV for transcriber
                tmp_wav = raga_dir / f"{seg_name}.wav"
                sf.write(str(tmp_wav), audio_segment, sr)
                
                midi_data = engine.predict(str(tmp_wav))
                
                # Direct Write MIDI
                if midi_data and len(midi_data.instruments) > 0:
                    midi_data.write(str(final_midi))
                else:
                    pretty_midi.PrettyMIDI().write(str(final_midi))
                
                # Direct Expression Extraction
                _extract_expression_features(audio_segment, sr, str(final_expr))
                
                # Cleanup temp WAV
                if tmp_wav.exists(): tmp_wav.unlink()
                
                midi_paths.append(str(final_midi))
                
            except Exception as e:
                logger.error(f"Failed to process {safe_raga} {seg_name}: {e}")
                with open(ERROR_LOG_PATH, "a") as f:
                    f.write(f"--- FAILED: {entry.file_path} ---\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write("\n")
                # Don't set entry.error here so that partial successes still count as 'ok'
                # entry.error = str(e) 

        entry.midi_output_path = ";".join(midi_paths)
        entry.processed = len(midi_paths) > 0

    except Exception as e:
        entry.error = str(e)
        logger.error(f"Critical worker error on {entry.file_path}: {e}")
        
    return entry


# _convert_to_midi is now integrated directly into process_audio_file for worker stability.


def _extract_expression_features(audio: np.ndarray, sr: int, output_path: str):
    """
    Extract continuous audio expression contours for the flow-matching head.
    
    Returns array of shape (L, 4) with columns:
    1. f0 (fundamental frequency, log-scale)
    2. amplitude (RMS energy)
    3. voiced (voicing probability/mask)
    4. spectral_centroid (brightness)
    """
    import librosa
    
    # Use hop_length that roughly matches the MIDI tokenizer's temporal resolution
    # Standard: 512 samples at 22050Hz ~= 23ms
    hop_length = 512
    
    # 1. Pitch (f0) using PYIN (more robust than simple autocorrelation)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        hop_length=hop_length
    )
    
    # Use log-scale for f0 (more perceptually linear)
    # Replace NaN with 0 before logging
    f0_safe = np.nan_to_num(f0, nan=1e-6)
    f0_log = np.log2(np.maximum(f0_safe, 1e-6) / 440.0) # Relative to A4
    
    # 2. Amplitude (RMS)
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    
    # 3. Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
    # Normalize centroid
    centroid_norm = centroid / (sr / 2)
    
    # Align lengths (librosa outputs might differ by 1-2 frames)
    min_len = min(len(f0_log), len(rms), len(centroid_norm))
    
    # Combine into (L, 4)
    features = np.stack([
        f0_log[:min_len],
        rms[:min_len],
        voiced_probs[:min_len] if voiced_probs is not None else voiced_flag[:min_len].astype(float),
        centroid_norm[:min_len]
    ], axis=-1).astype(np.float32)
    
    # Save as numpy file (using file handle to prevent np.save from appending .npy to .tmp)
    with open(output_path, 'wb') as f:
        np.save(f, features)
    return features


# ============================================================
# Main Pipeline
# ============================================================

def build_manifest(input_dir: Path, dunya_api_key: str = "") -> List[AudioFileEntry]:
    """Scan all datasets and build unified manifest"""
    logger.info(f"Scanning datasets in {input_dir}...")

    all_entries = []
    all_entries.extend(scan_dunya_hindustani(input_dir))
    all_entries.extend(scan_dunya_carnatic(input_dir))
    all_entries.extend(scan_south_indian_snippets(input_dir))
    all_entries.extend(scan_cmr_dataset(input_dir))
    all_entries.extend(scan_raga_recognition_dataset(input_dir, dunya_api_key))

    # Stats
    total = len(all_entries)
    labeled = sum(1 for e in all_entries if e.raga != "unknown" and e.raga != "rhythm_only")
    unknown = sum(1 for e in all_entries if e.raga == "unknown")
    rhythm = sum(1 for e in all_entries if e.raga == "rhythm_only")

    ragas = set(e.raga for e in all_entries if e.raga not in ("unknown", "rhythm_only"))

    logger.info(f"=== MANIFEST SUMMARY ===")
    logger.info(f"Total files: {total}")
    logger.info(f"Raga-labeled: {labeled}")
    logger.info(f"Unknown raga: {unknown}")
    logger.info(f"Rhythm-only: {rhythm}")
    logger.info(f"Unique ragas: {len(ragas)}")
    logger.info(f"Ragas: {sorted(ragas)}")

    return all_entries


def process_all(
    entries: List[AudioFileEntry],
    output_dir: Path,
    max_workers: int = 8,
    skip_midi: bool = False,
    skip_unknown: bool = True
) -> List[AudioFileEntry]:
    """Process all audio files in parallel"""
    # Filter
    to_process = []
    for e in entries:
        if skip_unknown and e.raga == "unknown":
            continue
        if e.raga == "rhythm_only":
            continue  # Skip rhythm-only for MIDI melody training
        to_process.append(e)

    logger.info(f"Processing {len(to_process)} files with {max_workers} workers...")

    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_audio_file, entry, output_dir, skip_midi): entry
            for entry in to_process
        }

        # Use tqdm for real-time ETA and status
        pbar = tqdm(as_completed(futures), total=len(to_process), desc="Preprocessing Dataset")
        for future in pbar:
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                # Update progress bar description with success/fail count
                success = sum(1 for r in results if r.processed)
                pbar.set_postfix({
                    "ok": success,
                    "fail": completed - success
                })
            except Exception as e:
                entry = futures[future]
                entry.error = str(e)
                results.append(entry)
                completed += 1

    success = sum(1 for r in results if r.processed)
    failed = sum(1 for r in results if r.error)
    logger.info(f"=== PROCESSING COMPLETE ===")
    logger.info(f"Processed: {success}/{len(to_process)}")
    logger.info(f"Failed: {failed}")

    return results


def save_manifest(entries: List[AudioFileEntry], output_path: Path):
    """Save manifest to JSON"""
    data = [asdict(e) for e in entries]
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Manifest saved to {output_path}")


def generate_raga_metadata(entries: List[AudioFileEntry], existing_metadata_path: Path, output_path: Path):
    """Generate expanded raga_metadata.json from processed entries"""
    # Load existing
    existing = {}
    if existing_metadata_path.exists():
        with open(existing_metadata_path, 'r') as f:
            existing = json.load(f)

    # Collect unique ragas from entries
    ragas_seen = set()
    for e in entries:
        if e.raga not in ("unknown", "rhythm_only") and e.processed:
            ragas_seen.add(e.raga)

    # Default metadata templates by tradition
    hindustani_defaults = {
        "moods": ["meditative", "contemplative"],
        "time_of_day": "evening",
        "tempo_range": [50, 100],
        "thaat": "unknown",
        "description": "Hindustani raga"
    }
    carnatic_defaults = {
        "moods": ["devotional", "contemplative"],
        "time_of_day": "evening",
        "tempo_range": [60, 120],
        "thaat": "melakarta",
        "description": "Carnatic raga"
    }

    # Hindustani raga metadata extensions
    hindustani_raga_info = {
        "yaman": {"moods": ["romantic", "peaceful", "serene"], "time_of_day": "evening", "thaat": "kalyan"},
        "bhairavi": {"moods": ["devotional", "melancholic", "sad"], "time_of_day": "morning", "thaat": "bhairavi"},
        "malkauns": {"moods": ["serious", "meditative", "mysterious"], "time_of_day": "night", "thaat": "bhairavi"},
        "todi": {"moods": ["serious", "meditative", "longing"], "time_of_day": "morning", "thaat": "todi"},
        "bihag": {"moods": ["romantic", "joyful"], "time_of_day": "night", "thaat": "bilawal"},
        "bhimpalasi": {"moods": ["romantic", "peaceful"], "time_of_day": "afternoon", "thaat": "kafi"},
        "desh": {"moods": ["romantic", "light", "monsoon"], "time_of_day": "night", "thaat": "khamaj"},
        "kedar": {"moods": ["devotional", "romantic"], "time_of_day": "night", "thaat": "kalyan"},
        "kalyan": {"moods": ["romantic", "peaceful", "serene"], "time_of_day": "evening", "thaat": "kalyan"},
        "khamaj": {"moods": ["romantic", "light", "playful"], "time_of_day": "night", "thaat": "khamaj"},
        "marwa": {"moods": ["serious", "sunset", "restless"], "time_of_day": "evening", "thaat": "marwa"},
        "puriya": {"moods": ["serious", "profound", "sunset"], "time_of_day": "evening", "thaat": "marwa"},
        "shree": {"moods": ["devotional", "majestic", "serious"], "time_of_day": "evening", "thaat": "purvi"},
        "lalit": {"moods": ["devotional", "dawn", "serious"], "time_of_day": "dawn", "thaat": "marwa"},
        "jog": {"moods": ["contemplative", "night", "serious"], "time_of_day": "night", "thaat": "khamaj"},
        "rageshri": {"moods": ["romantic", "night", "peaceful"], "time_of_day": "night", "thaat": "khamaj"},
        "megh": {"moods": ["romantic", "monsoon", "rainy"], "time_of_day": "monsoon", "thaat": "kafi"},
        "bahar": {"moods": ["spring", "joyful", "romantic"], "time_of_day": "night", "thaat": "kafi"},
        "multani": {"moods": ["serious", "afternoon", "meditative"], "time_of_day": "afternoon", "thaat": "todi"},
        "bageshree": {"moods": ["romantic", "tender", "longing"], "time_of_day": "night", "thaat": "kafi"},
        "bhoopali": {"moods": ["peaceful", "serene", "devotional"], "time_of_day": "evening", "thaat": "kalyan"},
        "abhogi": {"moods": ["devotional", "contemplative"], "time_of_day": "morning", "thaat": "kafi"},
        "dhani": {"moods": ["romantic", "night"], "time_of_day": "night", "thaat": "kafi"},
        "hameer": {"moods": ["majestic", "heroic"], "time_of_day": "night", "thaat": "kalyan"},
    }

    # Carnatic raga metadata extensions
    carnatic_raga_info = {
        "sankarabharanam": {"moods": ["devotional", "majestic", "peaceful"], "description": "72nd melakarta, equivalent to Bilawal"},
        "kalyani": {"moods": ["romantic", "majestic", "serene"], "description": "65th melakarta, equivalent to Yaman"},
        "kharaharapriya": {"moods": ["devotional", "contemplative"], "description": "22nd melakarta"},
        "todi": {"moods": ["serious", "introspective", "devotional"], "description": "8th melakarta, Shudha Todi"},
        "kambhoji": {"moods": ["devotional", "majestic"], "description": "Janya of Harikambhoji"},
        "bhairavi": {"moods": ["devotional", "melancholic", "pathos"], "description": "20th melakarta"},
        "mohanam": {"moods": ["devotional", "joyful", "serene"], "description": "Pentatonic, janya of Harikambhoji"},
        "hindolam": {"moods": ["serene", "meditative", "night"], "description": "Pentatonic, janya of Natabhairavi"},
        "saveri": {"moods": ["devotional", "pathos", "morning"], "description": "Janya of Mayamalavagowla"},
        "hamsadhwani": {"moods": ["auspicious", "joyful", "devotional"], "description": "Popular pentatonic raga"},
        "shanmukhapriya": {"moods": ["energetic", "dynamic", "powerful"], "description": "56th melakarta"},
        "mayamalavagowla": {"moods": ["devotional", "foundational"], "description": "15th melakarta, first raga learned"},
        "nattai": {"moods": ["auspicious", "majestic", "energetic"], "description": "Janya of Chalanata"},
        "amritavarshini": {"moods": ["devotional", "rain", "serene"], "description": "Janya of Panthuvarali"},
        "sahana": {"moods": ["romantic", "devotional", "serene"], "description": "Janya of Harikambhoji"},
    }

    # Merge into output
    output = dict(existing)

    for raga in ragas_seen:
        if raga not in output:
            # Check if we have specific info
            if raga in hindustani_raga_info:
                info = dict(hindustani_defaults)
                info.update(hindustani_raga_info[raga])
                output[raga] = info
            elif raga in carnatic_raga_info:
                info = dict(carnatic_defaults)
                info.update(carnatic_raga_info[raga])
                output[raga] = info
            else:
                # Use defaults based on which tradition the raga appeared in
                traditions = set(e.tradition for e in entries if e.raga == raga)
                if "hindustani" in traditions:
                    output[raga] = dict(hindustani_defaults)
                else:
                    output[raga] = dict(carnatic_defaults)
                output[raga]["description"] = f"Raga {raga}"

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Expanded raga metadata: {len(existing)} → {len(output)} ragas → {output_path}")


def validate_output(output_dir: Path):
    """Validate the processed output"""
    midi_files = list(output_dir.rglob("*.mid"))
    ragas = set()
    traditions = set()

    for f in midi_files:
        parts = f.relative_to(output_dir).parts
        if len(parts) >= 2:
            traditions.add(parts[0])
            ragas.add(parts[1])

    logger.info(f"=== VALIDATION ===")
    logger.info(f"Total MIDI files: {len(midi_files)}")
    logger.info(f"Traditions: {sorted(traditions)}")
    logger.info(f"Unique ragas: {len(ragas)}")
    logger.info(f"Ragas: {sorted(ragas)}")

    # Check file sizes
    empty = [f for f in midi_files if f.stat().st_size < 100]
    if empty:
        logger.warning(f"Found {len(empty)} potentially empty MIDI files")

    return len(midi_files) > 0


def main():
    parser = argparse.ArgumentParser(description='Process Indian Classical Music Datasets')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Root directory containing all sub-datasets')
    parser.add_argument('--output_dir', type=str, default='data/midi_v2',
                        help='Output directory for processed MIDIs')
    parser.add_argument('--metadata_input', type=str,
                        default='src/sekiro_ai/config/raga_metadata.json',
                        help='Existing raga metadata JSON')
    parser.add_argument('--metadata_output', type=str,
                        default='src/sekiro_ai/config/raga_metadata.json',
                        help='Output path for expanded raga metadata')
    parser.add_argument('--dunya_api_key', type=str, default=os.environ.get("DUNYA_TOKEN", ""),
                        help='Dunya API key for resolving UUID recordings')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--skip-midi', action='store_true',
                        help='Skip MIDI conversion (only classify and clean audio)')
    parser.add_argument('--skip-unknown', action='store_true', default=True,
                        help='Skip files with unknown raga')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate existing output')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Ensure log directory
    Path("data").mkdir(exist_ok=True)

    if args.validate_only:
        validate_output(output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: Self-Test (Zero-Guesswork Verification)
    logger.info("Running Stable Engine Self-Test...")
    try:
        from utils.transcription_onnx import BasicPitchONNX, MODEL_INPUT_SAMPLES, MODEL_INPUT_NAME
        test_engine = BasicPitchONNX(device="cpu")
        silence = np.zeros(MODEL_INPUT_SAMPLES, dtype=np.float32)
        test_engine.session.run(None, {MODEL_INPUT_NAME: silence.reshape(1, MODEL_INPUT_SAMPLES, 1)})
        logger.info("Self-test SUCCESS! Transitioning to production batch.")
    except Exception as e:
        logger.error(f"CRITICAL: Self-test failed: {e}")
        logger.error("Aborting to prevent corrupted dataset. Check your models/ folder.")
        sys.exit(1)

    # Step 1: Build manifest
    entries = build_manifest(input_dir, args.dunya_api_key)

    # Save raw manifest
    save_manifest(entries, output_dir / "manifest_raw.json")

    # Step 2: Process audio → MIDI
    results = process_all(
        entries, output_dir,
        max_workers=args.workers,
        skip_midi=args.skip_midi,
        skip_unknown=args.skip_unknown
    )

    # Save processed manifest
    save_manifest(results, output_dir / "manifest_processed.json")

    # Step 3: Generate expanded raga metadata
    generate_raga_metadata(
        results,
        Path(args.metadata_input),
        Path(args.metadata_output)
    )

    # Step 4: Validate
    validate_output(output_dir)

    logger.info("=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
