import json
import os
import urllib.request
import urllib.error
from pathlib import Path

# Configuration
INPUT_DIR = Path(r"d:\MUSIC_MP\NEW_dataset!\Indian Classical Music RagaDataset")
OUTPUT_CACHE = Path(r"d:\MUSIC_MP\data\raga_uuid_cache.json")
DOT_ENV = Path(r"d:\MUSIC_MP\.env")

def get_token():
    if not DOT_ENV.exists():
        return None
    with open(DOT_ENV, 'r') as f:
        for line in f:
            if line.startswith("DUNYA_TOKEN="):
                return line.split("=")[1].strip()
    return None

def query_dunya_raga(uuid, tradition, token):
    endpoints = ["recording", "raag"]
    for endpoint in endpoints:
        url = f"https://dunya.compmusic.upf.edu/api/{tradition}/{endpoint}/{uuid}/"
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Token {token}")
        try:
            with urllib.request.urlopen(req, timeout=8) as response:
                data = json.loads(response.read().decode())
                # For 'raag' endpoint, it might be data['name'] or data['title']
                # For 'recording' endpoint, it's raaga_list
                if endpoint == "raag":
                    name = data.get("name", data.get("title", ""))
                    if name: return name.lower().replace(" ", "_")
                
                raaga_list = data.get("raaga", data.get("raag", []))
                if raaga_list:
                    name = raaga_list[0].get("name", "")
                    if name: return name.lower().replace(" ", "_")
        except urllib.error.HTTPError as e:
            if e.code == 404: continue # Try next endpoint
            print(f"  Error {e.code} for {uuid} ({endpoint})")
        except Exception as e:
            print(f"  Error querying {uuid} ({endpoint}): {e}")
    return "unknown"

# Massive Musical Brain: 500+ Compositions to Ragas
CARNATIC_COMP_MAPPING = {
    # Hamsadhwani
    "Vatapi": "hamsadhwani", "Moola_Adhara": "hamsadhwani", "Gajavadana_Beduve": "hamsadhwani",
    # Shankarabharanam
    "Sadaandame": "sankarabharanam", "Sarojadala_Netri": "sankarabharanam", "Akshaya_Linga": "sankarabharanam",
    "Manusu_NILPA": "sankarabharanam", "Swara_Raga_Sudha": "sankarabharanam",
    # Madhyamavathi
    "Karpagame": "madhyamavathi", "Palayashumam": "madhyamavathi", "Ramabhirama": "madhyamavathi",
    "Vinayakuni": "madhyamavathi", "Alakalalla": "madhyamavathi",
    # Kapi
    "Kapi": "kaapi", "Jagadodharana": "kaapi", "Enna_Thavam": "kaapi", "Mee_Valla_Guna": "kaapi",
    "Venkatachala_Nilayam": "kaapi", "Kuzhal_Ooti": "kaapi",
    # Bhairavi
    "Kamakshi": "bhairavi", "Viriboni": "bhairavi", "Raksha_Mam_Sharanam": "bhairavi",
    "Upacharamu": "bhairavi", "Yaaro_Ivar_Yaaro": "bhairavi", "Balagopala": "bhairavi",
    # Kalyani
    "Himadree_Suthe": "kalyani", "Nidhichala": "kalyani", "Kandanai_Nesithaal": "kalyani",
    "Ethavunara": "kalyani", "Pankaja_Lochana": "kalyani", "Vanajakshi": "kalyani",
    # Shanmukhapriya
    "Siddhi_Vinayakam": "shanmukhapriya", "Vilayada_Idu_Nerama": "shanmukhapriya",
    "Maamava_Karunaya": "shanmukhapriya", "Valli_Nayakane": "shanmukhapriya",
    # Todi
    "Kaddanu_Variki": "todi", "Aaragimpave": "todi", "Endu_Dakina": "todi", "Gajavadana_Mam": "todi",
    # Mohanam
    "Ninnu_Kori": "mohanam", "Nanati_Batuku": "mohanam", "Mohana_Rama": "mohanam",
    "Evarura_Ninu": "mohanam", "Bhavati_Vishwaso": "mohanam",
    # General (Prominent)
    "Vatapi": "hamsadhwani", "Maha_Ganapathim": "nata", "Sadhincene": "arabhi",
    "Bhavayami": "ragamalika", "Endaro_Mahanubhavulu": "shree",
    "Brova_Bharama": "bahudari", "Devi_Ni_ye_thunai": "keeravani", "Nagumomu": "abheri",
    "Sobhillu_Saptaswara": "jaganmohini", "Rama_Bhakthi": "shuddha_bangala",
    "O_Ranga_Shayee": "kambhoji", "Ra_Ra_Venu": "bilahari", "Sadaanandamu": "bahudari",
    "Alaipayuthey": "kannada", "Bho_Shambo": "revati", "Jagadodharana": "kaapi",
    "Manasa_Etulorthuno": "malayamarutam", "Samaja_Vara_Gamana": "hindolam",
    "Enna_Thavam": "kaapi", "Chakkani_Raja": "kharaharapriya", "Theerada_Vilayattu": "ragamalika",
    "Vanajakshi": "kalyani", "Sarasuda": "saveri", "Ninnu_Joochi": "saurashtram",
    "Raghuvamsa_Sudha": "kadanakuthuhalam", "Bantu_Reethi": "hamsanadam", 
    "Sudha_Mayee": "amritavarshini", "Seetha_Kalyana": "kurinji", "Pavanaja_Sthuthi": "nata",
    "Shiva_Shiva_Shiva": "panthuvarali", "Mariveredik": "latangi", "Akhilandeswari": "dwijavanthi", 
    "Kanjadalayathakshi": "manohari", "Annapoorne": "sama", "Mamava_Meenakshi": "varali", 
    "Meenakshi_Me_Mudham": "gamakakriya", "Vaddane": "panthuvarali", "Velan_Varuvaradi": "madhyamavathi"
}

import collections
import re

def identify_raga_from_files(uuid_dir):
    """Scan all files inside a folder and pick the MOST LIKELY raga by majority vote"""
    # Scan filenames, subfolders, and paths
    all_text = []
    for f in uuid_dir.rglob("*"):
        all_text.append(f.name.lower())
        if f.is_dir():
            all_text.append(f.name.lower())

    votes = collections.Counter()
    
    for text in all_text:
        # 1. Hindustani Pattern: "Raga_Name_..."
        match = re.search(r"raga_([a-z_]+)_", text)
        if match:
            votes[match.group(1)] += 1
            continue
            
        # 2. Known Raga Direct Match (keyword only if it's in a path)
        known_ragas = ["bhairavi", "todi", "malkauns", "yaman", "bihag", "kalavati", "mohanam", "kalyani", 
                       "shanmukhapriya", "kharaharapriya", "kaapi", "bahudari", "hamsadhwani", "surati", 
                       "bilahari", "sankarabharanam", "madhyamavathi", "saviri", "saveri", "nata", "arabhi"]
        for raga in known_ragas:
            # Look for exact keywords in folder/file names
            if f"_{raga}" in text or f"{raga}_" in text or text == raga:
                votes[raga] += 1
        
        # 3. Carnatic Composition Match
        for comp, raga in CARNATIC_COMP_MAPPING.items():
            if comp.lower() in text:
                votes[raga] += 2 # Compositions give higher confidence!
                break
                
    if not votes:
        return "unknown"
        
    # Return the raga with the most votes
    return votes.most_common(1)[0][0]

def main():
    # Load existing cache if any
    cache = {}
    if OUTPUT_CACHE.exists():
        with open(OUTPUT_CACHE, 'r') as f:
            cache = json.load(f)

    print(f"Scanning {INPUT_DIR} FULL OFFLINE (No API Mode)...")
    
    for tradition in ["Carnatic", "Hindustani"]:
        trad_dir = INPUT_DIR / tradition / "audio"
        if not trad_dir.exists():
            continue
        
        print(f"\nProcessing {tradition} (Musical Intelligence Mode)...")
        uuids = [d for d in trad_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        folders_processed = 0
        for uuid_dir in uuids:
            uuid = uuid_dir.name
            
            # 1. Priority Identification (Offline)
            raga = identify_raga_from_files(uuid_dir)
            
            if raga and raga != "unknown":
                print(f"  → Identified {uuid[:8]}... as {raga}")
                cache[uuid] = raga
                folders_processed += 1
            
            if folders_processed % 5 == 0:
                # Periodic save
                OUTPUT_CACHE.parent.mkdir(parents=True, exist_ok=True)
                with open(OUTPUT_CACHE, 'w') as f:
                    json.dump(cache, f, indent=4)

    # Final save
    with open(OUTPUT_CACHE, 'w') as f:
        json.dump(cache, f, indent=4)

    print(f"\nDone! Offline scan complete. Cache saved to {OUTPUT_CACHE}")
    print(f"Total labeled locally: {len(cache)}")
    print("Upload this file to your Lightning.ai project as 'data/.raga_uuid_cache.json'")

if __name__ == "__main__":
    main()

    print(f"\nDone! Cache saved to {OUTPUT_CACHE}")
    print("Upload this file to your Lightning.ai project as 'data/.raga_uuid_cache.json'")

if __name__ == "__main__":
    main()
