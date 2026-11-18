"""
fetch_data.py
-------------
Fetch electronic structure data (Structure, Band Structure, DOS) from the
Materials Project API and save it locally as JSON for offline analysis.

Supports querying by:
  - Material ID (e.g. "mp-149", "mp-2534", "mp-134")
  - Chemical Formula / Element (e.g. "Si", "GaAs", "Al", "Cu", "Ga", "MoS2")
"""

import os
import json
import argparse
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
from monty.json import MontyEncoder, MontyDecoder

RAW_DATA_DIR = os.path.join("data", "raw")

# Fast-path well-known ground state Materials Project IDs
COMMON_MATERIALS = {
    "si": "mp-149",       # Silicon (diamond)
    "ge": "mp-32",        # Germanium (diamond)
    "c": "mp-28",         # Carbon (diamond)
    "diamond": "mp-28",   # Carbon diamond
    "gaas": "mp-2534",    # Gallium Arsenide (zincblende)
    "gan": "mp-804",      # Gallium Nitride (wurtzite)
    "inp": "mp-969",      # Indium Phosphide
    "al": "mp-134",       # Aluminum (fcc)
    "cu": "mp-30",        # Copper (fcc)
    "fe": "mp-13",        # Iron (bcc)
    "au": "mp-81",        # Gold (fcc)
    "ag": "mp-124",       # Silver (fcc)
    "ga": "mp-142",       # Gallium (orthorhombic ground state)
    "ni": "mp-23",        # Nickel (fcc)
    "ti": "mp-72",        # Titanium (hcp)
    "pt": "mp-126",       # Platinum (fcc)
    "mg": "mp-153",       # Magnesium (hcp)
    "zn": "mp-79",        # Zinc (hcp)
    "mos2": "mp-2815",    # MoS2 (2H)
    "graphene": "mp-48",  # Graphite/Carbon
}


def get_api_key() -> Optional[str]:
    """Load the Materials Project API key from config.env, .env, or environment variables."""
    for env_file in ("config.env", ".env"):
        if os.path.exists(env_file):
            load_dotenv(dotenv_path=env_file)
            break
    api_key = os.environ.get("MP_API_KEY")
    return api_key


def get_cached_file_path(material_id: str, raw_dir: str = RAW_DATA_DIR) -> str:
    """Return standard cached JSON file path for a material ID."""
    clean_id = material_id.lower().strip()
    if not clean_id.startswith("mp-") and not clean_id.startswith("mvc-"):
        clean_id = f"mp-{clean_id}"
    return os.path.join(raw_dir, f"{clean_id}.json")


def is_cached(material_id: str, raw_dir: str = RAW_DATA_DIR) -> bool:
    """Check if raw data is already cached locally."""
    path = get_cached_file_path(material_id, raw_dir)
    return os.path.exists(path)


def list_cached_materials(raw_dir: str = RAW_DATA_DIR) -> List[Dict[str, Any]]:
    """List all materials currently cached in data/raw/."""
    os.makedirs(raw_dir, exist_ok=True)
    cached = []
    for fname in sorted(os.listdir(raw_dir)):
        if fname.endswith(".json") and not fname.endswith("_summary.json"):
            mp_id = fname[:-5]
            fpath = os.path.join(raw_dir, fname)
            try:
                data = load_raw(fpath)
                formula = data.get("structure").composition.reduced_formula if data.get("structure") else mp_id
                bs = data.get("bandstructure")
                is_metal = bs.is_metal() if bs else None
                gap = 0.0 if is_metal else (bs.get_band_gap()["energy"] if bs else None)
                cached.append({
                    "material_id": mp_id,
                    "formula": formula,
                    "is_metal": is_metal,
                    "band_gap": gap,
                    "file_path": fpath,
                })
            except Exception:
                cached.append({
                    "material_id": mp_id,
                    "formula": mp_id,
                    "file_path": fpath,
                })
    return cached


def load_raw(path: str) -> Dict[str, Any]:
    """Reload JSON saved with MontyEncoder back into pymatgen objects."""
    with open(path, "r") as f:
        data = json.load(f, cls=MontyDecoder)
    return data


def save_raw(data: Dict[str, Any], output_dir: str = RAW_DATA_DIR) -> str:
    """Save pymatgen objects to JSON using MontyEncoder."""
    os.makedirs(output_dir, exist_ok=True)
    mp_id = data.get("material_id", "material")
    out_path = os.path.join(output_dir, f"{mp_id}.json")
    with open(out_path, "w") as f:
        json.dump(data, f, cls=MontyEncoder, indent=2)
    return out_path


def search_materials_by_formula(formula: str, api_key: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search Materials Project for materials matching a chemical formula.
    Returns candidate summaries sorted by thermodynamic stability.
    """
    key = api_key or get_api_key()
    if not key:
        raise ValueError("MP_API_KEY is not configured. Please add it to config.env or .env.")

    from mp_api.client import MPRester

    with MPRester(key) as mpr:
        # Search by formula
        docs = mpr.materials.summary.search(
            formula=formula.strip(),
            fields=["material_id", "formula_pretty", "symmetry", "energy_above_hull", "is_metal", "band_gap"],
        )

        results = []
        for doc in docs:
            results.append({
                "material_id": str(doc.material_id),
                "formula": str(doc.formula_pretty),
                "crystal_system": str(doc.symmetry.crystal_system) if doc.symmetry else "Unknown",
                "spacegroup_symbol": str(doc.symmetry.symbol) if doc.symmetry else "Unknown",
                "energy_above_hull": float(doc.energy_above_hull) if doc.energy_above_hull is not None else 999.0,
                "is_metal": bool(doc.is_metal),
                "band_gap": float(doc.band_gap) if doc.band_gap is not None else 0.0,
            })
        
        # Sort by stability (lowest energy above hull first)
        results.sort(key=lambda x: x["energy_above_hull"])
        return results[:limit]


def fetch_material(material_id: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Fetch structure, band structure, and DOS for a specific Materials Project ID."""
    key = api_key or get_api_key()
    if not key:
        raise ValueError("MP_API_KEY is not configured. Please add it to config.env or .env.")

    from mp_api.client import MPRester

    clean_id = material_id.strip()
    with MPRester(key) as mpr:
        structure = mpr.get_structure_by_material_id(clean_id)
        
        bandstructure = None
        try:
            bandstructure = mpr.get_bandstructure_by_material_id(clean_id)
        except Exception as e:
            print(f"Notice: line bandstructure fetch returned: {e}")

        dos = None
        try:
            dos = mpr.get_dos_by_material_id(clean_id)
        except Exception as e:
            print(f"Notice: DOS fetch returned: {e}")

    if bandstructure is None:
        raise ValueError(f"Material {clean_id} does not have an electronic band structure available on Materials Project.")

    return {
        "material_id": clean_id,
        "structure": structure,
        "bandstructure": bandstructure,
        "dos": dos,
    }


def get_or_fetch(identifier: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Smart resolver:
      1. Check if identifier is already cached locally.
      2. Check fast-path lookup table for common elements/materials (e.g. Al -> mp-134, Ga -> mp-142).
      3. If direct MP ID, fetch from MP API.
      4. If formula, search candidate materials and fetch the most stable entry with electronic data.
    """
    clean_id = identifier.strip()
    norm_key = clean_id.lower().replace(" ", "")

    # 1. Check direct cached file
    direct_path = get_cached_file_path(clean_id)
    if os.path.exists(direct_path):
        print(f"Loading cached data from {direct_path}...")
        return load_raw(direct_path)

    # Check if a cached file matches the requested chemical formula
    for item in list_cached_materials():
        if item["formula"].lower() == norm_key or item["material_id"].lower() == norm_key:
            print(f"Found cached data for {clean_id} at {item['file_path']}...")
            return load_raw(item["file_path"])

    key = api_key or get_api_key()
    if not key:
        raise ValueError(
            f"Material '{identifier}' is not cached locally and no MP_API_KEY was found in config.env."
        )

    # 2. Check fast-path lookup dictionary
    if norm_key in COMMON_MATERIALS:
        target_mp_id = COMMON_MATERIALS[norm_key]
        # Check if fast-path ID is cached
        if is_cached(target_mp_id):
            return load_raw(get_cached_file_path(target_mp_id))
        try:
            print(f"Using known ground-state ID {target_mp_id} for '{clean_id}'...")
            data = fetch_material(target_mp_id, key)
            save_raw(data)
            return data
        except Exception as e:
            print(f"Fast-path fetch for {target_mp_id} failed ({e}); falling back to search...")

    # 3. Direct MP ID (e.g. mp-149, mp-134)
    if norm_key.startswith("mp-") or norm_key.startswith("mvc-") or norm_key.isdigit():
        target_id = norm_key if (norm_key.startswith("mp-") or norm_key.startswith("mvc-")) else f"mp-{norm_key}"
        print(f"Fetching {target_id} from Materials Project API...")
        data = fetch_material(target_id, key)
        save_raw(data)
        return data

    # 4. Search by formula across candidates
    print(f"Searching Materials Project for formula '{clean_id}'...")
    candidates = search_materials_by_formula(clean_id, key, limit=8)
    if not candidates:
        raise ValueError(f"No materials found matching formula '{clean_id}' on Materials Project.")

    # Try candidates in order of thermodynamic stability until one with bandstructure succeeds
    last_error = None
    for cand in candidates:
        target_id = cand["material_id"]
        try:
            print(f"Trying candidate {target_id} ({cand['formula']}, ΔE={cand['energy_above_hull']:.3f} eV/atom)...")
            data = fetch_material(target_id, key)
            save_raw(data)
            return data
        except Exception as err:
            last_error = err
            print(f"Candidate {target_id} lacked band structure: {err}. Trying next...")

    raise ValueError(f"Found {len(candidates)} structures for '{clean_id}', but none had available band structure data: {last_error}")
