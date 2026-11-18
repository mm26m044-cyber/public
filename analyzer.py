"""
analyzer.py
-----------
Unified Electronic Structure & Valley Physics Analyzer for metals and semiconductors.

Performs:
  1. Material Classification (Metal vs. Semiconductor/Insulator)
  2. Band Gap & Transition Analysis (Direct / Indirect, VBM, CBM)
  3. Conduction Band Valley Finding (Γ, L, Δ, X, K, etc.)
  4. Curvature & Effective Mass Extraction (m*/m0)
  5. Drude Carrier Mobility Estimation (μ = e·τ / m*)
  6. Density of States (DOS) at the Fermi Level N(E_F)
  7. Automated Persistent Storage of Output Data (JSON and CSV in data/processed/)

Usage:
    from analyzer import analyze_material
    results = analyze_material(data, save_output=True)
"""

import os
import csv
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import numpy as np

PROCESSED_DATA_DIR = os.path.join("data", "processed")

# Physical Constants
HBAR2_OVER_M0 = 7.61996       # ħ²/m0 in eV·Å²
E_CHARGE = 1.602176634e-19    # C
M0 = 9.1093837015e-31         # kg
DEFAULT_TAU_S = 0.2e-12       # 0.2 ps default scattering time


def calculate_dos_at_fermi(dos, efermi: float) -> float:
    """Calculate the total DOS value at the Fermi energy N(E_F) in states/eV."""
    if dos is None:
        return 0.0
    energies = np.array(dos.energies)
    densities = dos.get_densities()  # 1D array summed over spins
    # Interpolate at E_F
    dos_at_ef = float(np.interp(efermi, energies, densities))
    return max(0.0, dos_at_ef)


def extract_conduction_band(bs):
    """Return lowest conduction band array and VBM energy."""
    if bs.is_metal():
        return None, None
    vbm = bs.get_vbm()
    spin = next(s for s, bands in vbm["band_index"].items() if len(bands) > 0)
    cb_index = max(vbm["band_index"][spin]) + 1
    cb = np.asarray(bs.bands[spin][cb_index])
    return cb, vbm["energy"]


def find_conduction_valleys(bs) -> Dict[str, Dict[str, Any]]:
    """Locate local conduction band valley minima (Γ, L, Δ, and other high symmetry points)."""
    if bs.is_metal():
        return {}

    cb, e_vbm = extract_conduction_band(bs)
    valleys = {}

    def add_point(name: str, indices: List[int]):
        if not indices:
            return
        sub_energies = cb[indices]
        best_sub_idx = int(np.argmin(sub_energies))
        i = indices[best_sub_idx]
        kpt = bs.kpoints[i]
        valleys[name] = {
            "name": name,
            "index": int(i),
            "energy_ev": float(cb[i]),
            "gap_ev": float(cb[i] - e_vbm),
            "frac_coords": [float(c) for c in kpt.frac_coords],
            "cart_coords": [float(c) for c in kpt.cart_coords] if hasattr(kpt, "cart_coords") else [],
            "label": kpt.label or name,
        }

    # Helper to find index by label
    def get_indices_by_label(*labels):
        target_labels = {l.lower().replace("\\", "") for l in labels}
        return [
            i for i, k in enumerate(bs.kpoints)
            if k.label and k.label.replace("\\", "").lower() in target_labels
        ]

    # Γ valley
    gamma_idx = get_indices_by_label("gamma", "γ", "g")
    if gamma_idx:
        add_point("Gamma (Γ)", gamma_idx)

    # L valley
    l_idx = get_indices_by_label("l")
    if l_idx:
        add_point("L", l_idx)

    # X valley
    x_idx = get_indices_by_label("x")
    if x_idx:
        add_point("X", x_idx)

    # K valley
    k_idx = get_indices_by_label("k")
    if k_idx:
        add_point("K", k_idx)

    # Δ valley (along Γ - X segment)
    for b in getattr(bs, "branches", []):
        bname = b["name"].replace("\\", "").lower()
        if "gamma" in bname and "x" in bname:
            idx_list = list(range(b["start_index"], b["end_index"] + 1))
            min_sub_idx = int(np.argmin(cb[idx_list]))
            i = idx_list[min_sub_idx]
            kpt = bs.kpoints[i]
            valleys["Delta (Δ)"] = {
                "name": "Delta (Δ)",
                "index": int(i),
                "energy_ev": float(cb[i]),
                "gap_ev": float(cb[i] - e_vbm),
                "frac_coords": [float(c) for c in kpt.frac_coords],
                "label": kpt.label or "Δ",
                "branch_name": b["name"],
                "at_endpoint": bool(i in (b["start_index"], b["end_index"])),
            }
            break

    return valleys


def calculate_effective_mass_and_mobility(
    bs, valley_info: Dict[str, Any], tau_s: float = DEFAULT_TAU_S, step: int = 1
) -> Dict[str, Any]:
    """Calculate effective mass m*/m0 via 3-point finite difference curvature and Drude mobility."""
    if bs.is_metal():
        return {"effective_mass": None, "mobility_cm2_vs": None}

    i0 = valley_info["index"]
    # Find which branch contains i0
    matching_branch = None
    for b in getattr(bs, "branches", []):
        if b["start_index"] <= i0 <= b["end_index"]:
            matching_branch = b
            break

    if not matching_branch:
        return {"effective_mass": None, "mobility_cm2_vs": None, "note": "K-point not in a branch."}

    i_l, i_r = i0 - step, i0 + step
    if i_l < matching_branch["start_index"] or i_r > matching_branch["end_index"]:
        return {
            "effective_mass": None,
            "mobility_cm2_vs": None,
            "note": "Valley is on path boundary; symmetric derivative unavailable.",
        }

    cb, _ = extract_conduction_band(bs)
    k_dist = bs.distance
    h1 = k_dist[i0] - k_dist[i_l]
    h2 = k_dist[i_r] - k_dist[i0]

    if h1 <= 0 or h2 <= 0:
        return {"effective_mass": None, "mobility_cm2_vs": None, "note": "Zero k-distance step."}

    # d²E/dk² in eV·Å²
    d2e = 2.0 / (h1 + h2) * ((cb[i_r] - cb[i0]) / h2 - (cb[i0] - cb[i_l]) / h1)

    if d2e <= 0:
        return {
            "d2e_dk2": float(d2e),
            "effective_mass": None,
            "mobility_cm2_vs": None,
            "note": f"Curvature is non-positive ({d2e:.2f} eV·Å²); not a local parabolic minimum along this line.",
        }

    m_rel = HBAR2_OVER_M0 / d2e
    mobility = E_CHARGE * tau_s / (m_rel * M0) * 1e4  # cm²/(V·s)

    return {
        "d2e_dk2": float(d2e),
        "effective_mass": float(m_rel),
        "mobility_cm2_vs": float(mobility),
        "tau_ps": float(tau_s * 1e12),
    }


def analyze_material(
    data: Dict[str, Any], tau_s: float = DEFAULT_TAU_S, save_output: bool = True
) -> Dict[str, Any]:
    """
    Main analysis orchestrator for any material data dictionary.
    Returns complete physical properties and optionally persists results to data/processed/.
    """
    mp_id = data.get("material_id", "unknown")
    structure = data.get("structure")
    bs = data.get("bandstructure")
    dos = data.get("dos")

    formula = structure.composition.reduced_formula if structure else mp_id
    spacegroup = structure.get_space_group_info() if structure else ("Unknown", 0)

    # Basic metadata
    analysis = {
        "material_id": mp_id,
        "formula": formula,
        "formula_pretty": formula,
        "unit_cell_formula": structure.composition.formula if structure else formula,
        "spacegroup_symbol": spacegroup[0] if isinstance(spacegroup, (tuple, list)) else str(spacegroup),
        "spacegroup_number": spacegroup[1] if isinstance(spacegroup, (tuple, list)) and len(spacegroup) > 1 else None,
        "density_g_cm3": float(structure.density) if structure else None,
        "num_sites": int(len(structure)) if structure else None,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "is_metal": bool(bs.is_metal()) if bs else False,
    }

    if bs is None:
        analysis["error"] = "No band structure object available for analysis."
        return analysis

    analysis["efermi_bs_ev"] = float(bs.efermi)
    if dos:
        analysis["efermi_dos_ev"] = float(dos.efermi)
        analysis["dos_at_fermi"] = calculate_dos_at_fermi(dos, dos.efermi)

    if analysis["is_metal"]:
        # --- METALLIC MATERIAL ---
        analysis["classification"] = "Metal"
        analysis["band_gap_ev"] = 0.0
        analysis["band_gap_type"] = "Metallic (Zero Gap)"
        analysis["notes"] = [
            "Partially filled bands cross the Fermi level E_F.",
            f"Finite density of states at Fermi level N(E_F) = {analysis.get('dos_at_fermi', 0.0):.2f} states/eV/cell.",
            "Semiconductor valley gaps and VBM/CBM separation are not defined for metals.",
        ]
        analysis["valleys"] = {}
        analysis["effective_mass_summary"] = None
    else:
        # --- SEMICONDUCTOR / INSULATOR ---
        analysis["classification"] = "Semiconductor / Insulator"
        gap_info = bs.get_band_gap()
        cbm = bs.get_cbm()
        vbm = bs.get_vbm()

        analysis["band_gap_ev"] = float(gap_info["energy"])
        analysis["band_gap_type"] = "Direct" if gap_info["direct"] else "Indirect"
        analysis["transition"] = str(gap_info.get("transition", "N/A"))

        analysis["vbm"] = {
            "energy_ev": float(vbm["energy"]),
            "kpoint_frac": [float(c) for c in vbm["kpoint"].frac_coords],
            "label": vbm["kpoint"].label or "VBM",
        }
        analysis["cbm"] = {
            "energy_ev": float(cbm["energy"]),
            "kpoint_frac": [float(c) for c in cbm["kpoint"].frac_coords],
            "label": cbm["kpoint"].label or "CBM",
        }

        # Conduction valleys analysis
        valleys = find_conduction_valleys(bs)

        # Calculate effective mass & mobility for each valley
        for name, vinfo in valleys.items():
            mass_res = calculate_effective_mass_and_mobility(bs, vinfo, tau_s=tau_s)
            vinfo["effective_mass_m0"] = mass_res.get("effective_mass")
            vinfo["mobility_cm2_vs"] = mass_res.get("mobility_cm2_vs")
            vinfo["d2e_dk2"] = mass_res.get("d2e_dk2")

        analysis["valleys"] = valleys

        # Primary minimum mass (CBM / Delta if Si)
        primary_valley = valleys.get("Delta (Δ)") or valleys.get("Gamma (Γ)") or (list(valleys.values())[0] if valleys else None)
        if primary_valley and primary_valley.get("effective_mass_m0"):
            analysis["primary_effective_mass_m0"] = primary_valley["effective_mass_m0"]
            analysis["primary_mobility_cm2_vs"] = primary_valley["mobility_cm2_vs"]

    # Save to data/processed if requested
    if save_output:
        save_analysis_outputs(analysis)

    return analysis


def save_analysis_outputs(analysis: Dict[str, Any], output_dir: str = PROCESSED_DATA_DIR) -> Dict[str, str]:
    """
    Save the analysis results into structured files:
      1. JSON summary: data/processed/{mp_id}_summary.json
      2. CSV valley report: data/processed/{mp_id}_valleys.csv (if semiconductor)
    """
    os.makedirs(output_dir, exist_ok=True)
    mp_id = analysis.get("material_id", "material")

    # 1. JSON Report
    json_path = os.path.join(output_dir, f"{mp_id}_summary.json")
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2)

    saved_files = {"json": json_path}

    # 2. CSV Valley Table (for semiconductors)
    valleys = analysis.get("valleys", {})
    if valleys:
        csv_path = os.path.join(output_dir, f"{mp_id}_valleys.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Valley", "Gap_eV", "Energy_eV", "k_frac_coords", "m_star_m0", "Mobility_cm2_Vs", "d2E_dk2_eVA2"
            ])
            for vname, vdata in valleys.items():
                k_str = "[" + ", ".join(f"{c:.3f}" for c in vdata.get("frac_coords", [])) + "]"
                m_star = f"{vdata['effective_mass_m0']:.3f}" if vdata.get("effective_mass_m0") is not None else "N/A"
                mob = f"{vdata['mobility_cm2_vs']:.1f}" if vdata.get("mobility_cm2_vs") is not None else "N/A"
                d2e = f"{vdata['d2e_dk2']:.2f}" if vdata.get("d2e_dk2") is not None else "N/A"
                writer.writerow([
                    vname,
                    f"{vdata['gap_ev']:.3f}",
                    f"{vdata['energy_ev']:.3f}",
                    k_str,
                    m_star,
                    mob,
                    d2e,
                ])
        saved_files["csv"] = csv_path

    return saved_files
