"""
ewald_sphere.py
---------------
Ewald Sphere 3D & 2D Diffraction Simulator for metals and semiconductors.

Implements:
  1. Reciprocal lattice construction from crystal structure (b1, b2, b3 and G_hkl)
  2. Radiation wavelength helper (X-ray lines & relativistic de Broglie electrons)
  3. Ewald sphere geometric condition: ||k_i + G_hkl| - k0| <= epsilon
  4. Universal structure factor F_hkl = sum_j Z_j * exp(2*pi*i*(h*x_j + k*y_j + l*z_j))
  5. Filtering of systematic absences (extinction rules)
  6. 2D detector screen projection and Bragg 2-theta calculation
  7. Automated saving of diffraction table to data/processed/{mp_id}_diffraction.csv

Usage:
    from ewald_sphere import simulate_ewald_diffraction
    res = simulate_ewald_diffraction(structure, wavelength_angstrom=1.5406, beam_direction=[0, 0, 1])
"""

import os
import csv
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

PROCESSED_DATA_DIR = os.path.join("data", "processed")

# Radiation wavelength presets (in Angstroms)
RADIATION_PRESETS = {
    "Cu-Ka": {"name": "Cu Kα X-ray", "wavelength_angstrom": 1.5406, "type": "X-ray"},
    "Mo-Ka": {"name": "Mo Kα X-ray", "wavelength_angstrom": 0.7107, "type": "X-ray"},
    "Cr-Ka": {"name": "Cr Kα X-ray", "wavelength_angstrom": 2.2897, "type": "X-ray"},
    "Co-Ka": {"name": "Co Kα X-ray", "wavelength_angstrom": 1.7890, "type": "X-ray"},
    "TEM-200kV": {"name": "200 keV Electron (TEM)", "wavelength_angstrom": 0.02508, "type": "Electron"},
    "TEM-100kV": {"name": "100 keV Electron (TEM)", "wavelength_angstrom": 0.03701, "type": "Electron"},
    "TEM-300kV": {"name": "300 keV Electron (TEM)", "wavelength_angstrom": 0.01969, "type": "Electron"},
}


def electron_energy_to_wavelength(energy_kev: float) -> float:
    """
    Calculate relativistic de Broglie electron wavelength in Angstroms.
    lambda = h / sqrt(2*m0*E*(1 + E/(2*m0*c^2)))
    """
    if energy_kev <= 0:
        raise ValueError("Energy must be positive.")
    # Relativistic formula constant: lambda (Å) = 12.2643 / sqrt(V * (1 + 0.978466e-6 * V)) with V in Volts
    voltage_volts = energy_kev * 1000.0
    wavelength = 12.2643 / np.sqrt(voltage_volts * (1.0 + 0.978466e-6 * voltage_volts))
    return float(wavelength)


def build_reciprocal_lattice(structure) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute reciprocal lattice vectors b1, b2, b3 in Angstrom^-1 with 2*pi convention:
      b1 = 2*pi * (a2 x a3) / V
      b2 = 2*pi * (a3 x a1) / V
      b3 = 2*pi * (a1 x a2) / V
    Returns (direct_matrix [3x3], reciprocal_matrix [3x3]) where rows are a_i and b_i.
    """
    direct_lattice = structure.lattice.matrix  # rows are a1, a2, a3
    a1, a2, a3 = direct_lattice[0], direct_lattice[1], direct_lattice[2]
    vol = float(np.dot(a1, np.cross(a2, a3)))

    b1 = 2.0 * np.pi * np.cross(a2, a3) / vol
    b2 = 2.0 * np.pi * np.cross(a3, a1) / vol
    b3 = 2.0 * np.pi * np.cross(a1, a2) / vol

    reciprocal_lattice = np.vstack([b1, b2, b3])
    return direct_lattice, reciprocal_lattice


def compute_structure_factor(
    h: int, k: int, l: int, frac_coords: np.ndarray, atomic_numbers: np.ndarray
) -> complex:
    """
    Universal Structure Factor:
      F_hkl = sum_j Z_j * exp(2 * pi * i * (h*x_j + k*y_j + l*z_j))
    """
    phase = 2.0 * np.pi * (h * frac_coords[:, 0] + k * frac_coords[:, 1] + l * frac_coords[:, 2])
    f_complex = np.sum(atomic_numbers * np.exp(1j * phase))
    return complex(f_complex)


def get_detector_coordinate_system(beam_dir_unit: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create orthogonal 3D basis (e_x, e_y, e_z) for detector screen where e_z is along beam direction.
    """
    e_z = beam_dir_unit / np.linalg.norm(beam_dir_unit)
    # Choose arbitrary non-parallel reference vector to cross with
    ref = np.array([0.0, 1.0, 0.0]) if abs(e_z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e_x = np.cross(ref, e_z)
    e_x = e_x / np.linalg.norm(e_x)
    e_y = np.cross(e_z, e_x)
    e_y = e_y / np.linalg.norm(e_y)
    return e_x, e_y, e_z


def simulate_ewald_diffraction(
    structure,
    wavelength_angstrom: float = 1.5406,
    beam_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    max_index: int = 4,
    tolerance_rel: float = 0.04,
    detector_distance_mm: float = 100.0,
    material_id: Optional[str] = None,
    save_output: bool = True,
) -> Dict[str, Any]:
    """
    Run full Ewald sphere diffraction simulation.

    Parameters:
      structure: pymatgen Structure
      wavelength_angstrom: radiation wavelength (Å)
      beam_direction: [u, v, w] crystallographic incident beam direction
      max_index: maximum |h|, |k|, |l| index for reciprocal lattice search
      tolerance_rel: relative thickness of Ewald sphere (| |k_i + G| - k0 | / k0 <= tolerance)
      detector_distance_mm: physical distance from crystal to simulated detector screen (mm)
      material_id: material identifier for saving reports
      save_output: whether to auto-save CSV/JSON in data/processed/

    Returns:
      Comprehensive dictionary with Ewald geometry, all G_hkl nodes, and active reflections.
    """
    direct_lattice, reciprocal_lattice = build_reciprocal_lattice(structure)
    b1, b2, b3 = reciprocal_lattice[0], reciprocal_lattice[1], reciprocal_lattice[2]

    # Convert beam direction to Cartesian unit vector
    u, v, w = beam_direction
    u_cart = u * direct_lattice[0] + v * direct_lattice[1] + w * direct_lattice[2]
    norm_u = np.linalg.norm(u_cart)
    if norm_u < 1e-6:
        u_cart = np.array([0.0, 0.0, 1.0])
        norm_u = 1.0
    beam_unit = u_cart / norm_u

    # Wavenumber k0 = 2*pi / lambda (in Å^-1)
    k0 = 2.0 * np.pi / wavelength_angstrom
    k_incident = k0 * beam_unit
    sphere_center = -k_incident  # Center in reciprocal space so (0,0,0) is on sphere
    sphere_tolerance = tolerance_rel * k0

    # Extract basis coordinates and atomic numbers
    frac_coords = np.array([site.frac_coords for site in structure])
    atomic_numbers = np.array([float(site.specie.Z) for site in structure])

    # Detector coordinate system
    det_ex, det_ey, det_ez = get_detector_coordinate_system(beam_unit)

    # Scan reciprocal lattice grid
    h_range = range(-max_index, max_index + 1)
    k_range = range(-max_index, max_index + 1)
    l_range = range(-max_index, max_index + 1)

    all_nodes = []
    active_reflections = []

    # First pass: find maximum |F_hkl|^2 for normalization
    max_f_sq = 1.0

    for h in h_range:
        for k in k_range:
            for l in l_range:
                g_vec = h * b1 + k * b2 + l * b3
                g_norm = float(np.linalg.norm(g_vec))

                f_val = compute_structure_factor(h, k, l, frac_coords, atomic_numbers)
                f_sq = float(abs(f_val) ** 2)
                if f_sq > max_f_sq:
                    max_f_sq = f_sq

    # Second pass: classify points & collect active diffracted beams
    for h in h_range:
        for k in k_range:
            for l in l_range:
                if h == 0 and k == 0 and l == 0:
                    # Origin (transmitted beam)
                    all_nodes.append({
                        "hkl": [0, 0, 0],
                        "g_cart": [0.0, 0.0, 0.0],
                        "d_spacing_angstrom": None,
                        "f_sq": max_f_sq,
                        "status": "origin",
                    })
                    continue

                g_vec = h * b1 + k * b2 + l * b3
                g_norm = float(np.linalg.norm(g_vec))
                d_spacing = 2.0 * np.pi / g_norm if g_norm > 1e-6 else None

                # Distance from sphere center
                dist_to_center = float(np.linalg.norm(g_vec - sphere_center))
                delta_k = abs(dist_to_center - k0)
                on_sphere = delta_k <= sphere_tolerance

                f_val = compute_structure_factor(h, k, l, frac_coords, atomic_numbers)
                f_sq = float(abs(f_val) ** 2)
                is_systematic_absence = (f_sq / max_f_sq) < 1e-3

                status = "inactive"
                if is_systematic_absence:
                    status = "forbidden"
                elif on_sphere:
                    status = "diffracting"

                node_entry = {
                    "hkl": [int(h), int(k), int(l)],
                    "g_cart": [float(c) for c in g_vec],
                    "g_norm": g_norm,
                    "d_spacing_angstrom": float(d_spacing) if d_spacing else None,
                    "f_sq": f_sq,
                    "f_norm": float(f_sq / max_f_sq),
                    "dist_to_center": dist_to_center,
                    "delta_k": delta_k,
                    "status": status,
                }
                all_nodes.append(node_entry)

                # If point diffracts and is not systematically absent:
                if on_sphere and not is_systematic_absence:
                    # Outgoing wavevector k_f = k_i + G
                    k_f = k_incident + g_vec
                    k_f_norm = np.linalg.norm(k_f)
                    k_f_unit = k_f / k_f_norm if k_f_norm > 1e-6 else beam_unit

                    # Bragg 2-theta angle
                    cos_2theta = float(np.clip(np.dot(beam_unit, k_f_unit), -1.0, 1.0))
                    two_theta_rad = float(np.arccos(cos_2theta))
                    two_theta_deg = float(np.degrees(two_theta_rad))

                    # 2D flat detector screen projection (forward scattering)
                    cos_proj = float(np.dot(k_f_unit, det_ez))
                    det_x, det_y = None, None
                    if cos_proj > 0.05:  # Ray travels toward screen
                        scale = detector_distance_mm / cos_proj
                        det_x = float(scale * np.dot(k_f_unit, det_ex))
                        det_y = float(scale * np.dot(k_f_unit, det_ey))

                    active_reflections.append({
                        "hkl": [int(h), int(k), int(l)],
                        "hkl_str": f"({h} {k} {l})",
                        "d_spacing_angstrom": float(d_spacing) if d_spacing else 0.0,
                        "two_theta_deg": two_theta_deg,
                        "f_sq": f_sq,
                        "intensity_rel": float(f_sq / max_f_sq),
                        "g_cart": [float(c) for c in g_vec],
                        "k_f_cart": [float(c) for c in k_f],
                        "detector_x_mm": det_x,
                        "detector_y_mm": det_y,
                    })

    # Sort active reflections by 2-theta angle
    active_reflections.sort(key=lambda r: r["two_theta_deg"])

    # Formula & metadata
    formula = structure.composition.reduced_formula

    result = {
        "material_id": material_id or "material",
        "formula": formula,
        "wavelength_angstrom": float(wavelength_angstrom),
        "wavenumber_k0": float(k0),
        "beam_direction_indices": [float(x) for x in beam_direction],
        "beam_unit_cart": [float(x) for x in beam_unit],
        "sphere_center_cart": [float(x) for x in sphere_center],
        "sphere_radius": float(k0),
        "tolerance_rel": float(tolerance_rel),
        "detector_distance_mm": float(detector_distance_mm),
        "max_index": int(max_index),
        "num_total_nodes": len(all_nodes),
        "num_diffracting": len(active_reflections),
        "active_reflections": active_reflections,
        "all_nodes": all_nodes,
        "reciprocal_lattice_b": [
            [float(x) for x in b1],
            [float(x) for x in b2],
            [float(x) for x in b3],
        ],
    }

    if save_output and material_id:
        save_ewald_outputs(result)

    return result


def save_ewald_outputs(sim_result: Dict[str, Any], output_dir: str = PROCESSED_DATA_DIR) -> Dict[str, str]:
    """
    Persist diffraction simulation results to data/processed/.
      - CSV: data/processed/{mp_id}_diffraction.csv
      - JSON: data/processed/{mp_id}_ewald.json
    """
    os.makedirs(output_dir, exist_ok=True)
    mp_id = sim_result.get("material_id", "material")

    # 1. JSON Report (summary metadata without massive node arrays for compact speed)
    json_path = os.path.join(output_dir, f"{mp_id}_ewald.json")
    compact_summary = {
        "material_id": mp_id,
        "formula": sim_result.get("formula"),
        "wavelength_angstrom": sim_result.get("wavelength_angstrom"),
        "wavenumber_k0_angstrom_inv": sim_result.get("wavenumber_k0"),
        "beam_direction": sim_result.get("beam_direction_indices"),
        "tolerance_rel": sim_result.get("tolerance_rel"),
        "num_reflections_active": sim_result.get("num_diffracting"),
        "active_reflections": sim_result.get("active_reflections", []),
    }
    with open(json_path, "w") as f:
        json.dump(compact_summary, f, indent=2)

    # 2. CSV Table of Active Diffraction Spots
    csv_path = os.path.join(output_dir, f"{mp_id}_diffraction.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "h", "k", "l", "hkl_plane", "d_spacing_A", "two_theta_deg", "Structure_Factor_F2", "Rel_Intensity", "Detector_X_mm", "Detector_Y_mm"
        ])
        for ref in sim_result.get("active_reflections", []):
            h, k, l = ref["hkl"]
            det_x = f"{ref['detector_x_mm']:.2f}" if ref.get("detector_x_mm") is not None else "N/A"
            det_y = f"{ref['detector_y_mm']:.2f}" if ref.get("detector_y_mm") is not None else "N/A"
            writer.writerow([
                h, k, l,
                ref["hkl_str"],
                f"{ref['d_spacing_angstrom']:.4f}",
                f"{ref['two_theta_deg']:.2f}",
                f"{ref['f_sq']:.1f}",
                f"{ref['intensity_rel']:.3f}",
                det_x,
                det_y,
            ])

    return {"json": json_path, "csv": csv_path}
