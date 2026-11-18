"""
plot_combined.py

Step 4 of Band Explorer: plot band structure and DOS side by side,
sharing a common energy (y) axis, so the two can be visually matched
directly -- the "textbook style" combined figure.

Note: pymatgen's built-in BSDOSPlotter requires a CompleteDos object
(one with a .structure attribute, for per-element projections). The
Dos object mp-api returns here doesn't have that. This script also
avoids BSPlotter's internal bs_plot_data() helper, whose output
format has changed between pymatgen versions -- instead it works
directly from the band structure object's core attributes (.bands,
.kpoints, .branches), which are stable physics data rather than a
plotting-internal format, so this shouldn't break again on version
updates.
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
from monty.json import MontyDecoder

MATERIAL_ID = "mp-149"
RAW_PATH = os.path.join("data", "raw", f"{MATERIAL_ID}.json")
OUTPUT_PATH = os.path.join("data", "raw", f"{MATERIAL_ID}_combined.png")

# How far above/below the Fermi level to show -- keeps the plot readable
# instead of showing every deep core band.
ENERGY_WINDOW_EV = 4.0


def load_raw(path: str) -> dict:
    """Reload the JSON saved by fetch_data.py back into pymatgen objects."""
    with open(path) as f:
        data = json.load(f, cls=MontyDecoder)
    return data


def get_branch_path_data(bandstructure):
    """
    Compute k-path distance and collect band energies, one branch
    (k-path segment between two high-symmetry points) at a time,
    using only the band structure's core, stable attributes.

    Returns:
        branch_distances: list of 1D arrays, one per branch
        branch_energies: list of dicts {spin: 2D array [nbands, nkpts_in_branch]}
        ticks: {"distance": [...], "label": [...]}
    """
    kpoints = bandstructure.kpoints
    bands = bandstructure.bands  # {Spin: array [nbands, n_total_kpts]}
    efermi = bandstructure.efermi

    branch_distances = []
    branch_energies = []
    ticks_distance = []
    ticks_label = []
    offset = 0.0

    for branch in bandstructure.branches:
        start, end = branch["start_index"], branch["end_index"]
        branch_kpts = kpoints[start:end + 1]

        d = [offset]
        for i in range(1, len(branch_kpts)):
            step = np.linalg.norm(
                np.array(branch_kpts[i].cart_coords)
                - np.array(branch_kpts[i - 1].cart_coords)
            )
            d.append(d[-1] + step)
        d = np.array(d)
        branch_distances.append(d)
        offset = d[-1]

        branch_energies.append({
            spin: energies[:, start:end + 1] - efermi
            for spin, energies in bands.items()
        })

        ticks_distance.append(d[0])
        ticks_label.append(branch_kpts[0].label or "")

    # tick at the very last point of the whole path
    last_branch = bandstructure.branches[-1]
    ticks_distance.append(branch_distances[-1][-1])
    ticks_label.append(kpoints[last_branch["end_index"]].label or "")

    return branch_distances, branch_energies, {
        "distance": ticks_distance, "label": ticks_label,
    }


def plot_combined(bandstructure, dos, output_path: str) -> None:
    branch_distances, branch_energies, ticks = get_branch_path_data(bandstructure)

    fig, (ax_bands, ax_dos) = plt.subplots(
        1, 2, figsize=(9, 6), sharey=True,
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05},
    )

    # --- Left panel: band structure ---
    for dist, energy_dict in zip(branch_distances, branch_energies):
        for spin_bands in energy_dict.values():   # one 2D array per spin channel
            for band in spin_bands:                # each row = one band
                ax_bands.plot(dist, band, color="tab:blue", linewidth=1)

    ax_bands.axhline(0, color="black", linestyle="--", linewidth=0.8)  # E_F
    ax_bands.set_xlim(branch_distances[0][0], branch_distances[-1][-1])
    ax_bands.set_ylim(-ENERGY_WINDOW_EV, ENERGY_WINDOW_EV)
    ax_bands.set_xticks(ticks["distance"])
    ax_bands.set_xticklabels(ticks["label"])
    ax_bands.set_ylabel("E $-$ E$_F$ (eV)")
    ax_bands.set_title("Band structure")

    # --- Right panel: DOS, rotated so energy is on the shared y-axis ---
    # We shift by dos.efermi ourselves so both panels use the exact
    # same zero point as the band structure panel above.
    energies = dos.energies - dos.efermi
    densities = dos.get_densities()  # summed over spin channels automatically

    ax_dos.plot(densities, energies, color="tab:blue", linewidth=1)
    ax_dos.fill_betweenx(energies, 0, densities, alpha=0.2, color="tab:blue")
    ax_dos.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax_dos.set_xlim(left=0)
    ax_dos.set_xlabel("DOS (states/eV)")
    ax_dos.set_title("Density of states")

    # Sanity note: the band-structure and DOS calculations are separate
    # DFT tasks, so their reported Fermi levels can differ very slightly.
    print(f"  Band structure E_F: {bandstructure.efermi:.4f} eV")
    print(f"  DOS E_F:            {dos.efermi:.4f} eV")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved combined plot to: {output_path}")


def main():
    print(f"Loading {RAW_PATH} ...")
    data = load_raw(RAW_PATH)

    plot_combined(data["bandstructure"], data["dos"], OUTPUT_PATH)


if __name__ == "__main__":
    main()


