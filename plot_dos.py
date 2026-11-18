"""
plot_dos.py

Step 3 of Band Explorer: reload the raw JSON saved by fetch_data.py
and plot the total density of states (DOS).

Same pattern as load_and_plot.py, just for DOS instead of band structure.
"""

import json
import os

import matplotlib.pyplot as plt
from monty.json import MontyDecoder
from pymatgen.electronic_structure.plotter import DosPlotter

MATERIAL_ID = "mp-149"
RAW_PATH = os.path.join("data", "raw", f"{MATERIAL_ID}.json")
OUTPUT_PATH = os.path.join("data", "raw", f"{MATERIAL_ID}_dos.png")


def load_raw(path: str) -> dict:
    """Reload the JSON saved by fetch_data.py back into pymatgen objects."""
    with open(path) as f:
        data = json.load(f, cls=MontyDecoder)
    return data


def plot_dos(dos, output_path: str) -> None:
    """Plot the total DOS and save it to a PNG."""
    plotter = DosPlotter()
    plotter.add_dos("Total DOS", dos)
    plotter.get_plot()  # draws onto the current matplotlib figure
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")


def main():
    print(f"Loading {RAW_PATH} ...")
    data = load_raw(RAW_PATH)

    dos = data["dos"]
    bs = data["bandstructure"]

    # Sanity check: the gap read from the DOS should roughly match the
    # gap read from the band structure. They won't be exactly identical
    # (DOS gap depends on the energy grid spacing) but should be close.
    dos_gap = dos.get_gap()
    bs_gap = bs.get_band_gap()["energy"]
    print(f"  Gap from DOS:            {dos_gap:.3f} eV")
    print(f"  Gap from band structure: {bs_gap:.3f} eV")

    plot_dos(dos, OUTPUT_PATH)


if __name__ == "__main__":
    main()
