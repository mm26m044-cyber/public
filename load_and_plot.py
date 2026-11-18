"""
load_and_plot.py

Step 2 of Band Explorer: reload the raw JSON saved by fetch_data.py
and plot the band structure exactly as-is (no corrections yet).

This confirms the save -> reload -> plot round trip works before we
build anything (Varshni, strain, effective mass) on top of it.

Usage:
    python load_and_plot.py
"""

import json
import os

import matplotlib.pyplot as plt
from monty.json import MontyDecoder
from pymatgen.electronic_structure.plotter import BSPlotter

MATERIAL_ID = "mp-149"
RAW_PATH = os.path.join("data", "raw", f"{MATERIAL_ID}.json")
OUTPUT_PATH = os.path.join("data", "raw", f"{MATERIAL_ID}_bandstructure.png")


def load_raw(path: str) -> dict:
    """Reload the JSON saved by fetch_data.py back into pymatgen objects.

    MontyDecoder is the counterpart to the MontyEncoder used when saving —
    it reads the "@module"/"@class" markers embedded in the JSON and
    reconstructs the original Structure/BandStructure/Dos objects.
    """
    with open(path) as f:
        data = json.load(f, cls=MontyDecoder)
    return data


def plot_bandstructure(bandstructure, output_path: str) -> None:
    """Plot the raw band structure and save it to a PNG."""
    plotter = BSPlotter(bandstructure)
    plotter.get_plot()  # draws onto the current matplotlib figure
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")


def main():
    print(f"Loading {RAW_PATH} ...")
    data = load_raw(RAW_PATH)

    bs = data["bandstructure"]
    gap = bs.get_band_gap()
    print(f"  Reloaded band gap: {gap['energy']:.3f} eV "
          f"({'direct' if gap['direct'] else 'indirect'})")

    plot_bandstructure(bs, OUTPUT_PATH)


if __name__ == "__main__":
    main()
