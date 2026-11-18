"""
band_gaps.py
------------
Conduction-band valley gaps (Γ, Δ, L) from a pymatgen band structure.

Definition used throughout:
    valley gap = (lowest conduction-band energy in that valley) - (global VBM)

Typical use:
    from band_gaps import print_valley_gaps, find_valleys
    print_valley_gaps(bs)          # quick look
    valleys = find_valleys(bs)     # dict you can reuse in other modules
"""
import numpy as np


def conduction_band(bs):
    """Return (cb, e_vbm).

    cb     : energy (eV) of the lowest conduction band at every k-point
    e_vbm  : energy (eV) of the valence band maximum
    """
    if bs.is_metal():
        raise ValueError("Band structure is metallic; valley gaps are not defined.")
    vbm = bs.get_vbm()
    # pick the spin channel that actually holds the VBM
    spin = next(s for s, bands in vbm["band_index"].items() if len(bands) > 0)
    # several bands can share the VBM (e.g. degenerate at Γ): the top one is the largest index
    cb_index = max(vbm["band_index"][spin]) + 1
    cb = np.asarray(bs.bands[spin][cb_index])
    return cb, vbm["energy"]


def _indices_with_label(bs, names):
    """Indices of k-points whose label matches one of `names` (case-insensitive, backslashes ignored)."""
    names = {n.lower() for n in names}
    return [
        i for i, k in enumerate(bs.kpoints)
        if k.label and k.label.replace("\\", "").lower() in names
    ]


def find_valleys(bs):
    """Locate the Γ, L and Δ conduction-band valleys.

    Returns a dict: name -> {"index", "energy", "gap", "frac_coords", ...}
      "gamma" : lowest CB at Γ
      "L"     : lowest CB at L
      "delta" : minimum of the lowest CB along the Γ–X segment; also carries
                "branch" (the path segment) and "at_endpoint" (True if the
                minimum sits exactly on Γ or X rather than in between)
    A valley is left out if the path has no matching label/segment.
    """
    cb, e_vbm = conduction_band(bs)
    valleys = {}

    def add(name, idx):
        i = idx[int(np.argmin(cb[idx]))]
        valleys[name] = {
            "index": i,
            "energy": float(cb[i]),
            "gap": float(cb[i] - e_vbm),
            "frac_coords": bs.kpoints[i].frac_coords,
        }
        return i

    gamma = _indices_with_label(bs, ["gamma", "γ"])
    if gamma:
        add("gamma", gamma)

    l_pts = _indices_with_label(bs, ["l"])
    if l_pts:
        add("L", l_pts)

    for b in bs.branches:
        name = b["name"].replace("\\", "").lower()
        if "gamma" in name and "x" in name:
            idx = list(range(b["start_index"], b["end_index"] + 1))
            i = add("delta", idx)
            valleys["delta"]["branch"] = b
            valleys["delta"]["at_endpoint"] = i in (b["start_index"], b["end_index"])
            break

    return valleys


def valley_gaps(bs):
    """Just the numbers: {"gamma": eV, "L": eV, "delta": eV}."""
    return {name: v["gap"] for name, v in find_valleys(bs).items()}


def print_valley_gaps(bs):
    for name, v in find_valleys(bs).items():
        print(f"{name:>6}: gap = {v['gap']:.3f} eV   k (frac) = {np.round(v['frac_coords'], 3)}")
