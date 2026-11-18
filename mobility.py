"""
mobility.py
-----------
Electron effective mass from the second derivative of E(k), and a simple
mobility estimate  mu = e * tau / m*.

IMPORTANT: the band structure gives m*, but NOT the scattering time tau.
tau is an input (default is a placeholder of 0.2 ps, roughly what Si's
measured electron mobility implies at 300 K). Treat mobility as an estimate.

Also: a Γ–X line path only gives the LONGITUDINAL curvature at the Δ valley,
so for Si this mass is m_l (about 0.92 m0), not the conductivity mass.

Typical use:
    from mobility import print_electron_mobility, electron_mobility
    print_electron_mobility(bs)                    # default tau
    rows = electron_mobility(bs, tau_s=0.1e-12)    # list of dicts
"""
from band_gaps import conduction_band, find_valleys

HBAR2_OVER_M0 = 7.61996       # ħ²/m0 in eV·Å²
E_CHARGE = 1.602176634e-19    # C
M0 = 9.1093837015e-31         # kg
DEFAULT_TAU_S = 0.2e-12       # s, placeholder scattering time


def curvature(energy, k_dist, i0, branch, step=1):
    """d²E/dk² (eV·Å²) at index i0: 3-point finite difference, valid for uneven k-spacing.

    Uses points i0-step and i0+step, which must lie inside the same path segment.
    """
    i_l, i_r = i0 - step, i0 + step
    if i_l < branch["start_index"] or i_r > branch["end_index"]:
        raise ValueError(f"step={step} runs past the end of segment {branch['name']}.")
    h1 = k_dist[i0] - k_dist[i_l]
    h2 = k_dist[i_r] - k_dist[i0]
    return 2.0 / (h1 + h2) * ((energy[i_r] - energy[i0]) / h2 - (energy[i0] - energy[i_l]) / h1)


def mass_from_curvature(d2e):
    """m*/m0 = ħ² / (m0 · d²E/dk²)."""
    return HBAR2_OVER_M0 / d2e


def mobility_cm2_per_vs(m_rel, tau_s=DEFAULT_TAU_S):
    """mu = e·tau/m*, returned in cm²/(V·s)."""
    return E_CHARGE * tau_s / (m_rel * M0) * 1e4


def electron_mobility(bs, tau_s=DEFAULT_TAU_S, steps=(1, 2, 3)):
    """Effective mass and mobility at the Δ valley, for several finite-difference step sizes.

    Returns a list of dicts with keys: step, d2e, m_rel, mobility.
    m_rel and mobility are None if d²E/dk² <= 0 (the point is not a true minimum).
    """
    valleys = find_valleys(bs)
    if "delta" not in valleys:
        raise ValueError("No Γ–X path segment found in this band structure.")
    delta = valleys["delta"]
    if delta["at_endpoint"]:
        raise ValueError(
            "The minimum sits on the end of the Γ–X segment, so a symmetric "
            "second derivative can't be taken there."
        )

    cb, _ = conduction_band(bs)
    rows = []
    for step in steps:
        try:
            d2e = curvature(cb, bs.distance, delta["index"], delta["branch"], step)
        except ValueError:
            continue  # window too wide for this segment
        if d2e <= 0:
            rows.append({"step": step, "d2e": d2e, "m_rel": None, "mobility": None})
        else:
            m = mass_from_curvature(d2e)
            rows.append({"step": step, "d2e": d2e, "m_rel": m,
                         "mobility": mobility_cm2_per_vs(m, tau_s)})
    return rows


def print_electron_mobility(bs, tau_s=DEFAULT_TAU_S, steps=(1, 2, 3)):
    print(f"Assumed tau = {tau_s * 1e12:.2f} ps  (input, not from the band structure)")
    for r in electron_mobility(bs, tau_s, steps):
        if r["m_rel"] is None:
            print(f"step={r['step']}: d2E/dk2 = {r['d2e']:.2f} eV·Å² <= 0, not a minimum along this path")
        else:
            print(f"step={r['step']}: d2E/dk2 = {r['d2e']:.2f} eV·Å², "
                  f"m* = {r['m_rel']:.3f} m0, mu ≈ {r['mobility']:.0f} cm²/V·s")
