"""
visualizer.py
-------------
Interactive and responsive electronic structure visualization module using Plotly.

Generates:
  1. Interactive Band Structure Plot
  2. Interactive Density of States (DOS) Plot
  3. Combined Side-by-Side Band Structure + DOS Plot
  4. Raw JSON payloads for seamless frontend browser rendering

Usage:
    from visualizer import generate_plotly_combined_json, generate_plotly_bandstructure_json
"""

from typing import Dict, Any, List, Tuple
import numpy as np


def extract_branch_path_data(bs) -> Tuple[List[np.ndarray], List[Dict[Any, np.ndarray]], Dict[str, List]]:
    """
    Extract k-path distances, branch energies, and symmetry labels.
    Uses robust direct attribute extraction from pymatgen band structure.
    """
    kpoints = bs.kpoints
    bands = bs.bands
    efermi = bs.efermi

    branch_distances = []
    branch_energies = []
    ticks_distance = []
    ticks_label = []
    offset = 0.0

    for branch in getattr(bs, "branches", []):
        start, end = branch["start_index"], branch["end_index"]
        branch_kpts = kpoints[start:end + 1]

        d = [offset]
        for i in range(1, len(branch_kpts)):
            step = np.linalg.norm(
                np.array(branch_kpts[i].cart_coords)
                - np.array(branch_kpts[i - 1].cart_coords)
            )
            d.append(d[-1] + step)
        d_arr = np.array(d)
        branch_distances.append(d_arr)
        offset = d_arr[-1]

        branch_energies.append({
            spin: energies[:, start:end + 1] - efermi
            for spin, energies in bands.items()
        })

        lbl = branch_kpts[0].label or ""
        # Clean greek symbols for plot labels
        lbl = lbl.replace("\\Gamma", "Γ").replace("GAMMA", "Γ")
        ticks_distance.append(float(d_arr[0]))
        ticks_label.append(lbl)

    if getattr(bs, "branches", None):
        last_branch = bs.branches[-1]
        last_lbl = kpoints[last_branch["end_index"]].label or ""
        last_lbl = last_lbl.replace("\\Gamma", "Γ").replace("GAMMA", "Γ")
        ticks_distance.append(float(branch_distances[-1][-1]))
        ticks_label.append(last_lbl)

    return branch_distances, branch_energies, {
        "distance": ticks_distance,
        "label": ticks_label,
    }


def generate_plotly_bandstructure_dict(bs, energy_window: float = 4.0, highlights: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate Plotly figure dictionary for interactive Band Structure."""
    branch_distances, branch_energies, ticks = extract_branch_path_data(bs)
    traces = []

    # Plot band lines
    for b_idx, (dist, energy_dict) in enumerate(zip(branch_distances, branch_energies)):
        for spin, spin_bands in energy_dict.items():
            for band_idx, band in enumerate(spin_bands):
                # Filter out bands completely outside the display window
                if np.min(band) > energy_window + 2.0 or np.max(band) < -energy_window - 2.0:
                    continue
                traces.append({
                    "x": dist.tolist(),
                    "y": band.tolist(),
                    "type": "scatter",
                    "mode": "lines",
                    "line": {"color": "#3b82f6", "width": 1.5},
                    "hoverinfo": "y",
                    "showlegend": False,
                })

    # Add vertical lines at high symmetry points
    shapes = []
    for d in ticks["distance"]:
        shapes.append({
            "type": "line",
            "x0": d, "x1": d,
            "y0": -energy_window, "y1": energy_window,
            "line": {"color": "rgba(156, 163, 175, 0.4)", "width": 1, "dash": "dot"},
        })

    # Add horizontal Fermi level line
    shapes.append({
        "type": "line",
        "x0": branch_distances[0][0] if branch_distances else 0,
        "x1": branch_distances[-1][-1] if branch_distances else 1,
        "y0": 0, "y1": 0,
        "line": {"color": "#ef4444", "width": 1.5, "dash": "dash"},
    })

    layout = {
        "title": {"text": "Electronic Band Structure", "font": {"size": 16}},
        "xaxis": {
            "tickvals": ticks["distance"],
            "ticktext": ticks["label"],
            "range": [branch_distances[0][0], branch_distances[-1][-1]] if branch_distances else [0, 1],
            "showgrid": False,
            "zeroline": False,
        },
        "yaxis": {
            "title": "Energy E − E<sub>F</sub> (eV)",
            "range": [-energy_window, energy_window],
            "zeroline": False,
            "gridcolor": "rgba(229, 231, 235, 0.3)",
        },
        "shapes": shapes,
        "margin": {"l": 55, "r": 20, "t": 40, "b": 40},
        "hovermode": "closest",
        "template": "plotly_white",
    }

    return {"data": traces, "layout": layout}


def generate_plotly_dos_dict(dos, energy_window: float = 4.0) -> Dict[str, Any]:
    """Generate Plotly figure dictionary for interactive DOS."""
    if dos is None:
        return {"data": [], "layout": {}}

    efermi = dos.efermi
    energies = (np.array(dos.energies) - efermi).tolist()
    densities = dos.get_densities().tolist()

    trace = {
        "x": densities,
        "y": energies,
        "type": "scatter",
        "mode": "lines",
        "fill": "tozerox",
        "fillcolor": "rgba(59, 130, 246, 0.2)",
        "line": {"color": "#2563eb", "width": 1.8},
        "name": "Total DOS",
        "hoverinfo": "x+y",
    }

    shapes = [{
        "type": "line",
        "x0": 0, "x1": max(densities) * 1.05 if densities else 1,
        "y0": 0, "y1": 0,
        "line": {"color": "#ef4444", "width": 1.5, "dash": "dash"},
    }]

    layout = {
        "title": {"text": "Total Density of States (DOS)", "font": {"size": 16}},
        "xaxis": {"title": "DOS (states/eV/cell)", "rangemode": "tozero", "gridcolor": "rgba(229, 231, 235, 0.3)"},
        "yaxis": {"title": "Energy E − E<sub>F</sub> (eV)", "range": [-energy_window, energy_window], "gridcolor": "rgba(229, 231, 235, 0.3)"},
        "shapes": shapes,
        "margin": {"l": 55, "r": 20, "t": 40, "b": 40},
        "template": "plotly_white",
    }

    return {"data": [trace], "layout": layout}


def generate_plotly_combined_dict(bs, dos, energy_window: float = 4.0) -> Dict[str, Any]:
    """Generate side-by-side combined Band Structure + DOS plot for textbook-style viewing."""
    bs_data = generate_plotly_bandstructure_dict(bs, energy_window=energy_window)
    dos_data = generate_plotly_dos_dict(dos, energy_window=energy_window) if dos else None

    traces = []
    # Band structure traces on xaxis1, yaxis1
    for t in bs_data["data"]:
        t_copy = dict(t)
        t_copy["xaxis"] = "x"
        t_copy["yaxis"] = "y"
        traces.append(t_copy)

    # DOS trace on xaxis2, yaxis1 (sharing y-axis)
    if dos_data and dos_data["data"]:
        for t in dos_data["data"]:
            t_copy = dict(t)
            t_copy["xaxis"] = "x2"
            t_copy["yaxis"] = "y"
            traces.append(t_copy)

    # Shapes for BS
    shapes = []
    for s in bs_data["layout"].get("shapes", []):
        s_copy = dict(s)
        s_copy["xref"] = "x"
        s_copy["yref"] = "y"
        shapes.append(s_copy)

    # Add E_F horizontal line on DOS panel
    if dos_data:
        max_dos = max(dos_data["data"][0]["x"]) if dos_data["data"][0]["x"] else 1
        shapes.append({
            "type": "line",
            "xref": "x2", "yref": "y",
            "x0": 0, "x1": max_dos,
            "y0": 0, "y1": 0,
            "line": {"color": "#ef4444", "width": 1.5, "dash": "dash"},
        })

    layout = {
        "title": {"text": "Band Structure & Density of States", "font": {"size": 17, "weight": 600}},
        "xaxis": {
            "domain": [0.0, 0.72],
            "tickvals": bs_data["layout"]["xaxis"]["tickvals"],
            "ticktext": bs_data["layout"]["xaxis"]["ticktext"],
            "range": bs_data["layout"]["xaxis"]["range"],
            "showgrid": False,
            "zeroline": False,
        },
        "xaxis2": {
            "domain": [0.75, 1.0],
            "title": "DOS (states/eV)",
            "rangemode": "tozero",
            "gridcolor": "rgba(229, 231, 235, 0.3)",
        },
        "yaxis": {
            "title": "Energy E − E<sub>F</sub> (eV)",
            "range": [-energy_window, energy_window],
            "zeroline": False,
            "gridcolor": "rgba(229, 231, 235, 0.3)",
        },
        "shapes": shapes,
        "margin": {"l": 55, "r": 20, "t": 45, "b": 40},
        "hovermode": "closest",
        "template": "plotly_white",
    }

    return {"data": traces, "layout": layout}


def generate_plotly_ewald_3d(sim_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate clean, interactive, uncluttered 3D Plotly visualization of the Ewald sphere.
    Features:
      - Glass-like translucent sphere + 3 orthogonal great-circle wireframe rings
      - Prominently highlighted active Bragg-reflecting (hkl) nodes
      - Minimal background lattice (inactive & forbidden nodes default to legendonly/faint)
      - Crisp incident (k_i) and diffracted (k_f) ray vectors
    """
    traces = []

    c_x, c_y, c_z = sim_result["sphere_center_cart"]
    radius = sim_result["sphere_radius"]

    # 1. Translucent Glass Sphere Surface
    n_theta = 20
    n_phi = 30
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    th_grid, ph_grid = np.meshgrid(theta, phi)

    xs = c_x + radius * np.sin(th_grid) * np.cos(ph_grid)
    ys = c_y + radius * np.sin(th_grid) * np.sin(ph_grid)
    zs = c_z + radius * np.cos(th_grid)

    traces.append({
        "type": "surface",
        "x": xs.tolist(),
        "y": ys.tolist(),
        "z": zs.tolist(),
        "opacity": 0.08,
        "colorscale": [[0, "#38bdf8"], [1, "#818cf8"]],
        "showscale": False,
        "name": "Ewald Sphere Surface",
        "hoverinfo": "none",
    })

    # 2. Great Circle Guide Rings (Equator and Meridians for clean 3D depth)
    circle_t = np.linspace(0, 2 * np.pi, 80)
    # XY Equator
    traces.append({
        "type": "scatter3d",
        "mode": "lines",
        "x": (c_x + radius * np.cos(circle_t)).tolist(),
        "y": (c_y + radius * np.sin(circle_t)).tolist(),
        "z": [c_z] * len(circle_t),
        "line": {"color": "rgba(96, 165, 250, 0.6)", "width": 2},
        "name": "Sphere Equator / Rings",
        "showlegend": True,
        "hoverinfo": "none",
    })
    # XZ Meridian
    traces.append({
        "type": "scatter3d",
        "mode": "lines",
        "x": (c_x + radius * np.cos(circle_t)).tolist(),
        "y": [c_y] * len(circle_t),
        "z": (c_z + radius * np.sin(circle_t)).tolist(),
        "line": {"color": "rgba(96, 165, 250, 0.4)", "width": 1.5, "dash": "dot"},
        "showlegend": False,
        "hoverinfo": "none",
    })
    # YZ Meridian
    traces.append({
        "type": "scatter3d",
        "mode": "lines",
        "x": [c_x] * len(circle_t),
        "y": (c_y + radius * np.cos(circle_t)).tolist(),
        "z": (c_z + radius * np.sin(circle_t)).tolist(),
        "line": {"color": "rgba(96, 165, 250, 0.4)", "width": 1.5, "dash": "dot"},
        "showlegend": False,
        "hoverinfo": "none",
    })

    # 3. Incident Beam Vector k_i (Center -> Origin (0,0,0))
    traces.append({
        "type": "scatter3d",
        "mode": "lines+markers",
        "x": [c_x, 0.0],
        "y": [c_y, 0.0],
        "z": [c_z, 0.0],
        "line": {"color": "#ef4444", "width": 6},
        "marker": {"size": [5, 8], "color": ["#f87171", "#ffffff"], "symbol": ["circle", "diamond"]},
        "name": "Incident Beam kᵢ",
        "hoverinfo": "text",
        "text": ["Sphere Center (−kᵢ)", "Origin (0,0,0) — Transmitted Beam"],
    })

    # 4. Diffracted Wavevector Beams k_f (Center -> Active G_hkl)
    # Combine lines into a single trace for cleaner legend & faster rendering
    kf_x, kf_y, kf_z = [], [], []
    for ref in sim_result.get("active_reflections", []):
        gx, gy, gz = ref["g_cart"]
        kf_x.extend([c_x, gx, None])
        kf_y.extend([c_y, gy, None])
        kf_z.extend([c_z, gz, None])

    if kf_x:
        traces.append({
            "type": "scatter3d",
            "mode": "lines",
            "x": kf_x,
            "y": kf_y,
            "z": kf_z,
            "line": {"color": "#10b981", "width": 2.5, "dash": "dash"},
            "name": "Diffracted Beams k_f",
            "showlegend": True,
            "hoverinfo": "none",
        })

    # 5. Reciprocal Lattice Points (Filtered for clean view)
    diff_x, diff_y, diff_z, diff_text, diff_hover = [], [], [], [], []
    forb_x, forb_y, forb_z, forb_hover = [], [], [], []
    inact_x, inact_y, inact_z, inact_hover = [], [], [], []

    # Maximum bounding radius around sphere to exclude far-away unhelpful noise
    max_display_dist = radius * 1.5 + 2.0

    for node in sim_result.get("all_nodes", []):
        gx, gy, gz = node["g_cart"]
        h, k, l = node["hkl"]
        g_norm = node.get("g_norm", 0)

        # Skip points too far out to reduce clutter
        if g_norm > max_display_dist:
            continue

        hkl_str = f"({h} {k} {l})"
        d_str = f"{node['d_spacing_angstrom']:.3f} Å" if node.get("d_spacing_angstrom") else "N/A"
        f_sq_str = f"{node.get('f_sq', 0):.1f}"

        if node["status"] == "diffracting":
            diff_x.append(gx)
            diff_y.append(gy)
            diff_z.append(gz)
            diff_text.append(hkl_str)
            diff_hover.append(
                f"<b>Diffracting Spot: {hkl_str}</b><br>d-spacing: {d_str}<br>|F_hkl|²: {f_sq_str}<br>Distance from Sphere: {node['delta_k']:.3f} Å⁻¹"
            )
        elif node["status"] == "forbidden":
            forb_x.append(gx)
            forb_y.append(gy)
            forb_z.append(gz)
            forb_hover.append(
                f"<b>Forbidden Absence: {hkl_str}</b><br>d-spacing: {d_str}<br>|F_hkl|² ≈ 0 (Systematic Extinction)"
            )
        elif node["status"] == "inactive":
            # Only keep nearby inactive points for context
            if node.get("dist_to_center", 999) < radius + 3.0:
                inact_x.append(gx)
                inact_y.append(gy)
                inact_z.append(gz)
                inact_hover.append(
                    f"<b>Reciprocal Node: {hkl_str}</b><br>d-spacing: {d_str}<br>|F_hkl|²: {f_sq_str}"
                )

    # Active Diffracting Nodes (Prominent Glowing Green/Gold)
    if diff_x:
        traces.append({
            "type": "scatter3d",
            "mode": "markers+text",
            "x": diff_x, "y": diff_y, "z": diff_z,
            "text": diff_text,
            "textposition": "top center",
            "textfont": {"color": "#fbbf24", "size": 12, "family": "monospace", "weight": "bold"},
            "marker": {
                "size": 9,
                "color": "#10b981",
                "symbol": "circle",
                "line": {"color": "#fbbf24", "width": 2.5},
            },
            "name": "Diffracting Nodes (Active)",
            "hoverinfo": "text",
            "hovertext": diff_hover,
        })

    # Inactive Reciprocal Nodes (Clean subtle slate dots, toggleable via legend)
    if inact_x:
        traces.append({
            "type": "scatter3d",
            "mode": "markers",
            "x": inact_x, "y": inact_y, "z": inact_z,
            "marker": {"size": 3.5, "color": "#64748b", "opacity": 0.5},
            "name": "Inactive Nodes (Lattice Grid)",
            "visible": "legendonly",  # Kept in legend so user can toggle on if wanted!
            "hoverinfo": "text",
            "hovertext": inact_hover,
        })

    # Forbidden Systematic Absences (Faint gray, toggleable via legend)
    if forb_x:
        traces.append({
            "type": "scatter3d",
            "mode": "markers",
            "x": forb_x, "y": forb_y, "z": forb_z,
            "marker": {"size": 3, "color": "#475569", "opacity": 0.3},
            "name": "Forbidden Absences (F=0)",
            "visible": "legendonly",  # Kept in legend so user can toggle on if wanted!
            "hoverinfo": "text",
            "hovertext": forb_hover,
        })

    # Reciprocal Origin (0,0,0)
    traces.append({
        "type": "scatter3d",
        "mode": "markers+text",
        "x": [0.0], "y": [0.0], "z": [0.0],
        "text": ["(0 0 0)"],
        "textposition": "bottom center",
        "textfont": {"color": "#ef4444", "size": 12, "weight": "bold"},
        "marker": {"size": 9, "color": "#ef4444", "symbol": "diamond"},
        "name": "Origin (0 0 0)",
        "hoverinfo": "text",
        "hovertext": ["Reciprocal Origin (0 0 0) — Transmitted Beam"],
    })

    layout = {
        "title": {
            "text": f"3D Ewald Sphere & Reciprocal Lattice — {sim_result.get('formula', '')} (λ = {sim_result.get('wavelength_angstrom', 1.54):.4f} Å)",
            "font": {"size": 16},
        },
        "scene": {
            "xaxis": {"title": "k_x (Å⁻¹)", "gridcolor": "rgba(255,255,255,0.08)", "zerolinecolor": "#475569"},
            "yaxis": {"title": "k_y (Å⁻¹)", "gridcolor": "rgba(255,255,255,0.08)", "zerolinecolor": "#475569"},
            "zaxis": {"title": "k_z (Å⁻¹)", "gridcolor": "rgba(255,255,255,0.08)", "zerolinecolor": "#475569"},
            "aspectmode": "data",
            "bgcolor": "rgba(15, 23, 42, 0.95)",
            "camera": {
                "eye": {"x": 1.6, "y": 1.6, "z": 1.2}
            }
        },
        "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
        "legend": {
            "orientation": "h",
            "y": -0.05,
            "font": {"size": 11, "color": "#cbd5e1"},
            "bgcolor": "rgba(30, 41, 59, 0.6)",
            "bordercolor": "rgba(255, 255, 255, 0.1)",
            "borderwidth": 1
        },
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#94a3b8", "family": "system-ui, sans-serif"},
    }

    return {"data": traces, "layout": layout}


def generate_plotly_diffraction_2d(sim_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate 2D Simulated Flat Detector Screen Spot Pattern.
    """
    traces = []
    active = [r for r in sim_result.get("active_reflections", []) if r.get("detector_x_mm") is not None]

    # Detector center / direct beam crosshair
    traces.append({
        "x": [0.0],
        "y": [0.0],
        "type": "scatter",
        "mode": "markers+text",
        "marker": {"size": 14, "color": "#ef4444", "symbol": "cross"},
        "text": ["Direct Beam (0 0 0)"],
        "textposition": "top right",
        "textfont": {"color": "#ef4444", "size": 11},
        "name": "Direct Beam (0 0 0)",
        "hoverinfo": "text",
        "hovertext": ["Direct Transmitted Beam (0, 0)"],
    })

    if active:
        xs = [r["detector_x_mm"] for r in active]
        ys = [r["detector_y_mm"] for r in active]
        sizes = [max(8, min(28, 8 + 18 * np.sqrt(r.get("intensity_rel", 0.1)))) for r in active]
        colors = [r.get("intensity_rel", 0.5) for r in active]
        labels = [r["hkl_str"] for r in active]
        hover = [
            f"<b>Diffraction Spot {r['hkl_str']}</b><br>2θ: {r['two_theta_deg']:.2f}°<br>d-spacing: {r['d_spacing_angstrom']:.3f} Å<br>Relative Intensity: {r['intensity_rel']:.3f}<br>|F_hkl|²: {r['f_sq']:.1f}<br>Detector Position: ({r['detector_x_mm']:.1f}, {r['detector_y_mm']:.1f}) mm"
            for r in active
        ]

        traces.append({
            "x": xs,
            "y": ys,
            "type": "scatter",
            "mode": "markers+text",
            "marker": {
                "size": sizes,
                "color": colors,
                "colorscale": [[0, "#38bdf8"], [0.5, "#3b82f6"], [1, "#fbbf24"]],
                "showscale": True,
                "colorbar": {"title": "Rel. Intensity", "titleside": "right", "thickness": 12, "len": 0.75},
                "line": {"color": "#ffffff", "width": 1.2},
            },
            "text": labels,
            "textposition": "top center",
            "textfont": {"color": "#f8fafc", "size": 11, "family": "monospace"},
            "name": "Diffraction Spots",
            "hoverinfo": "text",
            "hovertext": hover,
        })

    # Determine plot bounds
    max_d = max([abs(x) for x in ([0.0] + [r["detector_x_mm"] for r in active])] + [abs(y) for y in ([0.0] + [r["detector_y_mm"] for r in active])])
    bound = max(50.0, max_d * 1.2)

    # 2-theta concentric ring indicators (10 deg, 20 deg, 30 deg, 40 deg...)
    shapes = []
    det_dist = sim_result.get("detector_distance_mm", 100.0)
    for angle_deg in [10, 20, 30, 45, 60]:
        if angle_deg < 85:
            r_mm = det_dist * np.tan(np.radians(angle_deg))
            if r_mm <= bound:
                shapes.append({
                    "type": "circle",
                    "xref": "x", "yref": "y",
                    "x0": -r_mm, "y0": -r_mm,
                    "x1": r_mm, "y1": r_mm,
                    "line": {"color": "rgba(148, 163, 184, 0.2)", "dash": "dot", "width": 1},
                })

    layout = {
        "title": {
            "text": f"2D Simulated Detector Screen Spot Pattern — Distance: {det_dist:.0f} mm",
            "font": {"size": 16},
        },
        "xaxis": {
            "title": "Detector X (mm)",
            "range": [-bound, bound],
            "gridcolor": "rgba(229, 231, 235, 0.15)",
            "zerolinecolor": "rgba(255, 255, 255, 0.2)",
            "scaleanchor": "y",
            "scaleratio": 1,
        },
        "yaxis": {
            "title": "Detector Y (mm)",
            "range": [-bound, bound],
            "gridcolor": "rgba(229, 231, 235, 0.15)",
            "zerolinecolor": "rgba(255, 255, 255, 0.2)",
        },
        "shapes": shapes,
        "margin": {"l": 55, "r": 20, "t": 45, "b": 45},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(15, 23, 42, 0.8)",
        "font": {"color": "#94a3b8", "family": "system-ui, sans-serif"},
    }

    return {"data": traces, "layout": layout}

