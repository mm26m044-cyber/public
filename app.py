"""
app.py
------
Band Explorer & Ewald Sphere Diffraction Web Application (v1.0)

A lightweight, zero-dependency web application powered by Python's http.server
and modern Plotly.js interactive charts.

Features:
  - Input any chemical element/formula (e.g. Si, GaAs, Al, Cu) or Material ID (e.g. mp-149)
  - Unified Electronic Structure: Band structure, DOS, conduction valleys, and carrier mobility
  - Interactive 3D Ewald Sphere & 2D Simulated Detector Spot Pattern
  - Universal Structure Factor (F_hkl) and systematic absence extinction rules
  - Automatic saving of calculation & diffraction data (JSON and CSV in data/processed/)

Usage:
    python app.py
    (Then open http://localhost:8050 in your browser)
"""

import os
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

from fetch_data import get_or_fetch, list_cached_materials, get_cached_file_path, is_cached
from analyzer import analyze_material, save_analysis_outputs
from visualizer import (
    generate_plotly_combined_dict,
    generate_plotly_bandstructure_dict,
    generate_plotly_dos_dict,
    generate_plotly_ewald_3d,
    generate_plotly_diffraction_2d,
)
from ewald_sphere import simulate_ewald_diffraction, RADIATION_PRESETS, electron_energy_to_wavelength

PORT = 8050

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Band Explorer & Ewald Sphere Simulator</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    body { background-color: #0f172a; color: #f1f5f9; font-family: system-ui, -apple-system, sans-serif; }
    .glass-card { background: rgba(30, 41, 59, 0.85); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 1rem; }
    .tab-btn.active { background: #3b82f6; color: white; }
    .mode-tab.active { background: #2563eb; color: white; border-color: #3b82f6; }
    .custom-scrollbar::-webkit-scrollbar { width: 6px; height: 6px; }
    .custom-scrollbar::-webkit-scrollbar-thumb { background: #475569; border-radius: 4px; }
  </style>
</head>
<body class="min-h-screen flex flex-col p-4 md:p-8">

  <!-- Header -->
  <header class="max-w-7xl w-full mx-auto mb-6 flex flex-col md:flex-row items-center justify-between gap-4">
    <div>
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center text-white text-xl shadow-lg shadow-blue-500/30">
          <i class="fa-solid fa-atom"></i>
        </div>
        <h1 class="text-2xl md:text-3xl font-bold tracking-tight text-white">Band Explorer <span class="text-xs px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30">v1.0</span></h1>
      </div>
      <p class="text-slate-400 text-sm mt-1">Electronic Structure, Valley Minima, and 3D Ewald Sphere Diffraction</p>
    </div>
    <div class="flex items-center gap-3">
      <span id="statusBadge" class="text-xs px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 flex items-center gap-1.5">
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> Ready
      </span>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-7xl w-full mx-auto flex-1 flex flex-col gap-6">

    <!-- Search Section -->
    <section class="glass-card p-5 shadow-xl">
      <div class="flex flex-col md:flex-row gap-3">
        <div class="relative flex-1">
          <i class="fa-solid fa-magnifying-glass absolute left-4 top-1/2 -translate-y-1/2 text-slate-400"></i>
          <input type="text" id="searchInput" placeholder="Enter Element, Formula, or Material ID (e.g. Si, Al, GaAs, Cu, mp-149)..." 
                 class="w-full bg-slate-800/80 border border-slate-700 rounded-xl pl-11 pr-4 py-3 text-white placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 text-base"
                 value="mp-149" onkeydown="if(event.key==='Enter') loadMaterial();">
        </div>
        <button onclick="loadMaterial()" id="searchBtn" class="bg-blue-600 hover:bg-blue-500 text-white font-medium px-6 py-3 rounded-xl transition flex items-center justify-center gap-2 shadow-lg shadow-blue-600/30">
          <i class="fa-solid fa-bolt"></i> <span>Analyze</span>
        </button>
      </div>

      <!-- Quick Preset Chips & Cached Materials -->
      <div class="mt-4 flex flex-wrap items-center gap-2 text-xs">
        <span class="text-slate-400 mr-1"><i class="fa-regular fa-lightbulb"></i> Presets:</span>
        <button onclick="setQuery('Si')" class="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition">Si (Semiconductor)</button>
        <button onclick="setQuery('GaAs')" class="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition">GaAs (Direct Gap)</button>
        <button onclick="setQuery('Al')" class="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition">Al (Metal)</button>
        <button onclick="setQuery('Cu')" class="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition">Cu (Metal)</button>
        <button onclick="setQuery('Ge')" class="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition">Ge (Semiconductor)</button>
        
        <div id="cachedBadges" class="ml-auto flex items-center gap-1.5 text-slate-400">
          <!-- Cached items injected dynamically -->
        </div>
      </div>
    </section>

    <!-- Top Navigation Tabs (Mode Switcher) -->
    <div class="flex items-center gap-3 border-b border-slate-700/60 pb-2">
      <button onclick="switchMainMode('electronic')" id="modeElectronicBtn" class="mode-tab active px-4 py-2.5 rounded-xl font-bold text-sm transition flex items-center gap-2 border border-slate-700 bg-slate-800 text-slate-300">
        <i class="fa-solid fa-chart-line text-blue-400"></i> Electronic Band Structure & Valleys
      </button>
      <button onclick="switchMainMode('ewald')" id="modeEwaldBtn" class="mode-tab px-4 py-2.5 rounded-xl font-bold text-sm transition flex items-center gap-2 border border-slate-700 bg-slate-800 text-slate-300">
        <i class="fa-solid fa-globe text-emerald-400"></i> Ewald Sphere & 3D/2D Diffraction
      </button>
    </div>

    <!-- Loading Spinner -->
    <div id="loadingIndicator" class="hidden glass-card p-12 text-center my-4">
      <div class="inline-block animate-spin w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full mb-3"></div>
      <p class="text-slate-300 font-medium text-lg">Computing electronic & diffraction physics...</p>
      <p class="text-slate-500 text-sm mt-1">Solving band structure, reciprocal lattice, and Ewald sphere geometry...</p>
    </div>

    <!-- Error Banner -->
    <div id="errorBanner" class="hidden bg-rose-500/20 border border-rose-500/40 rounded-xl p-4 text-rose-300 flex items-center gap-3">
      <i class="fa-solid fa-triangle-exclamation text-xl"></i>
      <span id="errorMessage">Error message here</span>
    </div>

    <!-- ==================== MODE 1: ELECTRONIC STRUCTURE ==================== -->
    <div id="electronicView" class="flex flex-col gap-6">

      <!-- Summary KPI Cards -->
      <section class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 md:gap-4">
        
        <!-- Formula & ID -->
        <div class="glass-card p-4 flex flex-col justify-between">
          <span class="text-xs text-slate-400 font-medium uppercase tracking-wider">Material</span>
          <div class="mt-2">
            <h3 id="cardFormula" class="text-xl font-bold text-white tracking-wide">--</h3>
            <p id="cardId" class="text-xs text-slate-400 mt-0.5">--</p>
          </div>
        </div>

        <!-- Classification -->
        <div class="glass-card p-4 flex flex-col justify-between">
          <span class="text-xs text-slate-400 font-medium uppercase tracking-wider">Type</span>
          <div class="mt-2">
            <h3 id="cardType" class="text-lg font-bold text-blue-400">--</h3>
            <p id="cardTypeDetail" class="text-xs text-slate-400 mt-0.5">--</p>
          </div>
        </div>

        <!-- Band Gap -->
        <div class="glass-card p-4 flex flex-col justify-between">
          <span class="text-xs text-slate-400 font-medium uppercase tracking-wider">Band Gap</span>
          <div class="mt-2">
            <div class="flex items-baseline gap-1">
              <h3 id="cardBandGap" class="text-2xl font-black text-white">--</h3>
              <span class="text-xs text-slate-400">eV</span>
            </div>
            <p id="cardGapNature" class="text-xs text-slate-400 mt-0.5">--</p>
          </div>
        </div>

        <!-- Space Group -->
        <div class="glass-card p-4 flex flex-col justify-between">
          <span class="text-xs text-slate-400 font-medium uppercase tracking-wider">Symmetry</span>
          <div class="mt-2">
            <h3 id="cardSpacegroup" class="text-lg font-bold text-white">--</h3>
            <p id="cardCrystalSys" class="text-xs text-slate-400 mt-0.5">--</p>
          </div>
        </div>

        <!-- Fermi Energy -->
        <div class="glass-card p-4 flex flex-col justify-between">
          <span class="text-xs text-slate-400 font-medium uppercase tracking-wider">Fermi Level E<sub>F</sub></span>
          <div class="mt-2">
            <div class="flex items-baseline gap-1">
              <h3 id="cardFermi" class="text-xl font-bold text-white">--</h3>
              <span class="text-xs text-slate-400">eV</span>
            </div>
            <p class="text-xs text-slate-400 mt-0.5">DFT Reference</p>
          </div>
        </div>

        <!-- DOS at E_F -->
        <div class="glass-card p-4 flex flex-col justify-between">
          <span class="text-xs text-slate-400 font-medium uppercase tracking-wider">DOS at E<sub>F</sub></span>
          <div class="mt-2">
            <div class="flex items-baseline gap-1">
              <h3 id="cardDosFermi" class="text-xl font-bold text-white">--</h3>
              <span class="text-xs text-slate-400">st/eV</span>
            </div>
            <p class="text-xs text-slate-400 mt-0.5">N(E<sub>F</sub>)</p>
          </div>
        </div>

      </section>

      <!-- Plot Section -->
      <section class="glass-card p-5 shadow-xl flex flex-col gap-4">
        <!-- Plot View Controls -->
        <div class="flex flex-wrap items-center justify-between gap-3 border-b border-slate-700/60 pb-3">
          <div class="flex items-center gap-2">
            <span class="text-xs text-slate-400 font-medium mr-2"><i class="fa-solid fa-chart-line"></i> View:</span>
            <button onclick="switchPlotTab('combined')" id="tabCombined" class="tab-btn active px-3 py-1.5 rounded-lg text-xs font-semibold transition bg-slate-800 text-slate-300">Combined (Band + DOS)</button>
            <button onclick="switchPlotTab('band')" id="tabBand" class="tab-btn px-3 py-1.5 rounded-lg text-xs font-semibold transition bg-slate-800 text-slate-300">Band Structure</button>
            <button onclick="switchPlotTab('dos')" id="tabDos" class="tab-btn px-3 py-1.5 rounded-lg text-xs font-semibold transition bg-slate-800 text-slate-300">DOS</button>
          </div>
          
          <div class="flex items-center gap-3">
            <label class="text-xs text-slate-400 flex items-center gap-2">
              Window: ±<span id="windowVal">4</span> eV
              <input type="range" id="windowSlider" min="2" max="10" step="1" value="4" class="w-20 accent-blue-500" oninput="document.getElementById('windowVal').innerText = this.value; renderActivePlot();">
            </label>
            <button onclick="downloadCombinedWhitePNG()" class="px-2.5 py-1.5 rounded-lg text-xs font-semibold transition bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 flex items-center gap-1.5 shadow" title="Download publication-quality side-by-side PNG with white background">
              <i class="fa-solid fa-file-image text-emerald-400"></i> Download PNG (White BG)
            </button>
          </div>
        </div>

        <!-- Interactive Plotly Chart Container -->
        <div id="plotContainer" class="w-full h-[520px] rounded-lg overflow-hidden bg-slate-900/50"></div>
      </section>

      <!-- Physics & Valley Breakdown / Metal Details Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">

        <!-- Left 2 Cols: Valleys or Metal Information -->
        <section class="lg:col-span-2 glass-card p-5 shadow-xl">
          <div class="flex items-center justify-between mb-4 border-b border-slate-700/60 pb-3">
            <h2 class="text-base font-bold text-white flex items-center gap-2">
              <i class="fa-solid fa-wave-square text-blue-400"></i>
              <span id="valleySectionTitle">Conduction Valleys & Band Extrema</span>
            </h2>
            <span id="valleyCountBadge" class="text-xs px-2.5 py-0.5 rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30"></span>
          </div>

          <!-- Semiconductor Valley Table -->
          <div id="valleyTableContainer" class="overflow-x-auto custom-scrollbar">
            <table class="w-full text-left text-xs text-slate-300">
              <thead class="bg-slate-800/80 text-slate-400 uppercase font-semibold">
                <tr>
                  <th class="p-2.5 rounded-l-lg">Valley</th>
                  <th class="p-2.5">Valley Gap (eV)</th>
                  <th class="p-2.5">Energy (eV)</th>
                  <th class="p-2.5">k-Coords (frac)</th>
                  <th class="p-2.5">m* (m<sub>0</sub>)</th>
                  <th class="p-2.5 rounded-r-lg">Mobility μ (cm²/V·s)</th>
                </tr>
              </thead>
              <tbody id="valleyTableBody" class="divide-y divide-slate-800">
                <!-- Rows injected dynamically -->
              </tbody>
            </table>
          </div>

          <!-- Metal Details Box (Visible only for metals) -->
          <div id="metalInfoBox" class="hidden p-4 rounded-xl bg-slate-800/60 border border-slate-700 text-sm text-slate-300 space-y-2">
            <div class="flex items-center gap-2 text-amber-400 font-semibold">
              <i class="fa-solid fa-circle-info"></i> Metallic System Physics
            </div>
            <p>This material has continuous bands crossing the Fermi level with <strong class="text-white">zero band gap</strong>.</p>
            <ul class="list-disc list-inside text-slate-400 space-y-1 text-xs">
              <li>High density of available electronic states at the Fermi level: <span id="metalDosValue" class="text-blue-300 font-mono">--</span> states/eV/cell.</li>
              <li>Conduction valley gaps and direct/indirect band transitions are not applicable to metals.</li>
              <li>Electronic transport is dominated by Fermi surface intra-band scattering rather than activated band transitions.</li>
            </ul>
          </div>
        </section>

        <!-- Right 1 Col: Mobility Calculator & Saved Output Box -->
        <section class="glass-card p-5 shadow-xl flex flex-col justify-between gap-4">
          
          <!-- Interactive Mobility Calculator -->
          <div>
            <div class="flex items-center justify-between mb-3 border-b border-slate-700/60 pb-2">
              <h3 class="text-sm font-bold text-white flex items-center gap-2">
                <i class="fa-solid fa-calculator text-purple-400"></i> Mobility Estimator
              </h3>
              <span class="text-xs text-slate-400">Drude Model</span>
            </div>

            <div class="space-y-3 text-xs">
              <div>
                <div class="flex justify-between text-slate-400 mb-1">
                  <span>Scattering Time (τ):</span>
                  <span id="tauDisplay" class="font-mono text-purple-300 font-bold">0.20 ps</span>
                </div>
                <input type="range" id="tauSlider" min="0.01" max="1.0" step="0.01" value="0.2" class="w-full accent-purple-500" oninput="updateMobilityFromTau(this.value)">
              </div>

              <div class="p-3 rounded-lg bg-slate-800/70 border border-slate-700/60 space-y-1.5">
                <div class="flex justify-between">
                  <span class="text-slate-400">Longitudinal m*:</span>
                  <span id="calcMass" class="font-mono text-white font-bold">--</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-slate-400">Estimated μ:</span>
                  <span id="calcMobility" class="font-mono text-emerald-400 font-bold">--</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Data Storage / Output Card -->
          <div class="border-t border-slate-700/60 pt-3">
            <h4 class="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <i class="fa-solid fa-floppy-disk text-emerald-400"></i> Stored Output Data
            </h4>
            <div class="text-[11px] text-slate-400 space-y-1">
              <div class="flex items-center justify-between">
                <span>Summary JSON:</span>
                <a id="downloadJsonLink" href="#" target="_blank" class="text-blue-400 hover:underline font-mono">summary.json</a>
              </div>
              <div class="flex items-center justify-between" id="csvRow">
                <span>Valley Table CSV:</span>
                <a id="downloadCsvLink" href="#" target="_blank" class="text-blue-400 hover:underline font-mono">valleys.csv</a>
              </div>
              <div class="flex items-center justify-between">
                <span>Combined Plot (White BG):</span>
                <button onclick="downloadCombinedWhitePNG()" class="text-emerald-400 hover:text-emerald-300 hover:underline font-mono flex items-center gap-1">
                  <i class="fa-solid fa-file-image"></i> combined.png
                </button>
              </div>
            </div>
          </div>

        </section>

      </div>

    </div>

    <!-- ==================== MODE 2: EWALD SPHERE & DIFFRACTION ==================== -->
    <div id="ewaldView" class="hidden flex flex-col gap-6">

      <!-- Ewald Parameter Control Bar -->
      <section class="glass-card p-5 shadow-xl grid grid-cols-1 md:grid-cols-4 gap-4 text-xs">
        
        <!-- Radiation Source -->
        <div>
          <label class="font-semibold text-slate-300 block mb-1.5"><i class="fa-solid fa-radiation text-amber-400 mr-1"></i> Radiation Source:</label>
          <select id="radSelect" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white focus:outline-none focus:border-blue-500" onchange="onRadiationChange()">
            <option value="1.5406">Cu Kα X-ray (1.5406 Å)</option>
            <option value="0.7107">Mo Kα X-ray (0.7107 Å)</option>
            <option value="2.2897">Cr Kα X-ray (2.2897 Å)</option>
            <option value="1.7890">Co Kα X-ray (1.7890 Å)</option>
            <option value="0.02508">200 keV Electron / TEM (0.0251 Å)</option>
            <option value="0.03701">100 keV Electron / TEM (0.0370 Å)</option>
            <option value="custom">Custom Wavelength (Å)...</option>
          </select>
          <div id="customWlRow" class="hidden mt-2 flex items-center gap-1.5">
            <input type="number" id="customWlInput" step="0.001" value="1.5406" class="w-full bg-slate-800 border border-slate-700 rounded p-1.5 text-white" placeholder="λ in Å">
            <button onclick="recalculateEwald()" class="px-2 py-1.5 bg-blue-600 rounded text-white">Apply</button>
          </div>
        </div>

        <!-- Incident Beam Direction [u v w] -->
        <div>
          <label class="font-semibold text-slate-300 block mb-1.5"><i class="fa-solid fa-compass text-blue-400 mr-1"></i> Beam Direction [u v w]:</label>
          <div class="grid grid-cols-3 gap-1.5">
            <input type="number" id="beamU" value="0" class="bg-slate-800 border border-slate-700 rounded-lg p-2 text-center text-white" onchange="recalculateEwald()">
            <input type="number" id="beamV" value="0" class="bg-slate-800 border border-slate-700 rounded-lg p-2 text-center text-white" onchange="recalculateEwald()">
            <input type="number" id="beamW" value="1" class="bg-slate-800 border border-slate-700 rounded-lg p-2 text-center text-white" onchange="recalculateEwald()">
          </div>
          <div class="flex gap-1 mt-1.5 text-[10px]">
            <button onclick="setBeamDir(0,0,1)" class="px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700">[001]</button>
            <button onclick="setBeamDir(1,1,0)" class="px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700">[110]</button>
            <button onclick="setBeamDir(1,1,1)" class="px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700">[111]</button>
            <button onclick="setBeamDir(1,0,0)" class="px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700">[100]</button>
          </div>
        </div>

        <!-- Tolerance / Mosaicity Slider -->
        <div>
          <div class="flex justify-between font-semibold text-slate-300 mb-1.5">
            <span><i class="fa-solid fa-layer-group text-purple-400 mr-1"></i> Tolerance (Δk/k₀):</span>
            <span id="tolVal" class="text-purple-300 font-mono">4.0%</span>
          </div>
          <input type="range" id="tolSlider" min="1" max="10" step="0.5" value="4" class="w-full accent-purple-500 mt-2" oninput="document.getElementById('tolVal').innerText = parseFloat(this.value).toFixed(1)+'%';" onchange="recalculateEwald()">
          <span class="text-[10px] text-slate-400">Models beam divergence & crystal mosaicity</span>
        </div>

        <!-- Detector Distance & Range -->
        <div>
          <div class="flex justify-between font-semibold text-slate-300 mb-1.5">
            <span><i class="fa-solid fa-ruler-combined text-emerald-400 mr-1"></i> Detector Distance:</span>
            <span id="detDistVal" class="text-emerald-300 font-mono">100 mm</span>
          </div>
          <input type="range" id="detDistSlider" min="50" max="300" step="10" value="100" class="w-full accent-emerald-500 mt-2" oninput="document.getElementById('detDistVal').innerText = this.value + ' mm';" onchange="recalculateEwald()">
          <span class="text-[10px] text-slate-400">Distance to simulated 2D screen</span>
        </div>

      </section>

      <!-- Ewald Plots Section -->
      <section class="glass-card p-5 shadow-xl flex flex-col gap-4">
        
        <div class="flex flex-wrap items-center justify-between gap-3 border-b border-slate-700/60 pb-3">
          <div class="flex items-center gap-2">
            <span class="text-xs text-slate-400 font-medium mr-2"><i class="fa-solid fa-eye"></i> View:</span>
            <button onclick="switchEwaldTab('3d')" id="tabEwald3D" class="tab-btn active px-3 py-1.5 rounded-lg text-xs font-semibold transition bg-slate-800 text-slate-300">
              <i class="fa-solid fa-cube mr-1 text-blue-400"></i> 3D Ewald Sphere & Reciprocal Grid
            </button>
            <button onclick="switchEwaldTab('2d')" id="tabEwald2D" class="tab-btn px-3 py-1.5 rounded-lg text-xs font-semibold transition bg-slate-800 text-slate-300">
              <i class="fa-solid fa-circle-dot mr-1 text-emerald-400"></i> 2D Simulated Detector Screen (Spots)
            </button>
          </div>
          
          <!-- 3D Visual Layer Toggles -->
          <div id="ewald3DControls" class="flex flex-wrap items-center gap-3 text-xs text-slate-400">
            <label class="flex items-center gap-1.5 cursor-pointer hover:text-slate-200">
              <input type="checkbox" id="chkSurface" checked onchange="renderActiveEwaldPlot()" class="accent-blue-500 rounded"> Sphere Surface
            </label>
            <label class="flex items-center gap-1.5 cursor-pointer hover:text-slate-200">
              <input type="checkbox" id="chkBeams" checked onchange="renderActiveEwaldPlot()" class="accent-emerald-500 rounded"> Diffracted Rays (k_f)
            </label>
            <label class="flex items-center gap-1.5 cursor-pointer hover:text-slate-200">
              <input type="checkbox" id="chkLattice" onchange="renderActiveEwaldPlot()" class="accent-slate-500 rounded"> Inactive Lattice
            </label>
            <label class="flex items-center gap-1.5 cursor-pointer hover:text-slate-200">
              <input type="checkbox" id="chkForbidden" onchange="renderActiveEwaldPlot()" class="accent-slate-500 rounded"> Extinctions (F=0)
            </label>
            <span id="ewaldReflCount" class="px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 font-semibold">0 Active Reflections</span>
          </div>
        </div>

        <!-- Plotly Container for Ewald 3D / 2D -->
        <div id="ewaldPlotContainer" class="w-full h-[540px] rounded-lg overflow-hidden bg-slate-900/50"></div>
      </section>

      <!-- Ewald Active Reflections Table & Export -->
      <section class="glass-card p-5 shadow-xl">
        <div class="flex items-center justify-between mb-4 border-b border-slate-700/60 pb-3">
          <div>
            <h2 class="text-base font-bold text-white flex items-center gap-2">
              <i class="fa-solid fa-table-list text-emerald-400"></i> Active Bragg Diffraction Reflections
            </h2>
            <p class="text-xs text-slate-400 mt-0.5">Satisfying Ewald sphere intersection (||kᵢ + G|| ≈ k₀) and structure factor F(hkl) ≠ 0</p>
          </div>
          <a id="downloadDiffractionCsvLink" href="#" target="_blank" class="text-xs px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium transition flex items-center gap-1.5 shadow">
            <i class="fa-solid fa-download"></i> Export CSV
          </a>
        </div>

        <div class="overflow-x-auto custom-scrollbar max-h-72">
          <table class="w-full text-left text-xs text-slate-300">
            <thead class="bg-slate-800/80 text-slate-400 uppercase font-semibold sticky top-0">
              <tr>
                <th class="p-2.5 rounded-l-lg">Plane (h k l)</th>
                <th class="p-2.5">d-spacing (Å)</th>
                <th class="p-2.5">Bragg 2θ (deg)</th>
                <th class="p-2.5">|F_hkl|² (Structure Factor)</th>
                <th class="p-2.5">Relative Intensity</th>
                <th class="p-2.5 rounded-r-lg">Detector Position (X, Y) mm</th>
              </tr>
            </thead>
            <tbody id="diffractionTableBody" class="divide-y divide-slate-800">
              <!-- Rows injected dynamically -->
            </tbody>
          </table>
        </div>
      </section>

    </div>

  </main>

  <footer class="max-w-7xl w-full mx-auto mt-8 text-center text-xs text-slate-500">
    Band Explorer v1.0 • Computational Solid-State Physics & Electronic Structure Analytics
  </footer>

  <script>
    let currentAnalysis = null;
    let currentEwaldData = null;
    let activePlotTab = 'combined';
    let activeEwaldTab = '3d';
    let currentMainMode = 'electronic';

    function setQuery(q) {
      document.getElementById('searchInput').value = q;
      loadMaterial();
    }

    function switchMainMode(mode) {
      currentMainMode = mode;
      document.querySelectorAll('.mode-tab').forEach(b => b.classList.remove('active'));
      if (mode === 'electronic') {
        document.getElementById('modeElectronicBtn').classList.add('active');
        document.getElementById('electronicView').classList.remove('hidden');
        document.getElementById('ewaldView').classList.add('hidden');
        renderActivePlot();
      } else {
        document.getElementById('modeEwaldBtn').classList.add('active');
        document.getElementById('electronicView').classList.add('hidden');
        document.getElementById('ewaldView').classList.remove('hidden');
        renderActiveEwaldPlot();
      }
    }

    function switchPlotTab(tab) {
      activePlotTab = tab;
      document.querySelectorAll('.tab-btn').forEach(b => {
        if (['tabCombined', 'tabBand', 'tabDos'].includes(b.id)) b.classList.remove('active');
      });
      if (tab === 'combined') document.getElementById('tabCombined').classList.add('active');
      if (tab === 'band') document.getElementById('tabBand').classList.add('active');
      if (tab === 'dos') document.getElementById('tabDos').classList.add('active');
      renderActivePlot();
    }

    function switchEwaldTab(tab) {
      activeEwaldTab = tab;
      document.getElementById('tabEwald3D').classList.toggle('active', tab === '3d');
      document.getElementById('tabEwald2D').classList.toggle('active', tab === '2d');
      renderActiveEwaldPlot();
    }

    function onRadiationChange() {
      const val = document.getElementById('radSelect').value;
      const customRow = document.getElementById('customWlRow');
      if (val === 'custom') {
        customRow.classList.remove('hidden');
      } else {
        customRow.classList.add('hidden');
        recalculateEwald();
      }
    }

    function setBeamDir(u, v, w) {
      document.getElementById('beamU').value = u;
      document.getElementById('beamV').value = v;
      document.getElementById('beamW').value = w;
      recalculateEwald();
    }

    async function loadCachedList() {
      try {
        const res = await fetch('/api/cached');
        const data = await res.json();
        const container = document.getElementById('cachedBadges');
        if (data.materials && data.materials.length > 0) {
          container.innerHTML = '<span class="mr-1">Cached:</span>' + data.materials.map(m => 
            `<button onclick="setQuery('${m.material_id}')" class="px-2 py-0.5 rounded bg-blue-950 text-blue-300 border border-blue-800 hover:bg-blue-900">${m.formula} (${m.material_id})</button>`
          ).join(' ');
        }
      } catch (e) {
        console.error('Failed to load cached materials:', e);
      }
    }

    async function loadMaterial() {
      const q = document.getElementById('searchInput').value.trim();
      if (!q) return;

      document.getElementById('loadingIndicator').classList.remove('hidden');
      document.getElementById('errorBanner').classList.add('hidden');
      document.getElementById('statusBadge').innerHTML = '<span class="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span> Computing...';

      try {
        const res = await fetch(`/api/material?q=${encodeURIComponent(q)}`);
        const data = await res.json();

        if (data.error) {
          throw new Error(data.error);
        }

        currentAnalysis = data;
        populateUI(data);
        renderActivePlot();
        loadCachedList();

        // Also run initial Ewald simulation
        await recalculateEwald();

        document.getElementById('statusBadge').innerHTML = '<span class="w-2 h-2 rounded-full bg-emerald-400"></span> Ready';
      } catch (err) {
        document.getElementById('errorMessage').innerText = err.message || 'An error occurred while analyzing material.';
        document.getElementById('errorBanner').classList.remove('hidden');
        document.getElementById('statusBadge').innerHTML = '<span class="w-2 h-2 rounded-full bg-rose-400"></span> Error';
      } finally {
        document.getElementById('loadingIndicator').classList.add('hidden');
      }
    }

    async function recalculateEwald() {
      if (!currentAnalysis) return;
      const q = currentAnalysis.material_id;
      
      let wl = parseFloat(document.getElementById('radSelect').value);
      if (document.getElementById('radSelect').value === 'custom') {
        wl = parseFloat(document.getElementById('customWlInput').value) || 1.5406;
      }
      
      const u = parseFloat(document.getElementById('beamU').value) || 0;
      const v = parseFloat(document.getElementById('beamV').value) || 0;
      const w = parseFloat(document.getElementById('beamW').value) || 1;
      const tol = (parseFloat(document.getElementById('tolSlider').value) || 4.0) / 100.0;
      const detDist = parseFloat(document.getElementById('detDistSlider').value) || 100.0;

      try {
        const url = `/api/ewald?q=${encodeURIComponent(q)}&wl=${wl}&u=${u}&v=${v}&w=${w}&tol=${tol}&dist=${detDist}`;
        const res = await fetch(url);
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        currentEwaldData = data;
        populateEwaldUI(data);
        renderActiveEwaldPlot();
      } catch (err) {
        console.error("Ewald calculation failed:", err);
      }
    }

    function populateUI(data) {
      const p = data.analysis;
      document.getElementById('cardFormula').innerText = p.formula || p.material_id;
      document.getElementById('cardId').innerText = p.material_id;

      document.getElementById('cardType').innerText = p.classification;
      document.getElementById('cardTypeDetail').innerText = p.is_metal ? 'Metallic conduction' : (p.band_gap_type + ' band gap');
      
      document.getElementById('cardBandGap').innerText = p.is_metal ? '0.00' : p.band_gap_ev.toFixed(3);
      document.getElementById('cardGapNature').innerText = p.is_metal ? 'No band gap' : (p.transition || 'Direct');

      document.getElementById('cardSpacegroup').innerText = p.spacegroup_symbol || 'Unknown';
      document.getElementById('cardCrystalSys').innerText = p.spacegroup_number ? `SG #${p.spacegroup_number}` : '';

      document.getElementById('cardFermi').innerText = p.efermi_bs_ev ? p.efermi_bs_ev.toFixed(2) : '--';
      document.getElementById('cardDosFermi').innerText = p.dos_at_fermi ? p.dos_at_fermi.toFixed(2) : '0.00';

      // Update storage download links
      document.getElementById('downloadJsonLink').href = `/data/processed/${p.material_id}_summary.json`;
      document.getElementById('downloadJsonLink').innerText = `${p.material_id}_summary.json`;

      const csvLink = document.getElementById('downloadCsvLink');
      const csvRow = document.getElementById('csvRow');
      if (!p.is_metal && Object.keys(p.valleys || {}).length > 0) {
        csvRow.classList.remove('hidden');
        csvLink.href = `/data/processed/${p.material_id}_valleys.csv`;
        csvLink.innerText = `${p.material_id}_valleys.csv`;
      } else {
        csvRow.classList.add('hidden');
      }

      // Valleys vs Metal Box
      const valleyTableContainer = document.getElementById('valleyTableContainer');
      const metalInfoBox = document.getElementById('metalInfoBox');
      const valleyTableBody = document.getElementById('valleyTableBody');
      const valleyCountBadge = document.getElementById('valleyCountBadge');

      if (p.is_metal) {
        valleyTableContainer.classList.add('hidden');
        metalInfoBox.classList.remove('hidden');
        document.getElementById('metalDosValue').innerText = (p.dos_at_fermi || 0).toFixed(2);
        valleyCountBadge.innerText = 'Metallic State';
        document.getElementById('calcMass').innerText = 'N/A (Metal)';
        document.getElementById('calcMobility').innerText = 'N/A (Metal)';
      } else {
        metalInfoBox.classList.add('hidden');
        valleyTableContainer.classList.remove('hidden');
        
        const valleys = Object.values(p.valleys || {});
        valleyCountBadge.innerText = `${valleys.length} Valley${valleys.length === 1 ? '' : 's'} Found`;
        
        valleyTableBody.innerHTML = valleys.map(v => {
          const kStr = v.frac_coords ? `[${v.frac_coords.map(c => c.toFixed(2)).join(', ')}]` : '--';
          const massStr = v.effective_mass_m0 ? v.effective_mass_m0.toFixed(3) : '--';
          const mobStr = v.mobility_cm2_vs ? Math.round(v.mobility_cm2_vs).toLocaleString() : '--';
          return `
            <tr class="hover:bg-slate-800/50 transition">
              <td class="p-2.5 font-semibold text-white">${v.name}</td>
              <td class="p-2.5 text-blue-400 font-mono font-bold">${v.gap_ev.toFixed(3)}</td>
              <td class="p-2.5 font-mono">${v.energy_ev.toFixed(3)}</td>
              <td class="p-2.5 font-mono text-slate-400">${kStr}</td>
              <td class="p-2.5 font-mono text-purple-300 font-semibold">${massStr}</td>
              <td class="p-2.5 font-mono text-emerald-400 font-semibold">${mobStr}</td>
            </tr>
          `;
        }).join('');

        updateMobilityFromTau(document.getElementById('tauSlider').value);
      }
    }

    function populateEwaldUI(data) {
      const sim = data.simulation;
      const count = sim.num_diffracting || 0;
      document.getElementById('ewaldReflCount').innerText = `${count} Active Reflection${count === 1 ? '' : 's'}`;
      document.getElementById('downloadDiffractionCsvLink').href = `/data/processed/${sim.material_id}_diffraction.csv`;

      const tbody = document.getElementById('diffractionTableBody');
      if (!sim.active_reflections || sim.active_reflections.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" class="p-4 text-center text-slate-500">No reflections intersect the Ewald sphere at this orientation & wavelength. Try increasing the tolerance slider or switching beam direction.</td></tr>`;
        return;
      }

      tbody.innerHTML = sim.active_reflections.map(r => {
        const detStr = (r.detector_x_mm !== null) ? `(${r.detector_x_mm.toFixed(1)}, ${r.detector_y_mm.toFixed(1)})` : 'Backscatter';
        const intPct = Math.round(r.intensity_rel * 100);
        return `
          <tr class="hover:bg-slate-800/50 transition">
            <td class="p-2.5 font-mono font-bold text-emerald-400">${r.hkl_str}</td>
            <td class="p-2.5 font-mono">${r.d_spacing_angstrom.toFixed(3)}</td>
            <td class="p-2.5 font-mono text-amber-300">${r.two_theta_deg.toFixed(2)}°</td>
            <td class="p-2.5 font-mono text-purple-300 font-bold">${r.f_sq.toFixed(1)}</td>
            <td class="p-2.5">
              <div class="flex items-center gap-2">
                <div class="w-16 bg-slate-700 rounded-full h-1.5 overflow-hidden">
                  <div class="bg-emerald-400 h-1.5 rounded-full" style="width: ${intPct}%"></div>
                </div>
                <span class="font-mono text-xs text-slate-300">${(r.intensity_rel).toFixed(3)}</span>
              </div>
            </td>
            <td class="p-2.5 font-mono text-slate-400">${detStr}</td>
          </tr>
        `;
      }).join('');
    }

    function updateMobilityFromTau(tauPs) {
      document.getElementById('tauDisplay').innerText = `${parseFloat(tauPs).toFixed(2)} ps`;
      if (!currentAnalysis || currentAnalysis.analysis.is_metal) return;

      const p = currentAnalysis.analysis;
      const primaryMass = p.primary_effective_mass_m0;
      if (primaryMass) {
        document.getElementById('calcMass').innerText = `${primaryMass.toFixed(3)} m₀`;
        const mu = (1.602176634e-19 * (parseFloat(tauPs) * 1e-12)) / (primaryMass * 9.1093837e-31) * 1e4;
        document.getElementById('calcMobility').innerText = `≈ ${Math.round(mu).toLocaleString()} cm²/V·s`;
      } else {
        document.getElementById('calcMass').innerText = '--';
        document.getElementById('calcMobility').innerText = '--';
      }
    }

    function renderActivePlot() {
      if (!currentAnalysis || !currentAnalysis.plots) return;
      const win = parseFloat(document.getElementById('windowSlider').value);
      const plotData = currentAnalysis.plots[activePlotTab];
      if (!plotData) return;

      const plotObj = JSON.parse(JSON.stringify(plotData));
      if (plotObj.layout && plotObj.layout.yaxis) {
        plotObj.layout.yaxis.range = [-win, win];
      }

      plotObj.layout.paper_bgcolor = 'rgba(0,0,0,0)';
      plotObj.layout.plot_bgcolor = 'rgba(15, 23, 42, 0.6)';
      plotObj.layout.font = { color: '#94a3b8', family: 'system-ui, sans-serif' };

      Plotly.react('plotContainer', plotObj.data, plotObj.layout, {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
      });
    }

    function downloadCombinedWhitePNG() {
      if (!currentAnalysis || !currentAnalysis.plots || !currentAnalysis.plots.combined) {
        alert("Please load a material first.");
        return;
      }

      const win = parseFloat(document.getElementById('windowSlider').value) || 4.0;
      const plotObj = JSON.parse(JSON.stringify(currentAnalysis.plots.combined));

      // 1. Publication-grade crisp white background styling
      plotObj.layout.paper_bgcolor = '#ffffff';
      plotObj.layout.plot_bgcolor = '#ffffff';
      plotObj.layout.font = { color: '#0f172a', family: 'system-ui, -apple-system, sans-serif', size: 12 };

      const formulaName = currentAnalysis.analysis.formula || currentAnalysis.material_id;
      plotObj.layout.title = {
        text: `${formulaName} (${currentAnalysis.material_id}) — Electronic Band Structure & Density of States`,
        font: { color: '#0f172a', size: 18, weight: 'bold' }
      };

      if (plotObj.layout.yaxis) {
        plotObj.layout.yaxis.range = [-win, win];
        plotObj.layout.yaxis.gridcolor = '#e2e8f0';
        plotObj.layout.yaxis.titlefont = { color: '#0f172a', size: 13, weight: 'bold' };
        plotObj.layout.yaxis.tickfont = { color: '#0f172a', size: 11 };
        plotObj.layout.yaxis.zeroline = true;
        plotObj.layout.yaxis.zerolinecolor = '#94a3b8';
      }

      if (plotObj.layout.xaxis) {
        plotObj.layout.xaxis.gridcolor = '#f1f5f9';
        plotObj.layout.xaxis.tickfont = { color: '#0f172a', size: 12, weight: 'bold' };
      }

      if (plotObj.layout.xaxis2) {
        plotObj.layout.xaxis2.gridcolor = '#e2e8f0';
        plotObj.layout.xaxis2.titlefont = { color: '#0f172a', size: 13, weight: 'bold' };
        plotObj.layout.xaxis2.tickfont = { color: '#0f172a', size: 11 };
      }

      // 2. Adjust trace colors for high contrast on white
      plotObj.data.forEach(t => {
        if (t.line && (t.line.color === '#3b82f6' || !t.line.color)) {
          t.line.color = '#1d4ed8'; // Deeper royal blue for bands
          t.line.width = 1.6;
        } else if (t.name === 'Total DOS' || t.xaxis === 'x2') {
          if (t.line) t.line.color = '#1d4ed8';
          t.fillcolor = 'rgba(29, 78, 216, 0.18)';
        }
      });

      // 3. Shapes (Fermi level line & symmetry dividers)
      if (plotObj.layout.shapes) {
        plotObj.layout.shapes.forEach(s => {
          if (s.line && s.line.color === '#ef4444') {
            s.line.color = '#dc2626'; // High contrast red for E_F
            s.line.width = 1.5;
          } else if (s.line && s.line.color && s.line.color.includes('156')) {
            s.line.color = 'rgba(148, 163, 184, 0.7)';
          }
        });
      }

      // 4. Render to off-screen element and trigger high-DPI download
      const tempContainer = document.createElement('div');
      tempContainer.style.position = 'fixed';
      tempContainer.style.left = '-9999px';
      tempContainer.style.top = '-9999px';
      tempContainer.style.width = '1400px';
      tempContainer.style.height = '800px';
      document.body.appendChild(tempContainer);

      Plotly.newPlot(tempContainer, plotObj.data, plotObj.layout, { responsive: false }).then(() => {
        Plotly.downloadImage(tempContainer, {
          format: 'png',
          width: 1400,
          height: 800,
          filename: `${currentAnalysis.material_id}_${formulaName}_band_dos_white`
        }).then(() => {
          document.body.removeChild(tempContainer);
        });
      });
    }

    function renderActiveEwaldPlot() {
      if (!currentEwaldData || !currentEwaldData.plots) return;

      const ewald3dControls = document.getElementById('ewald3DControls');
      if (activeEwaldTab === '3d') {
        ewald3dControls.classList.remove('hidden');
      } else {
        ewald3dControls.classList.add('hidden');
      }

      const plotData = (activeEwaldTab === '3d') ? currentEwaldData.plots.ewald_3d : currentEwaldData.plots.diffraction_2d;
      if (!plotData) return;

      const plotObj = JSON.parse(JSON.stringify(plotData));
      plotObj.layout.paper_bgcolor = 'rgba(0,0,0,0)';
      plotObj.layout.font = { color: '#94a3b8', family: 'system-ui, sans-serif' };

      // Apply 3D layer filters
      if (activeEwaldTab === '3d' && plotObj.data) {
        const showSurface = document.getElementById('chkSurface').checked;
        const showBeams = document.getElementById('chkBeams').checked;
        const showLattice = document.getElementById('chkLattice').checked;
        const showForbidden = document.getElementById('chkForbidden').checked;

        plotObj.data.forEach(trace => {
          const name = trace.name || '';
          if (name.includes('Sphere') || name.includes('Equator')) {
            trace.visible = showSurface;
          } else if (name.includes('Diffracted Beams')) {
            trace.visible = showBeams;
          } else if (name.includes('Inactive Nodes')) {
            trace.visible = showLattice;
          } else if (name.includes('Forbidden')) {
            trace.visible = showForbidden;
          }
        });
      }

      Plotly.react('ewaldPlotContainer', plotObj.data, plotObj.layout, {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
      });
    }

    // Auto-load on page start
    window.addEventListener('DOMContentLoaded', () => {
      loadCachedList();
      loadMaterial();
    });
  </script>
</body>
</html>
"""


class BandExplorerRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # 1. API: List cached materials
        if path == "/api/cached":
            cached = list_cached_materials()
            self._send_json({"materials": cached})
            return

        # 2. API: Query/Analyze material (Electronic Structure)
        if path == "/api/material":
            q = query.get("q", ["mp-149"])[0].strip()
            try:
                data = get_or_fetch(q)
                analysis = analyze_material(data, save_output=True)

                bs = data.get("bandstructure")
                dos = data.get("dos")

                plots = {
                    "combined": generate_plotly_combined_dict(bs, dos),
                    "band": generate_plotly_bandstructure_dict(bs),
                    "dos": generate_plotly_dos_dict(dos) if dos else None,
                }

                self._send_json({
                    "material_id": data.get("material_id"),
                    "analysis": analysis,
                    "plots": plots,
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)
            return

        # 3. API: Ewald Sphere & Diffraction simulation
        if path == "/api/ewald":
            q = query.get("q", ["mp-149"])[0].strip()
            try:
                wl = float(query.get("wl", [1.5406])[0])
                u = float(query.get("u", [0.0])[0])
                v = float(query.get("v", [0.0])[0])
                w = float(query.get("w", [1.0])[0])
                tol = float(query.get("tol", [0.04])[0])
                dist = float(query.get("dist", [100.0])[0])

                data = get_or_fetch(q)
                structure = data.get("structure")
                if not structure:
                    raise ValueError(f"No crystal structure found for {q}")

                sim_result = simulate_ewald_diffraction(
                    structure=structure,
                    wavelength_angstrom=wl,
                    beam_direction=(u, v, w),
                    tolerance_rel=tol,
                    detector_distance_mm=dist,
                    material_id=data.get("material_id"),
                    save_output=True,
                )

                plots = {
                    "ewald_3d": generate_plotly_ewald_3d(sim_result),
                    "diffraction_2d": generate_plotly_diffraction_2d(sim_result),
                }

                self._send_json({
                    "simulation": sim_result,
                    "plots": plots,
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)
            return

        # 4. Static file downloads from data/
        if path.startswith("/data/"):
            clean_path = path.lstrip("/")
            if os.path.exists(clean_path):
                self.send_response(200)
                if clean_path.endswith(".json"):
                    self.send_header("Content-type", "application/json")
                elif clean_path.endswith(".csv"):
                    self.send_header("Content-type", "text/csv")
                elif clean_path.endswith(".png"):
                    self.send_header("Content-type", "image/png")
                else:
                    self.send_header("Content-type", "application/octet-stream")
                self.end_headers()
                with open(clean_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "File Not Found")
                return

        # 5. Root / Web App UI
        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        self.send_error(404, "Endpoint Not Found")

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[BandExplorer Web] {self.address_string()} - {args[0]}")


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def start_server(preferred_port: int = PORT):
    import argparse
    parser = argparse.ArgumentParser(description="Band Explorer & Ewald Sphere Web Server")
    parser.add_argument("--port", type=int, default=preferred_port, help="Port to run web server on")
    args, _ = parser.parse_known_args()

    port = args.port
    server = None

    for p in range(port, port + 10):
        try:
            server = ReusableHTTPServer(("0.0.0.0", p), BandExplorerRequestHandler)
            port = p
            break
        except OSError as e:
            if "Address already in use" in str(e) or getattr(e, "errno", None) == 48:
                continue
            raise

    if server is None:
        raise RuntimeError(f"Could not bind to any port in range {port}-{port + 10}. Please close previous instances.")

    print(f"\n=======================================================")
    print(f"  ⚡ Band Explorer & Ewald Sphere Simulator running at http://localhost:{port}")
    print(f"  Modes: 1) Electronic Band Structure  2) 3D Ewald Sphere & 2D Diffraction")
    print(f"  Calculated output data is automatically stored in data/processed/")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server cleanly...")
        server.server_close()


if __name__ == "__main__":
    start_server()
