# Petrel → Python QC Workflow

A Python workflow to QC upscaled reservoir simulation grid properties exported from Petrel to identifying connectivity problems before running a simulation.

## Background

In reservoir simulation, one of the most common early-stage problems is poor connectivity between wells and the geocellular model — often caused by incorrect perforations, production allocation issues, or poor reservoir characterization in the upscaled grid. (SPE 164420)

This workflow reads Petrel-exported grid properties directly into Python (.EGRID & .GRDECL), flags problematic cells based on porosity cutoffs, and visualizes the results in 3D without needing Petrel open.

## Series Progress

- ✅ Part 1: Read + Visualize upscaled porosity (posted on LinkedIn)
- ✅ Part 2: Property QC + connectivity flag map (posted on LinkedIn)
- ✅ Part 3: Modify PERMX + Export (posted on LinkedIn)

## Tools Used

- Python 3.10
- library: xtgeo - reading/writing subsurface grid properties
- PyVista - 3D visualization
- matplotlib - histograms and statistics

## Files

- `Part 1: Importing & Visualizing` (load grid + porosity, generate 3D visualization)
- `Part 2: QCing` (QC cutoffs, flag map, well overlay)
- `Part 3: Modifying` (PERMX)

## Requirements

```bash
pip install xtgeo pyvista numpy matplotlib
```

## How to Use

This workflow requires your own Petrel-exported files:
- `.EGRID` — grid geometry (export via right-click Simulation grid → Export model → ECLIPSE keywords)
- `PORO.GRDECL` — porosity property (export via right-click property → Export → ECLIPSE keywords)

Update the cutoff values, grid dimensions, and well coordinates in the script to match your own dataset.

## About

Built as part of a self-directed learning project bridging Petrel and Python for reservoir simulation workflows. 
Documented step-by-step on LinkedIn; follow along for upcoming parts.

This is a work in progress, shared for learning purposes. 
Feedback and suggestions are very welcome!
