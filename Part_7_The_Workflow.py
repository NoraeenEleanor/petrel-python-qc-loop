"""
PETREL REMODEL WORKFLOW

Implementation of the SPE 164420 (Hoffman, 2013) completion-based
geomodel remodeling workflow (Steps 1-6)

Built and validated against the actual project files:
    - Well trajectories     : Petrel exported .DEV.file (vertical wells;
                                2-station export is valid when INCL == 0
                                throughout)
    - Completions           : a simple .ev event file (well, top, base,
                                event date, event type -- perf only)
    - Grid                  : ECLIPSE .EGRID
    - Property              : ECLIPSE .INIT (PORO, PERMX/Y/Z, NTG)
    

Result on the 26x31x7 sector model (5,327 active cells):
    - 53 cells intersect a perforation interval
    - 15 of those 53 (28%) fall below the 0.10 PORO cutoff
    - 19 cells (0.36% of grid) are changed  by IF(pseudo > actual, ...)
        because the default value 0.1234 sits above the cutoff
        
        
Validation (PETREL):
    - importing pseudo-log, Scale Up Well Logs Maximum, Property Calculator
        for both formula steps) and compared PERF_UP cell by cell against
        Python's results:
            - PETREL flagged 54 cells vs. Python's 53 [5,641 of 5,642 grid
                cells (99.98%)]
            - The one DISAGREEMENT: Well I04, Layer K2/K3: the grid's
                structural dip at the location makes the two cells'
                Z-ranges overlap, so a perforation sample sitting in that
                overlap is a boundary case either way (this script's
                point in hexahedron test resolves it to K2; meanwhile
                PETREL's Max scale-up resolves it to K3).

Tools: Python (lib: xtgeo)
Validation: Petrel

"""


from __future__ import annotations

import numpy as np
import pandas as pd
import xtgeo


#   CONFIGURATION

GRID_FILE   = "SECTOR_MODEL.egrid"
INIT_FILE   = "CASE.INIT"
EVENT_FILE  = "Well_Event.ev"
DEV_FILES   = {
    "I01":  "I01.DEV File",     "I02":  "I02.DEV File",
    "I03":  "I03.DEV File",     "I04":  "I04.DEV File",
    "P01":  "P01.DEV File",     "P02":  "P02.DEV File",
    "P03":  "P03.DEV File",     "P04":  "P04.DEV File",
    "P05":  "P05.DEV File",     "P06":  "P06.DEV File",
}

PROPERTY        = "PORO"
CUTOFF          = 0.10
DEFAULT_VALUE   = 0.1234      # unlikely in nature pseudo value, just above cutoff




#   STEP 1-2:   Parse wellheads (from .DEV_files) and completion (from .ev)

def parse_dev_wellhead(path: str) -> tuple[float, float]:
    """
    Pull (X, Y) wellhead location out of a PETREL exported .DEV.File.
    
    Sample PETREL's header, e.g.:
        # WELLHEAD X-COORD: 2900.00000000 (m)
        # WELLHEAD Y-COORD: 13100.00000000 (m)
    
    Reading straight from these 2 lines is more robust than parsing
    the first data row.
    
    """
    x = y = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "WELL HEAD X-COORD" in line.upper():
                x = float(line.split(":")[1].split("(")[0].strip())
            elif "WELL HEAD Y-COORD" in line.upper():
                y = float(line.split(":")[1].split("(")[0].strip())
            if x is not None and y is not None:
                break
    if x is None or y is None:
        raise ValueError(f"Could not find WELL HEAD X/Y-COORDINATE in {path}")
    return x, y



#   (as SPE 164420) STEP 1: Export completion data from database
def load_completion_events(event_file: str) -> dict[str, list[tuple[float, float]]]:
    """
    Parse the .ev file into {well:  [(md_top, md_base), ...]}.
    
    Adjust the column parsing herehere if your .ev layout differs
    
    """
    completions: dict[str, list[tuple[float, float]]] = {}
    current_well = None
    with open(event_file, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.upper().startswith("UNITS"):
                continue
            if line.upper().startswith("WELLNAME"):
                current_well = line.split()[-1].strip('"')
                completions.setdefault(current_well, [])
                continue
            
            # Event row: <date> <event_type> <md_top> <md_base> <extra...>
            parts = line.split()
            if current_well and len(parts) >=4 and "perf" in parts[1].lower():
                try:
                    top, base = float(parts[2]), float(parts[3])
                    completions[current_well].append((min(top, base), max(top, base)))
                except ValueError:
                    continue
    return completions




#   STEP 3-4:   Locate complted cells and upscale "Maximum"

def find_completed_cells(
    grid: "xtgeo.Grid",
    dfg: pd.DataFrame,
    wellheads: dict[str, tuple[float, float]],
    completions: dict[str, list[tuple [float, float]]],
    actual_prop: "xtgeo.GridProperty",
    step: float = 0.25,
) -> pd.DataFrame:
    """
    v3: exact cell intersection. Each perforated interval is sampled every
    'step' metres along the (vertical) wellbore and every sample point is
    located in the corner-point grid with xtgeo's get_ijk_from_points().
    Any cell touched by a perforation is flagged - same as PETREL's Scale
    Up Well Logs with Average method = Maximum
    
    """
    pts_rows = []
    for well, (wx, wy) in wellheads.items():
        for top, base in completions.get(well, []):
            for z in np.arange(top, base + step, step):
                pts_rows.append((well, wx, wy, z))
    pts_df = pd.DataFrame(pts_rows, columns=["well", "X", "Y", "Z"])
    pts = xtgeo.Points(pts_df[["X", "Y", "Z"]].values)
    ijk = grid.get_ijk_from_points(pts, activeonly=True, dataframe=True, includepoints=False)
    ijk["well"] = pts_df["well"].values
    ijk = ijk[ijk["IX"] > 0]
    cells = ijk.drop_duplicates(["well", "IX", "JY", "KZ"]).copy()
    
    vals = actual_prop.values
    cells["actual"] = [float(vals[i - 1, j - 1, k - 1]) for i, j, k in zip(cells.IX, cells.JY, cells.KZ)]
    cells = cells.rename(columns={"IX": "I", "JY": "J", "KZ": "K"})
    return cells[["well", "I", "J", "K", "actual"]].reset_index(drop=True)



#   (as SPE 164420) STEP 4: Upscale production logs
def build_perf_upscaled(grid: "xtgeo.Grid", cdf: pd.DataFrame) -> "xtgeo.GridProperty":
    """Step 4: PERF_UPSCALED = 1 in any cell a completion touches, else = 0"""
    perf = xtgeo.GridProperty(grid, name="PERF_UPSCALED",
                              values=np.zeros(grid.dimensions, dtype=np.int32))
    vals = perf.values
    for _, r in cdf.iterrows():
        vals[r.I - 1, r.J - 1, r.K - 1] = 1
    perf.values = vals
    return perf



#   STEP 5-6:   Pseudo-property and final remodeled property

#   (as SPE 164420) STEP 5: Create pseudo property
def build_pseudo_property(perf_upscaled: "xtgeo.GridProperty", default_value: float) -> "xtgeo.GridProperty":
    pseudo = perf_upscaled.copy()
    pseudo.name = f"{PROPERTY}_PSEUDO"
    pseudo.values = np.where(perf_upscaled.values == 1, default_value, 0.0)
    return pseudo


#   (as SPE 164420) STEP 6: Modify the geomodel proerties
def modify_property(actual_prop: "xtgeo.GridProperty", pseudo_prop: "xtgeo.GridProperty") -> "xtgeo.GridProperty":
    modified = actual_prop.copy()
    modified.name = f"{PROPERTY}_REMODEL"
    modified.values = np.where(pseudo_prop.values > actual_prop.values,
                               pseudo_prop.values, actual_prop.values)
    return modified



#   MAIN

def main():
    grid = xtgeo.grid_from_file(GRID_FILE, fformat="egrid")
    actual_prop = xtgeo.gridproperty_from_file(INIT_FILE, fformat="init", name=PROPERTY, grid=grid)
    
    xpr = xtgeo.GridProperties()
    xpr.append_props([actual_prop])
    grid.gridprops = xpr
    dfg = grid.get_dataframe(activeonly=True)
    
    wellheads = {w: parse_dev_wellhead(p) for w, p in DEV_FILES.items()}
    completions = load_completion_events(EVENT_FILE)
    
    cdf = find_completed_cells(grid, dfg, wellheads, completions, actual_prop)
    cdf["modified"] = cdf["actual"] < CUTOFF
    cdf["after"] = np.where(cdf["modified"], DEFAULT_VALUE, cdf["actual"])
    
    perf_upscaled   = build_perf_upscaled(grid, cdf)
    pseudo_prop     = build_pseudo_property(perf_upscaled, DEFAULT_VALUE)
    modified_prop   = modify_property(actual_prop, pseudo_prop)
    
    total_cells = int(actual_prop.values.count())
    n_completed = len(cdf)
    n_below_cutoff = int(cdf["modified"].sum())
    
    # what the SPE 164420's formula IF(pseudo > actual, pseudo, actual) really changes:
    # every completed cell below the DEFAULT VALUE (0.1234). i.e. the cutoff band
    # (0.10-0.11234) is raised as well
    
    changed     = ~np.isclose(modified_prop.values.filled(0), actual_prop.values.filled(0))
    n_changed   = int(changed.sum())
    cdf["changed_by_formula"] = cdf["actual"] < DEFAULT_VALUE
    
    
    
    print("=== WORKFLOW SUMMARY ===")
    print(f"Property remodeled              :   {PROPERTY}")
    print(f"Cutoff / default                :   {CUTOFF} / {DEFAULT_VALUE}")
    print(f"Total active grid cells         :   {total_cells}")
    print(f"Cells intersecting perfs        :   {n_completed}")
    print(f"Completed cells BELOW cutoff    :   {n_below_cutoff} ({100*n_below_cutoff/n_completed:.1f}% of completed)")
    print(f"Cells CHANGED by the formula)   :   {n_changed} ({100*n_changed/total_cells:.2f}% of grid)")
    print(f"    (formula raises every completed cell < default, incl. the 0.10-0.1234 band)")
    
    # ROFF (RMS binary) + GRDECL (Eclipse ASCII, keywords <= 8, safest for PETREL)
    modified_prop.to_file(f"{PROPERTY}_REMODEL_v3.roff", fformat="roff")
    perf_upscaled.to_file(f"PERF_UPSCALED_v3.roff", fformat="roff")
    
    g_perf = perf_upscaled.copy(); g_perf.name = "PERFUP"
    g_perf.to_file("PERF_UPSCALED_v3_GRDECL", fformat="grdecl")
    
    g_poro = modified_prop.copy(); g_poro.name = "POROMD"
    g_poro.to_file("PORO_REMODEL_v3_GRDECL", fformat="grdecl")
    
    cdf.sort_values(["well", "K"]).to_csv("qc_completed_cells_v3_csv", index=False)
    
    write_pseudo_logs(completions, "pseudo_logs_petrel.txt")
    print("\nSaved: *_v3.roff, *_v3.GRDECL, qc_completed_cells_v3_csv, pseudo_logs_petrel.text")
    
    
    
def write_pseudo_logs(completions, out_file):
    """Binary perforation logs (1 = perforated, 0 = unperforated)"""
    def md_range(path):
        mds = []
        for ln in open(path, errors="ignore"):
            p = ln.split()
            if p and not ln.startswith("#"):
                try:
                    mds.append(float(p[0]))
                except ValueError:
                    pass
        return min(mds), max(mds)
        
    lines = ["# Pseudo perforation logs from Well_Event.ev (1 = perf, 0 = unperf)"]
    for well, path in DEV_FILES.items():
        md0, md1 = md_range(path)
        
        pts = [(md0, 0.0)]
        for top, base in sorted(completions.get(well, [])):
            pts += [(top - 0.5, 0.0), (top, 1.0), (base, 1.0), (base + 0.5, 0.0)]
        
        pts.append((md1, 0.0))
        clean = []
        for md, v in pts:
            if clean and md <= clean[-1][0]:
                if v > clean[-1][1]:
                    clean[-1] = (clean[-1][0], v)
                continue
            clean.append((md, v))
        lines += [f"WELL. {well}", "CURVE. PERF"] + [f"{md:.2f} {v:.2f}" for md, v in clean]
    open(out_file, "w").write("\n".join(lines) + "\n")




if __name__ == "__main__":
    main()
        
    