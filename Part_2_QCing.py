import xtgeo
import pyvista as pv
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load data
print("Loading grid...")
grid = xtgeo.grid_from_file("SECTOR_MODEL.EGRID")

print("Loading PORO...")
poro = xtgeo.gridproperty_from_file(
    "PORO.GRDECL", fformat="grdecl", name="PORO", grid=grid
)
poro_arr    = poro.values
poro_filled = poro_arr.filled(0)
ni, nj, nk  = grid.dimensions
print(f"Grid: {grid.dimensions}")
print("Ready — generating flag map GIFs...\n")

# Histogram - cutoff values
CUTOFF_DEAD = 0.02
CUTOFF_TIGHT = 0.08
CUTOFF_GOOD = 0.15

poro_flat = poro_filled[poro_filled > 0].flatten()

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(poro_flat, bins=50, color="steelblue", edgecolor="white")

ax.axvline(x=CUTOFF_DEAD,  color="#FF32FE", linestyle="--",
           linewidth=2, label=f"Dead cutoff ({CUTOFF_DEAD})")
ax.axvline(x=CUTOFF_TIGHT, color="#3232FF",    linestyle="--",
           linewidth=2, label=f"Tight cutoff ({CUTOFF_TIGHT})")
ax.axvline(x=CUTOFF_GOOD,  color="#00FFFF",    linestyle="--",
           linewidth=2, label=f"Good cutoff ({CUTOFF_GOOD})")

ax.set_xlabel("Porosity (fraction)")
ax.set_ylabel("Cell count")
ax.set_title(
    "Porosity Distribution — Cutoff Definition\n"
    "Sector Model | Petrel-Exported Grid"
)
ax.legend()
plt.tight_layout()
plt.savefig("PORO_histogram_cutoff.png", dpi=150)
plt.show()
print(" Histogram saved: PORO_histogram_cutoff.png")

# Calculate actual percentages
total_cells = poro_filled.size
active_cells = (poro_filled > 0).sum()

n_dead      = ((poro_filled > 0) & (poro_filled < CUTOFF_DEAD)).sum()
n_tight     = ((poro_filled >= CUTOFF_DEAD) & (poro_filled < CUTOFF_TIGHT)).sum()
n_good      = ((poro_filled >=CUTOFF_TIGHT) & (poro_filled < CUTOFF_GOOD)).sum()
n_excellent = (poro_filled >= CUTOFF_GOOD).sum()
n_inactive  = (poro_filled == 0).sum()

pct_dead        = 100 * n_dead / total_cells
pct_tight       = 100 * n_tight / total_cells
pct_good        = 100 * n_good / total_cells
pct_excellent   = 100 * n_excellent / total_cells
pct_inactive    = 100 * n_inactive / total_cells

print(f"\n=== ACTUAL VALUES FOR FINAL ===")
print(f"Total cells:                {total_cells}")
print(f"Inactive:                   {n_inactive} ({pct_inactive:.1f}%)")
print(f"Dead        (<0.02):        {n_dead} ({pct_dead:.1f}%)")
print(f"Tight       (0.02-0.08):    {n_tight} ({pct_tight:.1f}%)")
print(f"Good        (0.08-0.15):    {n_good} ({pct_good:.1f}%)")
print(f"Excellent   (>0.15):        {n_excellent} ({pct_excellent:.1f}%)")

# Combined good reservoir (good + excellent)
n_reservoir     = n_good + n_excellent
pct_reservoir   = 100 * n_reservoir / total_cells
print(f"\nCombined reservoir (>0.08): {n_reservoir} ({pct_reservoir:.1f}%)")

# Build 3D flag array
def build_flag_3d(poro_filled):
    flag = np.zeros_like(poro_filled)
    flag[poro_filled > 0]                               = 4 # excellent
    flag[(poro_filled > 0) & (poro_filled < 0.15)]      = 3 # good
    flag[(poro_filled > 0) & (poro_filled < 0.08)]      = 2 # tight
    flag[(poro_filled > 0) & (poro_filled < 0.02)]      = 1 # dead
    flag[poro_filled == 0]                              = 0 # inactive
    return flag

flag_3d = build_flag_3d(poro_filled)

# Custom cmap - 5 discrete colors (preference)
flag_cmap = [
    "#FFFFFF",      # 0 = inactive (white)
    "#FF32FE",      # 1 = dead (magenta)
    "#3232FF",      # 2 = tight (blue)
    "#00FFFF",      # 3 = good (cyan)
    "#75CC0A",      # 4 = excellent (limegreen)
]


# Build wavy mesh for one layer
THICKNESS       = 0.3   # thickness of each layer
LAYER_SPACING   = 2.7   # spacing between layers
WAVE_AMP        = 0.75  # wave amplitude (0=flat, 0.3=very wavy)
WAVE_FREQ       = 0.30  # wave frequency

def build_layer_mesh(k, flag_3d, ni, nj,
                     thickness=THICKNESS, spacing=LAYER_SPACING,
                     wave_amp=WAVE_AMP, wave_freq=WAVE_FREQ,
                     reverse=True):
    
    """
    wavy slab mesh for layer k. Reverse=True (K=1 on top, K=7 at bottom)
    """
    # Reverse layer order - K=1 top, K=7 bottom
    if reverse:
        z_base = (nk - 1 - k) * spacing
    else:
        z_base = k * spacing
    
    # Create coordinate arrays
    i_arr = np.arange(ni, -1, -1)
    j_arr = np.arange(nj + 1)
    ii, jj = np.meshgrid(i_arr, j_arr, indexing="ij")
    
    # Wavy Z surface - sin + cos combination
    wave = (wave_amp *
            np.sin(wave_freq * ii) *
            np.cos(wave_freq * jj))
    
    # Top and bottom Z with wave
    z_top = z_base + thickness + wave
    z_bot = z_base + wave
    
    # Build structured grid points
    # PyVista StructuredGrid needs (ni+1) x (nj+1) x 2 points
    x_pts = np.tile(ii[:, :, np.newaxis], (1, 1, 2)).astype(float)
    y_pts = np.tile(jj[:, :, np.newaxis], (1, 1, 2)).astype(float)
    z_pts = np.stack([z_bot, z_top], axis=-1).astype(float)
    
    mesh = pv.StructuredGrid(x_pts, y_pts, z_pts)
    
    # Assign flag values - repeat for top+bot points
    flag_layer = flag_3d[:, :, k]
    
    # Cell data (ni x nj cells)
    mesh.cell_data["FLAG"] = flag_layer.flatten(order="F")
    
    return mesh

# Legend
def add_legend_and_title(plotter, title_text, bg="#BDB7B7", text_color="black"):
    plotter.add_legend(
        labels=[
            ["Inactive",                "white",],
            ["Dead (PORO < 0.02)",      "magenta"],
            ["Tight (0.02-0.08)",       "blue"],
            ["Good (0.08-0.15)",        "cyan"],
            ["Excellent (> 0.15)",      "limegreen"],
        ],
        bcolor=bg, face="rectangle", size=(0.18, 0.18)
    )
    plotter.add_text(
        title_text, 
        position="upper_left",
        font_size=10,
        color=text_color
    )

# Camera rotation
def get_camera_pos(angle, radius=100, elevation=35,
                   cx=13, cy=15, cz=8):
    return [
        (
            cx + radius * np.cos(np.radians(angle)),
            cy + radius * np.sin(np.radians(angle)),
            cz + elevation
        ),
        (cx, cy, cz),
        (0, 0, 1)
    ]
    
# All 7 layers stacked GIF
print("\nGenerating ALL LAYERS GIF...")

plotter_all = pv.Plotter(
    shape=(1,1), off_screen=True, window_size=[1200, 800]
)

# Add all 7 layers meshes
for k in range (nk):
    mesh = build_layer_mesh(k, flag_3d, ni ,nj, reverse=True)
    plotter_all.add_mesh(mesh, scalars="FLAG", cmap=flag_cmap, clim=[0, 4],
                         show_scalar_bar=False, show_edges=True,
                         edge_color="#EEEEEE", opacity=1.0
    )

# Grid info from Petrel Statistics
X_ORIGIN = 1059.04; Y_ORIGIN = 3314.52
CELL_SIZE_I = 351.60; CELL_SIZE_J = 343.91
NJ = 31     # from 26x31x7

# Well real world coord from Petrel
# Tabulating all wells:
wells_xy = {
    "I01": (2900,       13100,      "injector"),
    "I02": (9000,       4600,       "injector"),
    "I03": (7000,       10000,      "injector"),
    "I04": (3500,       7400,       "injector"),
    "P01": (10042.15,   13121.17,   "producer"),
    "P02": (9000,       10500,      "procuder"),
    "P03": (6800,       8000,       "producer"),
    "PO4": (5000,       4200,       "producer"),
    "PO5": (7500,       13500,      "producer"),
    "P06": (2200,       5000,       "producer")
}

wells_ij = {}
for name, (x, y, wtype) in wells_xy.items():
    i       = int((x - X_ORIGIN) / CELL_SIZE_I)
    j       = int((y - Y_ORIGIN) / CELL_SIZE_J)
    j_flip  = NJ - 1 - j
    wells_ij[name]  = (i, j_flip, wtype)
    print(f"{name}: ({x},{y}) → I={i}, J={j_flip}")
    
# Add wells
z_top       = (nk - 1) * LAYER_SPACING + THICKNESS + 5.0
z_bottom    = -3

for well_name, (wi, wj, wtype) in wells_ij.items():
    
    # Flip the well matched to the flipped grid
    wi_flipped = ni - wi
    
    colour = "#FF4444" if wtype == "injector" else "#FFD700"
    # Red = injector, Gold = producer
    
    # Vertical well line
    line = pv.Line(
        pointa=(float(wi_flipped), float(wj), z_bottom),
        pointb=(float(wi_flipped), float(wj), z_top)
    )
    plotter_all.add_mesh(line, color=colour, line_width=5,
                         render_lines_as_tubes=True)
    
    # Wellhead pointer, shape: sphere
    sphere = pv.Sphere(radius=0.4,
                       center=(float(wi_flipped), float(wj), z_top))
    plotter_all.add_mesh(sphere, color=colour)
    
    # Well label
    plotter_all.add_point_labels(
        np.array([[float(wi_flipped), float(wj), z_top + 1.0]]),
        [well_name], font_size=18, text_color="black",
        point_color=colour, point_size=2,
        shape_opacity=0, always_visible=True
    )                   

# Legend
plotter_all.add_legend(
    labels=[
        ["  Inactive",            "white"],
        ["  Dead (PORO < 0.02)",  "#FF32FE"],
        ["  Tight (0.02-0.08)",   "#3232FF"],
        ["  Good (0.08-0.15)",    "#00FFFF"],
        ["  Excellent (> 0.15)",  "#75CC0A"],
        ["  Injector well",       "#FF4444"],
        ["  Producer well",       "#FFD700"],
    ],
    bcolor="white", face="rectangle", size=(0.20, 0.20)
)

# Title
plotter_all.add_text(
    "Connectivity Flag Map - All 7 Layers\n"
    "Python QC | Petrel-exported Grid\n",

    position="upper_left", font_size=10, color="black"
) 

plotter_all.set_background("white")

# GIF rotation
plotter_all.open_gif(
    "FLAG_MAP_ALL_LAYERS_WELLS.gif",
    framerate=8
)

centre_x = ni / 2
centre_y = nj / 2
centre_z = (nk * LAYER_SPACING) / 2

for angle in range(0, 360, 2):
    radius = 80
    plotter_all.camera_position = [
        (
            centre_x + radius * np.cos(np.radians(angle)),
            centre_y + radius * np.sin(np.radians(angle)),
            centre_z + 45
        ),
        (centre_x, centre_y, centre_z),
        (0, 0, 1)
    ]
    plotter_all.write_frame()
    
plotter_all.close()
print("     Saved: FLAG_MAP_ALL_LAYERS_WELLS.gif")