"PART 3: Property Modification + Export to ECLIPSE"

import xtgeo
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

print("=" * 50)
print("PART 3 - Property Modification")
print("=" * 50)

# Load Data
print("\nLoading grid...")
grid = xtgeo.grid_from_file("SECTOR_MODEL.EGRID")

print("Loading PORO...")
poro = xtgeo.gridproperty_from_file("PORO.GRDECL", fformat="grdecl",
                                    name="PORO", grid=grid)

print("Loading PERMX...")
permx = xtgeo.gridproperty_from_file("PERMX.GRDECL", fformat="grdecl",
                                       name="PERMX", grid=grid)

poro_arr    = poro.values
poro_filled = poro_arr.filled(0)
perm_arr    = permx.values
perm_filled = perm_arr.filled(0)
ni, nj, nk  = grid.dimensions

print(f"\nGrid: {grid.dimensions}")
print(f"PORO - Min: {poro_filled.min():.4f} "
      f"Max: {poro_filled.max():.4f}")
print(f"PERMX - Min: {perm_filled.min():.2f} "
      f"Max: {perm_filled.max():.2f} mD")

# Cutoff values (taking from Part_2)
CUTOFF_DEAD = 0.02; CUTOFF_TIGHT = 0.08

# Step 1: Identify the problematic zones
print("\n--- Identifying problematic zones ---")

mask_dead  = (poro_filled > 0) & (poro_filled < CUTOFF_DEAD)
mask_tight = (poro_filled >= CUTOFF_DEAD) & \
             (poro_filled < CUTOFF_TIGHT)
             
print(f"Dead cells:  {mask_dead.sum()}")
print(f"Tight cells:  {mask_tight.sum()}")

# Step 2: Apply PERMX modification
print("\n--- Applying modifications ---")

perm_original = perm_filled.copy()
perm_modified = perm_filled.copy()

# Dead cells (PERM almost near 0)
PERMX_DEAD_VALUE = 9.07        # mD
perm_modified[mask_dead]    = PERMX_DEAD_VALUE

# Tight cells (reduce by multiplier)
TIGHT_MULTIPLIER = 0.5          # 50% of original
perm_modified[mask_tight]  *= TIGHT_MULTIPLIER

print(f"Dead zone PERMX     → {PERMX_DEAD_VALUE} mD")
print(f"Tight zone PERMX    → {TIGHT_MULTIPLIER * 100:.0f}% of original")

# Step 3: Before vs. After statistics
print("\n--- Per-Layer PERMX Before vs. After ---")
print(f"{'Layer' :<8} {'Before(mD)':>12} "
      f"{'After(mD)':>12} {'Reduction':>10}")
print("=" * 45)

for k in range(nk):
    before  = perm_original[:, :, k]
    after   = perm_modified[:, :, k]
    active  = before[before > 0]
    if len(active) == 0:
        continue
    mean_b  = active.mean()
    mean_a  = after[after > 0].mean()
    red     = 100 * (1 - mean_a / mean_b)
    print(f"K={k+1:<6} {mean_b:>12.2f} "
          f"{mean_a:>12.2f} {red:>9.1f}%")
    
# Colour range
perm_max = perm_original[perm_original > 0].max()
perm_min = 0.001

# Custom log normalizer
norm = mcolors.LogNorm(vmin=max(perm_min, 1),
                       vmax=perm_max)

# Step 4: Histogram before vs. after
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Before
active_before = perm_original[perm_original > 0].flatten()
ax1.hist(active_before, bins=50, color="steelblue", edgecolor="white", log=False)
ax1.set_xlabel("PERMX (mD)")
ax1.set_xscale("log"); ax1.set_xticks([10, 100, 1000])
ax1.get_xaxis().set_major_formatter(
    mticker.LogFormatter(base=10, labelOnlyBase=False))
ax1.set_ylabel("Cell count")
ax1.set_title("PERMX Distribution\nBEFORE Modification")
ax1.axvline(x=active_before.mean(), color="red", linestyle="--",
            label=f"Mean: {active_before.mean():.1f} mD")
ax1.legend()

# After
active_after = perm_modified[perm_modified > 0].flatten()
ax2.hist(active_after, bins=50, color="darkorange", edgecolor="white")
ax2.set_xlabel("PERMX (mD)")
ax2.set_xscale("log"); ax2.set_xticks([10, 100, 1000])
ax2.get_xaxis().set_major_formatter(
    mticker.LogFormatter(base=10, labelOnlyBase=False))
ax2.set_ylabel("Cell count")
ax2.set_title("PERMX Distribution\nAFTER Modification")
ax2.axvline(x=active_after.mean(), color="red", linestyle="--",
            label=f"Mean: {active_after.mean():.1f} mD")
ax2.legend()

plt.suptitle(
    "PERMX Before vs. After Modification\n"
    "Dead zones: 9.07 mD | "
    "Tight zones: 50% of original",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig("PERMX_histogram_comparison.png", dpi=150)
plt.show()
print("\n   Histogram saved: PERMX_histogram_comparison.png")

# Step 5: Export the adjusted PERMX
print("\n --- Exporting adjusted PERMX ---")

permx_adjusted = xtgeo.GridProperty(
    grid,
    values=np.ma.MaskedArray(
        perm_modified,
        mask=(perm_modified == 0)
    ),
    name="PERMX"
)

permx_adjusted.to_file("PERMX_ADJUSTED.GRDECL", fformat="grdecl")

print(" Exported: PERMX_ADJUSTED.GRDECL")
print(" → Ready for ECLIPSE .DATA file!")

# Step 6: 3D visualization before vs. after
print("\nGenerating Before vs. After GIF...")

LAYER_SPACING = 2.7; WAVE_AMP= 0.75; WAVE_FREQ = 0.3; THICKNESS = 0.3

def build_perm_mesh(perm_data, k, ni, nj):
    z_base  = (nk - 1 - k) * LAYER_SPACING
    i_arr   = np.arange(ni, -1, -1)
    j_arr   = np.arange(nj + 1)
    ii, jj  = np.meshgrid(i_arr, j_arr, indexing="ij")
    
    wave    = WAVE_AMP * np.sin(WAVE_FREQ * ii) * \
              np.cos(WAVE_FREQ * jj)
    z_top   = z_base + THICKNESS + wave
    z_bot   = z_base + wave
    
    x_pts   = np.tile(ii[:, :, np.newaxis], (1, 1, 2)).astype(float)
    y_pts   = np.tile(jj[:, :, np.newaxis], (1, 1, 2)).astype(float)
    z_pts   = np.stack([z_bot, z_top], axis=-1).astype(float)
    
    mesh    = pv.StructuredGrid(x_pts, y_pts, z_pts)
    mesh.cell_data["PERMX"] = \
        perm_data[:, :, k].flatten(order="F")
    return mesh

plotter = pv.Plotter(shape=(1,2), off_screen=True, window_size=[1600, 800])

centre_x = ni / 2
centre_y = nj / 2
centre_z = (nk * LAYER_SPACING) / 2

# LEFT (before)
plotter.subplot(0, 0)
for k in range(nk):
    mesh = build_perm_mesh(perm_original, k, ni, nj)
    plotter.add_mesh(mesh, scalars="PERMX", cmap="rainbow",
                     clim=[10, 1582], log_scale=True,
                     show_scalar_bar=False,
                     show_edges=True, edge_color="#333333", opacity=1.0)
    
plotter.add_text(
    "BEFORE Modification\nOriginal PERMX (mD)",
    position="upper_left", font_size=12, color="white"
)
plotter.set_background("#1A1A1A")

# RIGHT (after) 
plotter.subplot(0, 1)
for k in range (nk):
    mesh = build_perm_mesh(perm_modified, k, ni, nj)
    plotter.add_mesh(mesh, scalars="PERMX", cmap="rainbow",
                     clim=[10, 1582], log_scale=True,
                     show_scalar_bar=False,
                     show_edges=True, edge_color="#333333", opacity=1.0)

plotter.add_text(
    "AFTER Modification\nAdjusted PERMX (mD)",
    position="upper_left", font_size=12, color="white"
)
plotter.set_background("#1A1A1A")

# Shared scalar bar (on the LEFT panel)
plotter.subplot(0, 0)
plotter.add_scalar_bar(title="PERMX (mD)", color="white", n_labels=4,
                       label_font_size=18, title_font_size=18,
                       fmt="%.0f", width=0.5, height=0.05,
                       position_x=0.27, position_y=0.85,
                       mapper=plotter.renderers[0].actors[
                           list(plotter.renderers[0].actors.keys())[-1]
                       ].GetMapper())   # link to mesh cmap

# GIF rotation
plotter.open_gif("PERM_BEFORE_AFTER.gif", framerate=8)

for angle in range(0, 360, 2):
    cam = [
        (
            centre_x + 80 * np.cos(np.radians(angle)),
            centre_y + 80 * np.sin(np.radians(angle)),
            centre_z + 45      
        ),
        (centre_x, centre_y, centre_z),
        (0, 0, 1)
    ]
    for col in [0, 1]:
        plotter.subplot(0, col)
        plotter.camera_position = cam
        
    plotter.write_frame()
    
plotter.close()
print(" Saved: PERMX_BEFORE_AFTER.gif")

print("\n=== DONE PART 3 ===")
print("Files generated:")
print("     PERMX_Histogram_Comparison.png")
print("     PERMX_Adjusted.GRDECL")
print("     PERMX_BEFORE_AFTER.gif")

# Calibration" Verification Block

print("\n--- VERIFICATION: Modification values check ---")

# Check dead cells (all must be 9.07)
dead_after = perm_modified[mask_dead]
print(f"Dead cells PERMX after:")
print(f"    Min: {dead_after.min():.4f} mD")
print(f"    Max: {dead_after.max():.4f} mD")
print(f"    All = 9.07? {np.all(dead_after == PERMX_DEAD_VALUE)}")

# Check tight cells (all must be 50% of original)
tight_before = perm_original[mask_tight]
tight_after = perm_modified[mask_tight]
ratio = tight_after / tight_before
print(f"\nTight cells PERMX ratio (should be 0.5):")
print(f"    Min ratio: {ratio.min():.4f}")
print(f"    Max ratio: {ratio.max():.4f}")
print(f"    Mean ratio: {ratio.mean():.4f}")
print(f"    All = 0.5? {np.allclose(ratio, TIGHT_MULTIPLIER)}")

# Check good cells - should remained UNCHANGED
good_mask   = poro_filled >= CUTOFF_TIGHT
good_before = perm_original[good_mask]
good_after  = perm_modified[good_mask]
print(f"\nGood cells PERMX (should be unchanged):")
print(f"    Same? {np.allclose(good_before, good_after)}")