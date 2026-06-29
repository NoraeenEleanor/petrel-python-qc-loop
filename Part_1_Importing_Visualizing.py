import xtgeo
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# Part 1: Load data
print("Loading grid...")
grid = xtgeo.grid_from_file("SECTOR_MODEL.EGRID")
print(f"Grid dimensions: {grid.dimensions}")

# Porosity data
print("Loading PORO...")
poro = xtgeo.gridproperty_from_file(
    "PORO.GRDECL", fformat="grdecl", name="PORO", grid=grid
)

poro_arr = poro.values
poro_filled = poro_arr.filled(0)

print(f"\n--- PORO QC ---")
print(f"Min:    {poro_arr.min():.4f}")
print(f"Max:    {poro_arr.max():.4f}")
print(f"Mean:   {poro_arr.mean():.4f}")

# Part 2: Adjust color to Petrel-like colormap (cmap=get is almost closest one or up to your preference)
petrel_colors = [
    "#00008B",   # dark blue (for low PORO)
    "#0000FF",   # blue,    
    "#00BFFF",   # deep sky blue
    "#00FFFF",   # cyan
    "#00FF80",   # cyan-green
    "#00FF00",   # green
    "#80FF00",   # yellow-green
    "#FFFF00",   # yellow
    "#FFB300",   # amber
    "#FFA500",   # orange
    "#FF0000",   # red (for high PORO)
]
petrel_cmap = LinearSegmentedColormap.from_list(
    "petrel_poro", petrel_colors, N=256
)

# Part 3: Modeling 2D Plot
plt.figure(figsize=(10,7))
plt.imshow(
    poro_arr[:, :, 0].T, cmap=petrel_cmap, origin="lower", vmin=0, vmax=0.30
)
plt.colorbar(label="Porosity (fraction)")
plt.title("Upscaled Porosity - Layer K=1")
plt.xlabel("I direction")
plt.ylabel("J direction")
plt.tight_layout()
plt.savefig("PORO_2D_Layer1.png", dpi=150)
plt.show()
print("2D plot saved!")

# Part 4: Building PyVista mesh
import pyvista as pv

ni, nj, nk = grid.dimensions    # 26, 31, 7

xc, yc, zc = np.meshgrid(
    np.arange(ni-1, -1, -1),
    np.arange(nj),
    np.arange(nk),
    indexing="ij"
)

mesh = pv.StructuredGrid(
    xc.astype(float),
    yc.astype(float),
    zc.astype(float)
)

mesh.point_data["PORO"] = poro_filled.flatten(order="F")

# Part 5: Static 3D screenshot
print("\nGenerating static 3D view...")
plotter_static = pv.Plotter(off_screen=False)
plotter_static.add_mesh(mesh, scalars="PORO", cmap=petrel_cmap, clim=[0, 0.28],
                 scalar_bar_args={"title": "Porosity (fraction)"},
                 show_edges=True, edge_color="grey", opacity=0.9)

plotter_static.add_axes()
plotter_static.set_background("white")
plotter_static.camera_position = [
    (60, -40, 35),      # camera position - right edge top
    (13, 15, 3),        # focal point - middle
    (0, 0, 1)           # up vector
]

plotter_static.screenshot("PORO_3D_static.png", window_size=[1200, 800])
plotter_static.close()
print("Static 3D saved → PORO_3D_static.png")

# Part 6: Rotating GIF
print("\nGenerating rotating GIF - please wait...")
plotter_gif = pv.Plotter(off_screen=False)
plotter_gif.add_mesh(mesh, scalars="PORO", cmap=petrel_cmap, clim=[0, 0.28],
                 scalar_bar_args={"title": "Porosity (fraction)"},
                 show_edges=True, edge_color="grey", opacity=0.9)

plotter_gif.add_axes()
plotter_gif.set_background("white")

# Adding title
plotter_gif.add_text(
    "Python-generated Visualization\nFrom Petrel-exported ECLIPSE Grid (.grdecl & .egrid)",
    position="upper_left", font_size=12, color="black", font="arial"
)

plotter_gif.show()

# Open GIF writer
plotter_gif.open_gif("PORO_3D_rotating.gif", framerate=15)

# rotation same as Petrel
for angle in range(0, 360, 2):
    plotter_gif.camera_position = [
        (
            60 * np.cos(np.radians(angle)) - 40 * np.sin(np.radians(angle)),
            60 * np.sin(np.radians(angle)) + 40 * np.cos(np.radians(angle)),
            35      # maintain elevation
        ),
        (13, 15, 3),    # always look at center
        (0, 0, 1)
    ]
    plotter_gif.write_frame()
    
plotter_gif.close()
print("Rotating GIF saved → PORO_3D_rotating.gif")
print("\nAll done! Files generated")
print("     PORO_2D_layer1.png")
print("     PORO_3D_static.png")
print("     poro_3D_rotating.gif")
