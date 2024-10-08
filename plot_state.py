#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def printerr(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


parser = argparse.ArgumentParser(
    description="Loads a .npz file containing a 'state' from array CA "
    "simulation and plots it using matplotlib. "
    "You can choose to display the plot and/or save it as an image file."
)
parser.add_argument("npz")
parser.add_argument(
    "--show-plot",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Show the plot",
)
parser.add_argument(
    "--save-image",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Save the generate plot to a file",
)
parser.add_argument(
    "--output-file",
    default=None,
    help="The file to save the plot to. If no filename is provided, the name of the npz file will be used",
)
parser.add_argument(
    "--plot-diffusing",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Plot the diffusing particles. Default is False",
)
parser.add_argument(
    "--crop-aggregate",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Crop the aggregate such that it is always zoomed in and centered. Default is False",
)
parser.add_argument("--dpi", default=300, type=int, help="The DPI of the saved image")
args = parser.parse_args()

npz_file = Path(args.npz).resolve()
if not npz_file.exists():
    printerr(f"File {npz_file} does not exist")
    sys.exit(2)

with np.load(npz_file) as data:
    state = data["state"]
    if not args.plot_diffusing:
        state[state == 1] = 0
    if args.crop_aggregate:
        x, y = np.where(state == 2)
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        state = state[min_x:max_x, min_y:max_y]
    plt.imshow(state)
    plt.set_cmap("viridis")

if args.save_image:
    if args.output_file is None:
        output_file = npz_file.with_suffix(".png")
    else:
        output_file = Path(args.output_file)
    plt.savefig(output_file, dpi=args.dpi)

if args.show_plot:
    plt.show()
