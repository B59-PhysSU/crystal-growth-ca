import argparse
import enum
import os
import shutil
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numba import njit


# each enum variant is a state of the cell
# 0 - empty
# 1 - diffusing
# 2 - aggregated
class CellState(enum.Enum):
    EMPTY = 0
    DIFFUSING = 1
    AGGREGATED = 2


def prepare_initial_state(sim_size, initial_diffusing_fraction):
    K_diffusing = int(initial_diffusing_fraction * sim_size * sim_size)

    # prepare the initial state
    initial_state = np.zeros((sim_size, sim_size), dtype=np.uint8)
    initial_diffusing = np.random.randint(0, sim_size, (K_diffusing, 2))
    initial_state[initial_diffusing[:, 0], initial_diffusing[:, 1]] = (
        CellState.DIFFUSING.value
    )
    initial_state[sim_size // 2, sim_size // 2] = CellState.AGGREGATED.value
    initial_diffusing = np.argwhere(initial_state == CellState.DIFFUSING.value)
    return initial_state, initial_diffusing


@njit
def pbc(i: int, n: int) -> int:
    return i % n


@njit
def pbc_indices(i: int, j: int, n: int) -> Tuple[int, int]:
    return pbc(i, n), pbc(j, n)


@njit(parallel=True)
def aggregation_step(
    state: np.ndarray, diffusing_list: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    new_state = state.copy()

    keep_mask = np.ones(len(diffusing_list), dtype=np.uint8)
    for cell_idx in range(len(diffusing_list)):
        cell_i, cell_j = diffusing_list[cell_idx]

        neighbour_states = np.array(
            [
                state[pbc_indices(cell_i + 1, cell_j, state.shape[0])],
                state[pbc_indices(cell_i - 1, cell_j, state.shape[0])],
                state[pbc_indices(cell_i, cell_j + 1, state.shape[0])],
                state[pbc_indices(cell_i, cell_j - 1, state.shape[0])],
            ]
        )

        if np.any(neighbour_states == CellState.AGGREGATED.value):
            new_state[cell_i, cell_j] = CellState.AGGREGATED.value
            keep_mask[cell_idx] = 0
            print(f"Cell at ({cell_i}, {cell_j}) aggregated")

    diffusing_list = diffusing_list[keep_mask.astype(np.bool_)]

    return new_state, diffusing_list


@njit
def diffuse_one(state: np.ndarray, diffusing_list: np.ndarray, idx: int):
    i, j = diffusing_list[idx]

    empty_neighbors = []
    for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        ni, nj = pbc_indices(i + di, j + dj, state.shape[0])
        if state[ni, nj] == CellState.EMPTY.value:
            empty_neighbors.append((ni, nj))

    if empty_neighbors:
        new_i, new_j = empty_neighbors[np.random.randint(len(empty_neighbors))]
        state[new_i, new_j] = CellState.DIFFUSING.value
        state[i, j] = CellState.EMPTY.value
        diffusing_list[idx] = [new_i, new_j]

    return diffusing_list


@njit
def diffuse_all(state: np.ndarray, diffusing_list: np.ndarray):
    for idx in range(len(diffusing_list)):
        diffuse_one(state, diffusing_list, idx)


@njit
def nds(state: np.ndarray, diffusing_list: np.ndarray, nds_count: int):
    for _ in range(nds_count):
        diffuse_all(state, diffusing_list)


@njit
def step_simulation(state, diffusing, nds_count=5):
    nds(state, diffusing, nds_count)  # nds works in place
    state, diffusing = aggregation_step(state, diffusing)
    return state, diffusing


# argparse cli that takes
# N - simulation size (default == 100)
# D - initial fraction of diffusing cells (default == 0.1)
# T - number of time steps (default == 100)
# save_path - path to save the final state as an image
# returns the argparse parsed namespace directly
def cli(args=None):
    parser = argparse.ArgumentParser(
        description="Run a cellular automata for A2A aggregation"
    )
    parser.add_argument(
        "-N",
        type=int,
        default=100,
        help="The size of the simulation grid (default 100)",
    )
    parser.add_argument(
        "-D",
        type=float,
        default=0.1,
        help="The initial fraction of diffusing cells (default 0.1)",
    )
    parser.add_argument(
        "--nds",
        type=int,
        default=5,
        help="The number of diffusion steps to take in each time step (default 5)",
    )
    parser.add_argument(
        "-T",
        type=int,
        default=100,
        help="The number of time steps to run the simulation for (default 100)",
    )

    cwd = Path(os.getcwd())
    default_dir = cwd / "ca_output"
    parser.add_argument(
        "--save-dir",
        type=str,
        default=default_dir,
        help="The directory where frames of the simulation will be saved",
    )
    return parser.parse_args(args)


def prepare_save_dir(save_dir):
    save_dir = Path(save_dir)
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def main():
    args = cli()
    print(
        f"Starting simulation with N={args.N}, D={args.D}, T={args.T}, nds={args.nds}"
    )
    save_dir = prepare_save_dir(args.save_dir)
    state, diffusing = prepare_initial_state(args.N, args.D)
    n_diffusing_initial = diffusing.shape[0]

    fig, ax = plt.subplots()
    for i in range(args.T):
        state, diffusing = step_simulation(state, diffusing, nds_count=args.nds)
        ax.imshow(state, cmap="viridis")
        plt.savefig(save_dir / f"frame_{i}.png", dpi=300)

        print(f"Step {i + 1}. Number of diffusing cells: {diffusing.shape[0]}")

        # break if 1% of the original number of diffusing cells are left
        if diffusing.shape[0] < n_diffusing_initial * 0.01:
            print(
                f"Stopping early at step {i + 1} as less than 1% of the original number of diffusing cells remain"
            )
            break


if __name__ == "__main__":
    main()
