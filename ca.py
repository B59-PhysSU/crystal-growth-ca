import argparse
import enum
import os
import shutil
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange


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
    initial_state[
        initial_diffusing[:, 0], initial_diffusing[:, 1]
    ] = CellState.DIFFUSING.value
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


@njit(parallel=True)
def calculate_moves(state: np.ndarray, diffusing_list: np.ndarray):
    """First pass: calculate desired moves for each particle."""
    move_targets = -np.ones((diffusing_list.shape[0], 2), dtype=np.int32)  # Store target moves (-1 means no move)

    for idx in prange(diffusing_list.shape[0]):  # Parallel loop over all diffusing particles
        i, j = diffusing_list[idx]
        empty_neighbors = []

        # Check north, south, east, west neighbors
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ni, nj = pbc_indices(i + di, j + dj, state.shape[0])
            if state[ni, nj] == CellState.EMPTY.value:
                empty_neighbors.append((ni, nj))

        if empty_neighbors:
            # Randomly choose an empty neighbor to move to
            move_targets[idx] = empty_neighbors[np.random.randint(len(empty_neighbors))]

    return move_targets

@njit(parallel=True)
def resolve_conflicts(state: np.ndarray, diffusing_list: np.ndarray, move_targets: np.ndarray):
    """Second pass: resolve conflicts and apply moves."""
    target_map = -np.ones(state.shape, dtype=np.int32)  # Map to track which cell is being targeted
    move_mask = np.zeros(len(diffusing_list), dtype=np.uint8)  # Mask to mark valid moves

    # First, resolve conflicts: mark one particle per target cell
    for idx in prange(len(diffusing_list)):
        target = move_targets[idx]
        if target[0] == -1:  # No move for this particle
            continue
        ni, nj = target
        if target_map[ni, nj] == -1:  # No conflict, mark this particle's move
            target_map[ni, nj] = idx
            move_mask[idx] = 1

    # Second, apply valid moves
    for idx in prange(len(diffusing_list)):
        if move_mask[idx]:  # Only apply moves that were resolved
            i, j = diffusing_list[idx]
            ni, nj = move_targets[idx]
            state[ni, nj] = CellState.DIFFUSING.value  # Move particle
            state[i, j] = CellState.EMPTY.value  # Clear old position
            diffusing_list[idx] = [ni, nj]  # Update particle position

@njit
def diffuse_all_parallel(state: np.ndarray, diffusing_list: np.ndarray):
    """Diffuse particles in parallel with conflict resolution."""
    # Step 1: Calculate desired moves for all particles
    move_targets = calculate_moves(state, diffusing_list)
    
    # Step 2: Resolve conflicts and apply valid moves
    resolve_conflicts(state, diffusing_list, move_targets)
    np.random.shuffle(diffusing_list)  # Shuffle diffusing list to avoid bias

diffuse_all = diffuse_all_parallel

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
    parser.add_argument(
        "--save-npz",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Save states as npz files along with images",
    )
    return parser.parse_args(args)


def prepare_save_dir(save_dir, make_npz_dir=False):
    save_dir = Path(save_dir)
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)
    if make_npz_dir:
        os.makedirs(save_dir / "npz", exist_ok=True)
    return save_dir


def main():
    args = cli()
    print(
        f"Starting simulation with N={args.N}, D={args.D}, T={args.T}, nds={args.nds}"
    )
    save_dir = prepare_save_dir(args.save_dir, make_npz_dir=args.save_npz)
    state, diffusing = prepare_initial_state(args.N, args.D)
    n_diffusing_initial = diffusing.shape[0]

    if args.save_npz:
        print(
            f"Saving states as npz files to {save_dir / 'npz'}. "
            f"Keys for npz files are 'state' and 'diffusing'"
        )

    _fig, ax = plt.subplots()
    for i in range(args.T):
        state, diffusing = step_simulation(state, diffusing, nds_count=args.nds)

        ax.imshow(state, cmap="viridis")
        plt.savefig(save_dir / f"frame_{i}.png", dpi=300)
        if args.save_npz:
            np.savez_compressed(
                save_dir / "npz" / f"frame_{i}.npz", state=state, diffusing=diffusing
            )

        print(f"Step {i + 1}. Number of diffusing cells: {diffusing.shape[0]}")

        if diffusing.shape[0] < n_diffusing_initial * 0.05:
            print(
                f"Stopping early at step {i + 1} as less than 5% of the original "
                f"number of diffusing cells remain"
            )
            break


if __name__ == "__main__":
    main()
