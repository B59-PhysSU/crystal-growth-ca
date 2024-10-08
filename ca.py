import argparse
import enum
import os
import shutil
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange, set_num_threads


# each enum variant is a state of the cell
# 0 - empty
# 1 - diffusing
# 2 - aggregated
class CellState(enum.Enum):
    EMPTY = 0
    DIFFUSING = 1
    AGGREGATED = 2


def prepare_initial_state(sim_size, initial_diffusing_fraction):
    assert sim_size > 50, "Simulation size must be greater than 50"
    K_diffusing = int(initial_diffusing_fraction * sim_size * sim_size)

    # prepare the initial state
    initial_state = np.zeros((sim_size, sim_size), dtype=np.uint8)
    initial_diffusing = np.random.randint(0, sim_size, (K_diffusing, 2))
    initial_state[initial_diffusing[:, 0], initial_diffusing[:, 1]] = (
        CellState.DIFFUSING.value
    )
    # make the initial state a plus-shape (four kinks)
    initial_state[sim_size // 2, sim_size // 2] = CellState.AGGREGATED.value
    initial_state[sim_size // 2 - 1, sim_size // 2] = CellState.AGGREGATED.value
    initial_state[sim_size // 2 + 1, sim_size // 2] = CellState.AGGREGATED.value
    initial_state[sim_size // 2, sim_size // 2 - 1] = CellState.AGGREGATED.value
    initial_state[sim_size // 2, sim_size // 2 + 1] = CellState.AGGREGATED.value
    initial_diffusing = np.argwhere(initial_state == CellState.DIFFUSING.value)
    return initial_state, initial_diffusing


@njit
def pbc(i: int, n: int) -> int:
    return i % n


@njit
def pbc_indices(i: int, j: int, n: int) -> Tuple[int, int]:
    return pbc(i, n), pbc(j, n)


# aggregation (kink-generation) step
# if any neighbour of a diffusing cell is aggregated, the diffusing cell becomes aggregated
@njit(parallel=True)
def aggregation_step(
    state: np.ndarray, diffusing_list: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    new_state = state.copy()
    new_diffusing_list = diffusing_list.copy()

    keep_mask = np.ones(len(new_diffusing_list), dtype=np.uint8)
    for cell_idx in prange(len(new_diffusing_list)):
        cell_i, cell_j = new_diffusing_list[cell_idx]

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

    new_diffusing_list = new_diffusing_list[keep_mask.astype(np.bool_)]

    return new_state, new_diffusing_list


# attach to a kink position
# where the four options for kink are:
#   X       X        A         A
# X D A   A D X    A D X     X D A
#   A       A        X         X
# where A -> aggregated, D -> diffusing, X -> Any state (aggregated, diffusing, empty)
# only when the diffusing cell is at the kink position, it becomes aggregated
@njit(parallel=True)
def attach_to_kink_step(
    state: np.ndarray, diffusing_list: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    new_state = state.copy()
    new_diffusing_list = diffusing_list.copy()
    keep_mask = np.ones(len(new_diffusing_list), dtype=np.uint8)

    for cell_idx in prange(len(new_diffusing_list)):
        cell_i, cell_j = new_diffusing_list[cell_idx]

        # Neighbors in the 4 directions
        n_up = state[pbc_indices(cell_i - 1, cell_j, state.shape[0])]
        n_down = state[pbc_indices(cell_i + 1, cell_j, state.shape[0])]
        n_left = state[pbc_indices(cell_i, cell_j - 1, state.shape[0])]
        n_right = state[pbc_indices(cell_i, cell_j + 1, state.shape[0])]

        is_kink = False

        #   X
        # X D A
        #   A
        is_kink |= (n_right == CellState.AGGREGATED.value) and (
            n_down == CellState.AGGREGATED.value
        )
        #   X
        # A D X
        #   A
        is_kink |= (n_left == CellState.AGGREGATED.value) and (
            n_down == CellState.AGGREGATED.value
        )
        #   A
        # A D X
        #   X
        is_kink |= (n_left == CellState.AGGREGATED.value) and (
            n_up == CellState.AGGREGATED.value
        )
        #   A
        # X D A
        #   X
        is_kink |= (n_right == CellState.AGGREGATED.value) and (
            n_up == CellState.AGGREGATED.value
        )
        
        if is_kink:
            print(f"Cell at ({cell_i}, {cell_j}) attached to kink")
            new_state[cell_i, cell_j] = CellState.AGGREGATED.value
            keep_mask[cell_idx] = 0

    # Final filtering: keep only the diffusing cells that did not aggregate
    new_diffusing_list = new_diffusing_list[keep_mask.astype(np.bool_)]

    return new_state, new_diffusing_list


@njit(parallel=True)
def calculate_moves(state: np.ndarray, diffusing_list: np.ndarray):
    move_targets = -np.ones(
        diffusing_list.shape, dtype=np.int32
    )  # Store target moves (-1 means no move)

    for idx in prange(diffusing_list.shape[0]):
        i, j = diffusing_list[idx]
        empty_neighbors = []

        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ni, nj = pbc_indices(i + di, j + dj, state.shape[0])
            if state[ni, nj] == CellState.EMPTY.value:
                empty_neighbors.append((ni, nj))

        if empty_neighbors:
            move_targets[idx] = empty_neighbors[np.random.randint(len(empty_neighbors))]

    return move_targets


@njit(parallel=True)
def resolve_conflicts(
    state: np.ndarray,
    diffusing_list: np.ndarray,
    move_targets: np.ndarray,
    target_map: np.ndarray,
    move_mask: np.ndarray,
):
    target_map.fill(-1)  # Reset target map
    move_mask.fill(0)  # Reset move mask
    # target_map shows which cells in the grid have already been claimed by a particle
    # target map values of -1 mean that the cell is not claimed, otherwise the value is the index of the particle in the diffusing_list
    # that will claim the cell at i, j
    # move_mask shows which particles have already moved to a new cell for faster processing (value == 0 means not moved)

    # First, resolve conflicts: mark one particle per target cell
    for idx in prange(len(diffusing_list)):
        target = move_targets[idx]
        if target[0] == -1:  # No move for this particle (it had no empty neighbors)
            continue

        # from here on particle wants to jump to an neighbouring empty cell
        ni, nj = target
        if (
            target_map[ni, nj] == -1
        ):  # There is no other particle that has already claimed this cell (-1)
            target_map[ni, nj] = idx  # claim the cell
            move_mask[idx] = 1  # Mark this particle as resolved (it got to move)

    # Second, apply valid moves
    for idx in prange(len(diffusing_list)):
        if move_mask[idx]:  # if this particle claimed a cell
            i, j = diffusing_list[idx]  # get the current position
            ni, nj = move_targets[idx]  # get the target position
            state[ni, nj] = CellState.DIFFUSING.value  # move to target position
            state[i, j] = CellState.EMPTY.value  # leave the current position
            diffusing_list[idx] = [ni, nj]  # Update particle position


@njit
def diffuse_all(
    state: np.ndarray,
    diffusing_list: np.ndarray,
    target_map: np.ndarray,
    move_mask: np.ndarray,
):

    # Step 1: Calculate desired moves for all particles
    move_targets = calculate_moves(state, diffusing_list)

    # Step 2: Resolve conflicts and apply valid moves
    resolve_conflicts(state, diffusing_list, move_targets, target_map, move_mask)
    np.random.shuffle(diffusing_list)  # Shuffle diffusing list to avoid bias


@njit
def nds(
    state: np.ndarray,
    diffusing_list: np.ndarray,
    nds_count: int,
    target_map: np.ndarray,
    move_mask: np.ndarray,
):
    for _ in range(nds_count):
        diffuse_all(state, diffusing_list, target_map, move_mask)


@njit(parallel=True)
def step_simulation(
    state, diffusing, target_map, move_mask, nds_count, attach_to_kink_probability
):
    nds(state, diffusing, nds_count, target_map, move_mask)  # nds works in place

    # split diffusing_list randomly in two sublists with attach_to_kink_probability
    # probability of a particle going to the attach_to_kink_step sub-list
    # and 1 - attach_to_kink_probability probability of a particle going to the aggregation_step sub-list
    attach_to_kink_mask = (
        np.random.rand(diffusing.shape[0]) < attach_to_kink_probability
    )
    diffusing_kink = diffusing[attach_to_kink_mask]
    diffusing_agg = diffusing[~attach_to_kink_mask]

    state, diffusing_agg = aggregation_step(state, diffusing_agg)
    state, diffusing_kink = attach_to_kink_step(state, diffusing_kink)

    # concatenate the two sublists back together and shuffle
    diffusing = np.concatenate((diffusing_agg, diffusing_kink))
    np.random.shuffle(diffusing)
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
    parser.add_argument(
        "--pA2k",
        type=float,
        default=0.8,
        help="Probability of applying the a2k rule to a diffusing cell during CA run (default 0.8)",
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
    parser.add_argument(
        "--save-plots",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Save plots of the state at each time step",
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
    print(f"Starting with the following arguments:\n{args}")
    save_dir = prepare_save_dir(args.save_dir, make_npz_dir=args.save_npz)
    state, diffusing = prepare_initial_state(args.N, args.D)

    # pre-allocate target map and move mask to avoid re-allocating memory
    # on every diffusion step
    target_map = -np.ones(state.shape, dtype=np.int32)
    move_mask = np.zeros(len(diffusing), dtype=np.uint8)

    n_diffusing_initial = diffusing.shape[0]

    if args.save_npz:
        print(
            f"Saving states as npz files to {save_dir / 'npz'}. "
            f"Keys for npz files are 'state' and 'diffusing'"
        )
    if args.save_plots:
        print("Warning saving plots will slow down the simulation!")

    _fig, ax = plt.subplots()
    for i in range(args.T):
        state, diffusing = step_simulation(
            state,
            diffusing,
            target_map,
            move_mask,
            nds_count=args.nds,
            attach_to_kink_probability=args.pA2k,
        )

        if args.save_plots:
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
