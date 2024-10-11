import numpy as np
from numba import njit

# Made with claude... Need to understand better


@njit
def is_valid_bounded(x, y, min_x, max_x, min_y, max_y):
    """Check if the (x, y) coordinates are valid within the bounded region."""
    return min_x <= x <= max_x and min_y <= y <= max_y


@njit
def find_articulation_points_bounded_core(
    grid, min_x, max_x, min_y, max_y, empty_value=0
):
    """Numba-optimized core function to find articulation points within a bounded region."""
    full_size = grid.shape[0]

    # Initialize arrays for the bounded region
    bound_height = max_x - min_x + 1
    bound_width = max_y - min_y + 1

    discovery_time = np.full((bound_height, bound_width), -1)
    low = np.full((bound_height, bound_width), -1)
    visited = np.zeros((bound_height, bound_width), dtype=np.bool_)
    articulation_points = np.zeros(
        (full_size, full_size), dtype=np.bool_
    )  # Full size output
    child_count = np.zeros((bound_height, bound_width), dtype=np.int32)

    # Parent array using separate x and y components
    parent_x = np.full((bound_height, bound_width), -1, dtype=np.int32)
    parent_y = np.full((bound_height, bound_width), -1, dtype=np.int32)

    # Directions for 8-connectivity
    dirs_x = np.array([-1, 1, 0, 0, -1, -1, 1, 1])
    dirs_y = np.array([0, 0, -1, 1, -1, 1, -1, 1])

    time = 0
    max_stack_size = bound_height * bound_width

    # Pre-allocate stack arrays
    stack_x = np.zeros(max_stack_size, dtype=np.int32)
    stack_y = np.zeros(max_stack_size, dtype=np.int32)
    stack_dir = np.zeros(max_stack_size, dtype=np.int32)
    stack_state = np.zeros(max_stack_size, dtype=np.int32)
    stack_size = 0

    # Process each unvisited EMPTY cell within the bounded region
    for i in range(min_x, max_x + 1):
        for j in range(min_y, max_y + 1):
            # Convert to bounded coordinates
            bound_i = i - min_x
            bound_j = j - min_y

            if grid[i, j] == empty_value and not visited[bound_i, bound_j]:
                # Initialize stack with start node
                stack_x[0] = bound_i
                stack_y[0] = bound_j
                stack_dir[0] = -1
                stack_state[0] = 0
                stack_size = 1

                while stack_size > 0:
                    # Get current stack top
                    x = stack_x[stack_size - 1]
                    y = stack_y[stack_size - 1]
                    dir_idx = stack_dir[stack_size - 1]
                    state = stack_state[stack_size - 1]

                    if state == 0:
                        # First visit to this node
                        visited[x, y] = True
                        discovery_time[x, y] = time
                        low[x, y] = time
                        time += 1
                        stack_dir[stack_size - 1] = 0
                        stack_state[stack_size - 1] = 1
                        continue

                    if dir_idx >= len(dirs_x):  # len(dirs)
                        # Finished processing all neighbors
                        stack_size -= 1
                        if stack_size > 0:  # If not root
                            px = parent_x[x, y]
                            py = parent_y[x, y]
                            if px >= 0:  # Valid parent
                                low[px, py] = min(low[px, py], low[x, y])

                                # Check if parent is an articulation point
                                if (
                                    parent_x[px, py] >= 0
                                    and low[x, y] >= discovery_time[px, py]
                                ):
                                    # Convert back to full grid coordinates when marking articulation points
                                    articulation_points[px + min_x, py + min_y] = True
                        continue

                    # Process next neighbor
                    nx_bound = x + dirs_x[dir_idx]
                    ny_bound = y + dirs_y[dir_idx]
                    nx_full = nx_bound + min_x
                    ny_full = ny_bound + min_y
                    stack_dir[stack_size - 1] = dir_idx + 1

                    if (
                        is_valid_bounded(nx_full, ny_full, min_x, max_x, min_y, max_y)
                        and grid[nx_full, ny_full] == empty_value
                    ):
                        if not visited[nx_bound, ny_bound]:
                            parent_x[nx_bound, ny_bound] = x
                            parent_y[nx_bound, ny_bound] = y
                            child_count[x, y] += 1

                            # Push new node to stack
                            stack_x[stack_size] = nx_bound
                            stack_y[stack_size] = ny_bound
                            stack_dir[stack_size] = -1
                            stack_state[stack_size] = 0
                            stack_size += 1
                        elif parent_x[x, y] != nx_bound or parent_y[x, y] != ny_bound:
                            low[x, y] = min(
                                low[x, y], discovery_time[nx_bound, ny_bound]
                            )

                # Check root for articulation point
                if child_count[bound_i, bound_j] > 1:
                    # Convert back to full grid coordinates
                    articulation_points[i, j] = True

    return articulation_points


@njit
def find_articulation_points(
    grid, empty_value=0, min_x=None, max_x=None, min_y=None, max_y=None
):
    """Wrapper function to call the Numba-optimized bounded core function."""
    if min_x is None or max_x is None or min_y is None or max_y is None:
        min_x, max_x = 0, grid.shape[0] - 1
        min_y, max_y = 0, grid.shape[1] - 1

    return find_articulation_points_bounded_core(
        grid, min_x, max_x, min_y, max_y, empty_value
    )


@njit
def get_crop_values(grid, value, pad=10):
    """Crop the grid around the bounding box of the given value."""
    x, y = np.where(grid == value)
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    min_x = max(min_x - pad, 0)
    max_x = min(max_x + pad, grid.shape[0] - 1)
    min_y = max(min_y - pad, 0)
    max_y = min(max_y + pad, grid.shape[1] - 1)

    return [min_x, max_x, min_y, max_y]


# smoke test
def smoke():
    zeroes = np.zeros((10, 10))

    zeroes[5, 5] = 1
    zeroes[5, 6] = 1
    zeroes[6, 5] = 1
    zeroes[6, 6] = 1
    crop = get_crop_values(zeroes, 1, pad=2)
    art_points = find_articulation_points(
        zeroes,
        empty_value=0,
        min_x=crop[0],
        max_x=crop[1],
        min_y=crop[2],
        max_y=crop[3],
    )

    assert crop == [3, 8, 3, 8]
    assert np.any(art_points == np.zeros((10, 10), dtype=np.bool_))

    grid = np.zeros((30, 30), dtype=np.int32)
    FULL = 1

    grid[18, 8] = FULL
    grid[18, 7] = FULL
    grid[18, 6] = FULL

    # Left wall
    grid[17, 6] = FULL
    grid[16, 6] = FULL
    grid[15, 6] = FULL
    grid[14, 6] = FULL
    grid[13, 6] = FULL
    grid[12, 6] = FULL

    # Right wall
    grid[17, 8] = FULL
    grid[16, 8] = FULL
    grid[15, 8] = FULL
    grid[14, 8] = FULL
    grid[13, 8] = FULL
    grid[12, 8] = FULL

    articulation_points = find_articulation_points(grid)
    expected_coords = np.array([[11, 7], [12, 7], [13, 7], [14, 7], [15, 7], [16, 7]])
    
    # plot the grid and highlight articulation points, optionally showing the bounding box
    from matplotlib import pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots(figsize=(8, 8))
    size = grid.shape[0]
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_xticks(np.arange(0, size, 1))
    ax.set_yticks(np.arange(0, size, 1))
    ax.grid(which="both", color="black", linestyle="-", linewidth=0.5)
    for i in range(size):
        for j in range(size):
            if grid[i, j] == FULL:
                rect = patches.Rectangle((j, i), 1, 1, facecolor="black")
                ax.add_patch(rect)
    for i in range(size):
        for j in range(size):
            if articulation_points[i, j]:
                rect = patches.Rectangle((j, i), 1, 1, facecolor="red")
                ax.add_patch(rect)
    plt.show()
    x, y = np.where(articulation_points)
    assert np.all(np.isin(np.column_stack((x, y)), expected_coords))

if __name__ == "__main__":
    smoke()
