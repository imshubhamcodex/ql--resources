import matplotlib.pyplot as plt
import numpy as np

def plot_policy_with_values(state_values, policy, grid_size, terminal_states, forbidden, title="Optimal Policy with State Values"):
    rows, cols = grid_size
    grid = np.zeros((rows, cols))

    for r in range(rows):
        for c in range(cols):
            grid[r, c] = state_values.get((r, c), 0)

    plt.figure(figsize=(7,7))
    plt.imshow(grid, cmap="plasma", origin="upper")
    plt.colorbar(label="State Value (V)")

    action_to_delta = {
        "UP": (0.2, 0),
        "DOWN": (-0.2, 0),
        "LEFT": (0, -0.2),
        "RIGHT": (0, 0.2)
    }

    for r in range(rows):
        for c in range(cols):
            if (r, c) in forbidden:
                # Forbidden cells marked X
                plt.text(c, r, "X", color="red", ha="center", va="center", fontsize=18, fontweight="bold")
                continue

            if terminal_states and (r, c) in terminal_states:
                # Mark terminal states with T
                plt.text(c, r, "T", color="lime", ha="center", va="center", fontsize=18, fontweight="bold")
                continue

            if (r, c) not in policy or policy[(r, c)] is None:
                continue

            p = policy[(r, c)]
            best_action = max(p, key=p.get) if isinstance(p, dict) else p
            dy, dx = action_to_delta[best_action]
            plt.arrow(c, r, dx, -dy, head_width=0.2, head_length=0.2, fc="black", ec="black")

            # overlay state value
            plt.text(c, r, f"{round(state_values.get((r, c), 0), 2)}", color="red",
                     ha="center", va="center", fontweight="bold")

    plt.title(title)
    plt.xticks(range(cols))
    plt.yticks(range(rows))
    plt.show()


def print_result(state_values, policy, grid_size, forbidden, stochastic=False):
    rows, cols = grid_size
    print("Optimal State Values (V*):")
    header = "\t" + "\t".join([f"Col {c}" for c in range(cols)])
    print(header)
    for r in range(rows):
        row_values = []
        for c in range(cols):
            row_values.append(f"{round(state_values[(r, c)], 2):>5}")
        print(f"Row {r}:\t" + "\t".join(row_values))

    print("\nOptimal Policy (π*):")
    if stochastic:
        header = "\t" + "\t".join([f"Col {c}       " for c in range(cols)])
    else:
        header = "\t" + "\t".join([f"Col {c}" for c in range(cols)])
    print(header)
    for r in range(rows):
        row_policy = []
        for c in range(cols):
            p = policy[(r, c)]
            if p is None:
                if stochastic:
                    row_policy.append("TERMINAL")
                else:
                    row_policy.append("TERM")
            else:
                if stochastic:
                    best_action = max(p, key=p.get)
                    row_policy.append(f"{best_action}:{round(p[best_action],3)}")
                else:
                    row_policy.append(p)
        print(f"Row {r}:\t" + "\t".join(row_policy))
