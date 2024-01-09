import numpy as np

# 5*5
grid_size = 5
num_states = grid_size * grid_size
# left, right, up, down
num_actions = 4
discount_factor = 0.9
num_iterations = 100

# Function to check if a state is within the grid
def is_valid_state(row, col):
    return 0 <= row < grid_size and 0 <= col < grid_size

# Function to initialize the policy arbitrarily
def initialize_policy():
    return np.random.randint(num_actions, size=num_states)

# Function to perform policy evaluation
def policy_evaluation(policy):
    V = np.zeros(num_states)

    while True:
        delta = 0
        for s in range(num_states):
            this_col = s % grid_size
            this_row = s // grid_size

            if (this_row == 1 and this_col == 1):
                V[s] = 10
                continue

            v = 0
            action = policy[s]
            if (action == 0):
                new_col = this_col - 1
                new_row = this_row
            elif (action == 1):
                new_col = this_col + 1
                new_row = this_row
            elif (action == 2):
                new_col = this_col
                new_row = this_row - 1
            elif (action == 3):
                new_col = this_col
                new_row = this_row + 1

            if_valid = is_valid_state(new_row, new_col)
            if (if_valid):
                if (new_row == 2 and new_col == 4):
                    r = -2
                else:
                    r = -1
                v = r + discount_factor * V[new_row * grid_size + new_col]

            delta = max(delta, abs(V[s] - v))
            V[s] = v

        if delta < 1e-6:
            break

    return V

# Function to extract the optimal policy
def extract_optimal_policy(V):
    policy = np.zeros(num_states, dtype=int)

    for s in range(num_states):
        this_col = s % grid_size
        this_row = s // grid_size
        max_value = -np.inf
        for a in range(num_actions):
            if (a == 0):
                new_col = this_col - 1
                new_row = this_row
            elif (a == 1):
                new_col = this_col + 1
                new_row = this_row
            elif (a == 2):
                new_col = this_col
                new_row = this_row - 1
            elif (a == 3):
                new_col = this_col
                new_row = this_row + 1

            if_valid = is_valid_state(new_row, new_col)
            if (if_valid):
                if (new_row == 2 and new_col == 4):
                    r = -2
                else:
                    r = -1
                value = r + discount_factor * V[new_row * grid_size + new_col]

                if value > max_value:
                    max_value = value
                    policy[s] = a

    return policy

# Function for policy iteration
def policy_iteration(num_iterations):
    policy = initialize_policy()

    for _ in range(num_iterations):
        V = policy_evaluation(policy)
        new_policy = extract_optimal_policy(V)

        if (new_policy == policy).all():
            print(_)
            break

        policy = new_policy

    return policy

# Main function
def main():
    optimal_policy = policy_iteration(num_iterations)
    for i in range(grid_size):
        print(optimal_policy[i*grid_size:i*grid_size+grid_size])

if __name__ == "__main__":
    main()
