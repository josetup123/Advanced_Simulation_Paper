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

# Function to perform value iteration to find the optimal value function
def value_iteration(num_iterations):
    V = np.zeros(num_states)
    theta = 1e-6

    for i in range(num_iterations):
        new_V = np.copy(V)
        delta = 0
        for s in range(num_states):
            this_col = s % grid_size
            this_row = s // grid_size
            # terminal
            if (this_row == 1 and this_col == 1):
                continue
            max_V = -10
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
                    elif (new_row == 1 and new_col == 1):
                        r = 10
                    else:
                        r = -1
                    if (max_V <= r + discount_factor * V[new_row*grid_size+new_col]):
                        max_V = r + discount_factor * V[new_row*grid_size+new_col]
            new_V[s] = max_V
            delta = max(delta, abs(max_V - V[s]))
        if delta < theta:
            print(i)
            break
        V = new_V

    return V

# Function to extract the optimal policy
def extract_policy(V):
    policy = np.zeros(num_states)
    for s in range(num_states):
        this_col = s % grid_size
        this_row = s // grid_size
        max_V = -10
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
                if (max_V <= V[new_row*grid_size+new_col]):
                    max_V = V[new_row*grid_size+new_col]
                    policy[s] = a

    return policy

# Main function
def main():
    optimal_value_function = value_iteration(num_iterations)
    optimal_policy = extract_policy(optimal_value_function)
    print("State value:")
    for i in range(grid_size):
        print(optimal_value_function[i*grid_size:i*grid_size+grid_size])
    
    print("Policy:")
    for i in range(grid_size):
        print(optimal_policy[i*grid_size:i*grid_size+grid_size])
    


if __name__ == "__main__":
    main()
