import numpy as np

def create_probability_matrix(num_rows):
    # Initialize an empty matrix with the desired shape
    matrix = np.zeros((num_rows, num_rows), dtype=float)

    # Populate the matrix such that each row sums to 1
    for i in range(num_rows):
        # Generate random probabilities for each state in the row
        row_probabilities = np.random.rand(num_rows)
        # Normalize the probabilities to make the row sum to 1
        row_probabilities /= row_probabilities.sum()
        # Assign the normalized probabilities to the matrix row
        matrix[i, :] = row_probabilities

    return np.mat(matrix)

# Set the number of rows (states)
num_rows = 4

# Create the probability matrix
result_matrix = create_probability_matrix(num_rows)

# Print the result
print(result_matrix)