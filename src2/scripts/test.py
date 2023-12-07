import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.stats import norm

# Sample data (replace with your actual data)
num_locations = 5
num_suppliers = 3



# Create a fake cost matrix with random values
np.random.seed(42)
fake_cost_matrix = np.random.randint(1, 10, size=(num_locations, num_locations))
cost_matrix = {(i, j): fake_cost_matrix[i][j] for i in range(num_locations) for j in range(num_locations)}

# Create other necessary data
locations = range(num_locations)
suppliers = range(num_suppliers)
demand_avg = {i: 10 for i in locations}  # Average demand
demand_std_dev = {i: 2 for i in locations}  # Standard deviation of demand
M = 1000  # A big-M constant
alpha = 1.0  # Weighting factor for cost
beta = 1.0   # Weighting factor for resource allocation
confidence_level_low = 0.05
confidence_level_high = 0.05

# Create a Gurobi model
model = gp.Model("StochasticRouteAllocation")
model.setParam('OutputFlag', 1)
# Decision variables
x = model.addVars(locations, locations, vtype=GRB.BINARY, name="x")
z = model.addVars(locations, suppliers, vtype=GRB.BINARY, name="z")
y = model.addVars(locations, vtype=GRB.CONTINUOUS, name="y")

# Objective function
obj_expr = alpha * gp.quicksum(cost_matrix[i, j] * x[i, j] for i in locations for j in locations) - beta * gp.quicksum(y[i] for i in locations)
           
model.setObjective(obj_expr, GRB.MINIMIZE)

# Connectivity constraints
model.addConstrs((x.sum(i, '*') == 1 for i in locations), name="connectivity_out")
model.addConstrs((x.sum('*', j) == 1 for j in locations), name="connectivity_in")

# Supplier assignment constraints
model.addConstrs((z.sum(i, '*') >= 1 for i in locations), name="supplier_assignment")

# Resource allocation constraints
model.addConstrs((y[i] >= demand_avg[i] + M * gp.quicksum(x[i, j] for j in locations) for i in locations), name="resource_allocation")

# Stochastic demand constraints
model.addConstrs((y[i] >= demand_avg[i] - demand_std_dev[i] * norm.ppf(1 - confidence_level_low) for i in locations), name="stochastic_low")
model.addConstrs((y[i] <= demand_avg[i] + demand_std_dev[i] * norm.ppf(1 - confidence_level_high) for i in locations), name="stochastic_high")

# Confidence level constraints
model.addConstrs((y[i] - demand_avg[i] >= -demand_std_dev[i] * norm.ppf(1 - confidence_level_low) for i in locations), name="confidence_low")
model.addConstrs((demand_avg[i] - y[i] >= -demand_std_dev[i] * norm.ppf(1 - confidence_level_high) for i in locations), name="confidence_high")

# Solve the model
model.optimize()

# Print the solution
if model.status == GRB.OPTIMAL:
    print("\nOptimal Solution:")
    for i, j in x:
        if x[i, j].x > 0.5:
            print(f"Route from {i} to {j}")
    for i, s in z:
        if z[i, s].x > 0.5:
            print(f"Supplier {s} serves location {i}")
    for i in y:
        print(f"Resource allocation at location {i}: {y[i].x}")
else:
    print("\nNo optimal solution found.")
