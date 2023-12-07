from gurobipy import *

# Create a new model
model = Model("CombinedCostSupplyChain")

# Parameters
num_suppliers = 3
num_demand_points = 4

S = [100, 150, 200]  # Supply capacities
D = [80, 120, 150, 100]  # Demand at each point
combined_param = [[11, 18, 21, 14], [36, 44, 51, 53], [54, 62, 69, 71]]  # Combined transportation cost and distance

# Variables
x = {}
for i in range(num_suppliers):
    for j in range(num_demand_points):
        x[i, j] = model.addVar(vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}")

# Objective function
model.setObjective(quicksum(combined_param[i][j] * x[i, j] for i in range(num_suppliers) for j in range(num_demand_points)), GRB.MINIMIZE)

# Supply constraints
for i in range(num_suppliers):
    model.addConstr(quicksum(x[i, j] for j in range(num_demand_points)) <= S[i], f"supply_{i}")

# Demand constraints
for j in range(num_demand_points):
    model.addConstr(quicksum(x[i, j] for i in range(num_suppliers)) == D[j], f"demand_{j}")

# Optimize the model
model.optimize()

# Print the results
if model.status == GRB.OPTIMAL:
    print("Optimal Solution Found:")
    for i in range(num_suppliers):
        for j in range(num_demand_points):
            print(f"x_{i}_{j} = {x[i, j].x}")
else:
    print("No solution found")
