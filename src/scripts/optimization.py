import pandas as pd
import gurobipy as gp 
from gurobipy import GRB
import os
import numpy as np
import sys
import argparse
import random
import mysql.connector
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from sqlalchemy.orm import sessionmaker

from math import radians, sin, cos, sqrt, atan2


import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.stats import norm


# - - If we were to open 1,604 stores, what would be the total cost? (Use population * distance as the cost for a city.)
# - - If we were to open 1,257 stores, what would be the total cost? (Use population * distance as the cost for a city.)
# - - If we were to open 1,192 stores, what would be the total cost? (Use population * distance as the cost for a city.)



def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    rad_lat1 = radians(lat1)
    rad_lon1 = radians(lon1)
    rad_lat2 = radians(lat2)
    rad_lon2 = radians(lon2)

    # Calculate differences in coordinates
    d_lat = rad_lat2 - rad_lat1
    d_lon = rad_lon2 - rad_lon1

    # Haversine formula
    a = sin(d_lat / 2) ** 2 + cos(rad_lat1) * cos(rad_lat2) * sin(d_lon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Calculate the distance
    distance = R * c * 1000

    return distance



def main():


    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)

    #HERE WE READ THE TABLES:

    engine = create_engine('mysql+mysqlconnector://root:ilab301@smartshots.ise.utk.edu:3306/DB', echo=False)


    # # Replace 'your_table_name' with your actual table name
    # table_name = 'heatpoints_optimized'
    # # Print the DataFrame
    # Read data from MySQL into a DataFrame
    # query=f"SELECT * FROM {table_name}"
    # heatpoints_optimized = pd.DataFrame(engine.connect().execute(text(query)))
    # print(heatpoints_optimized)



    # Replace 'your_table_name' with your actual table name
    table_name = 'firestations'
    # Print the DataFrame
    # Read data from MySQL into a DataFrame
    query=f"SELECT * FROM {table_name}"
    firestations_optimized = pd.DataFrame(engine.connect().execute(text(query)))
    print(firestations_optimized)
    # firestations=firestations_optimized.copy()


    # Replace 'your_table_name' with your actual table name
    table_name = 'firepoints_optimized'
    # Print the DataFrame
    # Read data from MySQL into a DataFrame
    query=f"SELECT * FROM {table_name}"
    firepoints_optimized = pd.DataFrame(engine.connect().execute(text(query)))
    

    # Replace 'your_table_name' with your actual table name
    table_name = 'firestations'
    # Print the DataFrame
    # Read data from MySQL into a DataFrame
    query=f"SELECT * FROM {table_name}"
    firestations = pd.DataFrame(engine.connect().execute(text(query)))
    
    #THIS IS GUROBI INPUT!




    # Save the Gurobi output (including the log) to a file #+str(N)+"_"+str(P)
    # with open("../gurobi_output"+".log", "w") as f:
    #     # Redirect standard output to the file
    #     sys.stdout = f
    #     # Redirect standard error to the file
    #     sys.stderr = f



    print(firestations_optimized)
    print(firepoints_optimized)







    #NAMING:

    firestations_optimized['NAME']="FS"+"_"+firestations_optimized['latitude'].astype('str')+"_"+firestations_optimized['longitude'].astype('str')
    names_firestations_optimized=list(firestations_optimized['NAME'].values)
    firepoints_optimized['NAME']=list("FP"+"_"+firepoints_optimized['latitude'].astype('str')+"_"+firepoints_optimized['longitude'].astype('str'))
    names_firepoints_optimized=list(firepoints_optimized['NAME'].values)

    




    print("calculation")
    # Function to calculate Haversine distance between two points
    # Calculate the distance matrix
    num_points1 = len(firestations_optimized)
    print(num_points1)
    num_points2 = len(firepoints_optimized)
    print(num_points2)
    # cost_matrix = np.zeros((num_points1, num_points2))

    # for i in range(num_points1):
    #     for j in range(i, num_points2):
    #         lat1, lon1 = firestations_optimized.iloc[i,0], firestations_optimized.iloc[i,1]
    #         lat2, lon2 = firepoints_optimized.iloc[j,0], firepoints_optimized.iloc[j,1]
    #         distance = haversine(lat1, lon1, lat2, lon2)
    #         cost_matrix[i][j] = distance 
    #         cost_matrix[j][i] = distance 

    # Initialize an empty distance matrix
    cost_matrix = pd.DataFrame(index=firestations_optimized.index, columns=firepoints_optimized.index)

    # Iterate over rows of both dataframes to calculate distances
    for i, row1 in firestations_optimized.iterrows():
        for j, row2 in firepoints_optimized.iterrows():
            cost_matrix.at[i, j] = haversine(row1['latitude'], row1['longitude'], row2['latitude'], row2['longitude'])

    # Display the distance matrix
    cost_matrix=cost_matrix.to_numpy()
    # print(cost_matrix)

    # distance_matrix now contains the distances between all points
    print("printingh cost matrix")
    print(cost_matrix)

    print(cost_matrix.shape)







    model = gp.Model("DIGITAL_TWIN_FIRE_REPONSE")

    #GUROBI WILL HANDLE A CODING SYSTEM THAT CONSITS OF FUNCTION_LTITUDE_LONGITUDE
    """
    FIREPOITNS FP ARE DEMANS AND FIRESTATIONS FS ARE WAREHOURSES

    """






    ################################################









    # for row in data.iterrows():
    #     m.addVar('x')

    #INDEXES!
    # suppliers=[i for i in range(len(firestations_optimized.iloc[:,0].values))]
    suppliers=list(firestations_optimized['fire_index'].values)
    print("SUPPLIERS: "+ str(len(suppliers)))
    #list(data.iloc[:,0].values)
    # locations=[i for i in range(len(firepoints_optimized.iloc[:,0].values))]
    locations=list(firepoints_optimized['fire_index'].values)
    print("LOCATIONS: "+ str(len(locations)))

    #list(data.iloc[:,0].values)



    # print(suppliers)
    # print(locations)
    # input()

    model = gp.Model("StochasticRouteAllocation")

    S = suppliers#[100, 150, 200]  # Supply capacities
    D_mean = locations#[80, 120, 150, 100]  # Mean demand
    D_stddev = [random.randint(1, 20) for _ in range(len(locations))] #[10, 15, 20, 12]  # Standard deviation of demand
    combined_param = cost_matrix
    #[[11, 18, 21, 14], [36, 44, 51, 53], [54, 62, 69, 71]]  # Combined transportation cost and distance



    num_suppliers=len(S)
    num_demand_points=len(D_mean)

    # Variables
    x = {}
    for i in range(num_suppliers):
        for j in range(num_demand_points):
            x[i, j] = model.addVar(vtype=GRB.CONTINUOUS,name=f"x_{names_firestations_optimized[i]}_{names_firepoints_optimized[j]}")

    # Random variables for stochastic demand
    D = {}
    for j in range(num_demand_points):
        print()
        D[j] = model.addVar(lb=D_mean[j] - 3 * D_stddev[j], ub=D_mean[j] + 3 * D_stddev[j], vtype=GRB.CONTINUOUS, name=f"D_{names_firepoints_optimized[j]}")#


    # Supply constraints
    for i in range(num_suppliers):
        model.addConstr(gp.quicksum(x[i, j] for j in range(num_demand_points)) <= S[i])#, f"supply_{i}"

    # Demand constraints
    for j in range(num_demand_points):
        model.addConstr(gp.quicksum(x[i, j] for i in range(num_suppliers)) == D[j])#, f"demand_{j}"


    # Objective function
    # combined_param[i][j] * 
    model.setObjective(gp.quicksum(combined_param[i][j] * x[i, j] for i in range(num_suppliers) for j in range(num_demand_points)), GRB.MINIMIZE)

    # Optimize the model
    model.optimize()
    model.printAttr('X')
    print("printingh cost matrix")
    # print(combined_param)
    print(combined_param.shape)


    

    # # Print the results
    # if model.status == GRB.OPTIMAL:
    #     print("Optimal Solution Found:")
    #     for i in range(num_suppliers):
    #         for j in range(num_demand_points):
    #             print(f"x_{i}_{j} = {x[i, j].x}")
    # else:
    #     print("No solution found")



    # # Create other necessary data
    # # locations = range(num_locations) #DEMAND j
    # # suppliers = range(num_suppliers) #SUPLIER i
    # demand_avg = firepoints_optimized['fire_index'].values
    # # {i: 10 for i in locations}  # Average demand
    # demand_std_dev = {i: 2 for i in locations}  # Standard deviation of demand
    # M = 1000  # A big-M constant
    # alpha = 1.0  # Weighting factor for cost
    # beta = 1.0   # Weighting factor for resource allocation
    # confidence_level_low = 0.05
    # confidence_level_high = 0.05

    # # Create a Gurobi model
    # # model = gp.Model("StochasticRouteAllocation")
    # model.setParam('OutputFlag', 1)



    # Decision variables
    # x = model.addVars(locations, locations, vtype=GRB.BINARY, name="x") #
    # z = model.addVars(locations, suppliers, vtype=GRB.BINARY, name="z") #ROUTE SELECTION
    # y = model.addVars(locations, vtype=GRB.CONTINUOUS, name="y")

    # # Objective function
    # obj_expr = alpha * gp.quicksum(cost_matrix[i, j] * x[i, j] for i in locations for j in locations) - beta * gp.quicksum(y[i] for i in locations)
            
    # model.setObjective(obj_expr, GRB.MINIMIZE)

    # # Connectivity constraints
    # model.addConstrs((x.sum(i, '*') == 1 for i in locations), name="connectivity_out")
    # model.addConstrs((x.sum('*', j) == 1 for j in locations), name="connectivity_in")

    # # Supplier assignment constraints
    # model.addConstrs((z.sum(i, '*') >= 1 for i in locations), name="supplier_assignment")

    # # Resource allocation constraints
    # model.addConstrs((y[i] >= demand_avg[i] + M * gp.quicksum(x[i, j] for j in locations) for i in locations), name="resource_allocation")

    # Stochastic demand constraints
    # model.addConstrs((y[i] >= demand_avg[i] - demand_std_dev[i] * norm.ppf(1 - confidence_level_low) for i in locations), name="stochastic_low")
    # model.addConstrs((y[i] <= demand_avg[i] + demand_std_dev[i] * norm.ppf(1 - confidence_level_high) for i in locations), name="stochastic_high")

    # Confidence level constraints
    # model.addConstrs((y[i] - demand_avg[i] >= -demand_std_dev[i] * norm.ppf(1 - confidence_level_low) for i in locations), name="confidence_low")
    # model.addConstrs((demand_avg[i] - y[i] >= -demand_std_dev[i] * norm.ppf(1 - confidence_level_high) for i in locations), name="confidence_high")


    # # Decision variables
    # x = model.addVars(locations, locations, vtype=GRB.BINARY, name="x")
    # z = model.addVars(locations, suppliers, vtype=GRB.BINARY, name="z")
    # y = model.addVars(locations, vtype=GRB.CONTINUOUS, name="y")

    # # Objective function
    # obj_expr = alpha * gp.quicksum(cost_matrix[i, j] * x[i, j] for i in locations for j in locations) \
    #         - beta * gp.quicksum(y[i] for i in locations)
    # model.setObjective(obj_expr, GRB.MINIMIZE)

    # # Connectivity constraints
    # model.addConstrs((x.sum(i, '*') == 1 for i in locations), name="connectivity_out")
    # model.addConstrs((x.sum('*', j) == 1 for j in locations), name="connectivity_in")

    # # Supplier assignment constraints
    # model.addConstrs((z.sum(i, '*') >= 1 for i in locations), name="supplier_assignment")

    # # Resource allocation constraints
    # model.addConstrs((y[i] >= M * gp.quicksum(x[i, j] for j in locations) for i in locations), name="resource_allocation")



    # # Solve the model
    # model.optimize()
    # # model.computeIIS()

    rows=[]



    # Print the solution
    if model.status == GRB.OPTIMAL:
        print("\nOptimal Solution:")
        for i, j in x:
            if x[i, j].x > 0.0:
                print(f"Route from {i} to {j}: " + str(x[i,j].x))
                # print(x[i,j].VarName)
                # input()
                latitude=str(x[i,j].VarName).split("_")[2]
                longitude=str(x[i,j].VarName).split("_")[3]
                fire_index=firestations_optimized.iloc[i,2].astype(int)
                activated=int(1)
                fire_index_assigned=int(x[i,j].x)
                latitude_destination=str(x[i,j].VarName).split("_")[-2]
                longitude_destination=str(x[i,j].VarName).split("_")[-1]
                rows.append({'latitude':latitude,'longitude':longitude,'fire_index':fire_index,'fire_index_assigned':fire_index_assigned,'activated':activated,'latitude_destination':latitude_destination,'longitude_destination':longitude_destination})
        data=pd.DataFrame(data=rows)
        # print(data)

        data['ID']=data['latitude'].astype(str)+"_"+data['longitude'].astype(str)
        firestations['ID']=firestations['latitude'].astype(str)+"_"+firestations['longitude'].astype(str)
        # print(firestations)
        # print(data)



        #HERE MERGE 2 TABLES

        firestations=firestations[['ID','latitude','longitude','fire_index']].merge(data[['ID','fire_index_assigned','activated','latitude_destination','longitude_destination']],on='ID',how='left')

        firestations.drop(columns=['ID'],inplace=True)
        
        
        #REPLACING NULLS:
        firestations.fire_index_assigned.fillna(0, inplace=True)
        firestations.activated.fillna(0, inplace=True)
        firestations.latitude_destination.fillna(999999, inplace=True)
        firestations.longitude_destination.fillna(999999, inplace=True)


        # print(firestations)
        print("CHECK THIS")

        engine = create_engine('mysql+mysqlconnector://root:ilab301@smartshots.ise.utk.edu:3306/DB', echo=False)

        firestations.to_sql(name='firestations_optimized', con=engine, if_exists ='replace', index=False)
        



        # for i, s in z:
        #     if z[i, s].x > 0.5:
        #         print(f"Supplier {s} serves location {i}")
        # for i in y:
        #     print(f"Resource allocation at location {i}: {y[i].x}")
        exit(0)
    else:
        print("\nNo optimal solution found.")
        exit(-1)


    # #MATRICES AND VECTORS!
    # population=list(data.iloc[:,5].values)
    # distance_matrix


    # LIMIT=list(data.iloc[:,-1].values)
    # #LOCATION i
    # #CITX j

    # #Variables

    # #WHETHER WE SERVE A STORE WE JUST OPEN
    # X=m.addVars(len(i_),len(j_),name='X',vtype=GRB.INTEGER)
    # Y=m.addVars(len(i_),name='Y',vtype=GRB.BINARY)



    # #EACH CUSTOMER MUST BE ASSIGNED TO EXACTLX ONE FACILITX:
    # # m.addConstrs(gp.quicksum(X[i,j] for i in i_ )==1 for j in j_)
    # for j in j_:
    #     m.addConstr(gp.quicksum(X[i, j] for i in i_) == 1)



    # #THE NUMBER OF FACILITIES TO BE LOCATED IS FIXED:
    # # m.addConstrs(gp.quicksum(X[i,j] for i in i_ )==N)
    # m.addConstr(gp.quicksum(Y[i] for i in i_) == N)



    # #LIMITING
    # for j in j_:
    #     m.addConstrs((X[i,j]<=Y[i]) for i in i_  )


    # #MAKING SURE AT MOST I HAVE THE LIMIT OF STORES IN AXIS

    # # m.addConstrs(gp.quicksum(X[i,j] for i in i_ ) <= LIMIT[i]  for j in  j_  )


    # m.setObjective( gp.quicksum(population[i]*X[i,j]*distance_matrix[i,j] for i in i_ for j in j_)) 
    # m.ModelSense = GRB.MINIMIZE
    # m.Params.NonConvex = 2 
    # m.params.NumericFocus = 3
    # # m.computeIIS()


    # m.optimize() 

    # m.printAttr('X')

    
















    #     # # Revert standard output and standard error
    #     # sys.stdout = sys.__stdout__
    #     # sys.stderr = sys.__stderr__



if __name__ == "__main__":
    main()













