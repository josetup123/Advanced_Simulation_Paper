import pandas as pd
import gurobipy as gp 
from gurobipy import GRB
import os
import numpy as np
import sys
import argparse
from parser import *


# - - If we were to open 1,604 stores, what would be the total cost? (Use population * distance as the cost for a city.)
# - - If we were to open 1,257 stores, what would be the total cost? (Use population * distance as the cost for a city.)
# - - If we were to open 1,192 stores, what would be the total cost? (Use population * distance as the cost for a city.)


def main():

    parser = argparse.ArgumentParser(description="Your script description here")

    parser.add_argument("--p", type=int, help="Path to the input file")
    parser.add_argument("--n", type=int, help="Path to the output file")

    args = parser.parse_args()

    # Access the argument values
    P = args.p
    N = args.n


    # P=100#9999999999
    # N=30#1604 

    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)














    # Save the Gurobi output (including the log) to a file
    with open("gurobi_output"+str(N)+"_"+str(P)+".log", "w") as f:
        # Redirect standard output to the file
        sys.stdout = f
        # Redirect standard error to the file
        sys.stderr = f


        m = gp.Model("HW5")



        data=pd.read_excel('HD_LOW.xlsx')
        print(data.shape)
        # data = data[(data['lat'] > 25) & (data['lat'] < 49) & (data['lon'] > -124.8) & (data['lon'] < -66.9)]
        # data=data.drop_duplicates(subset='LocID',keep='first')
        data=data[['LocID', 'HD', 'LOW', 'city', 'STATE_NAME', 'population','lat', 'lon']]
        print(data.shape)
        data['SUM'] = data['HD'] + data['LOW']



        print(data.head())



        # Sample data frame with latitude and longitude columns
        # data = {
        #     'Latitude': [40.7128, 34.0522, 51.5074],
        #     'Longitude': [-74.0060, -118.2437, -0.1278]
        # }

        # df = pd.DataFrame(data)
        print("calculation")
        # Function to calculate Haversine distance between two points
        def haversine(lat1, lon1, lat2, lon2):
            # Radius of the Earth in kilometers
            R = 6371.0

            # Convert latitude and longitude from degrees to radians
            lat1 = np.radians(lat1)
            lon1 = np.radians(lon1)
            lat2 = np.radians(lat2)
            lon2 = np.radians(lon2)

            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            distance = R * c

            return distance

        # Calculate the distance matrix
        num_points = len(data)
        distance_matrix = np.zeros((num_points, num_points))

        for i in range(num_points):
            for j in range(i, num_points):
                lat1, lon1 = data.iloc[i]['lat'], data.iloc[i]['lon']
                lat2, lon2 = data.iloc[j]['lat'], data.iloc[j]['lon']
                distance = haversine(lat1, lon1, lat2, lon2)
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance

        # distance_matrix now contains the distances between all points
        print(distance_matrix)















        ################################################









        # for row in data.iterrows():
        #     m.addVar('x')

        #INDEXES!
        i_=[i for i in range(len(data.iloc[:,0].values))]
        #list(data.iloc[:,0].values)
        j_=[i for i in range(len(data.iloc[:,0].values))]

        #list(data.iloc[:,0].values)

        #MATRICES AND VECTORS!
        population=list(data.iloc[:,5].values)
        distance_matrix


        LIMIT=list(data.iloc[:,-1].values)
        #LOCATION i
        #CITX j

        #Variables

        #WHETHER WE SERVE A STORE WE JUST OPEN
        X=m.addVars(len(i_),len(j_),name='X',vtype=GRB.INTEGER)
        Y=m.addVars(len(i_),name='Y',vtype=GRB.BINARY)



        #EACH CUSTOMER MUST BE ASSIGNED TO EXACTLX ONE FACILITX:
        # m.addConstrs(gp.quicksum(X[i,j] for i in i_ )==1 for j in j_)
        for j in j_:
            m.addConstr(gp.quicksum(X[i, j] for i in i_) == 1)



        #THE NUMBER OF FACILITIES TO BE LOCATED IS FIXED:
        # m.addConstrs(gp.quicksum(X[i,j] for i in i_ )==N)
        m.addConstr(gp.quicksum(Y[i] for i in i_) == N)



        #LIMITING
        for j in j_:
            m.addConstrs((X[i,j]<=Y[i]) for i in i_  )


        #MAKING SURE AT MOST I HAVE THE LIMIT OF STORES IN AXIS

        # m.addConstrs(gp.quicksum(X[i,j] for i in i_ ) <= LIMIT[i]  for j in  j_  )


        m.setObjective( gp.quicksum(population[i]*X[i,j]*distance_matrix[i,j] for i in i_ for j in j_)) 
        m.ModelSense = GRB.MINIMIZE
        m.Params.NonConvex = 2 
        m.params.NumericFocus = 3
        # m.computeIIS()


        m.optimize() 

        m.printAttr('X')

        
















        # Revert standard output and standard error
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


    # format_csv("gurobi_output"+str(N)+"_"+str(P)+".log")


if __name__ == "__main__":
    main()





















    # # Sets and parameters
    # R = regions   # set of all regions

    # B = 30  # total amount ot avocado supplX

    # peak_or_not = 1 # 1 if it is the peak season; 1 if isn't
    # Xear = 2022

    # c_waste = 0.1 # the cost ($) of wasting an avocado
    # c_transport = {'Great_Lakes': .3,'Midsouth':.1,'Northeast':.4,'Northern_New_England':.5,'SouthCentral':.3,'Southeast':.2,'West':.2,'Plains':.2}
    # # the cost of transporting an avocado

    # # Get the lower and upper bounds from the dataset for the price and the number of products to be stocked 
    # a_min = {r: 0 for r in R} # minimum avocado price in each region 
    # a_max = {r: 2 for r in R} # maximum avocado price in each region 
    # b_min = dict(df.groupbX('region')['units_sold'].min())  # minimum number of avocados allocated to each region
    # b_max = dict(df.groupbX('region')['units_sold'].max())   # maximum number of avocados allocated to each region



    # p = m.addVars(R,name="p",lb=a_min, ub=a_max)   # price of avocados in each region
    # x = m.addVars(R,name="x",lb=b_min,ub=b_max)  # quantitX supplied to each region
    # s = m.addVars(R,name="s",lb=0)   # predicted amount of sales in each region for the given price
    # w = m.addVars(R,name="w",lb=0)   # excess wasteage in each region




    # m.setObjective(sum(p[r]*s[r] - c_waste*w[r] - c_transport[r]*x[r] for r in R)) 
    # m.ModelSense = GRB.MAXIMIZE



    # # m.Params.NonConvex = 2 
    # m.optimize() 