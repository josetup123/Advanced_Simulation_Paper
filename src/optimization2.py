import pandas as pd
import gurobipy as gp 
from gurobipy import GRB
import os
import numpy as np
import numba as nb

P=1000
N=300

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)


m = gp.Model("HW5")


data=pd.read_excel(r'H:\My Drive\Advanced_Simulation\HW5\HD_LOW.xlsx').iloc[:P,:]

data = data[(data['lat'] > 25) & (data['lat'] < 49) & (data['lon'] > -124.8) & (data['lon'] < -66.9)]

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
@nb.jit
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
print(distance_matrix.shape)















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
# distance_matrix


LIMIT=list(data.iloc[:,-1].values)
print(LIMIT)
#LOCATION i
#CITY j
#
#Variables
#OPEN OR NOT STORES
X=m.addVars(len(i_),name='X',vtype=GRB.INTEGER) #NUMBER OF CITIES OPEN


m.addConstr(gp.quicksum(X[i] for i in i_) == N ) 
m.addConstrs(X[i] <= LIMIT[i] for i in i_ )


m.setObjective(gp.quicksum(population[j]*distance_matrix[i,j]*X[i] for i in i_ for j in j_)) 
m.ModelSense = GRB.MINIMIZE
m.optimize() 
m.printAttr('X')














exit(0)







# # Sets and parameters
# R = regions   # set of all regions

# B = 30  # total amount ot avocado supply

# peak_or_not = 1 # 1 if it is the peak season; 1 if isn't
# year = 2022

# c_waste = 0.1 # the cost ($) of wasting an avocado
# c_transport = {'Great_Lakes': .3,'Midsouth':.1,'Northeast':.4,'Northern_New_England':.5,'SouthCentral':.3,'Southeast':.2,'West':.2,'Plains':.2}
# # the cost of transporting an avocado

# # Get the lower and upper bounds from the dataset for the price and the number of products to be stocked 
# a_min = {r: 0 for r in R} # minimum avocado price in each region 
# a_max = {r: 2 for r in R} # maximum avocado price in each region 
# b_min = dict(df.groupby('region')['units_sold'].min())  # minimum number of avocados allocated to each region
# b_max = dict(df.groupby('region')['units_sold'].max())   # maximum number of avocados allocated to each region



# p = m.addVars(R,name="p",lb=a_min, ub=a_max)   # price of avocados in each region
# x = m.addVars(R,name="x",lb=b_min,ub=b_max)  # quantity supplied to each region
# s = m.addVars(R,name="s",lb=0)   # predicted amount of sales in each region for the given price
# w = m.addVars(R,name="w",lb=0)   # excess wasteage in each region




# m.setObjective(sum(p[r]*s[r] - c_waste*w[r] - c_transport[r]*x[r] for r in R)) 
# m.ModelSense = GRB.MAXIMIZE



# # m.Params.NonConvex = 2 
# m.optimize() 