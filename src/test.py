import pandas as pd
import numpy as np

# Sample data frame with latitude and longitude columns
data = {
    'Latitude': [40.7128, 34.0522, 51.5074],
    'Longitude': [-74.0060, -118.2437, -0.1278]
}

df = pd.DataFrame(data)

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
num_points = len(df)
distance_matrix = np.zeros((num_points, num_points))

for i in range(num_points):
    for j in range(i, num_points):
        lat1, lon1 = df.iloc[i]['Latitude'], df.iloc[i]['Longitude']
        lat2, lon2 = df.iloc[j]['Latitude'], df.iloc[j]['Longitude']
        distance = haversine(lat1, lon1, lat2, lon2)
        distance_matrix[i][j] = distance
        distance_matrix[j][i] = distance

# distance_matrix now contains the distances between all points
print(distance_matrix.shape)