from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
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

# Example coordinates (latitude and longitude in degrees)
lat1, lon1 = 37.7749, -122.4194
lat2, lon2 = 34.0522, -118.2437

# Calculate the distance
distance = haversine_distance(lat1, lon1, lat2, lon2)

print(f"Distance: {distance} m")