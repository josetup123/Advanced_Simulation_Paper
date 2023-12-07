import requests
import pandas as pd
from datetime import datetime, timedelta
import subprocess

import pandas as pd


import datetime
import pytz
from datetime import datetime

import mysql.connector
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from sqlalchemy.orm import sessionmaker
"""
https://firms.modaps.eosdis.nasa.gov/academy/
https://firms.modaps.eosdis.nasa.gov/usfs/active_fire/#kml-kmz


https://wiki.earthdata.nasa.gov/display/FIRMS/2022/07/14/Wildfire+detection+in+the+US+and+Canada+within+a+minute+of+satellite+observation



LIKELY FORECAST:

National Fire Danger Rating System (NFDRS)

"""

import os




current_directory = os.path.dirname(__file__)
os.chdir(current_directory)
print(os.getcwd())

def delete_all_files_in_folder(folder_path):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Iterate over the list and delete each file
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

# Example: Delete all files from a folder named 'example_folder'
folder_path = '../data'
# delete_all_files_in_folder(folder_path)


current_date_time = datetime.utcnow()
current_date=current_date_time.strftime("%Y-%m-%d")

print(current_date)


# input()
# Let's set your map key that was emailed to you. It should look something like 'abcdef1234567890abcdef1234567890'
MAP_KEY = '00c7a6e7907b1f5faa49c87c989a9f32'
#MAP_KEY = 'abcdef1234567890abcdef1234567890'



# let's query data_availability to find out what date range is available for various datasets
# we will explain these datasets a bit later

# this url will return information about all supported sensors and their corresponding datasets
# instead of 'all' you can specify individual sensor, ex:LANDSAT_NRT
# da_url = 'https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/' + MAP_KEY + '/all'
# df = pd.read_csv(da_url)
# print(df)

# let's create a simple function that tells us how many transactions we have used.
# We will use this in later examples

def get_transaction_count(url) :
  count = 0
  try:
    df = pd.read_json(url,  typ='series')
    count = df['current_transactions']
  except:
    print ("Error in our call.")
  return count

# tcount = get_transaction_count()
# print ('Our current transaction count is %i' % tcount)



# # in this example let's look at VIIRS NOAA-20, entire world and the most recent day
# area_url = 'https://firms.modaps.eosdis.nasa.gov/api/area/csv/' + MAP_KEY + '/VIIRS_NOAA20_NRT/world/1'
# start_count = get_transaction_count()
# df_area = pd.read_csv(area_url)
# end_count = get_transaction_count()
# print ('We used %i transactions.' % (end_count-start_count))

# /api/area/csv/00c7a6e7907b1f5faa49c87c989a9f32/LANDSAT_NRT/world/1
#CHECK: https://firms.modaps.eosdis.nasa.gov/api/area
# in this example let's look at VIIRS NOAA-20, entire world and the most recent day #NOW LANDSAT_NRT
area_url = 'https://firms.modaps.eosdis.nasa.gov/api/area/csv/' + MAP_KEY + '/LANDSAT_NRT/world/1/'+str(current_date) #WORLD BECAUSE IT IS ONLY USA NAD CA


landsat = pd.read_csv(area_url)
# H:\My Drive\Advanced_Simulation\Advanced_Simulation_Paper\src\data
landsat.to_csv('../data/landsat.csv',index=False)




"""

Attribute	Short Description	Long Description
Satellite	Satellite	L8 = Landsat 8; L9 = Landsat 9
Latitude	Latitude (decimal degrees)	Latitude of pixel center
Longitude	Longitude (decimal degrees)	Longitude of pixel center
Path	Path	Path number as identified in the World Reference System-2 (WRS-2)
Row	Row	Row number as identified in the World Reference System-2 (WRS-2)
Track	Track	The pixel location in the along-track dimension of the Landsat path based on the OLI Line-of-Sight (LOS) coordinate system. Generally increases in value from north to south for daytime and nighttime overpasses.
Scan	Scan	The pixel location in the cross-track dimension of the Landsat path based on the OLI Line-of-Sight (LOS) coordinate system. Generally increases in value from west to east for daytime and nighttime overpasses.
Acquire_Time	Date and time of start of scan	Date and time of acquisition/overpass of the satellite (in UTC). The format is YYYY-MM-DD followed by HH:MM-SS. For example, 2022-07-23 10:09:00.
Confidence	Confidence	Value domain: H, M, and L:
H: Higher confidence
M: Medium confidence
L: Lower confidence
Version	Version (FIRMS reference only)	1.0.7 NRT 
DayNight	Day or Night	D= Daytime fire, N= Nighttime fire

"""



area_url = 'https://firms.modaps.eosdis.nasa.gov/api/area/csv/' + MAP_KEY + '/MODIS_NRT/world/1/'+str(current_date) # oR WORLD-126.848974, -118.848974, -63.885444, -76.384358

modis = pd.read_csv(area_url)
# H:\My Drive\Advanced_Simulation\Advanced_Simulation_Paper\src\data
modis.to_csv('../data/modis.csv',index=False)




"""

Attribute
Short Description
Long Description
Latitude	Latitude	Center of nominal 375 m fire pixel.
Longitude	Longitude	Center of nominal 375 m fire pixel.
Bright_ti4	Brightness temperature I-4	VIIRS I-4 channel brightness temperature of the fire pixel measured in Kelvin.
Scan	Along Scan pixel size	The algorithm produces approximately 375 m pixels at nadir. Scan and track reflect actual pixel size.
Track	Along Track pixel size	The algorithm produces approximately 375 m pixels at nadir. Scan and track reflect actual pixel size.
Acq_Date	Acquisition Date	Date of VIIRS acquisition.
Acq_Time	Acquisition Time	Time of acquisition/overpass of the satellite (in UTC).
Satellite	Satellite	N= Suomi National Polar-orbiting Partnership (Suomi NPP).
Confidence	Confidence	
This value is based on a collection of intermediate algorithm quantities used in the detection process. It is intended to help users gauge the quality of individual hotspot/fire pixels. Confidence values are set to low, nominal, and high. Low confidence daytime fire pixels are typically associated with areas of Sun glint and lower relative temperature anomaly (<15 K) in the mid-infrared channel I4. Nominal confidence pixels are those free of potential Sun glint contamination during the day and marked by strong (>15 K) temperature anomaly in either day or nighttime data. High confidence fire pixels are associated with day or nighttime saturated pixels.

Please note: Low confidence nighttime pixels occur only over the geographic area extending from 11° E to 110° W and 7° N to 55° S. This area describes the region of influence of the South Atlantic Magnetic Anomaly which can cause spurious brightness temperatures in the mid-infrared channel I4 leading to potential false positive alarms. These have been removed from the NRT data distributed by FIRMS.

Version	Version (collection and source)	
Version identifies the collection (e.g., VIIRS Collection 1 or VIIRS Collection 2), and source of data processing (Ultra Real-Time (URT suffix added to collection), Real-Time (RT suffix), Near Real-Time (NRT suffix) or Standard Processing (collection only). For example:

"2.0URT" - Collection 2 Ultra Real-Time processing.
"2.0RT" - Collection 2 Real-Time processing.
"1.0NRT" - Collection 1 Near Real-Time processing.
"1.0" - Collection 1 Standard processing.

Bright_ti5	Brightness temperature I-5	I-5 Channel brightness temperature of the fire pixel measured in Kelvin.
FRP	Fire Radiative Power	
FRP depicts the pixel-integrated fire radiative power in megawatts (MW). Given the unique spatial and spectral resolution of the data, the VIIRS 375 m fire detection algorithm was customized and tuned to optimize its response over small fires while balancing the occurrence of false alarms. Frequent saturation of the mid-infrared I4 channel (3.55-3.93 µm) driving the detection of active fires requires additional tests and procedures to avoid pixel classification errors. As a result, sub-pixel fire characterization (e.g., fire radiative power [FRP] retrieval) is only viable across small and/or low-intensity fires. Systematic FRP retrievals are based on a hybrid approach combining 375 and 750 m data. In fact, starting in 2015 the algorithm incorporated additional VIIRS channel M13 (3.973-4.128 µm) 750 m data in both aggregated and unaggregated format.

Type*	Inferred hot spot type	0 = presumed vegetation fire
1 = active volcano
2 = other static land source
3 = offshore detection (includes all detections over water)
DayNight	Day or Night	
D= Daytime fire, N= Nighttime fire

"""




area_url = 'https://firms.modaps.eosdis.nasa.gov/api/area/csv/' + MAP_KEY + '/VIIRS_NOAA20_NRT/world/1/'+str(current_date)

viirs = pd.read_csv(area_url)
# H:\My Drive\Advanced_Simulation\Advanced_Simulation_Paper\src\data
viirs.to_csv('../data/viirs.csv',index=False)



"""

Attribute
Short Description
Long Description
Latitude	Latitude	Center of nominal 375 m fire pixel.
Longitude	Longitude	Center of nominal 375 m fire pixel.
Bright_ti4	Brightness temperature I-4	VIIRS I-4 channel brightness temperature of the fire pixel measured in Kelvin.
Scan	Along Scan pixel size	The algorithm produces approximately 375 m pixels at nadir. Scan and track reflect actual pixel size.
Track	Along Track pixel size	The algorithm produces approximately 375 m pixels at nadir. Scan and track reflect actual pixel size.
Acq_Date	Acquisition Date	Date of VIIRS acquisition.
Acq_Time	Acquisition Time	Time of acquisition/overpass of the satellite (in UTC).
Satellite	Satellite	N= Suomi National Polar-orbiting Partnership (Suomi NPP).
Confidence	Confidence	
This value is based on a collection of intermediate algorithm quantities used in the detection process. It is intended to help users gauge the quality of individual hotspot/fire pixels. Confidence values are set to low, nominal, and high. Low confidence daytime fire pixels are typically associated with areas of Sun glint and lower relative temperature anomaly (<15 K) in the mid-infrared channel I4. Nominal confidence pixels are those free of potential Sun glint contamination during the day and marked by strong (>15 K) temperature anomaly in either day or nighttime data. High confidence fire pixels are associated with day or nighttime saturated pixels.

Please note: Low confidence nighttime pixels occur only over the geographic area extending from 11° E to 110° W and 7° N to 55° S. This area describes the region of influence of the South Atlantic Magnetic Anomaly which can cause spurious brightness temperatures in the mid-infrared channel I4 leading to potential false positive alarms. These have been removed from the NRT data distributed by FIRMS.

Version	Version (collection and source)	
Version identifies the collection (e.g., VIIRS Collection 1 or VIIRS Collection 2), and source of data processing (Ultra Real-Time (URT suffix added to collection), Real-Time (RT suffix), Near Real-Time (NRT suffix) or Standard Processing (collection only). For example:

"2.0URT" - Collection 2 Ultra Real-Time processing.
"2.0RT" - Collection 2 Real-Time processing.
"1.0NRT" - Collection 1 Near Real-Time processing.
"1.0" - Collection 1 Standard processing.

Bright_ti5	Brightness temperature I-5	I-5 Channel brightness temperature of the fire pixel measured in Kelvin.
FRP	Fire Radiative Power	
FRP depicts the pixel-integrated fire radiative power in megawatts (MW). Given the unique spatial and spectral resolution of the data, the VIIRS 375 m fire detection algorithm was customized and tuned to optimize its response over small fires while balancing the occurrence of false alarms. Frequent saturation of the mid-infrared I4 channel (3.55-3.93 µm) driving the detection of active fires requires additional tests and procedures to avoid pixel classification errors. As a result, sub-pixel fire characterization (e.g., fire radiative power [FRP] retrieval) is only viable across small and/or low-intensity fires. Systematic FRP retrievals are based on a hybrid approach combining 375 and 750 m data. In fact, starting in 2015 the algorithm incorporated additional VIIRS channel M13 (3.973-4.128 µm) 750 m data in both aggregated and unaggregated format.

Type*	Inferred hot spot type	0 = presumed vegetation fire
1 = active volcano
2 = other static land source
3 = offshore detection (includes all detections over water)
DayNight	Day or Night	
D= Daytime fire, N= Nighttime fire

"""

# VIIRS_NOAA20_NRT/-125, 24, -66, 49/1




# now let's see how many transactions we use by querying this end point

# start_count = get_transaction_count()
# pd.read_csv(da_url)
# end_count = get_transaction_count()
# print ('We used %i transactions.' % (end_count-start_count))

# now remember, after 10 minutes this will reset

# 
# df_area

# def get_firms_data(api_key, start_date, end_date, latitude, longitude, radius):
#     base_url = "https://nrt4.modaps.eosdis.nasa.gov/api/v2/fire/geocell"
    
#     # Convert dates to the required format
#     start_date_str = start_date.strftime("%Y-%m-%d")
#     end_date_str = end_date.strftime("%Y-%m-%d")
    
#     # Define parameters for the API request
#     params = {
#         "api_key": api_key,
#         "start_date": start_date_str,
#         "end_date": end_date_str,
#         "latitude": latitude,
#         "longitude": longitude,
#         "radius": radius,
#     }

#     # Make the API request
#     response = requests.get(base_url, params=params)

#     # Check if the request was successful (status code 200)
#     if response.status_code == 200:
#         return response.json()
#     else:
#         print(f"Error {response.status_code}: {response.text}")
#         return None

# # Replace 'YOUR_API_KEY' with your actual FIRMS API key
# api_key = '00c7a6e7907b1f5faa49c87c989a9f32'

# # Specify the date range for the query
# start_date = datetime.now() - timedelta(days=7)  # 7 days ago
# end_date = datetime.now()

# # Specify the location and radius for the query
# latitude = 34.0522  # Example: Los Angeles, CA
# longitude = -118.2437
# radius = 5  # Radius in degrees

# # Make the FIRMS API request
# firms_data = get_firms_data(api_key, start_date, end_date, latitude, longitude, radius)

# # Process FIRMS data and store it in a DataFrame
# if firms_data:
#     df_columns = ['date', 'confidence', 'brightness', 'latitude', 'longitude']
#     df_data = []

#     for fire_event in firms_data['fires']:
#         date_str = fire_event['date']
#         date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
#         confidence = fire_event['confidence']
#         brightness = fire_event['brightness']
#         lat = fire_event['latitude']
#         lon = fire_event['longitude']

#         df_data.append([date, confidence, brightness, lat, lon])

#     # Create a DataFrame
#     df = pd.DataFrame(df_data, columns=df_columns)

#     # Display the DataFrame
#     print("FIRMS Data:")
#     print(df)
# else:
#     print("Failed to retrieve FIRMS data.")

# subprocess.run(["python", "parseData.py"], check=True)
# subprocess.call("parseData.py", shell=False)















def format_time(time_value):
    return time_value.zfill(4)


# import os




# current_directory = os.path.dirname(__file__)
# os.chdir(current_directory)
# print(os.getcwd())



# landsat=pd.read_csv('../data/landsat.csv')
if landsat.shape[0]>0:
    print(landsat)
    landsat['datetime_column'] = pd.to_datetime(landsat['acq_date'].astype('str') + ' ' + landsat['acq_time'].astype('str').apply(format_time), format='%Y-%m-%d %H%M')
    landsat['time_sinceupdate'] = (current_date_time-landsat['datetime_column']).dt.total_seconds() / 3600
    # Convert UTC to local timezone
    utc_timezone = pytz.utc
    local_timezone = pytz.timezone('America/New_York')  # Replace 'YOUR_LOCAL_TIMEZONE' with your local timezone
    landsat['datetime_column'] = landsat['datetime_column'].dt.tz_localize(utc_timezone).dt.tz_convert(local_timezone)
    # Apply the function to the specified column
    landsat['datetime_column'] = landsat['datetime_column'].dt.strftime("%Y-%m-%d %H%M")
    
    landsat=landsat[['latitude','longitude','datetime_column','confidence','daynight','satellite','time_sinceupdate']]
else:
    print("LANDSAT NOT AVAILABLE")


# modis=pd.read_csv('../data/modis.csv')
if modis.shape[0]>0:
    print(modis)
    modis['datetime_column'] = pd.to_datetime(modis['acq_date'].astype('str') + ' ' + modis['acq_time'].astype('str').apply(format_time), format='%Y-%m-%d %H%M')
    modis['time_sinceupdate'] = (current_date_time-modis['datetime_column']).dt.total_seconds() / 3600
    # Convert UTC to local timezone
    utc_timezone = pytz.utc
    local_timezone = pytz.timezone('America/New_York')  # Replace 'YOUR_LOCAL_TIMEZONE' with your local timezone
    modis['datetime_column'] = modis['datetime_column'].dt.tz_localize(utc_timezone).dt.tz_convert(local_timezone)
    # Apply the function to the specified column
    modis['datetime_column'] = modis['datetime_column'].dt.strftime("%Y-%m-%d %H%M")
    modis=modis[['latitude','longitude','datetime_column','confidence','daynight','satellite','time_sinceupdate']]
else:
    print("MODIS NOT AVAILABLE")


# viirs=pd.read_csv('../data/viirs.csv')
if viirs.shape[0]>0:
    print(viirs)
    viirs['datetime_column'] = pd.to_datetime(viirs['acq_date'].astype('str') + ' ' + viirs['acq_time'].astype('str').apply(format_time), format='%Y-%m-%d %H%M')
    viirs['time_sinceupdate'] = (current_date_time-viirs['datetime_column']).dt.total_seconds() / 3600
    # Convert UTC to local timezone
    utc_timezone = pytz.utc
    local_timezone = pytz.timezone('America/New_York')  # Replace 'YOUR_LOCAL_TIMEZONE' with your local timezone
    viirs['datetime_column'] = viirs['datetime_column'].dt.tz_localize(utc_timezone).dt.tz_convert(local_timezone)
    # Apply the function to the specified column
    viirs['datetime_column'] = viirs['datetime_column'].dt.strftime("%Y-%m-%d %H%M")
    viirs=viirs[['latitude','longitude','datetime_column','confidence','daynight','satellite','time_sinceupdate']]
else:
    print("VIIRS NOT AVAILABLE")

dataframes=[]
for i in [landsat,modis,viirs]: #
    if i.shape[0]>0:
        dataframes.append(i)
data=pd.concat(dataframes)


#WHOLE USA
# data = data[(data['latitude'] > 25) & (data['latitude'] < 49) & (data['longitude'] > -124.8) & (data['longitude'] < -66.9)]
#TENNESSEE
data = data[(data['latitude'] > 34.980322) & (data['latitude'] < 36.681860) & (data['longitude'] > -90.314831) & (data['longitude'] < -81.669813)]

# data = data[(data['latitude'] > 33.980322) & (data['latitude'] < 35.681860) & (data['longitude'] >  -93.314831 ) & (data['longitude'] < -79.669813  )]
#APPALACHIAN REGION
# data = data[(data['latitude'] > 34) & (data['latitude'] < 37) & (data['longitude'] > -90) & (data['longitude'] < -82)]
print(data)

# data.to_excel('../data/data.xlsx',index=False)

engine = create_engine('mysql+mysqlconnector://root:ilab301@smartshots.ise.utk.edu:3306/DB', echo=False)






data.to_sql(name='heatpoints', con=engine, if_exists ='replace', index=False)
Session = sessionmaker(bind=engine)
session = Session()
session.execute(text('''TRUNCATE DB.heatpoints_optimized;'''))
session.commit()


session.execute(text('''INSERT INTO DB.heatpoints_optimized SELECT * FROM DB.heatpoints;'''))
session.commit()
session.close()

# with engine.begin() as connection:
#     connection.execute('INSERT INTO DB.heatpoints_optimized SELECT * FROM DB.heatpoints;')



# with engine.connect() as con:
#     statement = text("""

# INSERT INTO heatpoints_optimized
# SELECT * FROM heatpoints



# """)
#     con.execute(statement)




exit(0)
