import requests
import pandas as pd
from datetime import datetime, timedelta
import subprocess
import geojson
import pandas as pd
import numpy as np

import datetime
import pytz
from datetime import datetime

import mysql.connector
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from sqlalchemy.orm import sessionmaker



from pykml import parser
import geopandas as gpd


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

current_date_time = datetime.utcnow()
current_date=current_date_time.strftime("%Y-%m-%d")

print(current_date)

# fireStations=pd.read_csv('../data/Fire_Stations.csv')
fireStations = gpd.read_file('../data/Fire_Stations.geojson')
# print(fireStations.head())
fireStations=fireStations.get_coordinates(ignore_index=True)
# fireStations['']
# print(fireStations.city)
# print(fireStations.state)

# var coordinates = featfireStationsure.getGeometry().getCoordinates();

# 
fireStations=fireStations.rename(columns={'x':'longitude','y':'latitude'})


# fireStations=fireStations[['X','Y','LOADDATE','ADDRESS','CITY','STATE','ZIPCODE']]

# #CREATED COLUMNS:
fireStations['fire_index']=np.random.randint(100, size=len(fireStations))



#TAKE A SAMPLE:



# #TENNESSEE
fireStations = fireStations[(fireStations['latitude'] > 34.980322) & (fireStations['latitude'] < 36.681860) & (fireStations['longitude'] > -90.314831) & (fireStations['longitude'] < -81.669813)]

# fireStations=fireStations.sample(100)
# fireStations = fireStations[(fireStations['latitude'] > 34.980322) & (fireStations['latitude'] < 36.681860) & (fireStations['longitude'] > -90.314831) & (fireStations['longitude'] < -81.669813)]


# fireStations.to_csv('../data/fireStations.csv',index=False)
engine = create_engine('mysql+mysqlconnector://root:ilab301@smartshots.ise.utk.edu:3306/DB', echo=False)
fireStations.to_sql(name='firestations', con=engine, if_exists ='replace', index=False)
print(fireStations)
