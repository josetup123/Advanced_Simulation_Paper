import pandas as pd


import datetime
import pytz
from datetime import datetime



def format_time(time_value):
    return time_value.zfill(4)


import os




current_directory = os.path.dirname(__file__)
os.chdir(current_directory)
print(os.getcwd())



landsat=pd.read_csv('../data/landsat.csv')
if landsat.shape[0]>0:
    print(landsat)
    landsat['datetime_column'] = pd.to_datetime(landsat['acq_date'].astype('str') + ' ' + landsat['acq_time'].astype('str').apply(format_time), format='%Y-%m-%d %H:%M')
    # Convert UTC to local timezone
    utc_timezone = pytz.utc
    local_timezone = pytz.timezone('America/New_York')  # Replace 'YOUR_LOCAL_TIMEZONE' with your local timezone
    landsat['datetime_column'] = landsat['datetime_column'].dt.tz_localize(utc_timezone).dt.tz_convert(local_timezone)
    # Apply the function to the specified column
    landsat['datetime_column'] = landsat['datetime_column'].dt.strftime("%Y-%m-%d %H:%M")
    landsat=landsat[['latitude','longitude','datetime_column','confidence','daynight','satellite']]
else:
    print("LANDSAT NOT AVAILABLE")


modis=pd.read_csv('../data/modis.csv')
if modis.shape[0]>0:
    print(modis)
    modis['datetime_column'] = pd.to_datetime(modis['acq_date'].astype('str') + ' ' + modis['acq_time'].astype('str').apply(format_time), format='%Y-%m-%d %H:%M')
    # Convert UTC to local timezone
    utc_timezone = pytz.utc
    local_timezone = pytz.timezone('America/New_York')  # Replace 'YOUR_LOCAL_TIMEZONE' with your local timezone
    modis['datetime_column'] = modis['datetime_column'].dt.tz_localize(utc_timezone).dt.tz_convert(local_timezone)
    # Apply the function to the specified column
    modis['datetime_column'] = modis['datetime_column'].dt.strftime("%Y-%m-%d %H:%M")
    modis=modis[['latitude','longitude','datetime_column','confidence','daynight','satellite']]
else:
    print("MODIS NOT AVAILABLE")


viirs=pd.read_csv('../data/viirs.csv')
if viirs.shape[0]>0:
    print(viirs)
    viirs['datetime_column'] = pd.to_datetime(viirs['acq_date'].astype('str') + ' ' + viirs['acq_time'].astype('str').apply(format_time), format='%Y-%m-%d %H:%M')
    # Convert UTC to local timezone
    utc_timezone = pytz.utc
    local_timezone = pytz.timezone('America/New_York')  # Replace 'YOUR_LOCAL_TIMEZONE' with your local timezone
    viirs['datetime_column'] = viirs['datetime_column'].dt.tz_localize(utc_timezone).dt.tz_convert(local_timezone)
    # Apply the function to the specified column
    viirs['datetime_column'] = viirs['datetime_column'].dt.strftime("%Y-%m-%d %H:%M")
    viirs=viirs[['latitude','longitude','datetime_column','confidence','daynight','satellite']]
else:
    print("VIIRS NOT AVAILABLE")

dataframes=[]
for i in [landsat,modis,viirs]:
    if i.shape[0]>0:
        dataframes.append(i)
data=pd.concat(dataframes)




#TENNESSEE
# data = data[    (data['latitude'] > 35.0003) & (data['latitude'] < 36.6781) & (data['longitude'] > -90.3131) & (data['longitude'] < -81.6469)]
print(data)
data.to_excel('../data/data.xlsx',index=False)






# utc_time = datetime.datetime.utcnow()
# local_time = utc_time.astimezone()
