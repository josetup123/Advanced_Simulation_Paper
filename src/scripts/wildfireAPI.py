# from arcgis.gis import GIS
# from arcgis.mapping import WebMap
# import pandas as pd

# https://services9.arcgis.com/RHVPKKiFTONKtxq3/ArcGIS/rest/services/USA_Wildfires_v1/FeatureServer/query?layerDefs=&geometry=&geometryType=esriGeometryEnvelope&inSR=&spatialRel=esriSpatialRelIntersects&outSR=&datumTransformation=&applyVCSProjection=false&returnGeometry=true&maxAllowableOffset=&geometryPrecision=&returnIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&returnZ=false&returnM=false&sqlFormat=none&f=html&token=
# import urllib, urllib2
# param = {'where':'1=1',’outFields’:’*’,’f’:’json’}
# url = ‘http://coagisweb.cabq.gov/…/APD_Incidents/MapServer/0/query ?’ + urllib.urlencode(param)
# rawreply = urllib2.urlopen(url).read()



# #https://www.arcgis.com/home/item.html?id=d957997ccee7408287a963600a77f61f

# # Connect to your GIS (ArcGIS Online or ArcGIS Enterprise)
# gis = GIS("https://services9.arcgis.com/RHVPKKiFTONKtxq3/arcgis/rest/services")

# # Access the web map by specifying its item ID
# web_map_item_id = "d957997ccee7408287a963600a77f61f"
# web_map_item = gis.content.get(web_map_item_id)

# # Access the layers in the web map
# web_map = WebMap(web_map_item)
# layers = web_map.layers

# # Assuming you want to query the first layer in the web map
# target_layer = layers[0]

# # Perform a query on the target layer
# query_result = target_layer.query(where="1=1", out_fields="*")

# # Convert the features to a pandas DataFrame
# features_df = pd.DataFrame.from_records([feature.attributes for feature in query_result.features])

# # Print the DataFrame
# print(features_df)