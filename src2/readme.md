

docker run --name mysql -d     -p3306:3306     -eMYSQL_ROOT_PASSWORD=ilab301    --restart unless-stopped    mysql:8

I've created this database for the simulation project "root@smartshots.ise.utk.edu:3306" with password ilab301. My data is now being pushed to this database once the python script is executed.








OSMR

docker run -t -v /home/ilab/osmr:/data osrm/osrm-backend osrm-extract -p /opt/car.lua /data/north-america-latest.osm.pbf


docker run -t -v /home/ilab/osmr:/data osrm/osrm-backend osrm-partition /data/north-america-latest.osrm
docker run -t -v /home/ilab/osmr:/data osrm/osrm-backend osrm-customize /data/north-america-latest.osrm




docker run --name osrm -t -i -p 5000:5000 -v c:/docker:/data osrm/osrm-backend osrm-routed --algorithm mld /data/berlin-latest.osrm


curl "http://smartshots.ise.utk.edu:5000/route/v1/driving/13.388860,52.517037;13.385983,52.496891?steps=true"


docker start osrm

