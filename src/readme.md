

docker run --name mysql -d     -p3306:3306     -eMYSQL_ROOT_PASSWORD=ilab301    --restart unless-stopped    mysql:8

I've created this database for the simulation project "root@smartshots.ise.utk.edu:3306" with password ilab301. My data is now being pushed to this database once the python script is executed.