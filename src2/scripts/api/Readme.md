

docker rm -f simu_bk

docker run  --privileged  --gpus all -p 8000:8000  --name simu_bk --cap-add=NET_ADMIN --cap-add=SYS_ADMIN  -d -it -v /home/ilab/Advanced_Simulation_Paper:/app/data:z  rest_api /bin/bash



sudo docker exec -it  simu_bk   bash


pip install django &&
pip install djangorestframework &&
django-admin startproject rest_api

cd rest_api/

chmod 777 -R  /app

python manage.py startapp api_simu


python manage.py runserver 0.0.0.0:8000

<!-- DB changes -->
python manage.py makemigrations
python manage.py migrate


######NEW 


django-admin startproject drf .


django-admin startapp api 

