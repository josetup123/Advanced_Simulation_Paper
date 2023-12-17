
# Wildfire Management Digital Twin Framework

## Overview

This repository introduces a comprehensive digital twin framework for wildfire management, aiming to optimize the allocation of firefighting entities and minimize the impact of wildfires on the population. The framework incorporates innovative technologies, including deep reinforcement learning, real-time satellite imaging, and a routing engine for enhanced decision-making.

## Key Features

- **Parameterized Experiments:** The framework allows for strategic allocation of firefighting entities to neighboring areas based on a proposed fire station index, enhancing the efficiency of wildfire response.

- **Deep Reinforcement Learning:** Utilizes deep reinforcement learning to determine optimal policies for clustered fire stations, with the goal of minimizing the impact of wildfires on the population.

- **Real-time Data Integration:** Incorporates real-time data from satellite imaging and a routing engine, providing up-to-date information for improved decision-making in wildfire management scenarios.

- **Holistic and Adaptive Response System:** Develops a holistic and adaptive response system that integrates deep reinforcement learning, ensuring effective and dynamic wildfire management strategies.

## Usage

To use the framework, follow the instructions in the documentation provided in the 'docs' directory. This includes guidelines for setting up the digital twin, running parameterized experiments, and leveraging deep reinforcement learning for optimal policy determination.


## Stage 1
<div align="center">
  <a href="https://youtu.be/vVTLdvKKL_E">
    <img  alt="Wildfire Management Framework Overview Stage 2" style="width:50%;">
  </a>
</div>


## Stage 2
<div align="center">
  <a href="https://www.youtube.com/watch?v=QCFEepq0dFw">
    <img alt="Wildfire Management Framework Overview Stage 2" style="width:50%;">
  </a>
</div>

## Getting Started

Clone the repository to your local machine:

```bash
git clone https://github.com/josetup123/Advanced_Simulation_Paper.git
cd wildfire-management-digital-twin
```

## Research Paper

<!-- [get the PDF]({{ site.url }}Latex/main.pdf) -->

https://drive.google.com/file/d/1lsHO2JNt7FeSHVg7-EQd4wqf2incnyRB/view?usp=sharing

<!-- https://docs.google.com/viewer?url=1lsHO2JNt7FeSHVg7-EQd4wqf2incnyRB -->
<!-- [embed]Latex/main.pdf[/embed] -->


## Additional Commands
```bash
docker run --name mysql -d     -p3306:3306     -eMYSQL_ROOT_PASSWORD=ilab301    --restart unless-stopped    mysql:8

I've created this database for the simulation project "root@smartshots.ise.utk.edu:3306" with password ilab301. My data is now being pushed to this database once the python script is executed.








OSMR

docker run -t -v /home/ilab/osmr:/data osrm/osrm-backend osrm-extract -p /opt/car.lua /data/north-america-latest.osm.pbf


docker run -t -v /home/ilab/osmr:/data osrm/osrm-backend osrm-partition /data/north-america-latest.osrm
docker run -t -v /home/ilab/osmr:/data osrm/osrm-backend osrm-customize /data/north-america-latest.osrm




docker run --name osrm -t -i -p 5000:5000 -v c:/docker:/data osrm/osrm-backend osrm-routed --algorithm mld /data/berlin-latest.osrm


curl "http://smartshots.ise.utk.edu:5000/route/v1/driving/13.388860,52.517037;13.385983,52.496891?steps=true"


docker start osrm






