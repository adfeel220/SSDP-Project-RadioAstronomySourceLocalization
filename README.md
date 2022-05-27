# SSDP-Project-RadioAstronomySourceLocalization
This repository contains the project of **EPFL COM-500 Statistical Signal and Data Processing through Applications** in 2021-2022 spring semester.

Project topic: **Radio Astronomy Source Localization**
Team member: Chun-Tso Tsai, Majdouline Ait Yahia, Koloina Randrianavony

## Abstract
This project aims to implement and compare the difference between DOA (Direction of Arrival) algorithms, namely MUSIC and Bluebild. We tested the algorithms using real astronomical data and our own generated data.

One can refer to the [report](report.pdf) for more details.

## File Structure
- `bluebild.py`: Contains the implementation for the Bluebild algorithm.
- `music.py`: Contains the implementation for the MUSIC (MUltiple SIgnal Classification) algorithm.
- `data_sim.ipynb`: The notebook which generates simulated data and test it.
- `RI_project.ipynb`: The main notebook which tests the algorithms in different settings.

The data used in this project comes from [here](https://moodle.epfl.ch/pluginfile.php/3067678/mod_resource/content/2/RA_source_localization.zip).
The package `ImoT-tools` for plotting spherical data can be found [here](https://pypi.org/project/ImoT-tools/) and it supports `pip` command.