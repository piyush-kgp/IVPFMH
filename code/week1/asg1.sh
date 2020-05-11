#!/bin/bash

python3 quantize.py --img_path lena.jpg
python3 spatial_averaging.py --img_path lena.jpg
python3 rotate.py --img_path lena.jpg
python3 scaling.py --img_path lena.jpg
