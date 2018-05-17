#!/bin/bash

echo "Training GBM..."
./lightgbm config_file=gbtrain.conf > gblog.txt
./draw.py gblog.txt

echo "Training DBM..."
./lightgbm config_file=dbtrain.conf > dblog.txt
./draw.py dblog.txt

display db_pic.png &
display gb_pic.png &

