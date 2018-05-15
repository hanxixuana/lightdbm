#!/bin/bash

./lightgbm config_file=gbtrain.conf > gblog.txt
./lightgbm config_file=dbtrain.conf > dblog.txt

./draw.py dblog.txt
./draw.py gblog.txt

display db_pic.png &
display gb_pic.png &

