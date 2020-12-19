#!/bin/bash
wget -O best_model.zip https://github.com/xco1014/all/releases/download/0.0/model.zip
unzip best_model.zip 
python3 hw5.py $1 $2