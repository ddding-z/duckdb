#!/bin/bash

set -e

# python bike_sharing_demand/train_bike_sharing_demand_rf.py -tn 100 -td 10
python train_expedia_rf.py -tn 100 -td 10
python train_flights_rf.py -tn 100 -td 10
python train_hospital_rf.py -tn 100 -td 10
python train_medical_charges_rf.py -tn 100 -td 10
python train_nyc-taxi-green-dec-2016_rf.py -tn 100 -td 10
python train_wine_quality_rf.py -tn 100 -td 10