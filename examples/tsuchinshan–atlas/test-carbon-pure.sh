#!/usr/bin/env bash

config="./config-carbon-pure.json"
scales=(
	1.125
	1.250
	1.375
	# 1.500
)

# for data in ./fracval/N*.dat; do
# for data in ./pyfracval/*.dat; do
for data in ./pyfracval/*_N1_*.dat ./pyfracval/*_N2_*.dat ./pyfracval/*_N4_*.dat; do
	echo "$config/$data"
	yasf compute --config "$config" --cluster "$data" --cluster-dimensional-scale 1.125 --backend mstm
	yasf compute --config "$config" --cluster "$data" --cluster-dimensional-scale 1.250 --backend mstm
	yasf compute --config "$config" --cluster "$data" --cluster-dimensional-scale 1.375 --backend mstm
	# yasf compute --config "$config" --cluster "$data" --cluster-dimensional-scale 1.500 --backend mstm
done
