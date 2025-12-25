#!/usr/bin/env bash

config="./config-olivine-pure.json"
scales=(
	1.125
	1.250
	1.375
	# 1.500
)

for scale in "${scales[@]}"; do
	echo "Processing with scale: $scale"

	# for data in ./fracval/N*.dat; do
	# for data in ./pyfracval/*.dat; do
	for data in ./pyfracval/*_N1_*.dat ./pyfracval/*_N2_*.dat ./pyfracval/*_N4_*.dat; do
		echo "$config/$data with scale $scale"
		yasf compute --config "$config" --cluster "$data" --cluster-dimensional-scale "$scale" --backend mstm
	done
done
