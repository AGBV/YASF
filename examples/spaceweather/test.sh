#!/usr/bin/env bash

configs=(
	"./config.json"
)
scales=(
	# 0.025
	0.1
)

for config in "${configs[@]}"; do
	for scale in "${scales[@]}"; do
		echo "Processing with scale: $scale"

		for data in ./pyfracval/*.dat; do
			# for data in ./pyfracval/*_N1_*.dat ./pyfracval/*_N2_*.dat ./pyfracval/*_N4_*.dat; do
			echo "$config/$data with scale $scale"
			uv run yasf compute --config "$config" --cluster "$data" --cluster-dimensional-scale "$scale" --backend mstm
			# break
		done
		# break
	done
	# break
done
