#!/usr/bin/env bash

configs=(
	"./config-olivine-pure.json"
	# "./config-carbon-pure.json"
)
scales=(
	1.125
	1.250
	1.375
	1.500
)

for config in "${configs[@]}"; do
	for scale in "${scales[@]}"; do
		echo "Processing with scale: $scale"

		# for data in ./fracval/N*.dat; do
		for data in ./pyfracval/*.dat; do
			# for data in ./pyfracval/*_N1_*.dat ./pyfracval/*_N2_*.dat ./pyfracval/*_N4_*.dat; do
			echo "$config/$data with scale $scale"
			uv run yasf compute --config "$config" --cluster "$data" --cluster-dimensional-scale "$scale" --backend mstm
		done
	done
done
