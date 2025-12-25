#!/usr/bin/env bash

config="./config-olivine75-carbon25.json"

# for data in ./fracval/N*.dat; do
for data in ./pyfracval/*.dat; do
	echo "$config/$data"
	yasf compute --config "$config" --cluster "$data" --backend mstm
done
