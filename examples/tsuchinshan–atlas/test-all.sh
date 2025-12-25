#!/usr/bin/env bash

./test-olivine-pure.sh
./test-carbon-pure.sh

# for config in ./*.json; do

# 	if [[ $config == *"enstatite"* ]]; then
# 		echo "Skipping enstatite"
# 		continue
# 	fi

# 	# for data in ./*.csv
# 	# do
# 	#   echo "$config/$data"
# 	#   yasf compute --config "$config" --cluster "$data" --backend mstm
# 	# done

# 	# for data in ./fracval/N*.dat; do
# 	for data in ./pyfracval/*.dat; do
# 		echo "$config/$data"
# 		yasf compute --config "$config" --cluster "$data" --backend mstm
# 	done

# done
