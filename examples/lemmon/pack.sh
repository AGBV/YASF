#!/usr/bin/env bash

folders=(
	out-630
	out-675
	out-700
)

for folder in "${folders[@]}"; do
	echo "Packing $folder"

	rm "${folder}.zip"
	nix run nixpkgs#zip -- -j "${folder}".zip "${folder}"/*
done
