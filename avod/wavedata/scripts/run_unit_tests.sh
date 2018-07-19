#!/usr/bin/env bash

cd "$(dirname "$0")"
cd ..

echo "Running unit tests in $(pwd)/wavedata"
coverage run --source wavedata -m unittest discover -b --pattern "*_test.py"

#coverage report -m
