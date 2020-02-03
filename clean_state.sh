#!/bin/bash

echo 'Deleting logs/ and output/'
rm -r ./logs
rm -r ./output

echo 'Recreating logs/ and output/'
mkdir ./logs
mkdir -p ./output/logs

