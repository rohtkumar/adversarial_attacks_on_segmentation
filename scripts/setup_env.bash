#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
cd ../../
conda activate env

jupyter notebook --no-browser --port=8889 

