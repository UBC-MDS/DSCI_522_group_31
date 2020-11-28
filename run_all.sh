# run_all.sh
# Tran Doan Khanh Vu, 28 Nov 2020
#
# This driver script downloads the data and splits into
# training and test data set. This script takes no arguments.
#
# Usage: bash run_all.sh

# Step 1: download data
python src/01_download_data.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv --out_file=data/raw/online_shoppers_intention.csv

# Step 2: Split data
python src/02_cleanup_data.py --in_file=data/raw/online_shoppers_intention.csv --out_training_file=data/processed/train_data.csv --out_test_file=data/processed/test_data.csv