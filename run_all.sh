# run_all.sh
# Tran Doan Khanh Vu, 28 Nov 2020
#
# This driver script downloads the data and splits into
# training and test data set. This script takes no arguments.
#
# Usage: bash run_all.sh

# Step 1: download data
python src/01_download_data.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv --out_file=data/raw/online_shoppers_intention.csv

# Step 2: split data
python src/02_cleanup_data.py --in_file=data/raw/online_shoppers_intention.csv --out_training_file=data/processed/train_data.csv --out_test_file=data/processed/test_data.csv

# Step 3: generate EDA png files
python src/03_generate_eda.py --in_file=data/processed/train_data.csv --out_folder=img/eda/

# Step 4: build machine learning models
python src/04_build_model.py --data_path=data/processed --out_report_path=img/reports --random_state=2020 --tune_params=True