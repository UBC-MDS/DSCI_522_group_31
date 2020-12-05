# online purchasing intention data pipe
# author: Jingjing Zhi, Yazan Saleh
# date: 2020-12-04

all: reports/report.md

# download data
data/raw/online_shoppers_intention.csv: src/01_download_data.py
	python src/01_download_data.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv --out_file=data/raw/online_shoppers_intention.csv
    
# pre-process data (e.g., split into train & test)
data/processed/train_data.csv data/processed/test_data.csv: src/02_cleanup_data.py data/raw/online_shoppers_intention.csv
	python src/02_cleanup_data.py --in_file=data/raw/online_shoppers_intention.csv --out_training_file=data/processed/train_data.csv --out_test_file=data/processed/test_data.csv

# generate EDA png files
img/eda/class_imbalance.png img/eda/feature_correlation.png img/eda/feature_density.png : src/03_generate_eda.py data/processed/train_data.csv
	python src/03_generate_eda.py --in_file=data/processed/train_data.csv --out_folder=img/eda/

# tune model
data/processed/best_model.sav img/reports/classification_report.csv img/reports/classification_report_feature_selection.csv img/reports/confusion_matrix.png img/reports/confusion_matrix_feature_selection.png : src/04_build_model.py data/processed/train_data.csv data/processed/test_data.csv
	python src/04_build_model.py --data_path=data/processed --out_report_path=img/reports --random_state=2020 --tune_params=True

# write the report
reports/report.md : reports/report.Rmd img/eda/class_imbalance.png img/eda/feature_correlation.png img/eda/feature_density.png img/reports/classification_report.csv img/reports/classification_report_feature_selection.csv img/reports/confusion_matrix.png img/reports/confusion_matrix_feature_selection.png
	Rscript -e "rmarkdown::render('reports/report.Rmd')"

clean: 
	rm -rf data
	rm -rf img/eda
	rm -rf img/reports
	rm -rf reports/report.md
	rm -rf reports/report.html