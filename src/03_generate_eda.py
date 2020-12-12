# author: Tran Doan Khanh Vu
# date: 2020-11-27

"""Load a csv / feather training data file from a local input file and perform the EDA and write the final images to an output folder.

Usage: src/generate_eda.py --in_file=<in_file> --out_folder=<out_folder>

Options:
--in_file=<in_file>          The path and the filename and the extension where we want to load from our disk
--out_folder=<out_folder>    The path where we want to save the the final images in our disk
"""

import os
import pandas as pd
import altair as alt
from altair_saver import save
import chromedriver_binary
from selenium import webdriver
from docopt import docopt
import feather

from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

opt = docopt(__doc__)


def main(in_file, out_folder):
    print("Start EDA script")

    # Step 1: Read the data into Pandas data frame
    in_extension = in_file[in_file.rindex(".") + 1 :]

    print("Read in the file:", in_file)
    if in_extension == "csv":
        df = pd.read_csv(in_file)
    elif in_extension == "feather":
        df = feather.read_dataframe(
            in_file,
        )
    else:
        print("Unknown data type", in_file)
        return

    # Step 2: Create the path if it does not exist
    dirpath = os.path.dirname(out_folder)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    # Step 3: Generate the EDAs
    EDAs = ["class_imbalance", "feature_density", "feature_correlation"]

    # stop openning Chrome when saving plot
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("start-maximized")  #
    chrome_options.add_argument("disable-infobars")
    chrome_options.add_argument("--disable-extensions")
    driver = webdriver.Chrome(chrome_options=chrome_options)

    alt.data_transformers.disable_max_rows()

    for index, EDA in enumerate(EDAs):
        filepath = os.path.join(dirpath, EDA + ".png")
        numeric_features = [
            "Administrative",
            "Administrative_Duration",
            "Informational",
            "Informational_Duration",
            "ProductRelated",
            "ProductRelated_Duration",
            "BounceRates",
            "ExitRates",
            "PageValues",
            "SpecialDay",
        ]

        if index == 0:
            chart = (
                alt.Chart(df, title="Class imbalance")
                .encode(
                    x="count()",
                    y=alt.X("Revenue"),
                    color=alt.Color("Revenue", legend=None),
                )
                .mark_bar()
                .properties(width=300, height=50)
                .configure_axis(titleFontSize=20)
                .configure_axisX(labelFontSize=20)
                .configure_axisY(labelFontSize=15)
                .configure_title(fontSize=30)
            )
        elif index == 1:
            data = {
                "left_plot": numeric_features[:5],
                "right_plot": numeric_features[5:],
            }
            charts = {}
            for key, cols in data.items():
                charts[key] = (
                    alt.Chart(df, title="Density plot for numeric features")
                    .transform_fold(cols, as_=["Features", "value"])
                    .transform_density(
                        density="value",
                        bandwidth=0.3,
                        groupby=["Features", "Revenue"],
                        extent=[0, 8],
                    )
                    .mark_area(opacity=0.3)
                    .encode(
                        x=alt.X("value:Q", title="Value", axis=None),
                        y=alt.Y("density:Q", title="", axis=None),
                        row=alt.Row("Features:N"),
                        color="Revenue",
                    )
                    .properties(width=250, height=100)
                )
            chart = charts["left_plot"] | charts["right_plot"]
        elif index == 2:
            corr_df = (
                df[numeric_features].corr("spearman").stack().reset_index(name="corr")
            )
            chart = (
                alt.Chart(corr_df, title="Correlations of numeric features")
                .mark_circle()
                .encode(
                    x=alt.X("level_0", title=""),
                    y=alt.Y("level_1", title=""),
                    size=alt.Color("corr", title="Correlation"),
                    color="corr",
                )
                .properties(width=300, height=300)
                .configure_axis(labelFontSize=10)
                .configure_title(fontSize=15)
            )

        save(chart, filepath, method="selenium", webdriver=driver, scale_factor=2)
        print("Successfully saved chart", filepath)
    print("End EDA script")


if __name__ == "__main__":
    main(opt["--in_file"], opt["--out_folder"])
