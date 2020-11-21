# author: Tran Doan Khanh Vu
# date: 2020-11-21

"""Downloads a csv data from a website to a local filepath. The output file will be either a csv or a feather file format, and it is determined by the extension of the file name.

Usage: src/download_data.py --url=<url> --out_file=<out_file>

Options:
--url=<url>              The URL of the file we want to download (the extension must be csv)
--out_file=<out_file>    The path and the filename and the extension where we want to save the file in our disk
"""

import os
import pandas as pd
import requests
from docopt import docopt
import feather

opt = docopt(__doc__)

def main(url, out_file):
    # Step 1: Check if the url is valid
    request = requests.get(url)
    if request.status_code != 200:
        print('Web site does not exist')
        return
    
    # Step 2: Read the data into Pandas data frame
    input = pd.read_csv(url, header=None)
    
    # Step 3: Create the path if it does not exist
    dirpath = os.path.dirname(out_file)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
  
    # Step 4: Write the file locally based on the extension type
    extension = out_file[out_file.rindex(".")+1:]

    if extension == "csv":
        input.to_csv(out_file, index = False)
        print("Create csv file " + out_file)
    elif extension == "feather":
        feather.write_dataframe(input, out_file)
        print("Create feather file " + out_file)

if __name__ == "__main__":
    main(opt["--url"], opt["--out_file"])