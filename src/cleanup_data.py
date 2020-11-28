# author: Tran Doan Khanh Vu
# date: 2020-11-27

"""Load a csv / feather data file from a local input file and split into test and training data set and write to 2 separate local output files. The output file will be either a csv or a feather file format, which is determined by the extension.

Usage: python src/cleanup_data.py --in_file=<in_file> --out_training_file=<out_training_file> --out_test_file=<out_test_file> [--random_state=<random_state>] [--test_size=<test_size>]

Options:
--in_file=<in_file>                        The path and the filename and the extension where we want to load from our disk
--out_training_file=<out_training_file>    The path and the filename and the extension where we want to save the training data file in our disk
--out_test_file=<out_test_file>            The path and the filename and the extension where we want to save the test data file in our disk
--random_state=<random_state>              The random state that we want to use for splitting. [default: 2020]
--test_size=<test_size>                    The percentage of testing data split from the original dataframe [default: 0.25]
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from docopt import docopt
import feather

opt = docopt(__doc__)

def main(in_file, out_training_file, out_test_file, random_state, test_size):
    print("Start cleanup script")
    
    # Step 1: Read the data into Pandas data frame
    in_extension = in_file[in_file.rindex(".")+1:]
    
    print("Read in the file:", in_file)
    if in_extension == "csv":
        input = pd.read_csv(in_file)
    elif in_extension == "feather":
        input = feather.read_dataframe(path, )

        # convert the first row into headers
        input.columns = input.iloc[0]
        input = input[1:]
    else:
        print("Unknown data type", in_file)
        return
    
    # Step 2: Split data into training and test data
    train_df, test_df = train_test_split(input, test_size=float(test_size), random_state=int(random_state))
    print("Split the data frame with test_size=", test_size, " and random_state=" , random_state, sep="")
    
    # Step 3: Create the path if it does not exist
    for out_file, df in [(out_training_file, train_df), (out_test_file, test_df)]:
        dirpath = os.path.dirname(out_file)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
  
        # Step 4: Write the file locally based on the extension type
        extension = out_file[out_file.rindex(".")+1:]

        if extension == "csv":
            df.to_csv(out_file, index = False)
        elif extension == "feather":
            feather.write_dataframe(df, out_file)
        else:
            print("Unknown output data type", output)
            return
        
        print("Successfully created file", out_file, "with number of rows:", df.shape[0], "and columns:", df.shape[1])
    
    print("End cleanup script")
        

if __name__ == "__main__":
    main(opt["--in_file"], opt["--out_training_file"], opt["--out_test_file"], opt["--random_state"], opt["--test_size"])