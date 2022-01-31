import zipfile
import pandas as pd
import os

os.chdir("C:\\Users\\Olivera\\PycharmProjects\\BETH_data_analysis")
# Open compressed data folder and load test, train and validation data into pandas data frame
with zipfile.ZipFile('data.zip') as z:
    with z.open('labelled_testing_data.csv') as test:
        test_df = pd.read_csv(test)
    with z.open('labelled_training_data.csv') as train:
        train_df = pd.read_csv(train)
    with z.open('labelled_validation_data.csv') as validation:
        valid_df = pd.read_csv(validation)

data = [train_df, valid_df, test_df]