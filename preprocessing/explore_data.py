from preprocessing.load_data import data
import pandas as pd

# These options are used in order not to truncate the data frame while printing
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

train_df, valid_df, test_df = data[0], data[1], data[2]


# Print info of each data frame to see the schema and data types
print("Testing data: ")
print(test_df.info())
print("Training data: ")
print(train_df.info())
print("Validation data: ")
print(valid_df.info())

# Explore the data columns
print(test_df.head(10))
print(train_df.head(10))
print(valid_df.head(10))

# Explore data values and statistics
print(test_df.describe())
print(train_df.describe())
print(valid_df.describe())