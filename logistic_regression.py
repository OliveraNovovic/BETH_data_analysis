"""The idea to test this approach came from the blog:
    https://towardsdatascience.com/supervised-machine-learning-technique-for-anomaly-detection-logistic-regression-97fc7a9cacd4
    The code is original, but the steps are the same as in the example described in the blog"""

import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from preprocessing import load_data, preprocess_data

data = load_data.data

train_df, valid_df, test_df = data[0], data[1], data[2]

train = preprocess_data.preprocessing(train_df)
valid = preprocess_data.preprocessing(valid_df)
test = preprocess_data.preprocessing(test_df)

# Number of records labeled as suspicious and non suspicious in train data
sus = len(train[train['sus'] == 1])
non_sus = len(train[train['sus'] == 0])

print("Suspicious: ", sus)
print("Non suspicious: ", non_sus)

# Rescale features: eventId, argsNum, processName_encode to be in range -1 to 1
rob_scaler = RobustScaler()
train['eventId_scaled'] = rob_scaler.fit_transform(train['eventId'].values.reshape(-1, 1))
train['argsNum_scaled'] = rob_scaler.fit_transform(train['argsNum'].values.reshape(-1, 1))
train['processName_scaled'] = rob_scaler.fit_transform(train['processName_encode'].values.reshape(-1, 1))

valid['eventId_scaled'] = rob_scaler.fit_transform(valid['eventId'].values.reshape(-1, 1))
valid['argsNum_scaled'] = rob_scaler.fit_transform(valid['argsNum'].values.reshape(-1, 1))
valid['processName_scaled'] = rob_scaler.fit_transform(valid['processName_encode'].values.reshape(-1, 1))

test['eventId_scaled'] = rob_scaler.fit_transform(test['eventId'].values.reshape(-1, 1))
test['argsNum_scaled'] = rob_scaler.fit_transform(test['argsNum'].values.reshape(-1, 1))
test['processName_scaled'] = rob_scaler.fit_transform(test['processName_encode'].values.reshape(-1, 1))

# Drop the original columns and columns that won't be used for training
train.drop(['timestamp', 'threadId', 'hostName', 'eventId', 'argsNum', 'processName_encode'], axis=1, inplace=True)
valid.drop(['timestamp', 'threadId', 'hostName', 'eventId', 'argsNum', 'processName_encode'], axis=1, inplace=True)
test.drop(['timestamp', 'threadId', 'hostName', 'eventId', 'argsNum', 'processName_encode'], axis=1, inplace=True)

# Define X (features) and y (label) variables
X_train = train.loc[:, train.columns != 'sus']
y_train = train.loc[:, train.columns == 'sus']

X_valid = valid.loc[:, valid.columns != 'sus']
y_valid = valid.loc[:, valid.columns == 'sus']

X_test = test.loc[:, test.columns != 'sus']
y_test = test.loc[:, test.columns == 'sus']


def undersampling(df, sus):
    sus_indices = df[df.sus == 1].index
    non_sus_indices = df[df.sus == 0].index
    random_nonsus_indices = np.random.choice(non_sus_indices, sus, replace=False)
    random_nonsus_indices = np.array(random_nonsus_indices)
    under_sample_indices = np.concatenate([sus_indices, random_nonsus_indices])
    under_sample_data = df.iloc[under_sample_indices, :]
    X_undersample = under_sample_data.loc[:, under_sample_data.columns != 'sus']
    y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'sus']

    return X_undersample, y_undersample


# Undersampling of the train data
sus_train = len(train[train['sus'] == 1])
X_train_undersample, y_train_undersample = undersampling(train, sus_train)

# Undersampling of the validation data
sus_valid = len(valid[valid['sus'] == 1])
X_valid_undersample, y_valid_undersample = undersampling(valid, sus_valid)

# Build and fit the model
model = LogisticRegression()
model.fit(X_train_undersample, y_train_undersample)
y_pred = model.predict(X_test)

# Model evaluation
classification_report = classification_report(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)
print("CLASSIFICATION REPORT")
print(classification_report)
print("CONFUSION MATRIX")
print(confusion_matrix)


