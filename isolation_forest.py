from sklearn.metrics import precision_recall_fscore_support
from preprocessing import load_data, preprocess_data
from sklearn.ensemble import IsolationForest

data = load_data.data

train_df, valid_df, test_df = data[0], data[1], data[2]

train = preprocess_data.preprocessing(train_df)
valid = preprocess_data.preprocessing(valid_df)
test = preprocess_data.preprocessing(test_df)

# Split features for training and label
train_features = train[["eventId", "argsNum", "processId_bin", "parentProcessId_bin",
                        "userId_bin", "mountNamespace_bin", "returnValue_map", "processName_encode"]]
train_labels = train['sus']

valid_features = valid[["eventId", "argsNum", "processId_bin", "parentProcessId_bin",
                        "userId_bin", "mountNamespace_bin", "returnValue_map", "processName_encode"]]
valid_labels = valid['sus']

test_features = test[["eventId", "argsNum", "processId_bin", "parentProcessId_bin",
                        "userId_bin", "mountNamespace_bin", "returnValue_map", "processName_encode"]]
test_labels = test['evil']


def calculate_metric(labels, pred_label):
    # Transform the  label to -1 for where sus=1 and 1 where sus=0
    true_label = labels.apply(lambda x: -1 if x == 1 else 1)

    pr_re_fs = precision_recall_fscore_support(true_label, pred_label, average="weighted", pos_label=None)
    print("Precision: ", pr_re_fs[0])
    print("Recall: ", pr_re_fs[1])
    print("F1-Score: ", pr_re_fs[2])


# Fit the model on training data
if_model_clf = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', random_state=0).fit(
    train_features)

# Prediction on validation data
pred_valid_label = if_model_clf.predict(valid_features)

# Calculate metrices for prediction on validation data
print("Prediction on validation data: ")
calculate_metric(valid_labels, pred_valid_label)
print('\n')

# Prediction on test data
pred_test_label = if_model_clf.predict(test_features)

# Calculate metrices for prediction on test data
print("Prediction on test data ['evil'] label: ")
calculate_metric(test_labels, pred_test_label)

