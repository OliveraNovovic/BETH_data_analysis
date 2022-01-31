"""[1] 'Highnam, Kate, et al. "BETH dataset: Real cybersecurity data for anomaly detection research."
        TRAINING 763.66.88 (2021): 8.',
        http://www.gatsby.ucl.ac.uk/~balaji/udl2021/accepted-papers/UDL2021-paper-033.pdf"""

from sklearn.preprocessing import LabelEncoder


# Preprocessing suggested in the paper[1]:
def preprocessing(df):
    df = df.drop(['stackAddresses', 'args'], axis=1)
    df['processId_bin'] = df['processId'].apply(lambda x: 1 if x in [0, 1, 2] else 0)
    df['parentProcessId_bin'] = df['parentProcessId'].apply(lambda x: 1 if x in [0, 1, 2] else 0)
    df['userId_bin'] = df['userId'].apply(lambda x: 0 if x < 1000 else 1)
    df['mountNamespace_bin'] = df['mountNamespace'].apply(lambda x: 1 if x == 4026531840 else 0)
    df['returnValue_map'] = df['returnValue'].apply(lambda x: -1 if x < 0 else 0 if x == 0 else 1)

    label_encoder = LabelEncoder()
    df['processName_encode'] = label_encoder.fit_transform(df['processName'])

    df = df.drop(['processId', 'parentProcessId', 'userId', 'mountNamespace', 'returnValue',
                  'eventName', 'processName'], axis=1)

    return df



