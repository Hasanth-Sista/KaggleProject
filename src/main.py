import pandas as pd
import numpy as np
import json
from sklearn import model_selection,neural_network,metrics
from pandas.io.json import json_normalize

path = "C:/Users/sista/PycharmProjects/KaggleProject/KaggleProject-MachineLearning/train.csv"

testPath = "C:/Users/sista/PycharmProjects/KaggleProject/KaggleProject-MachineLearning/test.csv"

def load_df(csv_path=path, nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv(csv_path,
                     converters={column: json.loads for column in JSON_COLUMNS},
                     dtype={'fullVisitorId': 'str'},  # Important!!
                     nrows=nrows)

    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    # print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

c = load_df(path)

attributeMap = dict()
count = 0
def normalizeColumn(data, attributes):
    global count
    for i in sorted(data[attributes].unique(), reverse=True):
        if i not in attributeMap:
            attributeMap[i] = count
            count = count + 1
    data[attributes] = data[attributes].map(attributeMap)
    return data

# fullVisitorId and sessionId are object type

columnList = ['date', 'fullVisitorId', 'sessionId', 'visitId', 'visitNumber', 'visitStartTime']

for column in c:
    if column not in columnList:
        c[column] = c[column].astype('str')

for column in c:
    if column not in columnList:
        normalizeColumn(c, column)

# c.to_csv("output.csv", sep='\t', encoding='utf-8')

#  find columns with constant values
constant_columns = []
for col in c.columns:
    if len(c[col].value_counts()) == 1:
        constant_columns.append(col)

# remove columns with constant values
# for column in constant_columns:
#     del c[column]


test = load_df(testPath)
for column in test:
    if column not in columnList:
        test[column] = test[column].astype('str')

for column in test:
    if column not in columnList:
        normalizeColumn(test, column)

# test.to_csv("output1.csv", sep='\t', encoding='utf-8')

X_train,X_validation,y_train,y_validation=model_selection.train_test_split(c, c['totals.transactionRevenue'], test_size=0.20, random_state=0)
neural_net = neural_network.MLPClassifier(hidden_layer_sizes=(5,),activation="relu",alpha=0.0001)
neural_net.fit(X_train,y_train)
y_pred = neural_net.predict(X_validation)

print(metrics.accuracy_score(y_validation, y_pred))










# c = pd.read_csv(path)
# df = pd.DataFrame(c)
# deviceDataFrame = df['device']
# deviceDataFrame = pd.DataFrame(deviceDataFrame, columns=['device'])
#
# columns = ['browser', 'browserVersion', 'browserSize', 'operatingSystem',
#                                              'operatingSystemVersion', 'isMobile', 'mobileDeviceBranding',
#                                              'mobileDeviceModel', 'mobileInputSelector', 'mobileDeviceInfo',
#                                              'mobileDeviceMarketingName', 'flashVersion', 'language',
#                                              'screenColors', 'screenResolution', 'deviceCategory']
#

# df1 = pd.DataFrame(deviceDataFrame, columns=columns, dtype=np.str)
# print(df1)
# print(deviceDataFrame)
# json_normalize(deviceDataFrame, columns)
# pd.read_json(deviceDataFrame, orient='columns')

# df2 = pd.DataFrame(df['device'], index=None, columns=['browser', 'browserVersion', 'browserSize', 'operatingSystem',
#                                              'operatingSystemVersion', 'isMobile', 'mobileDeviceBranding',
#                                              'mobileDeviceModel', 'mobileInputSelector', 'mobileDeviceInfo',
#                                              'mobileDeviceMarketingName', 'flashVersion', 'language',
#                                              'screenColors', 'screenResolution', 'deviceCategory'])
#
#
# df1 = pd.DataFrame()
# df1 = df1.append(df2)
# print(df1)
# stdf = df['device'].apply(json.loads)
#
# jdata = json.dumps(deviceDataFrame)
# deviceDataFrame = pd.DataFrame(deviceDataFrame)
# print(deviceDataFrame)
# deviceDataFrame = pd.DataFrame(jdata)

