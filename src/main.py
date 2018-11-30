import pandas as pd
import numpy as np
import json
from sklearn import model_selection,neural_network,metrics
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.svm import SVC
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

path = "C:/Users/achan/PycharmProjects/KaggleProject-MachineLearning/train1.csv"

testPath = "C:/Users/achan/PycharmProjects/KaggleProject-MachineLearning/test2.csv"

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
# print(c.columns)
attributeMap = dict()
count = 0.0
def normalizeColumn(data, attributes):
    global count
    for i in sorted(data[attributes].unique(), reverse=True):
        if i not in attributeMap:
            attributeMap[i] = count
            count = count + 1.0
    data[attributes] = data[attributes].map(attributeMap)
    return data

# fullVisitorId and sessionId are object type

columnList = ['date', 'fullVisitorId', 'sessionId', 'visitId', 'visitNumber', 'visitStartTime', 'totals.transactionRevenue']

for column in c:
    if column not in columnList:
        c[column] = c[column].astype('str')

for column in c:
    if column not in columnList:
        normalizeColumn(c, column)


#  find columns with constant values
constant_columns = []
for col in c.columns:
    if len(c[col].value_counts()) == 1:
        constant_columns.append(col)

# remove columns with constant values
for column in constant_columns:
    del c[column]

test = load_df(testPath)
# print(test.columns)
for column in test:
    if column not in columnList:
        test[column] = test[column].astype('str')

for column in test:
    if column not in columnList:
        normalizeColumn(test, column)

for column in constant_columns:
    del test[column]
# print(c[c['totals.transactionRevenue'].isnull()])
c["totals.transactionRevenue"].astype('float')
c["totals.transactionRevenue"].fillna(100000, inplace=True)
y=c['totals.transactionRevenue'].values
print(y)
# y = pd.DataFrame({"transactions": c['totals.transactionRevenue']},dtype=np.int64)
# print(y)
# n_classes = y.totalTR.unique()
print(c)
print(test)

X_train,X_validation,y_train,y_validation=model_selection.train_test_split(c, y, test_size=0.20, random_state=0)

# print(c.columns)

# plot for  geoNetwork Continent

geoContinent = dict()
for i in c['geoNetwork.continent']:
    if i not in geoContinent.keys():
        geoContinent[i] = 1
    else:
        geoContinent[i] += 1

x = list(geoContinent.keys())
y = []
xLabel = []
for i in x:
    for key, val in attributeMap.items():
        if i == val:
            xLabel.append(key)
            y.append(geoContinent.get(i))

plt.pie(y, labels=xLabel)
plt.show()

#plot for device browsers

deviceBrowserColumns = dict()
for i in c['device.browser']:
    if i not in deviceBrowserColumns.keys():
        deviceBrowserColumns[i] = 1
    else:
        deviceBrowserColumns[i] += 1

x = deviceBrowserColumns.keys()
columnNames = []
reversed_dictionary = dict(map(reversed, attributeMap.items()))
for i in x:
    columnNames.append((reversed_dictionary.get(i)))

y = deviceBrowserColumns.values()
plt.plot(y,columnNames)
plt.show()

# plot for ChannelGrouping

channelGrouping = dict()
for i in c['channelGrouping']:
    if i not in channelGrouping.keys():
        channelGrouping[i] = 1
    else:
        channelGrouping[i] += 1

x = list(channelGrouping.keys())
y = []
xLabel = []
for i in x:
    for key, val in attributeMap.items():
        if i == val:
            xLabel.append(key)
            y.append(channelGrouping.get(i))

plt.bar(xLabel, y, alpha=0.5)
plt.setp(plt.xticks()[1], rotation=30, ha='right')
plt.show()

#plot for operatingsystem

deviceOsColumns = dict()
for i in c['device.operatingSystem']:
    if i not in deviceOsColumns.keys():
        deviceOsColumns[i] = 1
    else:
        deviceOsColumns[i] += 1
x = deviceOsColumns.keys()
columnNames = []
reversed_dictionary = dict(map(reversed, attributeMap.items()))
for i in x:
    columnNames.append((reversed_dictionary.get(i)))
y = deviceOsColumns.values()
plt.plot(y, columnNames, alpha=1)
plt.show()


#SVM
# print("supprt vector Machine")
# svm=SVC(kernel= 'sigmoid', gamma= 1e-1,C= 10,degree=2)
# svm.fit(X_train,y_train)
# y_pred=svm.predict(X_validation)
# print(metrics.accuracy_score(y_validation, y_pred))
# y_test_pred=svm.predict(test)
# print(y_test_pred)
# total = y_test_pred.sum()
# print(total)
# y_input=[]
# for i in y_test_pred:
#     y_input.append(np.log(y_test_pred))
# print(np.sqrt(((y_input - np.log(total)) ** 2).mean()))
# print('\n')

# #Random Forest
# print("Random Forest")
# # X_train,X_validation,y_train,y_validation=model_selection.train_test_split(c, y, test_size=0.25, random_state=0)
# random_forest=RandomForestClassifier(n_estimators=10,criterion= 'gini', max_depth= 3, max_features= 'sqrt')
# random_forest.fit(X_train,y_train)
# y_pred=random_forest.predict(X_validation)
# print(metrics.accuracy_score(y_validation, y_pred))
# y_test_pred = random_forest.predict(test)
# print(y_test_pred)
# total = y_test_pred.sum()
# print(total)
# y_input=[]
# for i in y_test_pred:
#     y_input.append(np.log(y_test_pred))
# print(np.sqrt(((y_input - np.log(total)) ** 2).mean()))
# print('\n')
#
#Neural network
print("Neural Network")
# X_train,X_validation,y_train,y_validation=model_selection.train_test_split(c, y, test_size=0.20, random_state=0)
result_val = pd.DataFrame({"fullVisitorId":X_validation['fullVisitorId'], 'transactionRevenue': X_validation["totals.transactionRevenue"]})
del X_train['totals.transactionRevenue']
del X_validation["totals.transactionRevenue"]
neural_net = neural_network.MLPClassifier(hidden_layer_sizes=(5,),activation="relu",alpha=0.0001)
neural_net.fit(X_train,y_train)
y_pred = neural_net.predict(X_validation)
# y_pred[y_pred <=1000000 ] = 0.0
result_val['predictedRevenue'] = y_pred
total = result_val['transactionRevenue'].sum()
print(total)
# print(total.values)
#print(np.sqrt(metrics.mean_squared_error(np.log(result_val['predictedRevenue']).values, total)))
print(metrics.accuracy_score(y_validation, y_pred))
y_test_pred = neural_net.predict(test)
# y_test_pred[y_test_pred <= 1000000] = 0
# print(y_test_pred)
# total = y_test_pred.sum()
# print(total)
# y_input=[]
# for i in y_test_pred:
#     y_input.append(np.log(y_test_pred))
# print(np.sqrt(((y_input - np.log(total)) ** 2).mean()))
# print('\n')

result = pd.DataFrame({"fullVisitorId":test['fullVisitorId']})
result['predictedRevenue'] = y_test_pred
result = result.groupby('fullVisitorId')['predictedRevenue'].sum().reset_index().o
result.columns = ["fullVisitorId", "predictedLogRevenue"]
result["predictedLogRevenue"] = np.log1p(result["predictedLogRevenue"])
result.to_csv('output',index=True)
print(result.head(6))

tuned_MLP_parameters = [{ 'hidden_layer_sizes': [(10,5,2), (20,10,7)],'activation': ['relu','tanh'],
                     'alpha': [0.001,0.003,0.05,0.02,0.04],'learning_rate': ['constant','adaptive'],'max_iter':[100]}]

clf = GridSearchCV(neural_network.MLPClassifier(), tuned_MLP_parameters, cv=5,scoring='accuracy')
clf.fit(X_train, y_train)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Detailed classification report:")
print()
y_true, y_pred = y_test_pred.ravel(), clf.predict(test)
print(classification_report(y_true, y_pred))
print("Accuracy Score:")
print(accuracy_score(y_true, y_pred))
print()

#
# #Adaboosting
# print("Adaboost")
# X_train,X_validation,y_train,y_validation=model_selection.train_test_split(c, y, test_size=0.20, random_state=0)
# adaboost=AdaBoostClassifier(n_estimators = 100, learning_rate= 0.5, algorithm='SAMME.R' ,random_state=1)
# adaboost.fit(X_train,y_train)
# y_pred = adaboost.predict(X_validation)
# print(metrics.accuracy_score(y_validation, y_pred))
# y_test_pred = adaboost.predict(test)
# print(y_test_pred)
# total = y_test_pred.sum()
# print(total)
# y_input=[]
# for i in y_test_pred:
#     y_input.append(np.log(y_test_pred))
# print(np.sqrt(((y_input - np.log(total)) ** 2).mean()))
# print('\n')
#


