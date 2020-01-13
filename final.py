import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error

#AUXILIAR FUNTIONS
def create_imputation(dataFrame):

    fixedDataFrame = dataFrame.copy()
    fixingValues = {}
    for columName in fixedDataFrame.columns:
        #Mean for float and integers
        if np.issubdtype(fixedDataFrame[columName], np.float) | np.issubdtype(fixedDataFrame[columName], np.integer):
            mean = fixedDataFrame[columName].mean()
            if pd.isna(mean): # The sum in columns where all the elemnts are NaN is NaN. We fill them with 0's
                fixingValues.update({columName: 0})
                fixedDataFrame[columName] = fixedDataFrame[columName].fillna(0)
            else:
                fixingValues.update({columName: mean})
                fixedDataFrame[columName] = fixedDataFrame[columName].fillna(mean)
        else: # Mode for objects
            mode = fixedDataFrame[columName].mode()[0]
            fixingValues.update({columName: mode})
            fixedDataFrame[columName] = fixedDataFrame[columName].fillna(mode)
    return fixedDataFrame, fixingValues
def fixTotalDataFrame (df):
    oneHot = pd.get_dummies(df["city"])
    df = df.drop(
        ["city", "reanalysis_sat_precip_amt_mm", "week_start_date",
         "reanalysis_tdtr_k", "reanalysis_avg_temp_k"], axis=1)
    df = df.join(oneHot)
    # Imputation
    df, _ = create_imputation(df)
    featureList = df.columns
    return df
def fixOneCityDataFrame(df):
    for col in df.columns:
        if col != "week_start_date":
            df[col] = df[col].fillna(df[col].mean())
    df = df.drop(
        ["week_start_date", "reanalysis_sat_precip_amt_mm",
         "reanalysis_tdtr_k", "reanalysis_avg_temp_k"], axis=1)
    return df


def crossValidation(model, X, y, k):
    kf = KFold(n_splits=k)
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # Model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        print("Mean average Error", mae)
        #count+=1
    #print(X_test["year"], X_test["weekofyear"])
    plt.plot(range(len(predictions)), predictions, label="Predictions")
    plt.plot(range(len(y_test)), y_test, label="Real Data")
    plt.xlabel("Week of the year")
    plt.ylabel("Number of cases")
    plt.legend()
    return mae

#FEATURES PRINTING
features = pd.read_csv("dengue_features_train.csv")
useful_label = features.drop(['city', 'year', 'weekofyear','week_start_date'], axis=1)
labels = pd.read_csv("dengue_labels_train.csv")
labels = labels["total_cases"]
#features, _ = create_imputation(features)


is_sj = features.loc[:, 'city'] == 'sj'
specificYear = features.loc[:, 'year'] == 2005
specificYear2 = features.loc[:, 'year'] == 2006
specificYear3 = features.loc[:, 'year'] == 2007

specificYear = specificYear | specificYear2 | specificYear3
#specificYear2 = specificYear.append(features.loc[:, 'year'] == 2003)

#for col in useful_label.columns:
#    print(col + ' --> ' + str(useful_label[col].isna().sum()))

for col in useful_label.columns:
    useful_label[col]=useful_label[col].fillna(useful_label[col].mean())


rain_sj = useful_label[is_sj][specificYear]["station_precip_mm"]
humidity_sj = useful_label[is_sj][specificYear]["reanalysis_relative_humidity_percent"]
temp = useful_label[is_sj][specificYear]["reanalysis_min_air_temp_k"]- 273
date_sj = features[is_sj][specificYear]["week_start_date"]

cases_sj = labels[is_sj][specificYear]

plt.title("2003, 2004, 2005 San Juan")
plt.plot (date_sj,humidity_sj, 'y-', label="humidity")
plt.plot(date_sj,rain_sj , label="rain")
plt.plot(date_sj,temp,  'g-', label = "Temp")
plt.plot(date_sj, cases_sj, 'r*', label= "cases")
plt.xlabel("week of the Year")
plt.legend(bbox_to_anchor=(0.0075, 0.81, 0.25, .102), loc='lower left',
           ncol=1, mode="expand", borderaxespad=0.)
plt.show()


#CORRELATION PLOT
scaler = MinMaxScaler()
normalized = scaler.fit_transform(useful_label)

plt.xlabel("reanalysis_dew_point_temp_k")
plt.ylabel ("reanalysis_specific_humidity_g_per_kg")
plt.plot(normalized[:,7],normalized[:,13],'*')
print("****************")
plt.show()
exit(0)
for colum in useful_label.columns:
    print(colum,"---->", useful_label[colum].describe()._values[1])

for i in useful_label.columns:
    print(i)

box = useful_label.boxplot()
box.set_ylabel("Total Power")
plt.xticks( rotation='vertical')
plt.boxplot(normalized)

#sns.pairplot(pd.DataFrame(normalized))

plt.show()

#ONE-CITY-ONE-MODEL CREATION
sanJuanFeatures = pd.read_csv("SANJUAN_feature_lables.csv")
IquitosFeatures = pd.read_csv("IQUITOS_Feature_labels.csv")
sanJuanCases = pd.read_csv("SANJUAN_totalCases.csv")
IquitosCases = pd.read_csv("IQUITOS_totalCases.csv")

sanJuanFeatures = fixOneCityDataFrame(sanJuanFeatures)
IquitosFeatures = fixOneCityDataFrame(IquitosFeatures)

#WHOLE DATAFRAME
train_featureOriginal = pd.read_csv("dengue_features_train.csv")
train_labelsOriginal = pd.read_csv("dengue_labels_train.csv")
train_feature_total = fixTotalDataFrame(train_featureOriginal)
train_labels_total = train_labelsOriginal["total_cases"]

#MODEL COMPARISON
maeIq =[]
maeSJ =[]
maeTot=[]

for i in range(20):
    rfcv = RandomForestRegressor(n_estimators=50, criterion="mse")
    maeIq.append(crossValidation(rfcv, IquitosFeatures, IquitosCases, 5))
    rfcv = RandomForestRegressor(n_estimators=75, criterion="mse")
    maeSJ.append(crossValidation(rfcv, sanJuanFeatures, sanJuanCases, 5))
    rfcv = RandomForestRegressor(n_estimators=100, criterion="mse")
    maeTot.append(crossValidation(rfcv, train_feature_total, train_labels_total, 5))

plt.plot(range(len(maeIq)), maeIq, label = "Iquitos model")
plt.plot(range(len(maeIq)), maeSJ, label = "San Juan model ")
plt.plot(range(len(maeIq)), maeTot, label = "Whole model" )
plt.xlabel("Number of iteration")
plt.ylabel("MEA")
plt.legend()

rfcv = RandomForestRegressor(n_estimators=100, criterion="mse", max_depth=120)
maeTot.append(crossValidation(rfcv, brenos_features, brenos_labels, 5))
fI = rfcv.feature_importances_
#plt.plot([3, 4, 5], maeList, "-")
plt.bar(brenos_features.columns[1:], fI[1:])
plt.xticks(brenos_features.columns[1:], brenos_features.columns[1:], rotation='vertical')
plt.ylabel("Importance")
plt.xlabel("Feature")
plt.show()