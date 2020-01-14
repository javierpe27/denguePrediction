import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error

#AUXILIAR FUNTIONS
def fixTotalDataFrame (df):
    oneHot = pd.get_dummies(df["city"])
    df = df.drop(
        ["city", "reanalysis_sat_precip_amt_mm", "week_start_date",
         "reanalysis_tdtr_k", "reanalysis_avg_temp_k"], axis=1)
    df = df.join(oneHot)
    # Imputation
    for col in df.columns:
        if col != "week_start_date":
            df[col] = df[col].fillna(df[col].mean())
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

def crossValidation(X, y, k):
    maes = []
    kf = KFold(n_splits=k)
    index = np.arange(len(X))
    for train_index, test_index in kf.split(X, y):
        rfcv = RandomForestRegressor(n_estimators=100, criterion="mse", max_depth=120)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # Model
        rfcv.fit(X_train, y_train)
        predictions = rfcv.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        maes.append(mae)
        print("MAE", mae)
    #print(X_test["year"], X_test["weekofyear"])
    #plt.plot(range(len(predictions)), predictions, label="Predictions")
    #plt.plot(range(len(y_test)), y_test, label="Real Data")
    #plt.xlabel("Week of the year")
    #plt.ylabel("Number of cases")
    #plt.legend()
    print("Mean average Error", np.mean(maes))
    return  rfcv



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
    rfcv = RandomForestRegressor(n_estimators=75, criterion="mse")
    maeIq.append(crossValidation(IquitosFeatures, IquitosCases, 5))
    rfcv = RandomForestRegressor(n_estimators=75, criterion="mse")
    maeSJ.append(crossValidation(sanJuanFeatures, sanJuanCases, 5))
    rfcv = RandomForestRegressor(n_estimators=100, criterion="mse", max_depth=120)
    maeTot.append(crossValidation(train_feature_total, train_labels_total, 5))

plt.plot(range(len(maeIq)), maeIq, label = "Iquitos model")
plt.plot(range(len(maeIq)), maeSJ, label = "San Juan model ")
plt.plot(range(len(maeIq)), maeTot, label = "Whole model" )
plt.xlabel("Number of iteration")
plt.ylabel("MAE")
plt.legend()



#rfcv = RandomForestRegressor(n_estimators=100, criterion="mse", max_depth=120)
rfcv = crossValidation( train_feature_total, train_labels_total, 5)

fI = rfcv.feature_importances_
#plt.plot([3, 4, 5], maeList, "-")
plt.bar(train_feature_total.columns, fI)
plt.xticks(train_feature_total.columns, train_feature_total.columns, rotation='vertical')
plt.ylabel("Importance")
plt.xlabel("Feature")
plt.show()