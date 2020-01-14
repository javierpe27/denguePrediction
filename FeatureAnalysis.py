
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