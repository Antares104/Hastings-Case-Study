import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import numpy as np
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

#Changed the excel file to a csv file to be read in pandas
df1 = pd.read_csv(r"C:\Users\User\.spyder-py3\casestudydata.csv")
df = pd.read_csv(r"C:\Users\User\.spyder-py3\casestudydata.csv")

#Initial cleaning, dropping duplicate rows and getting rid of values arent appropriate
df = df.drop_duplicates()
df = df[df['Driver1_Licence_Years'] >= 0]
df = df[df['Driver1_Claims'] >= 0]
df = df[df['Vehicle_Age'] >= 0]
df = df[df['Vehicle_Value'] >= 0]
df = df[df['Tax'] >= 0]
df = df[df['Vehicle_Annual_Mileage'] >= 0]
df = df[df['Credit_Score'] >= 0]
df = df[df['Days_to_Inception'] >= 0]
df = df[df['Premium'] >= 0]
df = df[df['Capped_Premium'] >= 0]
expected_values1 = ['Yes', 'No']
df = df[df['Driver1_Convictions'].isin(expected_values1)]
expected_values2 = ['Full UK', 'Automatic', 'Provisional UK']
df = df[df['Driver1_Licence_Type'].isin(expected_values2)]
expected_values3 = ['Annual', 'Monthly']
df = df[df['Payment_Type'].isin(expected_values3)]
expected_values4 = ['Married', 'Separated', 'Widowed', 'Single', 'Other', 'Divorced', 'Common Law', 'Civil Partnership']
df['Driver1_Marital_Status'] = df['Driver1_Marital_Status'].str.strip().str.title()
df = df[df['Driver1_Marital_Status'].isin(expected_values4)]

#Encoding and mapping the variables that are as a string type so that the models can use them
label_encoder = LabelEncoder()
df['Payment_Type_Encoded'] = label_encoder.fit_transform(df['Payment_Type'])
marital_status_mapping = {
    'Single': 1,
    'Married': 2,
    'Separated': 3,
    'Divorced': 4,
    'Widowed': 5,
    'Other': 6,
    'Common Law': 7,
    'Civil Partnership': 8
}
driver_licence_mapping = {
    'Full UK': 1,
    'Automatic': 2,
    'Provisional UK': 3
}
driver_convictions_encoded= {
    'Yes': 1,
    'No': 2
}

df['Driver1_Marital_Status'] = df['Driver1_Marital_Status'].map(marital_status_mapping)
df['Driver1_Licence_Type_Encoded'] = df['Driver1_Licence_Type'].map(driver_licence_mapping)
df['Driver1_Convictions_Encoded'] = df['Driver1_Convictions'].map(driver_convictions_encoded)
Premium = df['Premium']
Vehicle_Value = df['Vehicle_Value']
Vehicle_Age = df['Vehicle_Age']
Vehicle_Annual_Mileage = df['Vehicle_Annual_Mileage']
Driver_Age = df['Driver1_DOB']
Years_Having_Licence = df['Driver1_Licence_Years']
Number_Claims = df['Driver1_Claims']

# Converting driver date of birth to driver age, again so that it could be used in the models
df['Driver1_DOB'] = pd.to_datetime(df['Driver1_DOB'], dayfirst=True)
current_date = pd.to_datetime('today')
df['Driver_Age'] = current_date - df['Driver1_DOB']
df['Driver_Age'] = df['Driver_Age'] / pd.Timedelta(days=365)
df['Driver_Age'] = df['Driver_Age'].astype(int)
df.drop('Driver1_DOB', axis=1, inplace=True)
Driver_Age = df['Driver_Age']
Years_Having_Licence = df['Driver1_Licence_Years']
Number_Claims = df['Driver1_Claims']

# Converting quote date to quote month as all the quotes were in 2020 and this would make the data much easier to work with
df['Quote_Date'] = pd.to_datetime(df['Quote_Date'], format='%d/%m/%Y')
df['Month'] = df['Quote_Date'].dt.month
Quote_Date = df['Month']

Driver_Marital_Status = df['Driver1_Marital_Status']
Driver1_Licence_Type = df['Driver1_Licence_Type_Encoded']
Driver1_Convictions = df['Driver1_Convictions_Encoded']
Credit_Score = df['Credit_Score']
Days_to_Inception = df['Days_to_Inception']
Capped_Premium = df['Capped_Premium']
Tax = df['Tax']
Payment_Type = df['Payment_Type_Encoded']

# Inputting all my new clean data into a single dataframe so that further cleaning can take place
insurance_data = pd.DataFrame({
    'Premium': Premium,
    'Quote_Date': Quote_Date,
    'Vehicle_Value': Vehicle_Value,
    'Vehicle_Age': Vehicle_Age,
    'Vehicle_Annual_Mileage': Vehicle_Annual_Mileage,
    'Driver_Age': Driver_Age,
    'Driver_Marital_Status': Driver_Marital_Status,
    'Years_Having_Licence': Years_Having_Licence,
    'Number_Claims': Number_Claims,
    'Driver1_Licence_Type': Driver1_Licence_Type,
    'Driver1_Convictions': Driver1_Convictions,
    'Credit_Score': Credit_Score,
    'Days_to_Inception': Days_to_Inception,
    'Capped_Premium': Capped_Premium,
    'Tax': Tax,
    'Payment_Type': Payment_Type
})
insurance_data1 = insurance_data

#Z score method of removing extreme values
z_scores = stats.zscore(insurance_data1)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 2.5).all(axis=1)
Z_Score = insurance_data1[filtered_entries]

#IQR method of removing extreme values
Q1 = insurance_data.quantile(0.25)
Q3 = insurance_data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
IQR_data = insurance_data[((insurance_data >= lower_bound) & (insurance_data <= upper_bound)).all(axis=1)]

#Mean absolute deviation method of removing extreme values
column_medians = insurance_data.median()
absolute_deviations = (insurance_data - column_medians).abs()
mad = absolute_deviations.median()
threshold = 7.5 * mad
outliers = (absolute_deviations > threshold)
Mean_Absolute_Deviation = insurance_data[~outliers.any(axis=1)]

#Local Outlier Factor method of removing extreme values
features = ['Premium', 'Vehicle_Value', 'Vehicle_Age', 'Vehicle_Annual_Mileage', 'Driver_Age', 'Years_Having_Licence', 'Credit_Score', 'Capped_Premium', 'Tax']
X = insurance_data[features]
lof = LocalOutlierFactor(n_neighbors=500, contamination=0.0425)
outlier_scores = lof.fit_predict(X)
Local_Outlier_Factor = insurance_data[outlier_scores != -1]

#Isolation Forest method of removing extreme values
features = ['Premium', 'Vehicle_Value', 'Vehicle_Age', 'Vehicle_Annual_Mileage', 'Driver_Age', 'Years_Having_Licence', 'Credit_Score', 'Capped_Premium', 'Tax']
X = insurance_data[features]
isolation_forest = IsolationForest(n_estimators=500, contamination=0.15, random_state=42)
outlier_preds = isolation_forest.fit_predict(X)
Isolation_Forest = insurance_data[outlier_preds != -1]

#this can be changed to choose what dataset you are interested in testing
filtered_data2 = Local_Outlier_Factor

#XGBoost model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

X = filtered_data2[['Vehicle_Value', 'Vehicle_Age', 'Driver_Age', 'Years_Having_Licence', 'Days_to_Inception', 'Tax', 'Payment_Type']]
y = filtered_data2['Premium']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

#the params were the optimal params for the metrics that i could find
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.11,
    'max_depth': 5,
    'min_child_weight': 1,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'alpha': 0.0,
    'lambda': 0.5,
}
num_boost_round = 42
model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
y_pred = model.predict(dtest)
test_score = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

#evaluates and prints r^2, mse, rmse, and mae
#print("Test R^2 Score:", test_score)
#print("Mean Squared Error (MSE):", mse)
#print("Root Mean Squared Error (RMSE):", rmse)
#print("Mean Absolute Error (MAE):", mae)


input_params = {
    'Vehicle_Value': 14000, 'Vehicle_Age': 4, 'Driver_Age': 59, 'Years_Having_Licence': 42, 'Days_to_Inception': 14, 'Tax': 1400, 'Payment_Type': 0
}
input_df = pd.DataFrame([input_params])
dinput = xgb.DMatrix(input_df)
predictions1 = model.predict(dinput)

#evaluates input_params and gives a prediction for premium price
print("Predicted Premium Price:", predictions1[0])

#pygam model

from pygam import LinearGAM, s

X = filtered_data2[['Vehicle_Value', 'Vehicle_Age', 'Driver_Age', 'Years_Having_Licence', 'Days_to_Inception', 'Tax', 'Payment_Type']]
y = filtered_data2['Premium']
gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6))
gam.fit(X, y)
predictions = gam.predict(X)
r2 = r2_score(y, predictions)
print("R^2 score:", r2)
new_data = {
    'Vehicle_Value': [14000],
    'Vehicle_Age': [4],
    'Driver_Age': [59],
    'Years_Having_Licence': [42],
    'Days_to_Inception': [14],
    'Tax': [1400],
    'Payment_Type': [0]
}

#this evaluates the prediction for a premium price given the new_data
new_df = pd.DataFrame(new_data)
predictions = gam.predict(new_df)
print("Predicted Premium:", predictions[0])

#this prints the average of the two methods
#print("Predicted Premium", (predictions[0]+predictions1[0])/2)

#Other methods were coded in essentially the same way and have been emitted from this file as I only wanted to highlight the top preforming models. full results can be seen in the report in table 1


#Plots

filtered_data3 = Isolation_Forest

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
sns.boxplot(x='Driver1_Licence_Type', y='Premium', data=filtered_data3)
plt.xlabel('Driver1 Licence Type')
plt.ylabel('Premium Price')
plt.ylim(300, 1350)

plt.subplot(2, 2, 2)
sns.boxplot(x='Driver1_Convictions', y='Premium', data=filtered_data3)
plt.xlabel('Driver1 Convictions')
plt.ylabel('Premium Price')
plt.ylim(300, 1350)

plt.subplot(2, 2, 3)
sns.boxplot(x='Driver_Marital_Status', y='Premium', data=filtered_data3)
plt.xlabel('Marital Status')
plt.ylabel('Premium Price')
plt.ylim(300, 1350)

plt.subplot(2, 2, 4)
sns.boxplot(x='Quote_Date', y='Premium', data=filtered_data3)
plt.xlabel('Quote Month')
plt.ylabel('Premium Price')
plt.ylim(300, 1350)
plt.tight_layout()
#plt.show()

premiums_by_age = filtered_data3.groupby('Driver_Age')['Premium'].mean().reset_index()
filtered_data3 = filtered_data3.copy()
bin_size = 2
filtered_data3['Days_to_Inception_Group'] = pd.cut(filtered_data3['Days_to_Inception'],
                                                    bins=range(0, filtered_data3['Days_to_Inception'].max() + bin_size, bin_size),
                                                    right=False)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.boxplot(x='Days_to_Inception_Group', y='Premium', data=filtered_data3, ax=axes[0, 0], patch_artist=True)
for patch in axes[0, 0].artists:
    patch.set_alpha(0.5)
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].set_ylim(250, 1500)
axes[0, 0].set_title('Premiums vs Days to Inception')
axes[0, 0].set_xlabel('Days to Inception')
axes[0, 0].set_ylabel('Premium')

sns.boxplot(x='Payment_Type', y='Premium', data=filtered_data3, ax=axes[0, 1], order=sorted(filtered_data3['Payment_Type'].unique()))  # Specify the order of categories
axes[0, 1].set_title('Distribution of Premiums for Different Payment Types')
axes[0, 1].set_ylim(250, 1500)
axes[0, 1].set_xlabel('Payment Type')
axes[0, 1].set_ylabel('Premium')

featureshm = ['Premium', 'Vehicle_Value', 'Vehicle_Age', 'Vehicle_Annual_Mileage', 'Driver_Age', 'Years_Having_Licence', 'Credit_Score', 'Tax', 'Days_to_Inception']
selected_data = filtered_data3[featureshm]
correlation_matrix = selected_data.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=axes[1, 0])
axes[1, 0].set_title('Correlation Heatmap')
axes[1, 0].set_xticklabels(featureshm, rotation=45, ha='right')

sns.regplot(x='Driver_Age', y='Premium', data=premiums_by_age, scatter_kws={'s': 10}, ax=axes[1, 1])
axes[1, 1].set_title('Mean Premiums vs. Driver Age')
axes[1, 1].set_xlabel('Driver Age')
axes[1, 1].set_ylabel('Mean Premium')
axes[1, 1].set_xlim(20, 85)

plt.tight_layout()
plt.show()
