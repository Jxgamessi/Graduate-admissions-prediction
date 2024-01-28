import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


data = pd.read_csv('admission_data.csv')


X = data.drop(['Serial No.', 'Chance of Admit'], axis=1)
y = data['Chance of Admit']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)


random_test_data = pd.DataFrame({
    'GRE Score': [310],
    'TOEFL Score': [107],
    'University Rating': [3],
    'SOP': [3.5],
    'LOR': [4.0],
    'CGPA': [8.0],
    'Research': [1]
})


random_test_data_scaled = scaler.transform(random_test_data)


predicted_chance_of_admit = rf_model.predict(random_test_data_scaled)


admission_threshold = 0.7


admitted = predicted_chance_of_admit > admission_threshold

print('Random Test Input:')
print(random_test_data)
print(f'\nPredicted Chance of Admit: {predicted_chance_of_admit[0]:.4f}')

if admitted:
    print('Admission Status: Admitted')
else:
    print('Admission Status: Not Admitted')
