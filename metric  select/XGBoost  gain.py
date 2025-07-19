import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

# Read the Excel file
file_path = "C:\\Users\\17958\\Desktop\\train_4.0.xlsx"
try:
    data = pd.read_excel(file_path)
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()

# Define feature columns and target column
feature_columns = [
    'B', 'COM_RAT', 'Cyclic', 'D', 'Dcy*', 'DIT', 'DPT*', 'E', 'Inner', 'LCOM',
    'Level', 'LOC', 'N', 'NCLOC', 'NOAC', 'NOC', 'NOIC', 'OCmax', 'PDcy', 'PDpt',
    'STAT', 'SUB', 'TCOM_RAT', 'V', 'WMC', 'CBO', 'CLOC', 'Command', 'CONS', 'CSA',
    'CSO', 'CSOA', 'Dcy', 'DPT', 'INNER', 'jf', 'JLOC', 'Jm', 'Level*', 'MPC', 'n',
    'NAAC', 'NAIC', 'NOOC', 'NTP', 'OCavg', 'OPavg', 'OSavg', 'OSmax', 'Query',
    'RFC', 'TODO'
]
target_column = "1适合LLM"

# Verify all columns exist in the data
missing_features = [col for col in feature_columns if col not in data.columns]
if missing_features:
    print(f"The following feature columns are missing from the data: {missing_features}")
    exit()

if target_column not in data.columns:
    print(f"Target column '{target_column}' not found in the data.")
    exit()

# Remove rows with NaN values
data = data.dropna()
if data.empty:
    print("No data remains after removing rows with NaN values.")
    exit()

# Prepare data
X = data[feature_columns]
y = data[target_column]

# Ensure all features are numeric
if not all(X.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))):
    print("Non-numeric data found in features. Please encode or convert them first.")
    exit()

# Train XGBoost model
model = xgb.XGBClassifier()
model.fit(X, y)

# Get feature importance based on information gain
importance = model.get_booster().get_score(importance_type='gain')

# Convert to DataFrame and sort
importance_df = pd.DataFrame(importance.items(), columns=['Feature', 'Importance'])
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print feature importance
print(importance_df)

# Visualize feature importance
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance based on Information Gain')
plt.gca().invert_yaxis()
plt.show()