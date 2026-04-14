import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("/Users/manikantasai/Desktop/water_consumption_uncleaned_1200_rows.csv")

print(df.head())


# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Convert numeric columns
df['urban_water_consumption_lpcd'] = pd.to_numeric(df['urban_water_consumption_lpcd'], errors='coerce')
df['rural_water_consumption_lpcd'] = pd.to_numeric(df['rural_water_consumption_lpcd'], errors='coerce')
df['total_water_consumption_mld'] = pd.to_numeric(df['total_water_consumption_mld'], errors='coerce')

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Drop duplicates
df.drop_duplicates(inplace=True)

print("Cleaned Data:")
print(df.head())


urban_array = df['urban_water_consumption_lpcd'].values
print("Mean:", np.mean(urban_array))
print("Max:", np.max(urban_array))
print("Min:", np.min(urban_array))



# Line Plot
plt.figure()
df.groupby("year")['total_water_consumption_mld'].mean().plot()
plt.title("Year vs Avg Water Consumption")
plt.grid(True)
plt.show()

# Bar Plot
plt.figure()
df.groupby("state")['total_water_consumption_mld'].mean().sort_values().tail(10).plot(kind='bar')
plt.title("Top 10 Water Consuming States")
plt.grid(True)
plt.show()

# Histogram
plt.figure()
plt.hist(df['urban_water_consumption_lpcd'])
plt.title("Urban Water Consumption Distribution")
plt.grid(True)
plt.show()

# Boxplot
plt.figure()
sns.boxplot(x=df['urban_water_consumption_lpcd'])
plt.title("Outlier Detection (Urban)")
plt.grid(True)
plt.show()

# Heatmap
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.grid(True)
plt.show()

# Pairplot
sns.pairplot(df.select_dtypes(include=np.number))
plt.show()

print(df.describe())
print(df.corr(numeric_only=True))
print(df.cov(numeric_only=True))

z_scores = np.abs(stats.zscore(df.select_dtypes(include=np.number)))
print("Outliers:", np.sum(z_scores > 3))



# Shapiro Test
stat, p = stats.shapiro(df['urban_water_consumption_lpcd'])
print("Shapiro p-value:", p)

# T-Test
t_stat, p_val = stats.ttest_ind(
    df['urban_water_consumption_lpcd'],
    df['rural_water_consumption_lpcd']
)
print("T-test p-value:", p_val)

# Chi-Square
contingency = pd.crosstab(df['state'], df['year'])
chi2, p, dof, exp = stats.chi2_contingency(contingency)
print("Chi-square p-value:", p)

# Distribution Plot
plt.figure()
sns.histplot(df['urban_water_consumption_lpcd'], kde=True)
plt.title("Distribution Curve")
plt.grid(True)
plt.show()



X = df[['urban_water_consumption_lpcd', 'rural_water_consumption_lpcd']]
y = df['total_water_consumption_mld']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Prediction Plot
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")

plt.show()
