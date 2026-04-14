💧 Water Consumption Analysis (State-wise) – Data Science Project
📌 Project Overview

This project analyzes **state-wise water consumption in India** using Python.
It covers the complete **Data Science lifecycle**, including data cleaning, visualization, statistical analysis, and machine learning.

The dataset used is **uncleaned (real-world style)** with missing values, inconsistencies, and noise.

---

🎯 Objectives

* Analyze water consumption trends across states
* Compare **urban vs rural usage**
* Perform **data cleaning and preprocessing**
* Generate **visual insights using graphs**
* Apply **statistical tests**
* Build a **machine learning model** for prediction

---

📂 Dataset

* File: `water_consumption_uncleaned_1200_rows.csv`
* Contains:

  * State
  * Year
  * Urban Water Consumption (LPCD)
  * Rural Water Consumption (LPCD)
  * Total Water Consumption (MLD)
  * Source
  * Notes

⚠️ Dataset includes:

* Missing values
* Inconsistent naming
* Duplicate entries
* Mixed formats

---

🛠️ Technologies Used

* Python 🐍
* Pandas & NumPy
* Matplotlib & Seaborn
* SciPy (Statistical Analysis)
* Scikit-learn (Machine Learning)

---

🔧 Project Workflow

1️⃣ Data Collection

* Loaded dataset using Pandas

2️⃣ Data Cleaning

* Removed spaces and standardized column names
* Converted data types
* Handled missing values
* Removed duplicates

3️⃣ Data Analysis (NumPy + Pandas)

* Mean, Max, Min calculations
* Grouping and aggregation

4️⃣ Data Visualization

Graphs used:

* Line Plot (Year vs Consumption)
* Bar Chart (Top states)
* Histogram
* Box Plot (Outliers)
* Scatter Plot
* Heatmap (Correlation)
* Pairplot
* Distribution Plot

5️⃣ Exploratory Data Analysis (EDA)

* Summary statistics
* Correlation & covariance
* Outlier detection (Z-score)

6️⃣ Statistical Analysis

* Shapiro-Wilk Test (Normality)
* T-Test (Urban vs Rural)
* Chi-Square Test
* Distribution analysis

7️⃣ Machine Learning

* Model: Linear Regression
* Features:

  * Urban consumption
  * Rural consumption
* Target:

  * Total water consumption
* Evaluation:

  * Mean Squared Error (MSE)
  * R² Score

---

📊 Key Insights

* Urban consumption is generally higher than rural
* Strong correlation between urban & total consumption
* Some states show extreme outliers
* Data required cleaning due to inconsistencies

---

🚀 How to Run

1. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

2. Run the script:

```bash
python your_script_name.py
```

3. Make sure dataset path is correct:

```python
pd.read_csv("your_file_path.csv")
```

---

📌 Future Improvements

* Add advanced ML models (Random Forest, XGBoost)
* Deploy as web dashboard
* Real-time API integration
* Interactive visualizations (Plotly)

---

📚 Course Coverage

This project covers:

* Data Collection
* Data Cleaning
* NumPy & Pandas
* Visualization
* EDA
* Statistical Analysis
* Machine Learning

---

👨‍💻 Author

Sai Manikanta Sai

---

⭐ Conclusion

This project demonstrates a complete **Data Science workflow** and provides insights into water consumption patterns across Indian states.
