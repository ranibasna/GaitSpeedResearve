import numpy as np
import pandas as pd
import pyreadstat
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from scipy import stats
from DataProcessing import MICEImputer, DataProcessor, data_prepare, ElasticNetSelector, best_interaction_terms_with_quadratic_fixed, transform_target, transform_variables, plot_distributions, adjust_negative_diff, gradient_boosting_feature_selection_mlflow, compare_selected_features, find_optimal_cutoffs, add_polynomial_terms

import os

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR")
SCRIPTS_DIR = os.getenv("SCRIPTS_DIR")
REPORTS_DIR = os.getenv("REPORTS_DIR")

# Define your target column based on your dataset
target_column = 'Diff'

numerical_columns = [target_column, 'BalanceTest', 'Weight', 'calf_circum', 'HandGripStrength', 'CongnitiveFunction']
nominal_columns = ['sex','Smoking']
ordinal_columns = ['age_group','PhysicalActivity', 'LegPain', 'BackPain','AlkoholConsumption']
columns_to_scale = numerical_columns
columns_to_log = ['HandGripStrength','CongnitiveFunction','BalanceTest']
columns_to_poly = ['PhysicalActivity', 'LegPain']

    
# data prepaeration
file_path = os.path.join(DATA_DIR, 'data.sav')
df, meta = pyreadstat.read_sav(file_path)
df = data_prepare(df=df, target_column=target_column)

negative_diff_indices = df[df[target_column] <= 0].index
print(negative_diff_indices)

# Convert the negative values in the target variable to NA for imputing
df = adjust_negative_diff(df, column_name=target_column, method='remove')

negative_diff_indices = df[df[target_column] <= 0].index
print(negative_diff_indices)



# Data Processing

# Instantiate the MICEImputer and ElasticNetSelector
mice_imputer = MICEImputer(numerical_columns, nominal_columns, ordinal_columns)
# data_processor = DataProcessor(columns_to_scale, columns_to_log)
elastic_net_selector = ElasticNetSelector(target_column=target_column)

# First, apply MICE imputation
imputed_data = mice_imputer.fit_transform(df)

negative_diff_indices = imputed_data[imputed_data[target_column] <= 0].index
print(negative_diff_indices)


# Convert the nominal columns to category dtypes
for var in nominal_columns:
    imputed_data[var] = imputed_data[var].astype('category')

# Next, select features with ElasticNet
selected_features = elastic_net_selector.fit(imputed_data).selected_features_

# Test the function and log results to MLflow
selected_features_GrBoost, model_GrBoost = gradient_boosting_feature_selection_mlflow(data=imputed_data, target_column=target_column , threshold=0.008)


# Filter the imputed data to only include the selected features
selected_data = imputed_data[selected_features + [target_column]]
# selected_data = imputed_data[selected_features_GrBoost + [target_column]]

print(selected_data.head())
# Get the columns of df and selected_data as sets
df_columns = set(df.columns)

selected_data_columns = set(selected_data.columns)
# Find the difference between the two sets
columns_in_df_not_in_selected_data = df_columns - selected_data_columns
# Print the differences
print("Columns in df but not in selected_data:", columns_in_df_not_in_selected_data)

comparison_result = compare_selected_features(
    elastic_net=selected_features, 
    gr_boost=selected_features_GrBoost
)

for key, value in comparison_result.items():
    print(key)
    for subkey, subvalue in value.items():
        print(f"  {subkey}: {subvalue}")




# transform the target variable to be more normally distributed
selected_data = transform_target(selected_data, target_column=target_column, method='yeojohnson')
# selected_data = transform_target(selected_data, target_column=target_column, method='log')

# columns_to_log = ['HandGripStrength','CongnitiveFunction','BalanceTest']
columns_to_log = ['CongnitiveFunction','BalanceTest']
columns_to_transform = columns_to_log
# columns_to_transform = ['HandGripStrength','CongnitiveFunction']
# Choose the transformation method: 'log', 'boxcox', or 'yeojohnson'
transformation_method = 'yeojohnson'  
# Call the function to transform the specified columns
selected_data = transform_variables(selected_data, columns_to_transform, method=transformation_method)


# Scale the continuous predictors
from sklearn.preprocessing import StandardScaler
# Initialize the StandardScaler
scaler = StandardScaler()
# Fit the scaler on the continuous predictors and transform the data
selected_data[numerical_columns] = scaler.fit_transform(selected_data[numerical_columns])


# Convert the nominal columns to category dtypes
for var in nominal_columns:
    selected_data[var] = selected_data[var].astype('category')


# Copy the selected data
selected_data = selected_data.copy()

# Now, split the filtered data into features and target
X = selected_data.drop(columns=[target_column])
y = selected_data[target_column]

# save selected data as a csv file
selected_data.to_csv(os.path.join(DATA_DIR, 'selected_data.csv'), index=False)

import statsmodels.formula.api as smf
# Convert X into a string of plus-separated column names for the formula
predictors = ' + '.join(X.columns)
formula = f"{target_column} ~ {predictors}"


# Fit the mixed model
re_formula_handgripstrength = "1 + PhysicalActivity + HandGripStrength"
# re_formula_handgripstrength = "1 + PhysicalActivity"
model = smf.mixedlm(formula, selected_data, groups=selected_data["age_group"], re_formula=re_formula_handgripstrength).fit(maxiter=100000000)


print(model.summary())
# print(model.random_effects)

random_effects = model.random_effects



with open('model_summaries.txt', 'a') as f:
    f.write("Random Intercept and slop Results\n")
    f.write(model.summary().as_text())
    for group, effects in random_effects.items():
        random_intercept = effects[0]  # Random intercept for the group
        physical_activity_slope = effects['PhysicalActivity']  # Random slope for PhysicalActivity
        handgrip_strength_slope = effects['HandGripStrength']  # Random slope for HandGripStrength

        f.write(f"Group: {group}")
        f.write(f"  Random Intercept: {random_intercept}")
        f.write(f"  PhysicalActivity Random Slope: {physical_activity_slope}")
        f.write(f"  HandGripStrength Random Slope: {handgrip_strength_slope}\n")



import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'model' is your fitted mixed model and 'random_effects' contains the random effects
random_effects = model.random_effects

# Preparing data for plotting
groups = []
intercepts = []
physical_activity_slopes = []
handgrip_strength_slopes = []

for group, effects in random_effects.items():
    groups.append(group)
    intercepts.append(effects[0])
    physical_activity_slopes.append(effects['PhysicalActivity'])
    handgrip_strength_slopes.append(effects['HandGripStrength'])

# Plotting Random Intercepts
plt.figure(figsize=(10, 6))
sns.barplot(x=groups, y=intercepts, color='blue')
plt.xlabel('Age Group')
plt.ylabel('Random Intercept')
plt.title('Random Intercepts for Each Age Group')
plt.show()

# Plotting Random Slopes for PhysicalActivity and HandGripStrength
plt.figure(figsize=(12, 6))
sns.lineplot(x=groups, y=physical_activity_slopes, marker='o', label='PhysicalActivity')
sns.lineplot(x=groups, y=handgrip_strength_slopes, marker='o', label='HandGripStrength')
plt.xlabel('Age Group')
plt.ylabel('Random Slope')
plt.title('Random Slopes for PhysicalActivity and HandGripStrength Across Age Groups')
plt.legend()
plt.show()



## Q-Q PLot
import statsmodels.api as sm

fig, ax = plt.subplots(figsize=(16, 9))
sm.qqplot(model.resid, dist=stats.norm, line='s', ax=ax)
ax.set_title("Q-Q Plot")
plt.show()


