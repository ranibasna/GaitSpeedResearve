# import packages
import pandas as pd
# import pytest
import pyreadstat
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

import statsmodels.api as sm
from statsmodels.imputation import mice



# Functions


from scipy import stats
from scipy.stats import yeojohnson


def transform_variables(data, columns, method='log'):
    """
    Transforms specified variables in the dataframe using the specified method.

    Parameters:
    - data: pandas DataFrame containing the target variables.
    - columns: List of strings, names of the variables to be transformed in 'data'.
    - method: The method of transformation. Options are 'log', 'boxcox', 'yeojohnson'.

    Returns:
    - Transformed DataFrame.
    """
    # Make a copy of the DataFrame to avoid changing the original data
    transformed_data = data.copy()

    for column in columns:
        # Ensure that the target variable is positive if using log or boxcox
        if method in ['log', 'boxcox'] and any(transformed_data[column] <= 0):
            raise ValueError(f"Variable '{column}' must be positive for {method} transformation.")

        if method == 'log':
            # Log transformation
            transformed_data[column] = np.log(transformed_data[column] + 0)  # Shift by 1 to handle zeros
        elif method == 'boxcox':
            # Box-Cox transformation requires strictly positive values
            transformed_data[column], _ = stats.boxcox(transformed_data[column] + 0.5)  # Shift by 1 to handle zeros and non-positive values
        elif method == 'yeojohnson':
            # Yeo-Johnson transformation
            transformed_data[column], _ = yeojohnson(transformed_data[column])
        else:
            raise ValueError("Invalid method. Use 'log', 'boxcox', or 'yeojohnson'.")

    return transformed_data


# transformed_data = transform_variables(selected_data, ['column1', 'column2'], method='log')



def transform_target(data, target_column, method='log'):
    """
    Transforms the target variable in the dataframe using the specified method.

    Parameters:
    - data: pandas DataFrame containing the target variable.
    - target_column: String, the name of the target variable column in 'data'.
    - method: The method of transformation. Options are 'log', 'boxcox', 'yeojohnson'.

    Returns:
    - Transformed DataFrame.
    """
    # Make a copy of the DataFrame to avoid changing the original data
    transformed_data = data.copy()
    
    # Ensure that the target variable is positive if using log or boxcox
    if method in ['log', 'boxcox'] and any(transformed_data[target_column] + 1 <= 0):
        raise ValueError(f"Target variable must be positive for {method} transformation.")
    
    if method == 'log':
        # Log transformation
        transformed_data[target_column] = np.log(transformed_data[target_column] + 1) # Shift by 1 to handle zeros
    elif method == 'boxcox':
        # Box-Cox transformation requires strictly positive values
        transformed_data[target_column], _ = stats.boxcox(transformed_data[target_column] + 1) # Shift by 1 to handle zeros and non-positive values
    elif method == 'yeojohnson':
        # Yeo-Johnson transformation
        transformed_data[target_column], _ = yeojohnson(transformed_data[target_column])
    else:
        raise ValueError("Invalid method. Use 'log', 'boxcox', or 'yeojohnson'.")

    return transformed_data

def adjust_negative_diff(data, column_name='Diff', method='remove'):
    """
    Adjusts negative values in the specified column of a DataFrame by either removing them,
    setting them to zero, or converting them to NaN for imputation.

    Parameters:
    - data: pandas DataFrame containing the target variable.
    - column_name: Name of the column to adjust.
    - method: 'remove' to drop rows with negative values, 'zero' to set negative values to zero,
              'impute' to convert negative values to NaN for later imputation.

    Returns:
    - Adjusted DataFrame.
    """
    adjusted_data = data.copy()

    if method == 'remove':
        # Remove rows where the column has negative values
        adjusted_data = adjusted_data[adjusted_data[column_name] > 0]
    elif method == 'zero':
        # Set negative values in the column to zero
        adjusted_data.loc[adjusted_data[column_name] < 0, column_name] = 0
    elif method == 'impute':
        # Convert negative values to NaN for imputation
        adjusted_data.loc[adjusted_data[column_name] <= 0, column_name] = np.nan
    else:
        raise ValueError("Invalid method. Use 'remove', 'zero', or 'impute'.")

    return adjusted_data


from sklearn.tree import DecisionTreeRegressor

def find_optimal_cutoffs(data, predictor, target, max_depth=3):
    """
    Uses a decision tree regressor to find optimal cutoffs for binning a continuous predictor.

    Parameters:
    data -- pandas DataFrame containing the predictor and target variables.
    predictor -- String, the name of the continuous predictor variable to bin.
    target -- String, the name of the numerical target variable for prediction.
    max_depth -- Integer, the maximum depth of the decision tree.

    Returns:
    A Series with the binned predictor variable.
    """
    # Initialize the decision tree regressor with the specified maximum depth
    tree = DecisionTreeRegressor(max_depth=max_depth)

    # Fit the decision tree model
    tree.fit(data[[predictor]], data[target])

    # Extract the threshold values from the tree; these can be used as cutoffs
    thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]

    # Sort the thresholds and add -inf and inf to cover all possible values
    cutoffs = np.sort(np.unique(thresholds))

    # Use these cutoffs to bin the predictor variable
    binned_predictor = pd.cut(data[predictor], bins=np.concatenate(([-np.inf], cutoffs, [np.inf])), labels=False)

    return binned_predictor


def add_polynomial_terms(data, ordinal_columns, max_degree=2):
    """
    Adds polynomial terms (up to a specified degree) for ordinal variables already coded as numbers.

    Parameters:
    - data: pandas DataFrame containing the data.
    - ordinal_columns: List of column names for which to create polynomial terms.
    - max_degree: Maximum degree of polynomial terms to create.

    Returns:
    - DataFrame with additional polynomial term columns.
    """
    for column in ordinal_columns:
        for degree in range(2, max_degree + 1):  # Start at 2, we already have the linear term (degree 1)
            poly_col_name = f"{column}_poly{degree}"
            data[poly_col_name] = data[column] ** degree
    return data

# Example usage:
# Assuming 'selected_data' is your DataFrame and 'ordinal_var' is your ordinal variable already coded as numbers
# selected_data = add_polynomial_terms(selected_data, ['ordinal_var'], max_degree=3)
# Now 'selected_data' will have 'ordinal_var_poly2' and 'ordinal_var_poly3' as additional columns

def plot_distributions(data, columns, bins=30, figsize=(10, 6)):
    """
    Plots histograms for multiple variables in a dataframe.

    Parameters:
    data -- pandas DataFrame containing the variables.
    columns -- List of column names to plot.
    bins -- Number of bins in the histogram.
    figsize -- Size of the figure.
    """
    num_columns = len(columns)
    plt.figure(figsize=(figsize[0], figsize[1] * num_columns))

    for i, column in enumerate(columns, 1):
        plt.subplot(num_columns, 1, i)
        plt.hist(data[column], bins=bins, edgecolor='k', alpha=0.7)
        plt.title(f'Distribution of {column}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)

    plt.tight_layout()
    plt.show()



# define the function
def create_pairplot(df, num_cols, cat_col):
    # create the pair plot
    sns.pairplot(data=df, vars=num_cols, hue=cat_col)
    plt.title('Pair Plot of Numerical Columns with Color Encoding', fontsize=15)
    plt.show()



def perform_mice_imputation(data, numerical_columns, nominal_columns, ordinal_columns):
    """Performs MICE imputation on the given data."""
    
    # Identify the columns with missing data
    missing_columns = data.columns[data.isnull().any()].tolist()

    # Create a MICE instance
    imp = mice.MICEData(data)

    # Apply MICE imputation for each missing column
    for column in missing_columns:
        # Determine the formula
        formula = f"{column} ~ {' + '.join([col for col in data.columns if col != column])}"

        # Determine the model type based on data type
        if column in numerical_columns:
            model_class = sm.OLS
        elif column in nominal_columns:
            # If binary or ordinal, use MNLogit
            model_class = sm.MNLogit if len(data[column].dropna().unique()) > 2 else sm.Logit
        elif column in ordinal_columns:
            model_class = sm.MNLogit

        # Perform the imputation
        imp.set_imputer(column, formula=formula, model_class=model_class)

    # Get the imputed dataset
    data_imputed = imp.data

    return data_imputed


def test_missing_value_imputation(original_data):
    """Test that there are no missing values after imputation."""
    imputed_data = perform_mice_imputation(original_data)
    assert not imputed_data.isnull().any().any()
    """Test that the shape of the imputed data matches the original."""
    assert imputed_data.shape == original_data.shape
    """Test that the columns of the imputed data match the original."""
    assert set(imputed_data.columns) == set(original_data.columns)



def validate_imputation_statistically(original_data, imputed_data):
    """
    Compare basic statistics (mean, median, standard deviation) 
    for the original data and the imputed data.
    
    Parameters:
    - original_data (pd.DataFrame): Original data with missing values.
    - imputed_data (pd.DataFrame): Data after imputation.
    
    Returns:
    - pd.DataFrame: Comparison of statistics for original and imputed data.
    """
    # Identify columns with missing values
    columns_with_missing = original_data.columns[original_data.isnull().any()].tolist()
    
    # Calculate statistics for original and imputed data
    original_stats = original_data[columns_with_missing].describe().loc[['mean', '50%', 'std']]
    imputed_stats = imputed_data[columns_with_missing].describe().loc[['mean', '50%', 'std']]
    
    # Rename rows for clarity
    original_stats = original_stats.rename(index={'50%': 'median'})
    imputed_stats = imputed_stats.rename(index={'50%': 'median'})
    
    # Concatenate results for a side-by-side comparison
    comparison = pd.concat([original_stats, imputed_stats], axis=0, keys=['Original', 'Imputed'])
    
    return comparison

def validate_imputation_visually(original_data, imputed_data, categorical_columns, numerical_columns):
    """
    Produce side-by-side density plots for the original and imputed data 
    for each feature with missing values.
    
    Parameters:
    - original_data (pd.DataFrame): Original data with missing values.
    - imputed_data (pd.DataFrame): Data after imputation.
    """
    columns_with_missing = original_data.columns[original_data.isnull().any()].tolist()
    
    for col in columns_with_missing:
        # Check for variation in data
        if original_data[col].nunique() > 1 and imputed_data[col].nunique() > 1:
            plt.figure(figsize=(12, 6))
            
            sns.kdeplot(original_data[col], label='Original Data', color='blue')
            sns.kdeplot(imputed_data[col], label='Imputed Data', color='red')
            
            plt.title(f"Distribution of {col} - Original vs. Imputed")
            plt.legend()
            plt.show()

def compare_distributions_imp(original_data, imputed_data, numerical_columns, categorical_columns):
    """
    Enhanced function to compare distributions of each variable before and after imputation 
    using a predefined color palette.

    Parameters:
    - original_data: DataFrame with the original data
    - imputed_data: DataFrame with the imputed data
    """
    # Define the color palette again
    color_palette = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#8e44ad"]
    # List of columns to check
    columns_to_check = original_data.columns[original_data.isnull().any()]
    
    for column in columns_to_check:
        
        # If the variable is numerical
        if column in numerical_columns:
            plt.figure(figsize=(10, 6))
            
            # Improved KDE plots with better aesthetics and color palette
            sns.kdeplot(original_data[column], fill=True, color=color_palette[0], label="Original", lw=2)
            sns.kdeplot(imputed_data[column], fill=True, color=color_palette[1], label="Imputed", lw=2)
            plt.title(f'{column} Distribution Comparison', fontsize=16)
            plt.xlabel(column, fontsize=14)
            plt.ylabel('Density', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            plt.show()

        # If the variable is categorical (binary, nominal, or ordinal)
        elif column in categorical_columns:
            plt.figure(figsize=(10, 6))
            
            # Improved bar plots with better aesthetics and color palette
            original_data[column].value_counts(normalize=True).plot(kind='bar', position=0, width=0.4, color=color_palette[0], label='Original', alpha=0.7)
            imputed_data[column].value_counts(normalize=True).plot(kind='bar', position=1, width=0.4, color=color_palette[1], label='Imputed', alpha=0.7)
            plt.title(f'{column} Distribution Comparison', fontsize=16)
            plt.ylabel('Proportion', fontsize=14)
            plt.xlabel(column, fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(axis='y', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            plt.show()

# Variable selection functions
import numpy as np
import tempfile
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LassoCV, LinearRegression, ElasticNetCV
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Ensure the MLflow experiment exists and is set
mlflow.set_experiment("Feature_Selection_Experiment")

def lasso_feature_selection_mlflow(data, target_column, alpha_values=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]):
    """
    Perform feature selection using Lasso regression.
    Train a linear regression model on the selected features.
    Log the results to MLflow.
    """
    # Split data into train and test sets
    X = data.drop(columns=[target_column])
    y = data[target_column]
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start MLflow run
    with mlflow.start_run():
        # Set run name and tags
        mlflow.set_tag("mlflow.runName", "Lasso Feature Selection")
        mlflow.set_tag("method", "Lasso")
        mlflow.set_tag("target_variable", target_column)  # Logging the target variable name

        # Train a Lasso model for feature selection
        lasso = LassoCV(alphas=alpha_values, cv=5)
        lasso.fit(X_train, y_train)
        
        # Identify the selected features
        # selected_features = X.columns[lasso.coef_ != 0].tolist()
        
        # Rank features based on the absolute values of their coefficients
        lasso_coef = lasso.coef_
        ranked_features = [x for _, x in sorted(zip(np.abs(lasso_coef), feature_names), reverse=True) if _ != 0]

        
        # Train a linear regression model using the selected features
        lr = LinearRegression().fit(X_train[ranked_features], y_train)
        
        # Predict and calculate metrics for the linear regression model
        y_pred = lr.predict(X_test[ranked_features])
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Log metrics to MLflow
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MAE", mae)
        
        # Log model
        mlflow.sklearn.log_model(lr, "linear_regression_model")
        
        # Log selected features
        # mlflow.log_text("\n".join(selected_features), "selected_features.txt")
        # Save the selected features to a local text file and log it
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
            temp.write("\n".join(ranked_features).encode())
            temp_path = temp.name
        mlflow.log_artifact(temp_path, "selected_features.txt")
        
    return ranked_features, lr


def elastic_net_feature_selection_mlflow(data, target_column, 
                                         alpha_values=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10], 
                                         l1_ratio_values=[.1, .5, .7, .9, .95, .99, 1]):
    """
    Perform feature selection using Elastic Net regression.
    Train a linear regression model on the selected features.
    Log the results to MLflow.
    """
    # Split data into train and test sets
    X = data.drop(columns=[target_column])
    y = data[target_column]
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start MLflow run
    with mlflow.start_run():
        # Set run name and tags
        mlflow.set_tag("mlflow.runName", "Elastic Feature Selection")
        mlflow.set_tag("method", "Elastic")
        mlflow.set_tag("target_variable", target_column)  # Logging the target variable name
        
        # Train an Elastic Net model for feature selection
        elastic_net = ElasticNetCV(alphas=alpha_values, l1_ratio=l1_ratio_values, cv=5)
        elastic_net.fit(X_train, y_train)
        
        # Select features based on the Elastic Net model (non-zero coefficients)
        selected_features_mask = elastic_net.coef_ != 0
        selected_features = np.array(feature_names)[selected_features_mask].tolist()
        
        # Rank the selected features based on the absolute values of their coefficients
        elastic_coef = elastic_net.coef_[selected_features_mask]
        ranked_features = [feature for _, feature in sorted(zip(np.abs(elastic_coef), selected_features), reverse=True)]
        
        # Train a linear regression model using the selected features
        lr = LinearRegression().fit(X_train[ranked_features], y_train)
        
        # Predict and calculate metrics for the linear regression model
        y_pred = lr.predict(X_test[ranked_features])
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Log metrics to MLflow
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MAE", mae)
        
        # Log model
        mlflow.sklearn.log_model(lr, "linear_regression_model")
        
        # Log selected features
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
            temp.write("\n".join(ranked_features))
            temp_path = temp.name
        mlflow.log_artifact(temp_path, "selected_features.txt")
        
    return ranked_features, lr

# Usage example (assuming 'selected_data' is your DataFrame):
# ranked_features, lr_model = elastic_net_feature_selection_mlflow(selected_data, 'TargetColumn')


# def elastic_net_feature_selection_mlflow(data, target_column, alpha_values=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10], l1_ratio_values=[.1, .5, .7, .9, .95, .99, 1]):
#     """
#     Perform feature selection using Elastic Net regression.
#     Train a linear regression model on the selected features.
#     Log the results to MLflow.
#     """
#     # Split data into train and test sets
#     X = data.drop(columns=[target_column])
#     y = data[target_column]
#     feature_names = X.columns.tolist()
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Start MLflow run
#     with mlflow.start_run():
#         # Set run name and tags
#         mlflow.set_tag("mlflow.runName", "Elastic Feature Selection")
#         mlflow.set_tag("method", "Elastic")
#         mlflow.set_tag("target_variable", target_column)  # Logging the target variable name
#         # Train an Elastic Net model for feature selection
#         elastic_net = ElasticNetCV(alphas=alpha_values, l1_ratio=l1_ratio_values, cv=5)
#         elastic_net.fit(X_train, y_train)
        
#         # Identify the selected features
#         # selected_features = X.columns[elastic_net.coef_ != 0].tolist()
        
#         # Rank features based on the absolute values of their coefficients
#         elastic_coef = elastic_net.coef_
#         ranked_features = [x for _, x in sorted(zip(np.abs(elastic_coef), feature_names), reverse=True) if _ != 0]
        
#         # Train a linear regression model using the selected features
#         lr = LinearRegression().fit(X_train[ranked_features], y_train)
        
#         # Predict and calculate metrics for the linear regression model
#         y_pred = lr.predict(X_test[ranked_features])
#         mse = mean_squared_error(y_test, y_pred)
#         rmse = mean_squared_error(y_test, y_pred, squared=False)
#         r2 = r2_score(y_test, y_pred)
#         mae = mean_absolute_error(y_test, y_pred)
        
#         # Log metrics to MLflow
#         mlflow.log_metric("MSE", mse)
#         mlflow.log_metric("RMSE", rmse)
#         mlflow.log_metric("R2", r2)
#         mlflow.log_metric("MAE", mae)
        
#         # Log model
#         mlflow.sklearn.log_model(lr, "linear_regression_model")
        
#         # Log selected features
#         # Save the selected features to a local text file and log it
#         with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
#             temp.write("\n".join(ranked_features).encode())
#             temp_path = temp.name
#         mlflow.log_artifact(temp_path, "selected_features.txt")
        
#     return ranked_features, lr


def rfe_feature_selection_mlflow(data, target_column, n_features_to_select=10):
    """
    Perform feature selection using Recursive Feature Elimination (RFE).
    Train a linear regression model on the selected features.
    Log the results to MLflow.
    """
    # Split data into train and test sets
    X = data.drop(columns=[target_column])
    y = data[target_column]
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start MLflow run
    with mlflow.start_run():
        # Set run name and tags
        mlflow.set_tag("mlflow.runName", "RFE Feature Selection")
        mlflow.set_tag("method", "RFE")
        mlflow.set_tag("target_variable", target_column)  # Logging the target variable name
        # Initialize a base estimator for RFE
        estimator = LinearRegression()
        
        # Initialize and fit RFE
        selector = RFE(estimator, n_features_to_select=n_features_to_select)
        selector.fit(X_train, y_train)
        
        # Identify the selected features
        # selected_features = X.columns[selector.support_].tolist()
        # Rank features based on the RFE ranking_
        ranked_features = [x for _, x in sorted(zip(selector.ranking_, feature_names))]

        
        # Train a linear regression model using the selected features
        lr = LinearRegression().fit(X_train[ranked_features], y_train)
        
        # Predict and calculate metrics for the linear regression model
        y_pred = lr.predict(X_test[ranked_features])
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Log metrics to MLflow
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MAE", mae)
        
        # Log model
        mlflow.sklearn.log_model(lr, "linear_regression_model")
        
        # Log selected features
        # Save the selected features to a local text file and log it
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
            temp.write("\n".join(ranked_features).encode())
            temp_path = temp.name
        mlflow.log_artifact(temp_path, "selected_features.txt")
        
    return ranked_features, lr

    
def hist_gradient_boosting_permutation_importance_mlflow(data, target_column):
    """
    Perform feature selection using Histogram-based Gradient Boosting Regression 
    with permutation importance.
    Train a linear regression model on the selected features.
    Log the results to MLflow.
    """
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Training the histogram-based gradient boosting regressor
    hgb_regressor = HistGradientBoostingRegressor().fit(X_train, y_train)
    
    # Calculating permutation importances
    result = permutation_importance(hgb_regressor, X_test, y_test, n_repeats=10, random_state=42)
    importances = result.importances_mean

    # Selecting the top features based on their importances
    threshold = 0.005  # This can be adjusted based on your needs
    # selected_features = X.columns[importances > threshold].tolist()
    
    # Rank features based on their importances
    ranked_features = [x for _, x in sorted(zip(importances, feature_names), reverse=True) if _ > threshold]


    # Linear Regression on selected features
    lr = LinearRegression().fit(X_train[ranked_features], y_train)
    y_pred = lr.predict(X_test[ranked_features])
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", "HistGradientBoosting with Permutation Importance")
        mlflow.set_tag("method", "HistGradientBoosting + Permutation Importance")
        mlflow.set_tag("target_variable", target_column)  # Logging the target variable name
        # Logging the threshold parameter
        mlflow.log_param("threshold", threshold)
        
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MAE", mae)

        # Log model
        mlflow.sklearn.log_model(lr, "linear_regression_model")

        # Log selected features
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
            temp.write("\n".join(ranked_features).encode())
            temp_path = temp.name
        mlflow.log_artifact(temp_path, "selected_features.txt")
        
    return ranked_features, lr


def gradient_boosting_feature_selection_mlflow(data, target_column, threshold):
    """
    Perform feature selection using Gradient Boosting Regression.
    Train a linear regression model on the selected features.
    Log the results to MLflow.
    """
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Training the gradient boosting regressor
    gb_regressor = GradientBoostingRegressor().fit(X_train, y_train)
    
    # Extracting feature importances
    importances = gb_regressor.feature_importances_
    
    # Selecting the top features based on their importances
    threshold = threshold  # This can be adjusted based on your needs
    # selected_features = X.columns[importances > threshold].tolist()
    
    # Rank features based on their importances
    ranked_features = [x for _, x in sorted(zip(importances, feature_names), reverse=True) if _ > threshold]

    # Linear Regression on selected features
    lr = LinearRegression().fit(X_train[ranked_features], y_train)
    y_pred = lr.predict(X_test[ranked_features])
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", "Gradient Boosting Feature Selection")
        mlflow.set_tag("method", "Gradient Boosting")
        mlflow.set_tag("target_variable", target_column)  # Logging the target variable name
        # Logging the threshold parameter
        mlflow.log_param("threshold", threshold)

        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MAE", mae)

        # Log model
        mlflow.sklearn.log_model(lr, "linear_regression_model")

        # Log selected features
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
            temp.write("\n".join(ranked_features).encode())
            temp_path = temp.name
        mlflow.log_artifact(temp_path, "selected_features.txt")
        
    return ranked_features, lr

# Feature selection compare
def compare_selected_features(**feature_lists):
    """
    Compare lists of selected features from different methods.
    
    Parameters:
    - **feature_lists: Arbitrary number of named lists of features to compare.
      Each named list corresponds to a feature selection method.
      
    Returns:
    - A summary dictionary containing intersections, differences, and unions.
    """
    
    methods = list(feature_lists.keys())
    summary = {}

    # Pairwise comparisons
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            method_1 = methods[i]
            method_2 = methods[j]
            features_1 = set(feature_lists[method_1])
            features_2 = set(feature_lists[method_2])

            intersection = features_1.intersection(features_2)
            difference_1_2 = features_1.difference(features_2)
            difference_2_1 = features_2.difference(features_1)
            union = features_1.union(features_2)

            comparison_name = f"{method_1}_vs_{method_2}"
            summary[comparison_name] = {
                "intersection": intersection,
                f"{method_1}_not_in_{method_2}": difference_1_2,
                f"{method_2}_not_in_{method_1}": difference_2_1,
                "union": union
            }

    return summary
def ensemble_feature_selection(feature_sets):
    """
    Ensemble method for feature selection based on multiple feature selection methods.
    
    Parameters:
    - feature_sets: Dictionary where keys are method names and values are lists of selected features
    
    Returns:
    - ranked_features: Features ranked by their occurrence across methods
    - feature_matrix: Binary matrix indicating feature selection by method
    """
    
    # Count the number of times each feature appears across all methods
    all_features = [feature for feature_list in feature_sets.values() for feature in feature_list]
    feature_counts = pd.Series(all_features).value_counts()
    ranked_features = feature_counts.index.tolist()
    
    # Create a binary matrix
    feature_matrix = pd.DataFrame(0, index=ranked_features, columns=feature_sets.keys())
    for method, features in feature_sets.items():
        feature_matrix.loc[features, method] = 1
    
    return ranked_features, feature_matrix

# Regression functions
import statsmodels.api as sm
def best_interaction_terms_with_quadratic_fixed(X, y, max_interactions=5, QuadraticTerm=[]):
    """
    Finds the best interaction terms for a linear regression model based on the adjusted R-squared value.
    Can also introduce quadratic terms for specified variables.
    
    Parameters:
    - X: Predictor dataframe.
    - y: Response variable.
    - max_interactions: Maximum number of interaction terms to consider.
    - QuadraticTerm: List of variables for which quadratic terms should be added.
    
    Returns:
    - Model summary with the best interaction terms and specified quadratic terms.
    """
    # Copy the dataframe to avoid modifying the original
    X = X.copy()
    
    # Initial model with no interactions or quadratic terms
    X_with_const = sm.add_constant(X)
    base_model = sm.OLS(y, X_with_const).fit()
    best_adj_r2 = base_model.rsquared_adj
    best_interactions = []
    
    # Add the quadratic terms
    for var in QuadraticTerm:
        X[f"{var}_squared"] = X[var] ** 2
    
    # Loop through pairs of predictors to test interactions
    for i, col1 in enumerate(X.columns):
        for j, col2 in enumerate(X.columns[i + 1:]):
            X_temp = X.copy()
            interaction_term = f"{col1}_{col2}"
            X_temp[interaction_term] = X_temp[col1] * X_temp[col2]
            
            # Fit the model with the interaction term
            X_temp_with_const = sm.add_constant(X_temp)
            model = sm.OLS(y, X_temp_with_const).fit()
            
            # Check if this model is better than the previous best
            if model.rsquared_adj > best_adj_r2:
                best_adj_r2 = model.rsquared_adj
                best_interactions = [(col1, col2)]
            elif model.rsquared_adj == best_adj_r2:
                best_interactions.append((col1, col2))
            
            # If the maximum number of interactions is reached, break
            if len(best_interactions) == max_interactions:
                break
    
    # Use the best interaction terms to fit the final model
    for col1, col2 in best_interactions:
        interaction_term = f"{col1}_{col2}"
        X[interaction_term] = X[col1] * X[col2]
        
    X_with_const = sm.add_constant(X)
    final_model = sm.OLS(y, X_with_const).fit()
    
    return final_model



# Pipline Classes 
from sklearn.base import BaseEstimator, TransformerMixin

class MICEImputer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_columns, nominal_columns, ordinal_columns):
        self.numerical_columns = numerical_columns
        self.nominal_columns = nominal_columns
        self.ordinal_columns = ordinal_columns
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return perform_mice_imputation(X, self.numerical_columns, self.nominal_columns, self.ordinal_columns)


# Data Processing Transformer
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_scale=None, columns_to_log=None):
        self.columns_to_scale = columns_to_scale or []
        self.columns_to_log = columns_to_log or []

    def fit(self, X, y=None):
        # Nothing to do in fit for now
        return self

    def transform(self, X):
        X_processed = X.copy()

        # Check for NaN values before transformations
        if X_processed.isnull().values.any():
            print("NaN values detected. Please handle them before transformations.")
            # Handle NaNs here, e.g., fill with median
            for col in X_processed.columns:
                if X_processed[col].isnull().any():
                    X_processed[col].fillna(X_processed[col].median(), inplace=True)

        # Check for zeros or negative values in columns to be log-transformed
        for col in self.columns_to_log:
            if (X_processed[col] <= 0).any():
                print(f"Non-positive values detected in {col} for log transformation. Adding a small constant.")
                X_processed[col] += 1e-9  # Adjust this constant as necessary

        # Log-transforming
        for col in self.columns_to_log:
            if col in X_processed.columns:
                X_processed[col] = np.log1p(X_processed[col])
            else:
                print(f"Column {col} not found in DataFrame for log-transforming.")

        # Scaling
        for col in self.columns_to_scale:
            if col in X_processed.columns:
                X_processed[col] = (X_processed[col] - X_processed[col].mean()) / X_processed[col].std()
            else:
                print(f"Column {col} not found in DataFrame for scaling.")

        return X_processed

# Example usage:
# df is your DataFrame containing the data
# processor = DataProcessor(columns_to_scale=['your_column_to_scale'], columns_to_log=['your_column_to_log'])
# df_ready_for_modeling = processor.fit_transform(df)


# class DataProcessor(BaseEstimator, TransformerMixin):
#     def __init__(self, columns_to_scale=None, columns_to_log=None):
#         self.columns_to_scale = columns_to_scale or []
#         self.columns_to_log = columns_to_log or []
        
#     def fit(self, X, y=None):
#         # In this case, nothing to do in fit, but it could be expanded for other operations.
#         return self
    
#     def transform(self, X):
#         X_processed = X.copy()
        
#         # Scaling
#         for col in self.columns_to_scale:
#             X_processed[col] = (X_processed[col] - X_processed[col].mean()) / X_processed[col].std()
        
#         # Log-transforming (using log1p for numerical stability)
#         for col in self.columns_to_log:
#             X_processed[col] = np.log1p(X_processed[col])
        
#        return X_processed

class ElasticNetSelector(BaseEstimator, TransformerMixin):
    def __init__(self, target_column):
        self.target_column = target_column
        self.selected_features_ = None

    def fit(self, X, y=None):
        # We only use the ranked features returned by the function.
        # The function is assumed to use default alpha_values and l1_ratio_values.
        ranked_features, _ = elastic_net_feature_selection_mlflow(X, self.target_column)
        self.selected_features_ = ranked_features
        return self

    def transform(self, X):
        # Return the dataset with only the selected features
        return X.loc[:, self.selected_features_]
    


class InteractionQuadraticRegressor(BaseEstimator, TransformerMixin):
    def __init__(self, max_interactions=5, QuadraticTerm=[]):
        self.max_interactions = max_interactions
        self.QuadraticTerm = QuadraticTerm
        
    def fit(self, X, y):
        self.model = best_interaction_terms_with_quadratic_fixed(X, y, self.max_interactions, self.QuadraticTerm)
        return self

    def transform(self, X):
        # Return the predictions
        return self.model.predict(X)
    
    


def extract_coefficients(model, feature_names):
    """
    Extract coefficients from a trained linear regression model and format them for publication.
    
    Parameters:
    - model: Trained linear regression model
    - feature_names: List of feature names
    
    Returns:
    - formatted_coefficients: DataFrame containing coefficients in a format suitable for publication
    """
    
    # Extract coefficients and intercept from the model
    coefficients = model.coef_
    intercept = model.intercept_
    
    # Create a DataFrame
    coef_df = pd.DataFrame({
        'Feature': feature_names + ['Intercept'],
        'Coefficient': list(coefficients) + [intercept]
    })
    
    return coef_df

# Example usage:
# lasso_selected_features, lasso_model = lasso_feature_selection_mlflow(df, 'target_column')
# coef_table = extract_coefficients(lasso_model, lasso_selected_features)
# print(coef_table)

import statsmodels.formula.api as smf
def fit_and_evaluate_model(formula, data, re_formula):
    model = smf.mixedlm(formula, data, groups=data['Age_Gender'], re_formula=re_formula).fit(method='lbfgs', maxiter=1000000, reml=False)
    return model

def fit_and_evaluate_regression_model(formula, data):
    # Fit an OLS regression model with the provided formula
    model = smf.ols(formula, data).fit()
    return model

def create_effect_size_table(models, variables):
    data = []
    for var in variables:
        row = [var]
        for model in models:
            try:
                coef = model.params[var]
                p_value = model.pvalues[var]
                row.extend([coef, p_value])
            except KeyError:
                row.extend([None, None])
        data.append(row)

    columns = ['Variable']
    for i, model in enumerate(models):
        columns.extend([f'Model_{i+1}_Coef', f'Model_{i+1}_PValue'])

    return pd.DataFrame(data, columns=columns)

#
def compare_models_table(models):
    data = []
    for i, model in enumerate(models):
        row = [f'Model_{i+1}', model.aic, model.bic, model.llf]
        data.append(row)

    return pd.DataFrame(data, columns=['Model', 'AIC', 'BIC', 'Log-Likelihood'])
#
import matplotlib.pyplot as plt
import seaborn as sns

def plot_coefficients(models, variables):
    for var in variables:
        coefs = []
        yerr_lower = []
        yerr_upper = []

        for model in models:
            if var in model.params:
                coef = model.params[var]
                ci_lower, ci_upper = model.conf_int().loc[var]
                coefs.append(coef)
                yerr_lower.append(coef - ci_lower)
                yerr_upper.append(ci_upper - coef)
            else:
                coefs.append(None)
                yerr_lower.append(None)
                yerr_upper.append(None)

        fig, ax = plt.subplots()
        ax.errorbar(x=list(range(len(models))), y=coefs, yerr=[yerr_lower, yerr_upper], fmt='o')
        ax.set_xticks(list(range(len(models))))
        ax.set_xticklabels([f'Model_{i+1}' for i in range(len(models))])
        ax.set_title(f'Coefficients of {var}')
        ax.set_xlabel('Model')
        ax.set_ylabel('Coefficient')
        plt.show()

def plot_model_fit(models):
    aics = [model.aic for model in models]
    bics = [model.bic for model in models]

    fig, ax = plt.subplots()
    ax.plot(aics, label='AIC')
    ax.plot(bics, label='BIC')
    ax.set_xticks(list(range(len(models))))
    ax.set_xticklabels([f'Model_{i+1}' for i in range(len(models))])
    ax.set_title('Model Fit Statistics')
    ax.set_xlabel('Model')
    ax.set_ylabel('Value')
    ax.legend()
    plt.show()
#
def plot_interaction_effects(model, variable, interaction_terms):
    for term in interaction_terms:
        var_name = f'{variable}:{term}'
        if var_name in model.params:
            coef = model.params[var_name]
            ci = model.conf_int().loc[var_name].tolist()
            plt.errorbar(x=[term], y=[coef], yerr=[[ci[0]-coef], [ci[1]-coef]], fmt='o', label=var_name)
    
    plt.title(f'Interaction Effects of {variable}')
    plt.xlabel('Terms')
    plt.ylabel('Coefficient')
    plt.legend()
    plt.show()
    
import statsmodels.api as sm
def diagnostic_plots(model_result, figsize=(12, 8)):
    """
    Generate diagnostic plots for a fitted mixed model.

    Parameters:
    model_result -- The result object from a fitted mixed model in statsmodels
    figsize -- Tuple indicating the size of the figures to be plotted
    """
    # Residuals vs Fitted
    plt.figure(figsize=figsize)
    fitted_vals = model_result.fittedvalues
    residuals = model_result.resid
    sns.residplot(x=fitted_vals, y=residuals, lowess=True)
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    plt.show()

    # Q-Q plot
    plt.figure(figsize=figsize)
    sm.qqplot(residuals, line='s')
    plt.title('Q-Q plot of residuals')
    plt.show()

    # Scale-Location Plot (also known as Spread-Location Plot)
    plt.figure(figsize=figsize)
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(residuals))
    sns.regplot(x=fitted_vals, y=model_norm_residuals_abs_sqrt, lowess=True)
    plt.xlabel('Fitted values')
    plt.ylabel('√|Standardized residuals|')
    plt.title('Scale-Location')
    plt.show()
