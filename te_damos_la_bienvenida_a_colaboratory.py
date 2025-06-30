
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#read and writing
df = pd.read_csv('Breast_Cancer.csv')

#exploring data

# First 5 rows
df.head(5)

# Last 5 rows
df.tail()

# Basic statistics
df.describe()

# Data types
df.dtypes

# Shape (rows, columns)
df.shape

# Column names
df.columns



"""# Task
Tell me about this dataset.

Here is all the data you need:
"Breast_Cancer.csv"

## Data loading

### Subtask:
Load the "Breast_Cancer.csv" dataset into a pandas DataFrame.

**Reasoning**:
Load the "Breast_Cancer.csv" dataset into a pandas DataFrame and display the first few rows to verify.
"""

import pandas as pd

try:
    df = pd.read_csv('Breast_Cancer.csv')
    display(df.head())
except FileNotFoundError:
    print("Error: 'Breast_Cancer.csv' not found. Please ensure the file exists in the current directory or provide the correct path.")
    df = None  # Set df to None to indicate failure
except pd.errors.ParserError:
    print("Error: Unable to parse the CSV file. Please check the file format.")
    df = None
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    df = None

"""## Data exploration

### Subtask:
Explore the loaded dataset to understand its characteristics.

**Reasoning**:
Explore the loaded dataset by checking its shape, data types, descriptive statistics, missing values, and identifying the target variable.
"""

print(df.shape())

# 2. Examine the data types of each column
print("\nData types of each column:")
print(df.info())

# 3. Generate descriptive statistics for numerical features
print("\nDescriptive statistics for numerical features:")
display(df.describe())

# 4. Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 5. Identify the target variable
print("\nTarget variable: Status")

# 6. Identify irrelevant or redundant columns (at first glance)
print("\nPotentially irrelevant/redundant columns at first glance: None obvious initially.")

"""## Data visualization

### Subtask:
Visualize the data distributions and relationships between features.

**Reasoning**:
Create histograms for the numerical features to visualize their distributions.
"""

import matplotlib.pyplot as plt
import seaborn as sns

numerical_features = df.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

"""**Reasoning**:
Create bar charts for categorical features to show the frequency of different categories.


"""

import matplotlib.pyplot as plt
import seaborn as sns

categorical_features = df.select_dtypes(include=['object']).columns

plt.figure(figsize=(20, 25))

for i, col in enumerate(categorical_features):
    plt.subplot(5, 3, i + 1)
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

plt.tight_layout()
plt.show()

"""**Reasoning**:
Generate a correlation matrix heatmap for numerical features to visualize their relationships.


"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Import pandas as it's used for select_dtypes

plt.figure(figsize=(12, 10))

# Select only numerical features for the correlation matrix
numerical_df = df.select_dtypes(include=['number'])

# Calculate the correlation matrix on the numerical DataFrame
correlation_matrix = numerical_df.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

"""**Reasoning**:
The previous attempt to calculate the correlation matrix failed due to the presence of non-numeric data.  We need to select only the numeric columns for the correlation calculation.


"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Select only numerical features for the correlation matrix
numerical_df = df.select_dtypes(include=['number'])

# Calculate the correlation matrix
correlation_matrix = numerical_df.corr()

# Create the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap (Numerical Features)')
plt.show()

"""## Data analysis

### Subtask:
Analyze the distribution of the target variable and explore potential relationships between features and the target variable.

**Reasoning**:
Analyze the distribution of the target variable 'Status' and visualize it. Then, explore the relationship between numerical features and the target variable using descriptive statistics and boxplots.  Finally, examine the relationship between categorical features and the target variable using frequency counts and visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.stats import chi2_contingency

# 1. Analyze the distribution of the target variable 'Status'
print(df['Status'].value_counts())
plt.figure(figsize=(6, 4))
sns.countplot(x='Status', data=df)
plt.title('Distribution of Status')
plt.show()

# 2. Numerical features vs. target variable
numerical_features = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x='Status', y=feature, data=df)
    plt.title(f'{feature} vs. Status')

    # Perform statistical test (t-test or Mann-Whitney U test)
    group1 = df[df['Status'] == 'Alive'][feature]
    group2 = df[df['Status'] == 'Dead'][feature]
    if len(group1) > 0 and len(group2) > 0:
        if len(group1) >= 30 and len(group2) >= 30:
            t_statistic, p_value = ttest_ind(group1, group2)
        else:
            t_statistic, p_value = mannwhitneyu(group1, group2)
        print(f"Statistical test ({feature}):")
        print(f"  t-statistic: {t_statistic}")
        print(f"  p-value: {p_value}")

plt.tight_layout()
plt.show()

# 3. Categorical features vs. target variable
categorical_features = ['Race', 'Marital Status', 'T Stage ', 'N Stage', '6th Stage', 'differentiate', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status']

plt.figure(figsize=(20, 25))
for i, feature in enumerate(categorical_features):
    plt.subplot(5, 2, i + 1)
    observed = pd.crosstab(df[feature], df['Status'])
    chi2, p, dof, expected = chi2_contingency(observed)
    sns.countplot(x=feature, hue='Status', data=df)
    plt.title(f'{feature} vs. Status (Chi2 p={p:.3f})')
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

"""## Summary:

### Q&A
There were no explicit questions asked in the task. However, the analysis aimed to understand the dataset, identify potential relationships between features and the target variable ('Status'), and explore the distribution of the target variable. The analysis successfully addressed these implicit questions.

### Data Analysis Key Findings
* The target variable "Status" is imbalanced, with significantly more "Alive" cases (3408) than "Dead" cases (616).
* Statistically significant differences were observed between "Alive" and "Dead" groups for all numerical features (Age, Tumor Size, Regional Node Examined, Reginol Node Positive, and Survival Months). Larger tumor size, more positive regional nodes, and shorter survival months were associated with death.
* Visualizations and Chi-squared tests were performed for categorical features (Race, Marital Status, T Stage, N Stage, 6th Stage, differentiate, Grade, A Stage, Estrogen Status, and Progesterone Status) to assess relationships with the target variable.  The significance of these relationships is determined by examining the p-values from the Chi-squared tests.


### Insights or Next Steps
* Investigate the statistically significant relationships between categorical features and the target variable further using the Chi-squared test results.
* Consider more advanced modeling techniques to account for the class imbalance in the target variable and to potentially uncover more complex relationships between features and patient survival.

Investigate the statistically significant relationships between categorical features and the target variable further using the Chi-squared test results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import pandas as pd # Import pandas for crosstab

# 1. Analyze the distribution of the target variable 'Status'
print(df['Status'].value_counts())
plt.figure(figsize=(6, 4))
sns.countplot(x='Status', data=df)
plt.title('Distribution of Status')
plt.show()

# 2. Numerical features vs. target variable (Keeping the existing code for context, although not the focus of this request)
# ... (existing code for numerical features vs. target variable)

# 3. Categorical features vs. target variable
categorical_features = ['Race', 'Marital Status', 'T Stage ', 'N Stage', '6th Stage', 'differentiate', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status']

print("\nStatistical tests (Chi-squared) for Categorical Features vs. Status:")
for feature in categorical_features:
    print(f"\nAnalyzing relationship between '{feature}' and 'Status':")
    # Create a contingency table
    observed = pd.crosstab(df[feature], df['Status'])

    # Perform the Chi-squared test
    chi2, p, dof, expected = chi2_contingency(observed)

    print(f"  Chi-squared Statistic: {chi2:.4f}")
    print(f"  P-value: {p:.4f}")
    print(f"  Degrees of Freedom: {dof}")
    print(f"  Contingency Table:\n{observed}")

    # Visualize the relationship with a countplot
    plt.figure(figsize=(8, 6)) # Adjust figure size for better readability
    sns.countplot(x=feature, hue='Status', data=df)
    plt.title(f'{feature} vs. Status (Chi2 p={p:.4f})') # Display more precise p-value in title
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

"""Consider more advanced modeling techniques to account for the class imbalance in the target variable and to potentially uncover more complex relationships between features and patient survival.


```
# Tiene formato de código
```


"""

# Install necessary libraries
#!pip install scikit-learn imbalanced-learn

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Assuming 'df' DataFrame is already loaded from the previous steps

# Separate features (X) and target variable (y)
X = df.drop('Status', axis=1)
y = df['Status']

# Convert target variable to numeric for modeling (0 for Alive, 1 for Dead)
y = y.apply(lambda x: 1 if x == 'Dead' else 0)

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler() # Scale numerical features
categorical_transformer = OneHotEncoder(handle_unknown='ignore') # One-hot encode categorical features

# Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Stratify to maintain class distribution

# --- Modeling with SMOTE and different classifiers ---

# 1. Logistic Regression with SMOTE
print("--- Logistic Regression with SMOTE ---")
# Create a pipeline with SMOTE and Logistic Regression
log_reg_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)), # Apply SMOTE on the training data
    ('classifier', LogisticRegression(solver='liblinear', random_state=42)) # Use liblinear solver for small datasets
])

# Train the model
log_reg_pipeline.fit(X_train, y_train)

# Make predictions
y_pred_lr = log_reg_pipeline.predict(X_test)
y_prob_lr = log_reg_pipeline.predict_proba(X_test)[:, 1] # Probabilities for the positive class

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Alive', 'Dead']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

print(f"\nAUC-ROC Score: {roc_auc_score(y_test, y_prob_lr):.4f}")

# 2. Random Forest with SMOTE
print("\n--- Random Forest with SMOTE ---")
# Create a pipeline with SMOTE and Random Forest
rf_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)), # Apply SMOTE on the training data
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')) # Use class_weight='balanced' as an alternative/addition to SMOTE
])

# Train the model
rf_pipeline.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1] # Probabilities for the positive class

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Alive', 'Dead']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print(f"\nAUC-ROC Score: {roc_auc_score(y_test, y_prob_rf):.4f}")

# --- Further Steps ---
print("\n--- Next Steps Consideration ---")
print("*  Explore other resampling techniques (e.g., NearMiss, ADASYN).")
print("*  Experiment with different models (e.g., Gradient Boosting, SVMs).")
print("*  Perform hyperparameter tuning for the chosen models.")
print("*  Consider feature engineering based on the analysis (e.g., combining stages).")
print("*  If time-to-event data is relevant, explore survival analysis models like Cox Proportional Hazards.")

"""Consider feature engineering based on the analysis (e.g., combining stages)"""

# Assuming 'df' DataFrame is already loaded

# --- Feature Engineering Example: Combining Stages ---

# Create a new feature by combining 'T Stage ' and 'N Stage'
# This is a simple concatenation; a more sophisticated approach might be needed based on domain knowledge
df['TN_Stage'] = df['T Stage '] + '_' + df['N Stage']

# You can also create interaction terms, e.g., by multiplying or combining categories
# For example, a hypothetical interaction:
# df['Tumor_Size_by_Grade'] = df['Tumor Size'] * df['Grade'].astype('category').cat.codes # Example, requires Grade to be numeric or encoded

print("\nDataFrame with new 'TN_Stage' feature:")
print(df[['T Stage ', 'N Stage', 'TN_Stage', 'Status']].head())

# Now, when you proceed to modeling, include 'TN_Stage' in your features
# You'll need to update the identification of categorical features accordingly

# Example of updating categorical features list for preprocessing
# Make sure 'TN_Stage' is added and the original 'T Stage ' and 'N Stage' are optionally removed if you only want the combined feature
categorical_features_engineered = [col for col in df.select_dtypes(include=['object']).columns if col not in ['Status', 'T Stage ', 'N Stage']] + ['TN_Stage']
numerical_features_engineered = df.select_dtypes(include=['int64', 'float64']).columns.tolist() # Keep original numerical features

# Separate features (X) and target variable (y) using the updated features
X_engineered = df[numerical_features_engineered + categorical_features_engineered]
y_engineered = df['Status'].apply(lambda x: 1 if x == 'Dead' else 0)

# --- Proceed with Modeling using the engineered features ---
# You would then use X_engineered and y_engineered in your train_test_split and subsequent modeling steps
# (The rest of the modeling code from the previous example would follow here, using the new dataframes and feature lists)

# Example of how the ColumnTransformer would be updated:
# preprocessor_engineered = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numerical_features_engineered),
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_engineered)
#     ])

# X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(X_engineered, y_engineered, test_size=0.2, random_state=42, stratify=y_engineered)

# Now use X_train_eng, X_test_eng, etc. in your pipelines

"""*  Explore other resampling techniques (e.g., NearMiss, ADASYN).
*  Experiment with different models (e.g., Gradient Boosting, SVMs).
*  Perform hyperparameter tuning for the chosen models.
"""



# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import Pipeline as ImbPipeline

# Assuming 'df' DataFrame is already loaded from the previous steps

# Separate features (X) and target variable (y)
X = df.drop('Status', axis=1)
y = df['Status']

# Convert target variable to numeric for modeling (0 for Alive, 1 for Dead)
y = y.apply(lambda x: 1 if x == 'Dead' else 0)

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler() # Scale numerical features
categorical_transformer = OneHotEncoder(handle_unknown='ignore') # One-hot encode categorical features

# Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Stratify to maintain class distribution

# --- Experimenting with Different Resampling Techniques and Models ---

# Define different resampling techniques and models
resamplers = {
    'None': None, # No resampling
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    # 'NearMiss': NearMiss() # NearMiss can be computationally intensive and requires careful parameter tuning
}

classifiers = {
    'LogisticRegression': LogisticRegression(solver='liblinear', random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42) # probability=True is needed for roc_auc_score
}

results = {}

print("--- Exploring Resampling Techniques and Models ---")

for resampler_name, resampler in resamplers.items():
    for clf_name, classifier in classifiers.items():
        print(f"\nRunning with Resampler: {resampler_name}, Classifier: {clf_name}")

        if resampler is None:
            # Create a standard pipeline if no resampling is used
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', classifier)
            ])
        else:
            # Create an imbalanced-learn pipeline for resampling
            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('resampler', resampler),
                ('classifier', classifier)
            ])

        try:
            # Train the model
            pipeline.fit(X_train, y_train)

            # Make predictions
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            # Evaluate the model
            report = classification_report(y_test, y_pred, target_names=['Alive', 'Dead'], output_dict=True)
            auc_roc = roc_auc_score(y_test, y_prob)
            conf_matrix = confusion_matrix(y_test, y_pred)

            results[(resampler_name, clf_name)] = {
                'classification_report': report,
                'auc_roc': auc_roc,
                'confusion_matrix': conf_matrix
            }

            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Alive', 'Dead']))
            print("\nConfusion Matrix:")
            print(conf_matrix)
            print(f"\nAUC-ROC Score: {auc_roc:.4f}")

        except Exception as e:
            print(f"An error occurred during training or evaluation: {e}")
            results[(resampler_name, clf_name)] = {'error': str(e)}


# --- Example of Hyperparameter Tuning with GridSearchCV (using RandomForest and SMOTE) ---
print("\n--- Hyperparameter Tuning Example (RandomForest with SMOTE) ---")

# Define the pipeline for tuning
# We are tuning the classifier part of the pipeline
pipeline_for_tuning = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define the parameter grid to search
# The parameter names in the grid need to match the step name ('classifier') followed by '__' and the parameter name
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
# We use 'roc_auc' as the scoring metric for imbalanced data
# cv=5 means 5-fold cross-validation
grid_search = GridSearchCV(pipeline_for_tuning, param_grid, cv=5, scoring='roc_auc', n_jobs=-1) # n_jobs=-1 uses all available cores

# Perform the grid search on the training data
print("Performing Grid Search...")
try:
    grid_search.fit(X_train, y_train)

    print("\nBest parameters found:")
    print(grid_search.best_params_)

    print("\nBest AUC-ROC score on validation sets:")
    print(f"{grid_search.best_score_:.4f}")

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_pred_tuned = best_model.predict(X_test)
    y_prob_tuned = best_model.predict_proba(X_test)[:, 1]

    print("\nEvaluation of the best tuned model on the test set:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_tuned, target_names=['Alive', 'Dead']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_tuned))
    print(f"\nAUC-ROC Score: {roc_auc_score(y_test, y_prob_tuned):.4f}")

except Exception as e:
     print(f"An error occurred during Grid Search: {e}")
     print("Check the parameter grid and data compatibility.")

# --- Summary of Results (Optional) ---
print("\n--- Summary of Model Performance (AUC-ROC) ---")
for (resampler, clf), metrics in results.items():
    if 'error' not in metrics:
        print(f"Resampler: {resampler}, Classifier: {clf} - AUC-ROC: {metrics['auc_roc']:.4f}")
    else:
         print(f"Resampler: {resampler}, Classifier: {clf} - Error: {metrics['error']}")

print("\n--- Next Steps ---")
print("*  Analyze the performance metrics (especially AUC-ROC, Recall for 'Dead' class) for each combination.")
print("*  Select the best performing models and consider their interpretability.")
print("*  Further refine hyperparameter tuning for promising models.")
print("*  Consider feature importance analysis for tree-based models.")
print("*  Investigate potential feature engineering based on model insights.")

