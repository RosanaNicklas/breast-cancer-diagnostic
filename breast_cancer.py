# Standard library
import pickle
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Define resampling strategies
resampling = {
    'None': None,
    'SMOTE': SMOTE(random_state=42),
    'UnderSample': RandomUnderSampler(random_state=42),
    'SMOTE+Under': ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('under', RandomUnderSampler(random_state=42))
    ])
}
# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, roc_auc_score, 
                            confusion_matrix, precision_recall_curve, 
                            average_precision_score, f1_score, roc_curve, auc)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, auc


# Load the dataset
df = pd.read_csv('Breast_Cancer.csv')

# ======================
# Advanced Feature Engineering
# ======================

# 1. Create combined stage feature
df['Combined_Stage'] = df['T Stage '] + '_' + df['N Stage'] + '_' + df['6th Stage']

# 2. Create interaction terms
df['Size_Grade_Interaction'] = df['Tumor Size'] * df['Grade'].map({'I':1, 'II':2, 'III':3, 'IV':4})
df['Node_Ratio'] = df['Reginol Node Positive'] / (df['Regional Node Examined'] + 1)  # +1 to avoid division by zero

# 3. Create age bins
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 40, 50, 60, 70, 100], 
                        labels=['<40', '40-50', '50-60', '60-70', '70+'])

# 4. Create hormone status combination
df['Hormone_Status'] = df['Estrogen Status'] + '_' + df['Progesterone Status']

# 5. Create survival risk indicator (months < median)
median_survival = df['Survival Months'].median()
df['Short_Survival_Risk'] = (df['Survival Months'] < median_survival).astype(int)

# ======================
# Data Preparation
# ======================

# Convert target variable
y = df['Status'].map({'Dead':1, 'Alive':0})
X = df.drop('Status', axis=1)

# Identify feature types
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Custom log transform for numerical features
def log_transform(X):
    return np.log1p(X)

# Preprocessing pipelines
# Preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('log', FunctionTransformer(log_transform)),  # Log transform for numerical features
            ('scale', StandardScaler())
        ]), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# ======================
# Model Training with Advanced Techniques
# ======================

# After train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Calculate class weights
positive_count = sum(y_train)
negative_count = len(y_train) - positive_count
scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1

# Define models
models = {
    'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        random_state=42,
        use_label_encoder=False
    ),
    'LightGBM': LGBMClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1  # Suppresses LightGBM warnings
    ),
    'SVM': SVC(
        kernel='rbf',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
}

# Define resampling strategies
resampling = {
    'None': None,
    'SMOTE': SMOTE(random_state=42),
    'UnderSample': RandomUnderSampler(random_state=42),
    'SMOTE+Under': ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('under', RandomUnderSampler(random_state=42))
    ])
}

# Then proceed with your grid search
for model_name, model in models.items():
    for resample_name, resampler in resampling.items():  # Now this will work
        # Your training code here

# 3. Update parameter grids


# Define resampling strategies
# Replace the problematic pipeline with:
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

resampling = ImbPipeline([
    ('smote', SMOTE(sampling_strategy=0.5, random_state=42)),
    ('under', RandomUnderSampler(sampling_strategy=0.8, random_state=42))
])
# Grid search parameters
param_grids = {
    'RandomForest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    },
    'XGBoost': {
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [3, 6],
        'classifier__n_estimators': [100, 200]
    },
    'LightGBM': {
        'classifier__num_leaves': [31, 63],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__n_estimators': [100, 200]
    },
    'SVM': {
        'classifier__base_estimator__C': [0.1, 1, 10],
        'classifier__base_estimator__gamma': ['scale', 'auto']
    }
}

# Store results
results = []
best_score = 0
best_model = None

# Train and evaluate models
# Train and evaluate models
for model_name, model in models.items():
    for resample_name, resampler in resampling.items():  # Now using the dictionary
        print(f"\nTraining {model_name} with {resample_name} resampling...")
        
        try:
            # Create appropriate pipeline
            if resampler is None:
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
            else:
                pipeline = ImbPipeline([
                    ('preprocessor', preprocessor),
                    ('resampler', resampler),
                    ('classifier', model)
                ])
            
            # Get the correct parameter grid
            current_param_grid = {
                k.replace('classifier__', ''): v 
                for k, v in param_grids.get(model_name, {}).items()
            }
            
            # Configure and run GridSearchCV
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=current_param_grid,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            # ... rest of your evaluation code ...
            
        except Exception as e:
            print(f"Error training {model_name} with {resample_name}: {str(e)}")
            continue
            # Get best model and predictions
            best_estimator = grid_search.best_estimator_
            y_pred = best_estimator.predict(X_test)
            y_prob = best_estimator.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_roc = roc_auc_score(y_test, y_prob)
            ap_score = average_precision_score(y_test, y_prob)
            f1 = f1_score(y_test, y_pred)
            
            # Store results
            result = {
                'Model': model_name,
                'Resampling': resample_name,
                'Best_Params': grid_search.best_params_,
                'AUC_ROC': auc_roc,
                'Average_Precision': ap_score,
                'F1_Score': f1,
                'Confusion_Matrix': confusion_matrix(y_test, y_pred),
                'Classification_Report': classification_report(y_test, y_pred, output_dict=True)
            }
            results.append(result)
            
            print(f"Best AUC-ROC: {auc_roc:.4f}")
            print(f"Best params: {grid_search.best_params_}")
            
            # Update best model
            if auc_roc > best_score:
                best_score = auc_roc
                best_model = best_estimator
                print("New best model found!")
                
        except Exception as e:
            print(f"Error training {model_name} with {resample_name}: {str(e)}")

# ======================
# Save the Best Model
# ======================

if best_model is not None:
    # Save with pickle
    with open('best_breast_cancer_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    # Also save with joblib (more efficient for large models)
    joblib.dump(best_model, 'best_breast_cancer_model.joblib')
    
    print(f"\nBest model saved with AUC-ROC: {best_score:.4f}")
    
    # Print best model details
    print("\nBest model details:")
    print(f"Model type: {type(best_model.steps[-1][1]).__name__}")
    print(f"Resampling: {best_model.steps[-2][0] if len(best_model.steps) > 2 else 'None'}")
    print("Parameters:", best_model.get_params())
else:
    print("No model was successfully trained.")

# ======================
# Results Analysis
# ======================

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Sort by AUC-ROC
results_df = results_df.sort_values('AUC_ROC', ascending=False)

# Print top 5 models
print("\nTop 5 models:")
print(results_df[['Model', 'Resampling', 'AUC_ROC', 'F1_Score']].head())


# Ensure your ColumnTransformer preserves feature names
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    verbose_feature_names_out=False  # Add this
)

# Then get feature names after transformation
preprocessor.fit(X_train)
feature_names = preprocessor.get_feature_names_out()

# Plot feature importance for tree-based models
if hasattr(best_model.steps[-1][1], 'feature_importances_'):
    try:
        # Get feature names
        num_features = numerical_features
        cat_features = best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_features = np.concatenate([num_features, cat_features])
        
        # Get importances
        importances = best_model.steps[-1][1].feature_importances_
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'Feature': all_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', 
                   data=feature_importance.head(20))
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.show()
        
        # Save feature importance
        feature_importance.to_csv('feature_importances.csv', index=False)
    except Exception as e:
        print(f"Could not plot feature importance: {str(e)}")

# ======================
# Final Evaluation
# ======================

if best_model is not None:
    # Detailed evaluation on test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    print("\nFinal Evaluation on Test Set:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Alive', 'Dead'], 
                yticklabels=['Alive', 'Dead'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()