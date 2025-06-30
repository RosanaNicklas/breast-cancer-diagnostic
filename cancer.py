import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


#read and writing
df = pd.read_csv('Breast_Cancer.csv')

#exploring data

# First 5 rows
df.head(5)
"""
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


# selecting columns
df['Status']  # Single column
df[['Status', 'Age']]  # Multiple columns

# selecting rows
df.iloc[0]  # First row by index position
df.loc[0]  # First row by label (default is same as position

# Slicing
df[0:3]  # First 3 rows

# Condicional
# Example: Get rows where 'Survival Months' > 50
filtered_df = df[df['Survival Months'] > 50]

filtered_df = df[(df['Survival Months'] > 50) & (df['Status'] == 'Alive')]

survival_data = df['Survival Months']  # Returns a Series
survival_data = df[['Survival Months']]  # Returns a DataFrame

# Histogram of Survival Months
df['Survival Months'].hist(bins=20)
plt.xlabel('Survival Months')
plt.ylabel('Frequency')
plt.title('Distribution of Survival Months')
plt.show()

sns.boxplot(x=df['Survival Months'])
plt.show()

df[(df['Age'] > 25) & (df['Marital Status'] == 'Divorced')]  # Multiple conditions

#df.rename(columns={'old_name': 'new_name'}, inplace=True)

#df.drop('column_name', axis=1, inplace=True)




df.plot()  # Basic line plot
df['Age'].hist()  # Histogram
df.plot.scatter(x='Age', y='Status')  # Scatter plot"""