import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

 Load the Titanic dataset
df = pd.read_csv('train.csv')

 Display basic information
print("Initial data shape:", df.shape)
print(df.head())

 1. Data Cleaning
 Check for missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

 Fill missing Age values with the median
df['Age'].fillna(df['Age'].median(), inplace=True)

 Fill missing Embarked values with the mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

 Drop the Cabin column due to many missing values
df.drop('Cabin', axis=1, inplace=True)

 Verify missing values
print("\nMissing values after cleaning:")
print(df.isnull().sum())

 2. Handle Noisy Data - Binning Age
bins = [0, 12, 18, 35, 60, 100]
labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

 Display Age and AgeGroup
print(df[['Age', 'AgeGroup']].head())

 3. Data Integration - Simulate integration with Gender data
gender_data = df[['PassengerId', 'Sex']].copy()
gender_data.columns = ['PassengerId', 'Gender']

 Drop original column and re-integrate
df.drop('Sex', axis=1, inplace=True)
df = df.merge(gender_data, on='PassengerId', how='left')

 Display integrated data
print(df[['PassengerId', 'Gender']].head())

 4. Visualization (Optional)
sns.countplot(x='AgeGroup', hue='Survived', data=df)
plt.title('Survival Count by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Survived')
plt.tight_layout()
plt.savefig('survival_by_agegroup.png')  # Save figure
plt.show()

 5. Save cleaned data
df.to_csv('titanic_cleaned.csv', index=False)
print("\nCleaned data saved to titanic_cleaned.csv")
