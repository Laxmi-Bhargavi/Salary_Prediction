import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv('dataset.csv')

# Drop rows where target (Salary) is missing
df = df.dropna(subset=['Salary'])

# Features and target
X = df[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']]
y = df['Salary']

# Preview and check for missing values
print(df.head())
print("Missing values:\n", df.isnull().sum())

# Define categorical and numerical columns
categorical = ['Gender', 'Education Level', 'Job Title']
numerical = ['Age', 'Years of Experience']

# Preprocessor to handle missing data and encode categoricals
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical),
        ('num', SimpleImputer(strategy='mean'), numerical)
    ]
)

# Full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train model
model_pipeline.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

print("✅ Model trained and saved as model.pkl")