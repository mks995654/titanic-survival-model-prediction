import numpy as np    #To handle numerical data and arrays
import pandas as pd   #To organize data in table format and perform data manipulation
import matplotlib.pyplot as plt #To create visualizations and plots
from sklearn.datasets import load_iris #To load the Iris dataset, a commonly used dataset in machine learning for classification tasks
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder # To scale data and encode categorical variables
from sklearn.decomposition import PCA #Dimentions reduction technique
from sklearn.neighbors import KNeighborsClassifier # Nearest neighbor classification algorithm
from sklearn.pipeline import Pipeline
import seaborn as sns # To create graphs
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier # ML algorithm that uses multiple decision trees to make predictions
from sklearn.linear_model import LogisticRegression # ML algorithm used for classification tasks that models the relationship between a dependent variable and one or more independent variables using a logistic function
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay # matrix to compare predicted and actual values, and a report to evaluate the performance of a classification model

# Load data
titanic = sns.load_dataset('titanic')
titanic.head()

# Count each feature in dataset and identify how many missing values there are

titanic.count()

# Now drop some of the features that have a lot of missing values and are not relevant to our analysis

# From data set selected features and target variable (survived is target)
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'who', 'adult_male', 'alone']
target = 'survived'

x= titanic[features]
y= titanic[target]

# Now need to check how balanced is data base means how many people survived and how many did not survive.
# This is important because if the data is imbalanced, it can affect the performance of machine learning models.
# If one class is significantly more represented than the other, the model may have difficulty learning to predict the minority class accurately.

y.value_counts()   # From data only 38% of passengers survived, which indicates that the dataset is imbalanced. 

# Here in train test split we will use stratify to ensure that the proportion of classes in the target variable is maintained in both the training and testing sets.
# This is important when dealing with imbalanced datasets, as it helps to ensure that the model is trained on a representative sample of the data and can generalize well to unseen data.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# In data set we have both numerical and categorical features, so we need to preprocess them separately.

numerical_features = x_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = x_train.select_dtypes(include=['category' ,'object', 'string']).columns.tolist()


# separate preprocessing pipelines for both feature types

numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values by replacing them with the median value of each numerical feature
    ('scaler', StandardScaler())  # Scale numerical features to have a mean of 0 and a standard deviation of 1, which can improve the performance of many machine learning algorithms
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values by replacing them with the most frequent value of each categorical feature
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical features using one-hot encoding, which creates binary columns for each category and allows the model to interpret categorical data effectively
])

# Combine the numerical and categorical pipelines into a single preprocessor using ColumnTransformer, which applies the appropriate transformations to each type of feature
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Create a pipeline that combines the preprocessor with a machine learning model, in this case, a Random Forest Classifier, which is an ensemble learning method that uses multiple decision trees to make predictions and can handle both numerical and categorical data effectively
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])
param_grid = {
    # 'classifier__n_estimators': [50, 100],
    # 'classifier__max_depth': [None, 10, 20],
    # 'classifier__min_samples_split': [2, 5],
    'classifier__solver' : ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight' : [None, 'balanced']
}
# perform grid search to find the best hyperparameters for the Logistic Regression model using GridSearchCV, which exhaustively searches over specified parameter values for an estimator and uses cross-validation to evaluate the performance of each combination of parameters
model = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
model.fit(x_train, y_train)
model.param_grid = param_grid

y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred))

# Display confusion matrix to visualize the performance of the classification model by showing the true positive, true negative, false positive, and false negative predictions
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Titanic Classification Confusion Matrix')
plt.tight_layout()
plt.show()  

# Extract the logistic regression feature coefficients and plot their magnitude in a bar chart.

coefficients = model.best_estimator_.named_steps['classifier'].coef_[0]

# combine numerical and categorical feature names
model.best_estimator_['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_features) # Get the names of the one-hot encoded features for categorical variables
#feature_importances = model.best_estimator_['classifier'].feature_importances_ # Get the feature importances from the trained Random Forest model
feature_names = numerical_features + list(model.best_estimator_['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_features)) # Combine numerical feature names with one-hot encoded categorical feature names

# Create a DataFrame for the coefficients
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False, key=abs)  # Sort by absolute values

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Coefficient'].abs(), color='skyblue')
plt.gca().invert_yaxis()
plt.title('Feature Coefficient magnitudes for Logistic Regression model')
plt.xlabel('Coefficient Magnitude')
plt.show()

# Print test score
test_score = model.best_estimator_.score(x_test, y_test)
print(f"\nTest set accuracy: {test_score:.2%}")

