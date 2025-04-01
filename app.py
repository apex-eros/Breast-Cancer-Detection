from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

app = Flask(__name__, template_folder=os.path.abspath("templates"))

# Load dataset
url = 'C:/Users/Admin/Desktop/JAYESH/Projects/Breast Cancer/dataset/data.csv'
df = pd.read_csv(url)

# Preprocess data
df = df.drop(['Unnamed: 32', 'id'], axis=1)
df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

y = df[['diagnosis']]
x = df.drop('diagnosis', axis=1)

# Feature selection
k = 5
selector = SelectKBest(score_func=f_classif, k=k)
X_selected = selector.fit_transform(x, y.values.ravel())

sc = StandardScaler()
X_scaled = sc.fit_transform(X_selected)

# Convert selected features back to DataFrame
selected_features = x.columns[selector.get_support()]
X_selected_df = pd.DataFrame(X_scaled, columns=selected_features)

X_train, X_test, y_train, y_test = train_test_split(X_selected_df, y, test_size=0.2, random_state=42, stratify=y)

# Handling imbalanced data
smote = SMOTE(random_state=40)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train.values.ravel())

# Train Decision Tree Classifier
dt_params = {"criterion": ["gini", "entropy"]}
dt = DecisionTreeClassifier(random_state=42)
dt_random = RandomizedSearchCV(dt, dt_params, n_iter=10, cv=5, scoring="accuracy", random_state=42)
dt_random.fit(X_train_resampled, y_train_resampled)

@app.route('/')
def loadPage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def cancerPrediction():
    try:
        # Receive JSON data
        data = request.json
        new_data = [[
            float(data['perimeter_mean']),
            float(data['concave_points_mean']),
            float(data['radius_worst']),
            float(data['perimeter_worst']),
            float(data['concave_points_worst'])
        ]]
        
        new_df = pd.DataFrame(new_data, columns=selected_features)
        new_df_scaled = sc.transform(new_df)

        # Make a prediction
        prediction = dt_random.best_estimator_.predict(new_df_scaled)[0]
        proba = dt_random.best_estimator_.predict_proba(new_df_scaled)[:, 1][0]

        output1 = "The patient is diagnosed with Breast Cancer" if prediction == 1 else "The patient is not diagnosed with Breast Cancer"
        output2 = f": {proba * 100:.2f}%" if prediction == 1 else ": Low"

        return jsonify({"prediction": output1, "probability": output2})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
