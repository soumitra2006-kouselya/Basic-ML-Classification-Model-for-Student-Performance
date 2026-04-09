# STEP 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# STEP 2: Load dataset
data = pd.read_csv("dataset.csv")

# STEP 3: Separate input (features) and output (target)
X = data[["hours", "attendance"]]   # Features
y = data["pass"]                    # Target

# STEP 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# STEP 5: Create the model
model = LogisticRegression()

# STEP 6: Train the model
model.fit(X_train, y_train)

# STEP 7: Make predictions on test data
y_pred = model.predict(X_test)

# STEP 8: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# STEP 9: Test with new input (FIXED VERSION)
# Creating input with proper column names
sample = pd.DataFrame([[5, 70]], columns=["hours", "attendance"])

prediction = model.predict(sample)

# STEP 10: Display result
print("\nPrediction for [5 hrs, 70% attendance]:", "Pass" if prediction[0] == 1 else "Fail")