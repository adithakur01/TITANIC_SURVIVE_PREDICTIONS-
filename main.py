import tkinter as tk
from tkinter import messagebox
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load Titanic dataset (replace my dataset path with your dataset path)
data = pd.read_csv(r"c:\Users\Aditya\Downloads\train (1).csv")

# Features (X) and target variable (y)
X = data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = data['Survived']

# Convert categorical variables to numerical using .loc
X.loc[:, 'Sex'] = X['Sex'].map({'male': 0, 'female': 1})
X.loc[:, 'Embarked'] = X['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Handle missing values
X = X.dropna()
y = y.loc[X.index]

# Ensure X and y have the same number of samples

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # we split data into 80,20 percent 

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()

# Train the model
dt_classifier.fit(X_train, y_train)

# Evaluate the model on the training set
train_predictions = dt_classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)

# Evaluate the model on the testing set
test_predictions = dt_classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)

# Print model evaluation metrics
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("\nClassification Report for Testing Set:\n", classification_report(y_test, test_predictions))

# Function to predict survival and display result in GUI
def predict_survival():
    try:
        pclass = int(entry_pclass.get())
        sex = 1 if entry_sex.get().lower() == 'female' else 0
        age = float(entry_age.get())
        fare = float(entry_fare.get())
        embarked = entry_embarked.get().upper()

        # Make prediction
        prediction = dt_classifier.predict([[pclass, sex, age, fare, embarked]])

        # Display result
        if prediction[0] == 1:
            result_label.config(text="Survived")
        else:
            result_label.config(text="Not Survived")

    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter valid values.")

# GUI
root = tk.Tk()
root.title("Titanic Survival Prediction")

# Labels and Entry widgets
label_pclass = tk.Label(root, text="Pclass:", font=('Arial', 14, 'bold'))
entry_pclass = tk.Entry(root, font=('Arial', 14))

label_sex = tk.Label(root, text="Sex (male/female):", font=('Arial', 14, 'bold'))
entry_sex = tk.Entry(root, font=('Arial', 14))

label_age = tk.Label(root, text="Age:", font=('Arial', 14, 'bold'))
entry_age = tk.Entry(root, font=('Arial', 14))

label_fare = tk.Label(root, text="Fare:", font=('Arial', 14, 'bold'))
entry_fare = tk.Entry(root, font=('Arial', 14))

label_embarked = tk.Label(root, text="Embarked (S=0/C=1/Q=2):", font=('Arial', 14, 'bold'))
entry_embarked = tk.Entry(root, font=('Arial', 14))

result_label = tk.Label(root, text="")

# Button for prediction
predict_button = tk.Button(root, text="Predict Survival", command=predict_survival)

# Arrange widgets in the grid
label_pclass.grid(row=0, column=0)
entry_pclass.grid(row=0, column=1)

label_sex.grid(row=1, column=0)
entry_sex.grid(row=1, column=1)

label_age.grid(row=2, column=0)
entry_age.grid(row=2, column=1)

label_fare.grid(row=3, column=0)
entry_fare.grid(row=3, column=1)

label_embarked.grid(row=4, column=0)
entry_embarked.grid(row=4, column=1)

predict_button.grid(row=5, column=0, columnspan=2)

result_label.grid(row=6, column=0, columnspan=2)

# Start the GUI
root.mainloop()
