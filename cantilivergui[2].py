import tkinter as tk
from tkinter import ttk
import numpy as np
import joblib

# Load your trained model and scaler
model = joblib.load('logistic_regression_model2.pkl')
scaler = joblib.load('standard_scaler2.pkl')

# Function to make predictions in the GUI
def make_prediction():
    try:
        # Collect input values from GUI
        credit_score = float(credit_score_entry.get())
        age = float(age_entry.get())
        tenure = float(tenure_entry.get())
        balance = float(balance_entry.get())
        num_of_products = float(num_of_products_entry.get())
        gender = gender_var.get()
        has_credit_card = credit_card_var.get()
        is_active_member = active_member_var.get()
        estimated_salary = float(salary_entry.get())

        # Encode categorical variables
        gender = 1 if gender == "Female" else 0
        has_credit_card = 1 if has_credit_card == "Yes" else 0
        is_active_member = 1 if is_active_member == "Yes" else 0

        # Create input array
        data = np.array([[credit_score, age, tenure, balance, num_of_products, estimated_salary, 
                          gender, has_credit_card, is_active_member]])
        
        # Scale numerical values
        data = scaler.transform(data)

        # Make prediction using your model
        prediction = model.predict(data)
        result = "Leave" if prediction[0] > 0.5 else "Stay"
        prediction_label.config(text="Prediction: " + result)
    except Exception as e:
        prediction_label.config(text="Error: " + str(e))

# Create GUI window
root = tk.Tk()
root.title("Bank Customer Churn Prediction")

# Create labels and entry fields for input
ttk.Label(root, text="Credit Score:").grid(row=0, column=0, padx=5, pady=5)
credit_score_entry = ttk.Entry(root)
credit_score_entry.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(root, text="Age:").grid(row=1, column=0, padx=5, pady=5)
age_entry = ttk.Entry(root)
age_entry.grid(row=1, column=1, padx=5, pady=5)

ttk.Label(root, text="Tenure:").grid(row=2, column=0, padx=5, pady=5)
tenure_entry = ttk.Entry(root)
tenure_entry.grid(row=2, column=1, padx=5, pady=5)

ttk.Label(root, text="Balance:").grid(row=3, column=0, padx=5, pady=5)
balance_entry = ttk.Entry(root)
balance_entry.grid(row=3, column=1, padx=5, pady=5)

ttk.Label(root, text="Number of Products:").grid(row=4, column=0, padx=5, pady=5)
num_of_products_entry = ttk.Entry(root)
num_of_products_entry.grid(row=4, column=1, padx=5, pady=5)

ttk.Label(root, text="Gender:").grid(row=5, column=0, padx=5, pady=5)
gender_var = tk.StringVar()
gender_combobox = ttk.Combobox(root, textvariable=gender_var, values=["Male", "Female"])
gender_combobox.grid(row=5, column=1, padx=5, pady=5)
gender_combobox.current(1)

ttk.Label(root, text="Has Credit Card:").grid(row=6, column=0, padx=5, pady=5)
credit_card_var = tk.StringVar()
credit_card_combobox = ttk.Combobox(root, textvariable=credit_card_var, values=["Yes", "No"])
credit_card_combobox.grid(row=6, column=1, padx=5, pady=5)
credit_card_combobox.current(0)

ttk.Label(root, text="Active Member:").grid(row=7, column=0, padx=5, pady=5)
active_member_var = tk.StringVar()
active_member_combobox = ttk.Combobox(root, textvariable=active_member_var, values=["Yes", "No"])
active_member_combobox.grid(row=7, column=1, padx=5, pady=5)
active_member_combobox.current(0)

ttk.Label(root, text="Estimated Salary:").grid(row=8, column=0, padx=5, pady=5)
salary_entry = ttk.Entry(root)
salary_entry.grid(row=8, column=1, padx=5, pady=5)

# Create button to trigger prediction
predict_button = ttk.Button(root, text="Predict", command=make_prediction)
predict_button.grid(row=9, column=0, columnspan=2, padx=5, pady=5)

# Create label to display prediction
prediction_label = ttk.Label(root, text="")
prediction_label.grid(row=10, column=0, columnspan=2, padx=5, pady=5)

# Run GUI loop
root.mainloop()
