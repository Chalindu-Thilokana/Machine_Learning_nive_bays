from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Dataset
data = {
    'Age': ['<=30', '<=30', '31-40', '>40', '>40', '>40', '31-40', '<=30', '<=30', '>40', '<=30', '31-40', '31-40', '>40'],
    'Income': ['High', 'High', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 'Low', 'Medium', 'Medium', 'Medium', 'High', 'Medium'],
    'Student': ['No', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No'],
    'Credit_rating': ['Fair', 'Excellent', 'Fair', 'Fair', 'Fair', 'Excellent', 'Excellent', 'Fair', 'Fair', 'Fair', 'Excellent', 'Excellent', 'Fair', 'Excellent'],
    'Buys_computer': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Encode all columns and store encoders
encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop('Buys_computer', axis=1)
y = df['Buys_computer']

model = CategoricalNB()
model.fit(X, y)

# Input function with validation
def get_valid_input(prompt, options):
    options_lower = [o.lower() for o in options]
    while True:
        val = input(prompt).strip()
        if val.lower() in options_lower:
            # Return the original case value
            return options[options_lower.index(val.lower())]
        else:
            print(f"❌ Invalid input! Please enter one of {options}")

print("\n💻 Please enter details to predict whether the user will buy a computer:")

age = get_valid_input("Age (<=30 / 31-40 / >40): ", ['<=30', '31-40', '>40'])
income = get_valid_input("Income (Low / Medium / High): ", ['Low', 'Medium', 'High'])
student = get_valid_input("Student? (Yes / No): ", ['Yes', 'No'])
credit = get_valid_input("Credit Rating (Fair / Excellent): ", ['Fair', 'Excellent'])

test_sample = pd.DataFrame([[age, income, student, credit]], columns=['Age', 'Income', 'Student', 'Credit_rating'])

for col in test_sample.columns:
    test_sample[col] = encoders[col].transform(test_sample[col])

prediction = model.predict(test_sample)
result = encoders['Buys_computer'].inverse_transform(prediction)[0]

print(f"\n🧠 Prediction: Will the user buy a computer? ➜ {result}")
