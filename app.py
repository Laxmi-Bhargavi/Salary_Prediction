from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect input from form
    age = int(request.form['age'])
    gender = request.form['gender']
    education = request.form['education']
    job = request.form['job']
    experience = float(request.form['experience'])

    # Create input DataFrame
    input_df = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Education Level': education,
        'Job Title': job,
        'Years of Experience': experience
    }])

    # Predict
    salary = model.predict(input_df)[0]
    return render_template('index.html', predicted_salary=f"Estimated Salary: ₹{salary:,.2f}")

if __name__ == '__main__':
    app.run(debug=True)