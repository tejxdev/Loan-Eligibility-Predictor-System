from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('loan_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = np.array([data])
    result = model.predict(final_input)[0]
    output = 'Eligible' if result == 1 else 'Not Eligible'
    return render_template('index.html', prediction_text=f'Loan Status: {output}')

if __name__ == "__main__":
    app.run(debug=True)
