from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (only runs once)
nltk.download('stopwords')

app = Flask(__name__)

# ✅ Load models and vectorizer
nb_model = joblib.load('models/naive_bayes_model.pkl')
lr_model = joblib.load('models/logistic_regression_model.pkl')
rf_model = joblib.load('models/random_forest_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# ✅ Text cleaning function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

# ✅ ROUTES

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    name = request.form.get('name')
    email = request.form.get('email')
    subject = request.form.get('subject')
    message = request.form.get('message')

    # You can save this data or send an email if needed

    return render_template('contact_successfull.html', name=name)


# 🔹 Spam Detection form page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        cleaned_message = clean_text(message)
        data = vectorizer.transform([cleaned_message])

        # Run all models
        prediction_nb = nb_model.predict(data)
        prediction_lr = lr_model.predict(data)
        prediction_rf = rf_model.predict(data)

        result = {
            'Naive Bayes': 'Spam' if prediction_nb[0] == 1 else 'Not Spam',
            'Logistic Regression': 'Spam' if prediction_lr[0] == 1 else 'Not Spam',
            'Random Forest': 'Spam' if prediction_rf[0] == 1 else 'Not Spam'
        }

        return render_template('result.html', result=result, original=message)
    
    return render_template('predict_form.html')

if __name__ == '__main__':
    app.run(debug=True)
