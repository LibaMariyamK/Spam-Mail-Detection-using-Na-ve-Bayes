from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset and train the model
df = pd.read_csv('mail_data.csv')
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(df['Message'])
y_train = df['Category']

# Train Multinomial Naïve Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save the trained model and vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']

    # Load the model and vectorizer
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)

    # Transform input message
    message_vec = vectorizer.transform([message])

    # Make prediction
    prediction = model.predict(message_vec)[0]

    return render_template('index.html', message=message, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
