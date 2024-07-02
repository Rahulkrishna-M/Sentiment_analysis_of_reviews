from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
# Load your trained model
model = joblib.load('sentiment_analysis_model.pkl')

@app.route('/')
def home():
    return "Sentiment Analysis Model API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    review = data['review']
    prediction = model.predict([review])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
