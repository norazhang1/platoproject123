from flask import Flask, render_template, request

try:
    from transformers import pipeline
    emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
except ModuleNotFoundError:
    emotion_classifier = None

app = Flask(__name__)

advice_dict = {
    "joy": "Keep doing what brings you happiness.",
    "sadness": "Take deep breaths. It's okay to feel down.",
    "anger": "Try going for a walk to cool off.",
    "fear": "Ground yourself with your senses.",
    "love": "Express it! Do something kind.",
    "surprise": "Is it exciting or uncertain? Reflect on it.",
    "neutral": "You seem balanced. Recharge calmly."
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    user_input = request.form['text']
    if emotion_classifier is None:
        return render_template('result.html', emotion="error", score=0.0, text=user_input,
                               advice="Model not loaded. Check installation.")
    result = emotion_classifier(user_input)[0]
    emotion = result['label'].lower()
    score = round(result['score'], 2)
    advice = advice_dict.get(emotion, "Take a moment to reflect.")
    return render_template('result.html', emotion=emotion, score=score, text=user_input, advice=advice)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=81, debug=True)
