from flask import Flask, request, jsonify, render_template
import nltk
import random
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')  # This should already be done, but let's try again
nltk.download('punkt_tab')
nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)

# Load the chatbot text data
with open(r"D:\chatbot\chatbot\chatbot1.txt", 'r', errors='ignore') as f:
    raw = f.read().lower()

# Download necessary NLTK packages
nltk.download('punkt')
nltk.download('wordnet')

# Tokenize the data
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

# Lemmatizer for normalizing words
lemmer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


# Remove punctuation
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Define greeting responses
GREETING_INPUTS = ["hello", "hi", "greetings", "sup", "what's up", "hey"]
GREETING_RESPONSES = ["hi", "hey", "nods", "hi there", "hello", "I am glad! You are talking to me"]


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Chatbot response logic
def response(user_response):
    robo_response = ' '
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = sent_tokens[idx]
        return robo_response


# Flask route to serve the homepage (chatbot interface)
@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML page for chatbot


# Flask route to process user input and return chatbot response
@app.route('/get', methods=['POST'])
def chat():
    user_response = request.json.get("user_input", "").lower()
    print(f"Received user input: {user_response}")  # Log the input to the console

    if user_response:
        if user_response in ['Hi', 'Hello']:
            return jsonify({"response": "Chatbot: Hello,How can I help you!"})
        elif user_response in ['bye', 'exit']:
            return jsonify({"response": "Chatbot: Bye! Thanks!!!!"})
        elif user_response in ['thanks', 'thank you']:
            return jsonify({"response": "Chatbot: You are welcome.."})
        elif greeting(user_response) is not None:
            return jsonify({"response": f"Chatbot: {greeting(user_response)}"})
        else:
            bot_reply = response(user_response)
            return jsonify({"response": f"Chatbot: {bot_reply}"})
    else:
        return jsonify({"response": "Please provide input to chat with the bot."})


if __name__ == '__main__':
    app.run(debug=True)
