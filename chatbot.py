import json
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Load dataset
with open("intents.json") as f:
    data = json.load(f)

patterns = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

# 2. Text → Numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)  # features
y = np.array(tags)  # labels

# 3. Train model
model = MultinomialNB()
model.fit(X, y)

# 4. Chat function
def chatbot_response(text):
    X_test = vectorizer.transform([text])
    tag = model.predict(X_test)[0]

    # pick random response from that tag
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

# 5. Test
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Bot: Goodbye!")
        break
    print("Bot:", chatbot_response(user_input))
