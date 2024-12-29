import joblib

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

print("Movie Review Sentiment Chatbot")
print("Type 'exit' to quit.")

while True:
    user_input = input("Enter your movie review: ") 
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    vectorized_input = vectorizer.transform([user_input])
    sentiment = model.predict(vectorized_input)[0]
    print("Sentiment:", "Positive" if sentiment == 1 else "Negative")
