import json
import tensorflow as tf
import pickle
from conversation_management import ConversationManager

# Initialise your model and data
model = tf.keras.models.load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

conversation_manager = ConversationManager(model, intents, words, classes)

session_id = "user123"
while True:
    message = input("You: ")
    if message.lower() == "quit":
        break
    
    response = conversation_manager.generate_response(session_id, message)
    print("Bot:", response)