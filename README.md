# Learning ML/Python Chatbot 🤖

This is a learning project designed to understand the basics of machine learning and natural language processing using Python. It implements a simple chatbot that can be trained to recognise intents and carry out basic conversations.

## 🧠 What It Does

The project consists of two main components:

1. **Training Module (`train.py`)**
  - Processes training data from `intents.json`
  - Uses a neural network to learn patterns
  - Creates a model that can recognise user intents
  - Implements Bag of Words (BoW) approach for text processing

2. **Chatbot Module (`chatbot.py`)**
  - Uses the trained model to understand user input
  - Manages conversation flow
  - Extracts entities (like amounts, names)
  - Provides contextual responses

The chatbot can handle:
- Basic conversations
- Money transfers
- Balance enquiries
- Loan enquiries
- Multi-step conversations
- Context maintenance

## 📋 Requirements

- Python 3.8+
- TensorFlow (macOS optimized)
- NLTK (Natural Language Toolkit)
- NumPy

## 🛠 Environment Setup (macOS)

1. **Create and Activate Conda Environment**
  ```bash
  # Create new environment
  conda create -n tensorflow python=3.8

  # Activate environment
  conda activate tensorflow
  ```

2. **Install TensorFlow Dependencies**
  ```bash
  # Install base dependencies
  conda install tensorflow-deps
  
  # Install macOS optimized TensorFlow
  pip install tensorflow-macos
  
  # Install Metal acceleration (Apple Silicon)
  pip install tensorflow-metal
  ```
3. **Install Required Packages**
  ```bash
  # Install NLTK
  pip install nltk
  
  # Install NumPy
  pip install numpy
  ```

## 🎯 Training the Model
1. **Prepare Training Data**
  - Review intents.json
  - Add/modify intents as needed
  - Format:
    ```json
    jsonCopy{
      "intents": [
        {
          "tag": "greeting",
          "patterns": ["hi", "hello"],
          "responses": ["Hello!", "Hi there!"]
        }
      ]
    }
    ```
2. **Run Training**
  ```bash 
  python train.py
  ```
  This will:
  
  - Process the intents
  - Train the neural network
  - Save the model as 'chatbot_model.h5'
  - Save vocabulary as 'words.pkl'
  - Save classes as 'classes.pkl'

## 🤖 Running the Chatbot

1. **Start the Chatbot**
  ```bash
  python chatbot.py
  ```

2. **Interact with the Bot**
  - Type your messages
  - Bot will respond based on trained intents
  - Type "quit" to exit
