import numpy as np
import json
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import random
import nltk
from nltk.stem import WordNetLemmatizer
import pickle

# --- 📔 NLP Setup ---
# Download required NLTK (Natural Language Toolkit) components for text processing

nltk.download('punkt')  # Tokenizer data
# 🧠 How it works:
#   - A pre-trained model that divides text into sentences and words
#   - Uses unsupervised learning to recognize sentence boundaries
#   - Handles abbreviations (e.g., "Dr.", "Mr.") intelligently
#   - Can identify when periods are part of abbreviations vs end of sentences

nltk.download('punkt_tab')  # Additional tokenizer data
# 🧠 How it works:
#   - Contains language-specific abbreviation lists
#   - Includes special cases for different languages
#   - Helps improve tokenization accuracy for specific contexts
#   - Supplements the main Punkt tokenizer with extra rules

nltk.download('wordnet')  # Required for lemmatization
# 🧠 How it works:
#   - Used by the WordNetLemmatizer
#   - Contains relationships between words:
#     - Synonyms (similar meanings)
#     - Hypernyms (more general terms)
#     - Hyponyms (more specific terms)
#   - Helps reduce words to their base form

# Initialise WordNet lemmatizer for reducing words to base form
lemmatizer = WordNetLemmatizer()

# --- 📔 Data Loading and Initial Processing ---
# Load training intents from JSON file
# Expected format: {"intents": [{"tag": "greeting", "patterns": ["hi", "hello"], "responses": [...]}]}
with open('intents.json') as file:
    intents = json.load(file)

# Initialise data structures for NLP processing
words = []          # Will store all unique words from patterns
classes = []        # Will store all unique intent tags
documents = []      # Will store (pattern_word_list, intent_tag) tuples
ignore_chars = ['?', '!', '.', ',']  # Characters to filter out

# --- 📔 Text Processing Pipeline ---
# Process each intent and its patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenization: Convert pattern into list of words
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Store the word list and its associated intent tag
        documents.append((word_list, intent['tag']))
        # Track unique intent tags
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Text Normalisation:
# 1. Lemmatize each word (convert to base form)
# 2. Convert to lowercase
# 3. Remove punctuation
# 4. Remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_chars]
words = sorted(list(set(words)))

# Sort intent classes for consistency
classes = sorted(list(set(classes)))

# Save processed vocabulary and classes for future use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# --- 📔 Training Data Preparation ---
training = []
output_empty = [0] * len(classes)  # Template for one-hot encoded output

# I'm using Bag of Words model here, converting each pattern into training data
#
# 🧠 Bag of Words (BoW) Model Explanation:
#   - What it is:
#     - Text → Binary (0/1) vector representation
#     - Each position represents a word from vocabulary
#     - Order of words is lost (hence "bag")
#     - Only tracks presence/absence of words
for document in documents:
    # Initialise bag of words
    bag = []
    pattern_words = document[0]
    # Normalise pattern words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # Create bag of words representation
    # 1 if word exists in pattern, 0 if it doesn't
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
    
    # Create output row with 1 for current intent tag, 0 for others
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Randomise training data to prevent learning sequence patterns
random.shuffle(training)
training = np.array(training, dtype=object)

# Split into features (X) and labels (y)
train_x = list(training[:, 0])  # Bag of words
train_y = list(training[:, 1])  # Intent tags

# --- 📔 Neural Network Architecture ---
model = Sequential([
    # First Dense Layer
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    # 🧠 How it works:
    #   - Input shape matches your vocabulary size (e.g., 1000 words = 1000 inputs)
    #   - 128 neurons process different word combinations
    #   - ReLU (Rectified Linear Unit) activation (f(x) = max(0,x)):
    #     - Keeps positive values unchanged
    #     - Converts negative values to zero
    #     - Helps network learn non-linear patterns
    
    # First Dropout Layer
    Dropout(0.5),
    # 🧠 How it works:
    #   - Randomly turns off 50% of neurons during each training step
    #   - Forces network to learn multiple paths to correct answer
    #   - Prevents over-reliance on specific neurons
    #   - Like training multiple smaller networks
    
    # Second Dense Layer
    Dense(64, activation='relu'),
    # 🧠 How it works:
    #   - Reduces dimensionality from 128 to 64 neurons
    #   - Creates more abstract features from first layer
    #   - ReLU activation continues non-linear pattern recognition
    #   - Fewer neurons = more focused feature detection

    # Second Dropout Layer
    Dropout(0.5),
    # 🧠 How it works:
    #   - Same 50% dropout principle as first layer
    #   - Prevents co-adaptation between layers
    #   - Further reduces overfitting risk
    #   - Maintains network robustness
    
   # Output Layer
    Dense(len(train_y[0]), activation='softmax')
    # 🧠 How it works:
    #   - Number of neurons matches number of intent classes
    #   - Softmax activation:
    #     - Converts outputs to probabilities (0-1)
    #     - All probabilities sum to 1
    #     - Highest value = most likely intent
    #     - Example: [0.7, 0.2, 0.1] = 70% confident it's first class
])

# --- 📔 Model Compilation and Training ---
# Configure Stochastic Gradient Descent (SGD) optimiser
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
# 🧠 How Learning Rate Works:
#   - Controls how big steps the model takes when learning
#   - 0.01 = conservative learning:
#     - Smaller steps = more stable learning
#     - Less likely to miss optimal solutions
#     - Takes longer to train

# 🧠 How Momentum Works:
#   - Like a ball rolling down a hill, keeps moving in successful directions
#   - 0.9 = 90% of previous update is remembered
#   - Benefits:
#     - Speeds up training in consistent directions
#     - Helps escape local minima (small valleys in error landscape)
#     - Smooths out noisy updates

# 🧠 How Nesterov Momentum Works:
#   - Looks ahead before making updates
#   - Process:
#     1. First looks where momentum would take us
#     2. Then calculates gradient at that future position
#     3. Makes a more informed update
#   - Benefits:
#     - More accurate updates than standard momentum
#     - Better at navigating around curves
#     - Faster convergence to optimal solution

# Compile model with specific loss function and metrics
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# 🧠 How Categorical Cross-Entropy Loss Works:
#   - Measures how far predictions are from actual values
#   - Perfect prediction = 0 loss
#   - Worst prediction = infinite loss
#   - Formula: -∑(true_value * log(predicted_value))

# 🧠 How the Optimizer (SGD) Works Here:
#   - Uses loss value to update network weights
#   - Process:
#     1. Make prediction
#     2. Calculate loss
#     3. Update weights to reduce loss

# 🧠 How Accuracy Metric Works:
#   - Tracks percentage of correct predictions
#   - Calculation:
#     - Prediction is "correct" if highest probability matches true class
#   - Reported after each training epoch
#   - Helps monitor if model is actually learning

# Train the model with specific parameters
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
# 🧠 How Epochs Work:
#   - One epoch = one full pass through all training data
#   - 200 epochs means:
#     - Model sees each example 200 times
#     - Like reading a book 200 times to really understand it
#   - Process per epoch:
#     1. Process all training samples
#     2. Calculate average loss and accuracy
#     3. Report progress

# 🧠 How Batch Size Works:
#   - Batch size 5 means:
#     - Process 5 training examples at once
#     - Update model weights after each batch
#   - Small batch size (5) means:
#     - More frequent updates
#     - More volatile but potentially better learning
#     - Less memory required

# 🧠 How Verbose Output Works:
#   - verbose=1 shows:
#     - Progress bar for each epoch
#     - Loss value (how wrong the model is)
#     - Accuracy (percentage of correct predictions)
#     - Time taken per epoch

# Save trained model and training history
model.save('chatbot_model.h5', hist)

print("Model created and saved to file")