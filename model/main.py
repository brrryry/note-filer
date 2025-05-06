from flask import Flask, request, jsonify # type: ignore
from dotenv import load_dotenv # type: ignore
import pandas as pd
from data import num_categories, get_categories
import os

#import tensorflow as tf
#from tensorflow import keras

load_dotenv()

app = Flask(__name__)

@app.route("/message", methods=["POST"])
def mark_message():
    data = request.get_json() # get request body data

    # get req body data
    message = data.get("message")
    category = data.get("category")
    user = data.get("user")
    guild = data.get("guild")
    timestamp = data.get("timestamp")

    # error checking - just checking if anything in the body is missing or not a string.
    if not message: return jsonify({"error": "Message is required"}), 400
    if not category: return jsonify({"error": "Category is required"}), 400
    if not user: return jsonify({"error": "User is required"}), 400
    if not guild: return jsonify({"error": "Guild is required"}), 400
    if not timestamp: return jsonify({"error": "Timestamp is required"}), 400

    if not isinstance(message, str): return jsonify({"error": "Message must be a string"}), 400
    if not isinstance(category, str): return jsonify({"error": "Category must be a string"}), 400
    if not isinstance(user, str): return jsonify({"error": "User must be a string"}), 400
    if not isinstance(guild, str): return jsonify({"error": "Guild must be a string"}), 400
    if not isinstance(timestamp, str): return jsonify({"error": "Timestamp must be a string"}), 400

    # arbitrary length checks - just to make sure the data is not too long
    if len(message) > 500: return jsonify({"error": "Message is too long"}), 400
    if len(category) > 50: return jsonify({"error": "Category is too long"}), 400
    
    # check if the data folder exists. if not, create it.
    df = None

    if not os.path.exists(f"{os.getenv('DATA_FOLDER')}/{guild}.csv"): df = pd.DataFrame(columns=["message", "category", "user", "guild", "timestamp"])
    else: df = pd.read_csv(f"{os.getenv('DATA_FOLDER')}/{guild}.csv")

    current_categories = num_categories(df)

    df.loc[len(df)] = [message, category, user, guild, timestamp]
    df.to_csv(f"{os.getenv('DATA_FOLDER')}/{guild}.csv", index=False)
    return jsonify({"message": "Message marked successfully", "categories": num_categories(df), "newcategory": num_categories(df) > current_categories}), 200

@app.route("/predict", methods=["POST"])
def model_predict():
    data = request.get_json()

    # get req body data
    message = data.get("message")
    guild = data.get("guild")
    user = data.get("user")
    timestamp = data.get("timestamp")

    # TODO: given the message, predict the category using the server's model


# this is a test endpoint to get the number of categories in the dataframe
@app.route("/categories/<guild>", methods=["GET"])
def categories(guild):
    if not os.path.exists(f"{os.getenv('DATA_FOLDER')}/{guild}.csv"): return jsonify({"error": "Guild not found"}), 404 # this should not happen, but just a failsafe...

    df = pd.read_csv(f"{os.getenv('DATA_FOLDER')}/{guild}.csv")

    categories = get_categories(df)
    print(categories)

    if len(categories) == 0: return jsonify({"error": "No categories found"}), 404 # again, should not happen

    return jsonify({"categories": categories}), 200

@app.route("/train/<guild>", methods=["POST"])
def train_model(guild):
    if not os.path.exists(f"{os.getenv('DATA_FOLDER')}/{guild}.csv"): return jsonify({"error": "Guild not found"}), 404
    df = pd.read_csv(f"{os.getenv('DATA_FOLDER')}/{guild}.csv")
    categories = get_categories(df)

    return jsonify({"id": guild, "num_categories": len(categories)}), 200 

    # TODO: create a model and train it on the data here!


# this is a test endpoint.
@app.route("/")
def hello_world():
    return "Hello, World!"
#
# model.py
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
########################################################################################################################################################################################################################################################################################
#############################################################################################################################################
#beginning of model 
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
# simulation.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D, Conv1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import pickle
import re
import time
import random

# Ensure TensorFlow reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
set_seed()

# Set the data folder path
DATA_FOLDER = "data"

# Create data directory if it doesn't exist
os.makedirs(DATA_FOLDER, exist_ok=True)

# Download NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    print("NLTK resources already downloaded or could not be downloaded")

def create_test_csv():
    """Create a CSV file with expanded test data for training the model."""
    guild_id = "1354533600745361809"
    
    # Define test data - expanded version with more examples per category
    test_data = [
        # Badminton category
        ["I got a new badminton racket and it's amazing!", "badminton", "323297551887368204", "1354533600745361809", "1744939103027"],
        ["When is the next badminton tournament?", "badminton", "302923939154493441", "1354533600745361809", "1744939336172"],
        ["Does anyone want to play badminton this weekend?", "badminton", "323297551887368204", "1354533600745361809", "1744939676212"],
        ["The shuttle was out of bounds in that last game", "badminton", "446721983437619245", "1354533600745361809", "1744939780845"],
        ["I need to improve my backhand stroke", "badminton", "302923939154493441", "1354533600745361809", "1744939890321"],
        ["Just registered for the doubles badminton championship", "badminton", "323297551887368204", "1354533600745361809", "1744939900001"],
        ["My badminton coach says I need to work on my footwork", "badminton", "446721983437619245", "1354533600745361809", "1744939900002"],
        ["Looking for a good place to buy badminton equipment", "badminton", "302923939154493441", "1354533600745361809", "1744939900003"],
        ["The Lin Dan vs Lee Chong Wei match was incredible", "badminton", "323297551887368204", "1354533600745361809", "1744939900004"],
        ["Anyone have recommendations for badminton shoes?", "badminton", "446721983437619245", "1354533600745361809", "1744939900005"],
        ["I'm trying to master the jump smash in badminton", "badminton", "302923939154493441", "1354533600745361809", "1744939900006"],
        ["Who's watching the All England Open badminton championship?", "badminton", "323297551887368204", "1354533600745361809", "1744939900007"],
        
        # Astronomy category
        ["The new moon looks beautiful tonight", "astronomy", "323297551887368204", "1354533600745361809", "1744940123456"],
        ["I spotted Saturn through my telescope", "astronomy", "302923939154493441", "1354533600745361809", "1744940234567"],
        ["The lunar eclipse will be visible next Tuesday", "astronomy", "446721983437619245", "1354533600745361809", "1744940345678"],
        ["Has anyone seen the meteor shower last night?", "astronomy", "323297551887368204", "1354533600745361809", "1744940456789"],
        ["The International Space Station just passed over my house", "astronomy", "302923939154493441", "1354533600745361809", "1744940567890"],
        ["I'm learning about black holes and they're fascinating", "astronomy", "446721983437619245", "1354533600745361809", "1744940900001"],
        ["Just got a new telescope for stargazing", "astronomy", "323297551887368204", "1354533600745361809", "1744940900002"],
        ["The Milky Way is clearly visible tonight", "astronomy", "302923939154493441", "1354533600745361809", "1744940900003"],
        ["Did you know there are more stars in the universe than grains of sand on Earth?", "astronomy", "446721983437619245", "1354533600745361809", "1744940900004"],
        ["Learning about the different phases of the moon", "astronomy", "323297551887368204", "1354533600745361809", "1744940900005"],
        ["The James Webb Space Telescope images are mind-blowing", "astronomy", "302923939154493441", "1354533600745361809", "1744940900006"],
        ["Anyone interested in joining an astronomy club?", "astronomy", "446721983437619245", "1354533600745361809", "1744940900007"],
        
        # Gaming category - significantly expanded
        ["I'm looking for a new gaming headset", "gaming", "446721983437619245", "1354533600745361809", "1744940678901"],
        ["Has anyone played the new Elder Scrolls game?", "gaming", "323297551887368204", "1354533600745361809", "1744940789012"],
        ["My ping is terrible today", "gaming", "302923939154493441", "1354533600745361809", "1744940890123"],
        ["I just built a new gaming PC with RTX 4080", "gaming", "446721983437619245", "1354533600745361809", "1744941001234"],
        ["Does anyone have recommendations for good RPGs?", "gaming", "323297551887368204", "1354533600745361809", "1744941112345"],
        ["I'm stuck on this boss fight in Elden Ring", "gaming", "302923939154493441", "1354533600745361809", "1744941900001"],
        ["What's your favorite Nintendo Switch game?", "gaming", "446721983437619245", "1354533600745361809", "1744941900002"],
        ["Looking for people to join my Minecraft server", "gaming", "323297551887368204", "1354533600745361809", "1744941900003"],
        ["The new Zelda game has amazing open world mechanics", "gaming", "302923939154493441", "1354533600745361809", "1744941900004"],
        ["I need help with this puzzle in Portal 2", "gaming", "446721983437619245", "1354533600745361809", "1744941900005"],
        ["Anyone excited for the upcoming Starfield DLC?", "gaming", "323297551887368204", "1354533600745361809", "1744941900006"],
        ["The loot drop rates in this game are terrible", "gaming", "302923939154493441", "1354533600745361809", "1744941900007"],
        ["Just reached Diamond rank in League of Legends", "gaming", "446721983437619245", "1354533600745361809", "1744941900008"],
        ["Does anyone want to play co-op games this weekend?", "gaming", "323297551887368204", "1354533600745361809", "1744941900009"],
        ["The graphics on the PS5 version are incredible", "gaming", "302923939154493441", "1354533600745361809", "1744941900010"],
        
        # Cooking category - expanded
        ["I made a beef Wellington for dinner", "cooking", "302923939154493441", "1354533600745361809", "1744941223456"],
        ["What's the best way to cook risotto?", "cooking", "446721983437619245", "1354533600745361809", "1744941334567"],
        ["I'm trying to perfect my sourdough bread recipe", "cooking", "323297551887368204", "1354533600745361809", "1744941445678"],
        ["Does anyone have a good recipe for chocolate cake?", "cooking", "302923939154493441", "1354533600745361809", "1744941556789"],
        ["My new cast iron pan is amazing", "cooking", "446721983437619245", "1354533600745361809", "1744941667890"],
        ["Just learned how to make homemade pasta", "cooking", "323297551887368204", "1354533600745361809", "1744941900004"],
        ["Looking for tips on cooking the perfect steak", "cooking", "302923939154493441", "1354533600745361809", "1744941900005"],
        ["I'm taking a cooking class to learn French cuisine", "cooking", "446721983437619245", "1354533600745361809", "1744941900006"],
        ["Does anyone know a good substitute for buttermilk?", "cooking", "323297551887368204", "1354533600745361809", "1744941900007"],
        ["I finally mastered the art of making macarons", "cooking", "302923939154493441", "1354533600745361809", "1744941900008"],
        ["What's your favorite spice to cook with?", "cooking", "446721983437619245", "1354533600745361809", "1744941900009"],
        ["Just got an Instant Pot and love it", "cooking", "323297551887368204", "1354533600745361809", "1744941900010"],
        ["Any tips for meal prepping for the week?", "cooking", "302923939154493441", "1354533600745361809", "1744941900011"],
        
        # Outdoors category
        ["I'm going hiking in the mountains this weekend", "outdoors", "323297551887368204", "1354533600745361809", "1744941778901"],
        ["Has anyone tried the new hiking trail by the lake?", "outdoors", "302923939154493441", "1354533600745361809", "1744941889012"],
        ["I saw a deer in the park this morning", "outdoors", "446721983437619245", "1354533600745361809", "1744942000123"],
        ["What's the best tent for camping in the rain?", "outdoors", "323297551887368204", "1354533600745361809", "1744942111234"],
        ["I caught a huge fish on my trip last weekend", "outdoors", "302923939154493441", "1354533600745361809", "1744942222345"],
        ["Planning a kayaking trip down the river", "outdoors", "446721983437619245", "1354533600745361809", "1744942900001"],
        ["Just got back from an amazing camping trip", "outdoors", "323297551887368204", "1354533600745361809", "1744942900002"],
        ["Looking for good spots for bird watching", "outdoors", "302923939154493441", "1354533600745361809", "1744942900003"],
        ["I need recommendations for a good pair of hiking boots", "outdoors", "446721983437619245", "1354533600745361809", "1744942900004"],
        ["Anyone interested in a group trail run this weekend?", "outdoors", "323297551887368204", "1354533600745361809", "1744942900005"],
        ["The fall foliage on my hike was spectacular", "outdoors", "302923939154493441", "1354533600745361809", "1744942900006"],
        ["Just bought a new mountain bike for the trails", "outdoors", "446721983437619245", "1354533600745361809", "1744942900007"],
        
        # Programming category - expanded with more examples
        ["The new JavaScript framework looks promising", "programming", "446721983437619245", "1354533600745361809", "1744942333456"],
        ["I can't figure out this bug in my Python code", "programming", "323297551887368204", "1354533600745361809", "1744942444567"],
        ["Does anyone know how to optimize this SQL query?", "programming", "302923939154493441", "1354533600745361809", "1744942555678"],
        ["I just deployed my first React app", "programming", "446721983437619245", "1354533600745361809", "1744942666789"],
        ["Git merge conflicts are driving me crazy today", "programming", "323297551887368204", "1354533600745361809", "1744942777890"],
        ["Learning machine learning with TensorFlow", "programming", "302923939154493441", "1354533600745361809", "1744942900004"],
        ["What IDE do you recommend for Java development?", "programming", "446721983437619245", "1354533600745361809", "1744942900005"],
        ["Just finished my first open source contribution", "programming", "323297551887368204", "1354533600745361809", "1744942900006"],
        ["Having trouble with dependency injection in Spring Boot", "programming", "302923939154493441", "1354533600745361809", "1744942900007"],
        ["Does anyone use Docker for their development environment?", "programming", "446721983437619245", "1354533600745361809", "1744942900008"],
        ["I'm learning functional programming concepts", "programming", "323297551887368204", "1354533600745361809", "1744942900009"],
        ["How do you handle state management in large React apps?", "programming", "302923939154493441", "1354533600745361809", "1744942900010"],
        ["Just solved a difficult algorithm challenge", "programming", "446721983437619245", "1354533600745361809", "1744942900011"],
        ["What's your preferred testing framework?", "programming", "323297551887368204", "1354533600745361809", "1744942900012"],
        ["I'm trying to understand asynchronous programming", "programming", "302923939154493441", "1354533600745361809", "1744942900013"],
        
        # Music category
        ["The concert last night was amazing", "music", "302923939154493441", "1354533600745361809", "1744942888901"],
        ["I'm learning to play the guitar", "music", "446721983437619245", "1354533600745361809", "1744943000012"],
        ["Has anyone heard the new album by Taylor Swift?", "music", "323297551887368204", "1354533600745361809", "1744943111123"],
        ["I'm going to a jazz festival next month", "music", "302923939154493441", "1354533600745361809", "1744943222234"],
        ["What's your favorite music genre?", "music", "446721983437619245", "1354533600745361809", "1744943333345"],
        ["Looking for recommendations for good classical music", "music", "323297551887368204", "1354533600745361809", "1744943900001"],
        ["Just got tickets to the upcoming rock concert", "music", "302923939154493441", "1354533600745361809", "1744943900002"],
        ["I'm thinking about starting a band", "music", "446721983437619245", "1354533600745361809", "1744943900003"],
        ["Learning to read sheet music is challenging", "music", "323297551887368204", "1354533600745361809", "1744943900004"],
        ["What's your favorite instrument to play?", "music", "302923939154493441", "1354533600745361809", "1744943900005"],
        ["The acoustics in that concert hall are perfect", "music", "446721983437619245", "1354533600745361809", "1744943900006"],
        ["I just discovered this amazing indie band", "music", "323297551887368204", "1354533600745361809", "1744943900007"],
        ["Looking for a good music production software", "music", "302923939154493441", "1354533600745361809", "1744943900008"],
    ]
    

print("Starting advanced deep learning model training...")

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = str(42)

# Set the data folder path
DATA_FOLDER = "data"

# Create data directory if it doesn't exist
os.makedirs(DATA_FOLDER, exist_ok=True)

def text_preprocessing(text):
    """Advanced text preprocessing function for NLP tasks"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions (common in messages)
    text = re.sub(r'@\w+', '', text)
    
    # Remove punctuations and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def data_augmentation(df, augment_factor=2):
    """Generate augmented data to increase training set size and diversity"""
    print("Performing data augmentation...")
    augmented_data = []
    categories = df['category'].unique()
    
    # Simple synonym replacement and word order change
    for _, row in df.iterrows():
        message = row['message']
        category = row['category']
        user = row['user']
        guild = row['guild']
        timestamp = row['timestamp']
        
        # Only augment if message has more than 4 words
        words = message.split()
        if len(words) > 4:
            # Word order change - swap some words randomly
            for _ in range(augment_factor):
                words_copy = words.copy()
                
                # Choose two random positions to swap
                if len(words) > 5:
                    idx1, idx2 = np.random.choice(len(words), 2, replace=False)
                    words_copy[idx1], words_copy[idx2] = words_copy[idx2], words_copy[idx1]
                
                # Add small variations
                if np.random.random() > 0.5 and len(words) > 3:
                    # Remove a random word (not the first or last)
                    idx_to_remove = np.random.randint(1, len(words_copy) - 1)
                    words_copy.pop(idx_to_remove)
                
                augmented_message = ' '.join(words_copy)
                if augmented_message != message:  # Ensure it's different
                    new_timestamp = str(int(timestamp) + np.random.randint(1, 1000))
                    augmented_data.append([augmented_message, category, user, guild, new_timestamp])
    
    # Convert augmented data to DataFrame and concatenate with original
    if augmented_data:
        augmented_df = pd.DataFrame(augmented_data, columns=df.columns)
        return pd.concat([df, augmented_df], ignore_index=True)
    return df

def build_advanced_model(vocab_size, embedding_dim, max_sequence_length, num_classes):
    """
    Build a sophisticated neural network architecture combining CNN and LSTM layers
    """
    model = Sequential([
        # Embedding Layer
        Embedding(vocab_size, embedding_dim, input_length=max_sequence_length),
        
        # Spatial Dropout to reduce overfitting
        SpatialDropout1D(0.2),
        
        # Convolutional layer to capture n-gram features
        Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'),
        
        # Bidirectional LSTM layers
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
        
        # Dense layers
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile with Adam optimizer and learning rate
    opt = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    
    return model

def train_cross_validation():
    """
    Train a highly accurate model using cross-validation to ensure robustness
    """
    # Guild ID for the dataset
    guild_id = "1354533600745361809"
    model_save_path = f"{DATA_FOLDER}/{guild_id}_model"
    os.makedirs(model_save_path, exist_ok=True)
    
    # Check if data file exists
    data_file = f"{DATA_FOLDER}/{guild_id}.csv"
    if not os.path.exists(data_file):
        print("Data file not found. Creating test data...")
        from data import create_test_csv
        create_test_csv()
    
    # Load the data
    print("Loading and preprocessing data...")
    df = pd.read_csv(data_file)
    print(f"Original dataset size: {len(df)} examples")
    
    # Apply data augmentation to increase dataset size
    df = data_augmentation(df, augment_factor=2)
    print(f"Dataset size after augmentation: {len(df)} examples")
    
    # Apply text preprocessing
    print("Preprocessing text data...")
    df['preprocessed_message'] = df['message'].apply(text_preprocessing)
    
    # Prepare features and target
    X = df['preprocessed_message'].values
    y = df['category'].values
    
    # Get list of categories
    categories = df['category'].unique().tolist()
    print(f"Training model for {len(categories)} categories: {categories}")
    
    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    
    # Transform target to numerical values
    y_encoded = label_encoder.transform(y)
    
    # Save label encoder
    with open(f"{model_save_path}/label_encoder.pickle", 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Parameters for the model
    max_words = 15000
    max_sequence_length = 150
    embedding_dim = 256
    num_classes = len(categories)
    epochs = 100
    batch_size = 16
    n_splits = 5  # Number of folds for cross-validation
    
    # Tokenize text
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X)
    
    # Save tokenizer
    with open(f"{model_save_path}/tokenizer.pickle", 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Convert to sequences and pad
    X_sequences = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(X_sequences, maxlen=max_sequence_length, padding='post')
    
    # Get vocabulary size
    vocab_size = min(max_words, len(tokenizer.word_index) + 1)
    
    # Convert to one-hot encoding for categorical loss
    y_one_hot = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)
    
    # Start K-fold cross-validation
    print(f"\nStarting {n_splits}-fold cross-validation...")
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_no = 1
    fold_accuracies = []
    
    best_accuracy = 0
    best_model_path = f"{model_save_path}/best_model.h5"
    
    for train_idx, val_idx in kfold.split(X_padded):
        print(f"\n----- Training Fold {fold_no}/{n_splits} -----")
        
        # Split data
        X_train, X_val = X_padded[train_idx], X_padded[val_idx]
        y_train, y_val = y_one_hot[train_idx], y_one_hot[val_idx]
        
        # Build model
        model = build_advanced_model(vocab_size, embedding_dim, max_sequence_length, num_classes)
        
        if fold_no == 1:
            print("Model Architecture:")
            model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001),
            ModelCheckpoint(
                filepath=f"{model_save_path}/fold_{fold_no}_best.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        training_time = time.time() - start_time
        
        # Evaluate on validation set
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"Fold {fold_no} - Validation Accuracy: {accuracy:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        
        # Save accuracy
        fold_accuracies.append(accuracy)
        
        # Save the model if it's the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.save(best_model_path)
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")
        
        # Plot learning curves
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Fold {fold_no} - Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Fold {fold_no} - Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{model_save_path}/fold_{fold_no}_history.png")
        plt.close()
        
        fold_no += 1
    
    # Print cross-validation results
    print("\nCross-validation complete!")
    print(f"Accuracy scores for all folds: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    print(f"Average accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Best fold accuracy: {best_accuracy:.4f}")
    
    # Train final model on all data
    print("\nTraining final ensemble model on all data...")
    
    # Load best model
    final_model = load_model(best_model_path)
    
    # Fine-tune on all data
    final_model.fit(
        X_padded, y_one_hot,
        epochs=int(epochs/3),  # Fewer epochs for final training
        batch_size=batch_size,
        verbose=1,
        callbacks=[
            EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001)
        ]
    )
    
    # Save final model
    final_model.save(f"{model_save_path}/final_model")
    print(f"Final model saved to {model_save_path}/final_model")
    
    # Evaluate on test data
    print("\nEvaluating final model on test data...")
    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
        X_padded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    y_test_one_hot = tf.keras.utils.to_categorical(y_test_encoded, num_classes=num_classes)
    
    # Evaluate
    loss, accuracy = final_model.evaluate(X_test, y_test_one_hot, verbose=0)
    print(f"Final Test Accuracy: {accuracy:.4f}")
    
    # Predict on test data
    y_pred_probs = final_model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    class_names = label_encoder.classes_
    report = classification_report(y_test_encoded, y_pred, target_names=class_names)
    print(report)
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f"{model_save_path}/confusion_matrix.png")
    
    # Calculate additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_encoded, y_pred, average='weighted')
    
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    
    # Test predictions on some examples
    print("\nTesting predictions on examples:")
    test_messages = [
        "Anyone want to play badminton tomorrow evening?",
        "I'm looking for a good telescope for beginners",
        "The new game update has amazing graphics",
        "I need a good recipe for chocolate chip cookies",
        "Going hiking in the mountains next weekend",
        "Having trouble with my JavaScript code",
        "The concert yesterday was amazing"
    ]
    
    # Preprocess test messages
    test_messages_preprocessed = [text_preprocessing(msg) for msg in test_messages]
    test_sequences = tokenizer.texts_to_sequences(test_messages_preprocessed)
    test_padded = pad_sequences(test_sequences, maxlen=max_sequence_length, padding='post')
    
    # Predict
    predictions = final_model.predict(test_padded)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_categories = label_encoder.inverse_transform(predicted_classes)
    
    print("\nPrediction Examples:")
    for msg, category, confidence in zip(test_messages, predicted_categories, np.max(predictions, axis=1)):
        print(f"Message: {msg}")
        print(f"Predicted Category: {category}")
        print(f"Confidence: {confidence:.4f}")
        print()
    
    return final_model, best_accuracy

if __name__ == "__main__":
    # Run the training process
    train_cross_validation()
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
#End of model
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
# Set the data folder path
DATA_FOLDER = "data"

def get_categories(df):
    """
    Get unique categories from the dataframe.
    """
    return df['category'].unique().tolist()

def num_categories(df):
    """
    Get the number of unique categories in the dataframe.
    """
    return len(get_categories(df))

def train(guild_id, model_save_path=None, test_size=0.2, random_state=42, epochs=50, batch_size=16):
    """
    Train a deep learning NLP model to categorize messages using gradient descent and backpropagation.
    
    Parameters:
    -----------
    guild_id : str
        The ID of the guild/server to train the model for
    
    model_save_path : str, optional
        Directory where the trained model will be saved. If None, will use DATA_FOLDER/guild_id_model
    
    test_size : float
        Proportion of the dataset to be used as test set (default: 0.2)
    
    random_state : int
        Random seed for reproducible results
        
    epochs : int
        Number of training epochs
        
    batch_size : int
        Batch size for training
        
    Returns:
    --------
    model : trained model object
        The trained keras model
    accuracy : float
        Accuracy score on the test set
    """
    # Set default model save path if not provided
    if model_save_path is None:
        model_save_path = f"{DATA_FOLDER}/{guild_id}_model"
    
    # Check if data file exists
    data_file = f"{DATA_FOLDER}/{guild_id}.csv"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"No training data found for guild {guild_id}")
    
    # Load the data
    df = pd.read_csv(data_file)
    
    # Check if we have enough data
    categories = get_categories(df)
    if len(categories) < 2:
        raise ValueError("Need at least 2 categories to train a model")
    
    if len(df) < 10:
        raise ValueError("Need at least 10 messages to train a model")
    
    print(f"Training model with {len(df)} messages across {len(categories)} categories")
    
    # Prepare features and target
    X = df['message'].values
    y = df['category'].values
    
    # Split the data - handle small datasets
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        print("Warning: Some categories have too few examples for stratified sampling. Using regular split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    # Create label encoder for categories
    label_encoder = LabelEncoder()
    label_encoder.fit(categories)
    
    # Transform target to numerical values
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Convert to one-hot encoding for keras
    num_classes = len(categories)
    y_train_one_hot = tf.keras.utils.to_categorical(y_train_encoded, num_classes=num_classes)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test_encoded, num_classes=num_classes)
    
    # Tokenize text
    max_words = 10000  # Maximum vocab size
    max_sequence_length = 100  # Max length of each message
    
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    # Convert text to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences to ensure uniform length
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post')
    
    # Get vocabulary size (add 1 for OOV token)
    vocab_size = min(max_words, len(tokenizer.word_index) + 1)
    
    # Create model
    embedding_dim = 128
    
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_sequence_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model with gradient descent optimizer
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',  # Adam is a variant of gradient descent
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train the model using backpropagation
    print("\nTraining neural network with gradient descent and backpropagation...")
    history = model.fit(
        X_train_padded, y_train_one_hot,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_padded, y_test_one_hot),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_padded, y_test_one_hot)
    print(f"\nTest accuracy: {accuracy:.4f}")
    
    # Create directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)
    
    # Save the model, tokenizer, and label encoder
    model.save(f"{model_save_path}/keras_model")
    
    with open(f"{model_save_path}/tokenizer.pickle", 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(f"{model_save_path}/label_encoder.pickle", 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save training history as a plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{model_save_path}/training_history.png")
    
    print(f"Model saved to {model_save_path}")
    return model, accuracy

def predict_category(message, guild_id):
    """
    Predict the category of a message using the trained model for a specific guild.
    
    Parameters:
    -----------
    message : str
        The message to categorize
    
    guild_id : str
        The ID of the guild/server
        
    Returns:
    --------
    predicted_category : str
        The predicted category for the message
    confidence : float
        Confidence score for the prediction
    """
    model_path = f"{DATA_FOLDER}/{guild_id}_model"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found for guild {guild_id}. Run train({guild_id}) first.")
    
    # Load the model, tokenizer, and label encoder
    model = load_model(f"{model_path}/keras_model")
    
    with open(f"{model_path}/tokenizer.pickle", 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    with open(f"{model_path}/label_encoder.pickle", 'rb') as handle:
        label_encoder = pickle.load(handle)
    
    # Prepare the message
    max_sequence_length = 100
    sequences = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    
    # Get predictions
    prediction = model.predict(padded, verbose=0)[0]
    
    # Get the predicted class and confidence
    predicted_class_index = np.argmax(prediction)
    confidence = prediction[predicted_class_index]
    
    # Convert back to category name
    predicted_category = label_encoder.inverse_transform([predicted_class_index])[0]
    
    return predicted_category, confidence
def train(guild_id, model_save_path=None, test_size=0.2, random_state=42):
    """
    Train an NLP model to categorize messages into user-defined categories for a specific guild.
    
    Parameters:
    -----------
    guild_id : str
        The ID of the guild/server to train the model for
    
    model_save_path : str, optional
        Path where the trained model will be saved. If None, will use DATA_FOLDER/guild_id_model
    
    test_size : float
        Proportion of the dataset to be used as test set (default: 0.2)
    
    random_state : int
        Random seed for reproducible results
        
    Returns:
    --------
    model : trained model object
        The trained classification model
    accuracy : float
        Accuracy score on the test set
    """
    import os
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score
    
    # Set default model save path if not provided
    if model_save_path is None:
        model_save_path = f"{os.getenv('DATA_FOLDER')}/{guild_id}_model"
    
    # Check if data file exists
    data_file = f"{os.getenv('DATA_FOLDER')}/{guild_id}.csv"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"No training data found for guild {guild_id}")
    
    # Load the data
    df = pd.read_csv(data_file)
    
    # Check if we have enough data
    categories = get_categories(df)
    if len(categories) < 2:
        raise ValueError("Need at least 2 categories to train a model")
    
    if len(df) < 10:
        raise ValueError("Need at least 10 messages to train a model")
    
    # Prepare features and target
    X = df['message'].values
    y = df['category'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(df) > 20 else None
    )
    
    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(categories)
    
    # Transform the target
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Create and train the pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2))),
        ('classifier', OneVsRestClassifier(LinearSVC(random_state=random_state)))
    ])
    
    pipeline.fit(X_train, y_train_encoded)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    
    # Save the model and label encoder
    import joblib
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump((pipeline, label_encoder), model_save_path)
    
    print(f"Model trained for guild {guild_id} with accuracy: {accuracy:.2f}")
    
    return pipeline, accuracy

def predict_category(message, guild_id):
    """
    Predict the category of a message using the trained model for a specific guild.
    
    Parameters:
    -----------
    message : str
        The message to categorize
    
    guild_id : str
        The ID of the guild/server
        
    Returns:
    --------
    predicted_category : str
        The predicted category for the message
    confidence : float
        Confidence score for the prediction (if available)

    #paramrters input a string, output max(categories)
    #need ID for correct server DATA_FOLDER/ID.csv
    #pretrain for weights, not meant to be unsupervised
    #this is for supervision with test data
    Categories = file.read(/data)
    #interpet the names of the ategoies
    """
    import os
    import joblib
    
    model_path = f"{os.getenv('DATA_FOLDER')}/{guild_id}_model"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found for guild {guild_id}. Run train({guild_id}) first.")
    
    # Load the model and label encoder
    pipeline, label_encoder = joblib.load(model_path)
    
    # Make prediction
    prediction = pipeline.predict([message])[0]
    
    # Get predicted category name
    predicted_category = label_encoder.inverse_transform([prediction])[0]
    
    # Try to get confidence scores if the model supports it
    try:
        # Get confidence scores
        confidence_scores = pipeline.decision_function([message])[0]
        confidence = confidence_scores[prediction]
    except:
        confidence = None
    
    return predicted_category, confidence
