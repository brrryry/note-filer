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
