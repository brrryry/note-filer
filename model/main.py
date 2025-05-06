from flask import Flask, request, jsonify # type: ignore
from dotenv import load_dotenv # type: ignore
import pandas as pd
from data import num_categories, get_categories
import os
import threading
from model import predict_category, train

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

    # error checking - just checking if anything in the body is missing or not a string.
    if not message: return jsonify({"error": "Message is required"}), 400
    if not guild: return jsonify({"error": "Guild is required"}), 400
    if not user: return jsonify({"error": "User is required"}), 400
    if not timestamp: return jsonify({"error": "Timestamp is required"}), 400

    if not isinstance(message, str): return jsonify({"error": "Message must be a string"}), 400
    if not isinstance(guild, str): return jsonify({"error": "Guild must be a string"}), 400
    if not isinstance(user, str): return jsonify({"error": "User must be a string"}), 400
    if not isinstance(timestamp, str): return jsonify({"error": "Timestamp must be a string"}), 400

    predicted_category, confidence = predict_category(message, guild)
    print(predicted_category, confidence)

    if predicted_category is None: return jsonify({"error": "Model not trained yet"}), 400
    if confidence <= 0: return jsonify({"error": "Model not confident enough"}), 400

    return jsonify({"category": predicted_category, "confidence": confidence}), 200



# this is a test endpoint to get the number of categories in the dataframe
@app.route("/categories/<guild>", methods=["GET"])
def categories(guild):
    if not os.path.exists(f"{os.getenv('DATA_FOLDER')}/{guild}.csv"): return jsonify({"error": "Guild not found"}), 404 # this should not happen, but just a failsafe...

    df = pd.read_csv(f"{os.getenv('DATA_FOLDER')}/{guild}.csv")

    categories = get_categories(df)

    if len(categories) == 0: return jsonify({"error": "No categories found"}), 404 # again, should not happen

    return jsonify({"categories": categories}), 200

@app.route("/train/<guild>", methods=["POST"])
def train_model(guild):
    if not os.path.exists(f"{os.getenv('DATA_FOLDER')}/{guild}.csv"): return jsonify({"error": "Guild not found"}), 404
    df = pd.read_csv(f"{os.getenv('DATA_FOLDER')}/{guild}.csv")



    training_task = threading.Thread(target=train, args=(guild,))
    training_task.start()

    return jsonify({"message": "Model training started"}), 200
    

# this is a test endpoint.
@app.route("/")
def hello_world():
    return "Hello, World!"