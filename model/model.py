from flask import Flask, request, jsonify # type: ignore
from dotenv import load_dotenv # type: ignore
import pandas as pd
from data import num_categories, get_categories
import os
import threading

#import tensorflow as tf
#from tensorflow import keras

load_dotenv()

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
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D, Conv1D
from tensorflow.keras.layers import Input, Concatenate, BatchNormalization, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import pickle
import re
import time
import random
import pathlib

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

# Try to import NLTK (not essential for the main functionality)
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    print("NLTK resources already downloaded or could not be downloaded")

# Create a standalone predictor class
class TextCategoryPredictor:
    def __init__(self, models, tokenizer, label_encoder, max_sequence_length):
        self.models = models
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_sequence_length = max_sequence_length
    
    def preprocess_text(self, text):
        """Apply same preprocessing used during training"""
        return text_preprocessing(text)
    
    def predict(self, messages, return_confidence=False):
        """
        Predict categories for a list of messages
        
        Args:
            messages: List of text messages to classify
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predicted categories, and optionally confidence scores
        """
        # Preprocess messages
        preprocessed = [self.preprocess_text(msg) for msg in messages]
        
        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences(preprocessed)
        
        # Pad sequences
        padded = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        
        # Get predictions from all models
        all_predictions = []
        for model in self.models:
            preds = model.predict(padded)
            all_predictions.append(preds)
        
        # Compute ensemble predictions with weighted average
        weights = [1.2, 1.0, 0.8]  # Assign more weight to better models
        weighted_preds = []
        
        for i, preds in enumerate(all_predictions):
            weight = weights[i] if i < len(weights) else 1.0
            weighted_preds.append(preds * weight)
        
        # Average predictions
        ensemble_preds = np.sum(weighted_preds, axis=0) / np.sum(weights[:len(all_predictions)])
        
        # Get classes and confidences
        pred_classes = np.argmax(ensemble_preds, axis=1)
        pred_categories = self.label_encoder.inverse_transform(pred_classes)
        confidences = np.max(ensemble_preds, axis=1)
        
        if return_confidence:
            return pred_categories, confidences
        return pred_categories

def create_test_csv():
    """Create a CSV file with expanded test data for training the model."""
    guild_id = "1354533600745361809"
    
    # Define test data - expanded version with more examples per category
    test_data = [
        # Badminton category
    ["I bought a new Yonex Astrox racket for my tournament next week", "badminton", "323297551887368204", "1354533600745361809", "1744939900100"],
    ["My shuttle keeps hitting the net during clear shots", "badminton", "446721983437619245", "1354533600745361809", "1744939900101"],
    ["The badminton court floors were just refinished at the community center", "badminton", "302923939154493441", "1354533600745361809", "1744939900102"],
    ["Looking for a badminton partner who can play on Tuesday evenings", "badminton", "323297551887368204", "1354533600745361809", "1744939900103"],
    ["My backhand smash is getting better after practicing for a month", "badminton", "446721983437619245", "1354533600745361809", "1744939900104"],
    ["Does anyone know if the badminton league is still accepting new players?", "badminton", "302923939154493441", "1354533600745361809", "1744939900105"],
    ["I'm still sore from yesterday's intense badminton match", "badminton", "323297551887368204", "1354533600745361809", "1744939900106"],
    ["What's the best string tension for control-oriented badminton players?", "badminton", "446721983437619245", "1354533600745361809", "1744939900107"],
    ["The new feather shuttles are much better than the synthetic ones we used before", "badminton", "302923939154493441", "1354533600745361809", "1744939900108"],
    ["I need to work on my footwork to improve my court coverage", "badminton", "323297551887368204", "1354533600745361809", "1744939900109"],
    ["My mixed doubles partner and I won the local tournament last weekend", "badminton", "446721983437619245", "1354533600745361809", "1744939900110"],
    ["Should I get badminton-specific shoes or will regular indoor court shoes work?", "badminton", "302923939154493441", "1354533600745361809", "1744939900111"],
    ["Trying to master the around-the-head shot for backhand returns", "badminton", "323297551887368204", "1354533600745361809", "1744939900112"],
    ["The drop shot is my favorite badminton technique to surprise opponents", "badminton", "446721983437619245", "1354533600745361809", "1744939900113"],
    ["I've been watching Viktor Axelsen's matches to learn his deceptive shots", "badminton", "302923939154493441", "1354533600745361809", "1744939900114"],
    ["My wrist hurts after playing badminton for 3 hours straight", "badminton", "323297551887368204", "1354533600745361809", "1744939900115"],
    ["Our badminton club is hosting a fundraiser tournament next month", "badminton", "446721983437619245", "1354533600745361809", "1744939900116"],
    ["I need to practice my service more, it's too predictable", "badminton", "302923939154493441", "1354533600745361809", "1744939900117"],
    ["The crosscourt net shot is so difficult to master in badminton", "badminton", "323297551887368204", "1354533600745361809", "1744939900118"],
    ["Our team needs one more badminton player for the regional championships", "badminton", "446721983437619245", "1354533600745361809", "1744939900119"],
    ["I prefer playing singles rather than doubles in badminton", "badminton", "302923939154493441", "1354533600745361809", "1744939900120"],
    ["The shuttle was definitely in, I saw the line!", "badminton", "323297551887368204", "1354533600745361809", "1744939900121"],
    ["My defensive clears need to be deeper to push opponents to the back", "badminton", "446721983437619245", "1354533600745361809", "1744939900122"],
    ["Which grip do you use for your backhand shots?", "badminton", "302923939154493441", "1354533600745361809", "1744939900123"],
    ["I find it hard to maintain stamina through a full three-game match", "badminton", "323297551887368204", "1354533600745361809", "1744939900124"],
    ["The service rule changes have affected my playing style significantly", "badminton", "446721983437619245", "1354533600745361809", "1744939900125"],
    ["We need to book the badminton courts early on weekends", "badminton", "302923939154493441", "1354533600745361809", "1744939900126"],
    ["My jump smash is getting more powerful after weight training", "badminton", "323297551887368204", "1354533600745361809", "1744939900127"],
    ["Looking for recommendations on badminton training programs", "badminton", "446721983437619245", "1354533600745361809", "1744939900128"],
    ["The Malaysian badminton team always impresses me with their skills", "badminton", "302923939154493441", "1354533600745361809", "1744939900129"],
        
        # Astronomy category
    ["I just photographed the Andromeda galaxy with my new telescope", "astronomy", "323297551887368204", "1354533600745361809", "1744940900100"],
    ["Jupiter's Great Red Spot is visible tonight through my 8-inch reflector", "astronomy", "302923939154493441", "1354533600745361809", "1744940900101"],
    ["Has anyone tried astrophotography with a DSLR and star tracker?", "astronomy", "446721983437619245", "1354533600745361809", "1744940900102"],
    ["The Perseid meteor shower peaks this weekend, weather looks clear", "astronomy", "323297551887368204", "1354533600745361809", "1744940900103"],
    ["My goto mount keeps losing alignment when tracking deep sky objects", "astronomy", "302923939154493441", "1354533600745361809", "1744940900104"],
    ["I'm fascinated by the possibility of life on Europa due to its subsurface ocean", "astronomy", "446721983437619245", "1354533600745361809", "1744940900105"],
    ["The Ring Nebula looks amazing through my eyepiece with a UHC filter", "astronomy", "323297551887368204", "1354533600745361809", "1744940900106"],
    ["Anyone going to the dark sky preserve for the lunar eclipse?", "astronomy", "302923939154493441", "1354533600745361809", "1744940900107"],
    ["I've been reading about neutron stars and their incredible density", "astronomy", "446721983437619245", "1354533600745361809", "1744940900108"],
    ["Mars is especially bright in the eastern sky after midnight", "astronomy", "323297551887368204", "1354533600745361809", "1744940900109"],
    ["Light pollution in the city makes it hard to see anything but the brightest stars", "astronomy", "302923939154493441", "1354533600745361809", "1744940900110"],
    ["The Hubble Deep Field image shows galaxies from billions of years ago", "astronomy", "446721983437619245", "1354533600745361809", "1744940900111"],
    ["My star chart app helps me identify constellations I never noticed before", "astronomy", "323297551887368204", "1354533600745361809", "1744940900112"],
    ["Venus is visible as the evening star just after sunset today", "astronomy", "302923939154493441", "1354533600745361809", "1744940900113"],
    ["Black holes are regions where gravity is so strong that nothing can escape", "astronomy", "446721983437619245", "1354533600745361809", "1744940900114"],
    ["I'm planning to observe the Orion Nebula tonight if clouds clear", "astronomy", "323297551887368204", "1354533600745361809", "1744940900115"],
    ["The phases of the moon affect when certain objects are best visible", "astronomy", "302923939154493441", "1354533600745361809", "1744940900116"],
    ["Supernovas occur when massive stars collapse at the end of their lifecycle", "astronomy", "446721983437619245", "1354533600745361809", "1744940900117"],
    ["Saturn's rings are tilted at a good angle for viewing this season", "astronomy", "323297551887368204", "1354533600745361809", "1744940900118"],
    ["I need a better eyepiece to resolve the cloud bands on Jupiter", "astronomy", "302923939154493441", "1354533600745361809", "1744940900119"],
    ["The cosmic microwave background radiation is evidence of the Big Bang", "astronomy", "446721983437619245", "1354533600745361809", "1744940900120"],
    ["Comet watching requires patience and dark skies", "astronomy", "323297551887368204", "1354533600745361809", "1744940900121"],
    ["Our galaxy, the Milky Way, is a barred spiral galaxy", "astronomy", "302923939154493441", "1354533600745361809", "1744940900122"],
    ["Exoplanets are planets that orbit stars outside our solar system", "astronomy", "446721983437619245", "1354533600745361809", "1744940900123"],
    ["The Summer Triangle is formed by the stars Vega, Deneb, and Altair", "astronomy", "323297551887368204", "1354533600745361809", "1744940900124"],
    ["Red dwarf stars are the most common type in our galaxy", "astronomy", "302923939154493441", "1354533600745361809", "1744940900125"],
    ["Light from distant galaxies takes millions or billions of years to reach us", "astronomy", "446721983437619245", "1354533600745361809", "1744940900126"],
    ["Planetary nebulae form when stars like our Sun die and shed their outer layers", "astronomy", "323297551887368204", "1354533600745361809", "1744940900127"],
    ["The North Star, Polaris, stays nearly fixed in our sky as Earth rotates", "astronomy", "302923939154493441", "1354533600745361809", "1744940900128"],
    ["Gravitational lensing occurs when massive objects bend light from distant sources", "astronomy", "446721983437619245", "1354533600745361809", "1744940900129"],
        
        # Gaming category - significantly expanded
    ["Just reached Diamond rank after grinding for months", "gaming", "446721983437619245", "1354533600745361809", "1744941900100"],
    ["The new DLC adds three new playable characters to the roster", "gaming", "323297551887368204", "1354533600745361809", "1744941900101"],
    ["My gaming setup needs a better GPU to run this at max settings", "gaming", "302923939154493441", "1354533600745361809", "1744941900102"],
    ["The boss fight in the final dungeon is nearly impossible solo", "gaming", "446721983437619245", "1354533600745361809", "1744941900103"],
    ["Just pre-ordered the collector's edition with exclusive in-game items", "gaming", "323297551887368204", "1354533600745361809", "1744941900104"],
    ["My keyboard macros give me an edge in MMO combat", "gaming", "302923939154493441", "1354533600745361809", "1744941900105"],
    ["The crafting system in this game is so complex but rewarding", "gaming", "446721983437619245", "1354533600745361809", "1744941900106"],
    ["Anyone want to join my party for the raid tonight?", "gaming", "323297551887368204", "1354533600745361809", "1744941900107"],
    ["The developer nerfed my favorite weapon in the latest patch", "gaming", "302923939154493441", "1354533600745361809", "1744941900108"],
    ["I spent three hours farming materials for this legendary item", "gaming", "446721983437619245", "1354533600745361809", "1744941900109"],
    ["The community mod adds so many quality of life improvements", "gaming", "323297551887368204", "1354533600745361809", "1744941900110"],
    ["My stream schedule includes speedruns every Saturday", "gaming", "302923939154493441", "1354533600745361809", "1744941900111"],
    ["The end-game content lacks challenge for veteran players", "gaming", "446721983437619245", "1354533600745361809", "1744941900112"],
    ["Just unlocked all achievements after 200 hours of gameplay", "gaming", "323297551887368204", "1354533600745361809", "1744941900113"],
    ["My custom controller bindings help with advanced movement tech", "gaming", "302923939154493441", "1354533600745361809", "1744941900114"],
    ["The procedurally generated levels keep each playthrough fresh", "gaming", "446721983437619245", "1354533600745361809", "1744941900115"],
    ["I'm stuck on this puzzle, any hints without spoilers?", "gaming", "323297551887368204", "1354533600745361809", "1744941900116"],
    ["The open world feels alive with dynamic events and NPCs", "gaming", "302923939154493441", "1354533600745361809", "1744941900117"],
    ["My gaming chair makes long sessions much more comfortable", "gaming", "446721983437619245", "1354533600745361809", "1744941900118"],
    ["The lore in this game is deep and well-written", "gaming", "323297551887368204", "1354533600745361809", "1744941900119"],
    ["Having lag issues during peak hours, need to upgrade my internet", "gaming", "302923939154493441", "1354533600745361809", "1744941900120"],
    ["The inventory management system needs a complete overhaul", "gaming", "446721983437619245", "1354533600745361809", "1744941900121"],
    ["Found an exploit that lets you duplicate rare items", "gaming", "323297551887368204", "1354533600745361809", "1744941900122"],
    ["The art style makes up for the dated graphics engine", "gaming", "302923939154493441", "1354533600745361809", "1744941900123"],
    ["My clan is recruiting active players for weekly tournaments", "gaming", "446721983437619245", "1354533600745361809", "1744941900124"],
    ["The permadeath feature adds so much tension to every encounter", "gaming", "323297551887368204", "1354533600745361809", "1744941900125"],
    ["Need to optimize my build for the new meta after the patch", "gaming", "302923939154493441", "1354533600745361809", "1744941900126"],
    ["The soundtrack in this game is absolutely phenomenal", "gaming", "446721983437619245", "1354533600745361809", "1744941900127"],
    ["Just beat my personal best time for this speedrun category", "gaming", "323297551887368204", "1354533600745361809", "1744941900128"],
    ["The tutorial doesn't explain half the mechanics you need to know", "gaming", "302923939154493441", "1354533600745361809", "1744941900129"],
        
        # Cooking category - expanded
    ["My sourdough starter is finally active after a week of feeding", "cooking", "302923939154493441", "1354533600745361809", "1744941900100"],
    ["The slow cooker makes tender pulled pork with minimal effort", "cooking", "446721983437619245", "1354533600745361809", "1744941900101"],
    ["I finally mastered the technique for perfectly seared scallops", "cooking", "323297551887368204", "1354533600745361809", "1744941900102"],
    ["What's your favorite spice blend for roast chicken?", "cooking", "302923939154493441", "1354533600745361809", "1744941900103"],
    ["My homemade pasta sauce simmers for hours to develop flavor", "cooking", "446721983437619245", "1354533600745361809", "1744941900104"],
    ["The secret to fluffy pancakes is not overmixing the batter", "cooking", "323297551887368204", "1354533600745361809", "1744941900105"],
    ["I'm experimenting with fermented kimchi and hot sauce", "cooking", "302923939154493441", "1354533600745361809", "1744941900106"],
    ["My new chef's knife makes chopping vegetables so much easier", "cooking", "446721983437619245", "1354533600745361809", "1744941900107"],
    ["The recipe calls for saffron but it's so expensive", "cooking", "323297551887368204", "1354533600745361809", "1744941900108"],
    ["Low and slow is the key to perfect barbecue ribs", "cooking", "302923939154493441", "1354533600745361809", "1744941900109"],
    ["I made duck confit for the first time yesterday", "cooking", "446721983437619245", "1354533600745361809", "1744941900110"],
    ["My pie crust always breaks when I transfer it to the pan", "cooking", "323297551887368204", "1354533600745361809", "1744941900111"],
    ["Fresh herbs make such a difference in homemade soup", "cooking", "302923939154493441", "1354533600745361809", "1744941900112"],
    ["The key to crispy roast potatoes is parboiling them first", "cooking", "446721983437619245", "1354533600745361809", "1744941900113"],
    ["I'm taking a knife skills class at the culinary institute", "cooking", "323297551887368204", "1354533600745361809", "1744941900114"],
    ["My kitchen scale helps with precise baking measurements", "cooking", "302923939154493441", "1354533600745361809", "1744941900115"],
    ["Properly resting meat after cooking makes it juicier", "cooking", "446721983437619245", "1354533600745361809", "1744941900116"],
    ["I'm working on perfecting my hollandaise sauce technique", "cooking", "323297551887368204", "1354533600745361809", "1744941900117"],
    ["Homemade stock is worth the effort for risotto", "cooking", "302923939154493441", "1354533600745361809", "1744941900118"],
    ["Does anyone have tips for keeping macarons from cracking?", "cooking", "446721983437619245", "1354533600745361809", "1744941900119"],
    ["My bread didn't rise properly, might be old yeast", "cooking", "323297551887368204", "1354533600745361809", "1744941900120"],
    ["The sous vide makes perfect medium-rare steak every time", "cooking", "302923939154493441", "1354533600745361809", "1744941900121"],
    ["I just got a pasta roller attachment for my stand mixer", "cooking", "446721983437619245", "1354533600745361809", "1744941900122"],
    ["Marinating overnight really enhances the flavor of the chicken", "cooking", "323297551887368204", "1354533600745361809", "1744941900123"],
    ["My first attempt at croissants was time-consuming but worth it", "cooking", "302923939154493441", "1354533600745361809", "1744941900124"],
    ["The temperature of butter matters so much in baking", "cooking", "446721983437619245", "1354533600745361809", "1744941900125"],
    ["I need a better ventilation system for high-heat wok cooking", "cooking", "323297551887368204", "1354533600745361809", "1744941900126"],
    ["Salt, fat, acid, heat are the essential elements to balance", "cooking", "302923939154493441", "1354533600745361809", "1744941900127"],
    ["My homemade ice cream has much better texture than store-bought", "cooking", "446721983437619245", "1354533600745361809", "1744941900128"],
    ["The food processor saves so much time with prep work", "cooking", "323297551887368204", "1354533600745361809", "1744941900129"],
        
        # Outdoors category
    ["The mountain trails are gorgeous after fresh snowfall", "outdoors", "446721983437619245", "1354533600745361809", "1744942900100"],
    ["My new waterproof hiking boots made a huge difference on wet trails", "outdoors", "323297551887368204", "1354533600745361809", "1744942900101"],
    ["The wildlife viewing at the national park was incredible this weekend", "outdoors", "302923939154493441", "1354533600745361809", "1744942900102"],
    ["I'm looking for recommendations on lightweight backpacking tents", "outdoors", "446721983437619245", "1354533600745361809", "1744942900103"],
    ["The autumn colors along the ridge trail are at peak brilliance now", "outdoors", "323297551887368204", "1354533600745361809", "1744942900104"],
    ["My GPS watch tracks all my trail runs with elevation data", "outdoors", "302923939154493441", "1354533600745361809", "1744942900105"],
    ["The fly fishing was excellent on the north fork of the river", "outdoors", "446721983437619245", "1354533600745361809", "1744942900106"],
    ["Does anyone have bear spray I can borrow for my backpacking trip?", "outdoors", "323297551887368204", "1354533600745361809", "1744942900107"],
    ["The rock climbing routes on the west face are challenging but fun", "outdoors", "302923939154493441", "1354533600745361809", "1744942900108"],
    ["My hammock camping setup is so comfortable in the woods", "outdoors", "446721983437619245", "1354533600745361809", "1744942900109"],
    ["The sunrise from the summit was worth the pre-dawn hike", "outdoors", "323297551887368204", "1354533600745361809", "1744942900110"],
    ["I spotted three eagles during my kayaking trip yesterday", "outdoors", "302923939154493441", "1354533600745361809", "1744942900111"],
    ["The mountain bike trails are muddy after the rain, best to wait", "outdoors", "446721983437619245", "1354533600745361809", "1744942900112"],
    ["My ultralight backpacking setup weighs less than 15 pounds total", "outdoors", "323297551887368204", "1354533600745361809", "1744942900113"],
    ["The wildflowers in the alpine meadows are blooming now", "outdoors", "302923939154493441", "1354533600745361809", "1744942900114"],
    ["I'm planning a multi-day canoe trip down the river next month", "outdoors", "446721983437619245", "1354533600745361809", "1744942900115"],
    ["The stargazing is amazing when camping far from city lights", "outdoors", "323297551887368204", "1354533600745361809", "1744942900116"],
    ["My new trail running shoes have excellent grip on rocky terrain", "outdoors", "302923939154493441", "1354533600745361809", "1744942900117"],
    ["The hot springs are a perfect reward after a day of hiking", "outdoors", "446721983437619245", "1354533600745361809", "1744942900118"],
    ["I need to reapply waterproofing to my rain jacket before the trip", "outdoors", "323297551887368204", "1354533600745361809", "1744942900119"],
    ["The cross-country skiing conditions are perfect after fresh powder", "outdoors", "302923939154493441", "1354533600745361809", "1744942900120"],
    ["My water filter is essential for backcountry camping", "outdoors", "446721983437619245", "1354533600745361809", "1744942900121"],
    ["The trail conditions were more difficult than the guidebook suggested", "outdoors", "323297551887368204", "1354533600745361809", "1744942900122"],
    ["I saw a moose and her calf while hiking near the lake", "outdoors", "302923939154493441", "1354533600745361809", "1744942900123"],
    ["My 4-season tent held up perfectly during the overnight snowstorm", "outdoors", "446721983437619245", "1354533600745361809", "1744942900124"],
    ["The fall mushroom foraging has been excellent this year", "outdoors", "323297551887368204", "1354533600745361809", "1744942900125"],
    ["My trekking poles save my knees on steep downhill sections", "outdoors", "302923939154493441", "1354533600745361809", "1744942900126"],
    ["The coastal trail offers amazing views of the ocean", "outdoors", "446721983437619245", "1354533600745361809", "1744942900127"],
    ["I need to break in these new hiking boots before the long trek", "outdoors", "323297551887368204", "1354533600745361809", "1744942900128"],
    ["The desert blooms after rain are spectacular and short-lived", "outdoors", "302923939154493441", "1354533600745361809", "1744942900129"],
        
        # Programming category - expanded with more examples
    ["My recursive function is causing a stack overflow with large inputs", "programming", "302923939154493441", "1354533600745361809", "1744942900100"],
    ["The new TypeScript features make strongly typed code much cleaner", "programming", "446721983437619245", "1354533600745361809", "1744942900101"],
    ["I'm implementing a binary search tree for faster lookups", "programming", "323297551887368204", "1354533600745361809", "1744942900102"],
    ["My CI/CD pipeline automatically deploys after successful tests", "programming", "302923939154493441", "1354533600745361809", "1744942900103"],
    ["The memory leak was caused by forgetting to close database connections", "programming", "446721983437619245", "1354533600745361809", "1744942900104"],
    ["I'm learning Rust for its memory safety guarantees", "programming", "323297551887368204", "1354533600745361809", "1744942900105"],
    ["The microservices architecture makes our system more scalable", "programming", "302923939154493441", "1354533600745361809", "1744942900106"],
    ["My pull request finally got merged after addressing review comments", "programming", "446721983437619245", "1354533600745361809", "1744942900107"],
    ["The regex pattern matches emails but not with certain domains", "programming", "323297551887368204", "1354533600745361809", "1744942900108"],
    ["Code refactoring improved performance by 40% in our benchmarks", "programming", "302923939154493441", "1354533600745361809", "1744942900109"],
    ["I'm implementing pagination to handle large data sets efficiently", "programming", "446721983437619245", "1354533600745361809", "1744942900110"],
    ["The API rate limiting is preventing our stress tests from completing", "programming", "323297551887368204", "1354533600745361809", "1744942900111"],
    ["My database query needs optimization, it's causing timeouts", "programming", "302923939154493441", "1354533600745361809", "1744942900112"],
    ["I'm learning about design patterns to write more maintainable code", "programming", "446721983437619245", "1354533600745361809", "1744942900113"],
    ["The caching layer significantly reduced our server load", "programming", "323297551887368204", "1354533600745361809", "1744942900114"],
    ["My unit tests caught a regression before it reached production", "programming", "302923939154493441", "1354533600745361809", "1744942900115"],
    ["I'm experimenting with functional programming concepts in JavaScript", "programming", "446721983437619245", "1354533600745361809", "1744942900116"],
    ["The containerized environment ensures consistent builds across machines", "programming", "323297551887368204", "1354533600745361809", "1744942900117"],
    ["My front-end component library needs better documentation", "programming", "302923939154493441", "1354533600745361809", "1744942900118"],
    ["I'm implementing websockets for real-time data updates", "programming", "446721983437619245", "1354533600745361809", "1744942900119"],
    ["The authentication middleware handles JWT validation for API routes", "programming", "323297551887368204", "1354533600745361809", "1744942900120"],
    ["My code formatter automatically fixes style issues before commits", "programming", "302923939154493441", "1354533600745361809", "1744942900121"],
    ["I'm learning about blockchain development with smart contracts", "programming", "446721983437619245", "1354533600745361809", "1744942900122"],
    ["The parallel processing implementation reduced runtime from hours to minutes", "programming", "323297551887368204", "1354533600745361809", "1744942900123"],
    ["My debugging session revealed race conditions in the concurrent code", "programming", "302923939154493441", "1354533600745361809", "1744942900124"],
    ["I'm implementing a custom logger for better error tracking", "programming", "446721983437619245", "1354533600745361809", "1744942900125"],
    ["The static code analyzer catches potential bugs before runtime", "programming", "323297551887368204", "1354533600745361809", "1744942900126"],
    ["My API documentation is auto-generated from code comments", "programming", "302923939154493441", "1354533600745361809", "1744942900127"],
    ["I'm learning about accessibility standards for web development", "programming", "446721983437619245", "1354533600745361809", "1744942900128"],
    ["The database migration scripts need thorough testing before deployment", "programming", "323297551887368204", "1354533600745361809", "1744942900129"],
        
        # Music category
    ["I just learned how to play my favorite song on guitar", "music", "302923939154493441", "1354533600745361809", "1744943900100"],
    ["The new album from that indie band exceeded all my expectations", "music", "446721983437619245", "1354533600745361809", "1744943900101"],
    ["I'm struggling with bar chords but making progress slowly", "music", "323297551887368204", "1354533600745361809", "1744943900102"],
    ["The acoustics in that new concert venue are phenomenal", "music", "302923939154493441", "1354533600745361809", "1744943900103"],
    ["I've been practicing piano scales for an hour every day", "music", "446721983437619245", "1354533600745361809", "1744943900104"],
    ["The symphony orchestra's rendition of Beethoven was moving", "music", "323297551887368204", "1354533600745361809", "1744943900105"],
    ["My new audio interface improved my home recording quality", "music", "302923939154493441", "1354533600745361809", "1744943900106"],
    ["I'm looking for bandmates who can commit to weekly rehearsals", "music", "446721983437619245", "1354533600745361809", "1744943900107"],
    ["The vinyl record has a warmth that digital streaming lacks", "music", "323297551887368204", "1354533600745361809", "1744943900108"],
    ["My vocal range has expanded after taking lessons for six months", "music", "302923939154493441", "1354533600745361809", "1744943900109"],
    ["The music festival lineup this year is absolutely stacked", "music", "446721983437619245", "1354533600745361809", "1744943900110"],
    ["I'm learning to read sheet music but it's slow going", "music", "323297551887368204", "1354533600745361809", "1744943900111"],
    ["The producer completely transformed our rough demo tracks", "music", "302923939154493441", "1354533600745361809", "1744943900112"],
    ["My drum kit needs new heads and cymbal stands", "music", "446721983437619245", "1354533600745361809", "1744943900113"],
    ["The jazz ensemble improvisation was mind-blowing last night", "music", "323297551887368204", "1354533600745361809", "1744943900114"],
    ["I'm working on writing my first original composition", "music", "302923939154493441", "1354533600745361809", "1744943900115"],
    ["The bass player really holds the rhythm section together", "music", "446721983437619245", "1354533600745361809", "1744943900116"],
    ["My favorite headphones reveal details I never noticed before", "music", "323297551887368204", "1354533600745361809", "1744943900117"],
    ["The music theory class is helping me understand chord progressions", "music", "302923939154493441", "1354533600745361809", "1744943900118"],
    ["I'm saving up for a better violin with richer tone", "music", "446721983437619245", "1354533600745361809", "1744943900119"],
    ["The choir's harmonies gave me goosebumps during the performance", "music", "323297551887368204", "1354533600745361809", "1744943900120"],
    ["My playlist for road trips has the perfect energy", "music", "302923939154493441", "1354533600745361809", "1744943900121"],
    ["I'm experimenting with different tunings on my guitar", "music", "446721983437619245", "1354533600745361809", "1744943900122"],
    ["The electronic music producer uses interesting sampling techniques", "music", "323297551887368204", "1354533600745361809", "1744943900123"],
    ["My metronome practice is improving my timing dramatically", "music", "302923939154493441", "1354533600745361809", "1744943900124"],
    ["The horn section adds so much energy to the band's sound", "music", "446721983437619245", "1354533600745361809", "1744943900125"],
    ["I use music streaming services to discover new artists", "music", "323297551887368204", "1354533600745361809", "1744943900126"],
    ["My finger picking technique is getting smoother with practice", "music", "302923939154493441", "1354533600745361809", "1744943900127"],
    ["The songwriter's lyrics are so poetic and meaningful", "music", "446721983437619245", "1354533600745361809", "1744943900128"],
    ["I'm learning to play by ear instead of relying on tabs", "music", "323297551887368204", "1354533600745361809", "1744943900129"],
    ]

# Ensure TensorFlow reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
set_seed()

# Set the data folder path
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

def text_preprocessing(text):
    """Improved text preprocessing function for NLP tasks"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions (common in messages)
    text = re.sub(r'@\w+', '', text)
    
    # Keep punctuation as it could be meaningful for sentence structure
    # Only remove excessive punctuation
    text = re.sub(r'([!?.,;:])\1+', r'\1', text)
    
    # Keep numbers as they may be relevant for context
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_n_grams(text, n=2):
    """Create n-grams from text to capture phrases"""
    words = text.split()
    n_grams = []
    for i in range(len(words) - n + 1):
        n_grams.append('_'.join(words[i:i+n]))
    return n_grams

def data_augmentation(df, augment_factor=2):
    """Enhanced data augmentation with domain-specific techniques"""
    print("Performing enhanced data augmentation...")
    augmented_data = []
    categories = df['category'].unique()
    
    original_columns = df.columns.tolist()
    
    # Define category-specific keyword replacements
    category_keywords = {
        'astronomy': {
            'telescope': ['scope', 'observatory equipment', 'stargazing tool'],
            'planet': ['celestial body', 'astronomical object', 'world'],
            'galaxy': ['star system', 'celestial formation', 'cosmic structure'],
            'constellation': ['star pattern', 'celestial formation', 'star group']
        },
        'badminton': {
            'racket': ['racquet', 'paddle', 'gear'],
            'shuttle': ['shuttlecock', 'birdie', 'feather cork'],
            'court': ['playing area', 'badminton field', 'playing space'],
            'smash': ['power hit', 'overhead shot', 'attack shot']
        },
        'gaming': {
            'game': ['title', 'video game', 'release'],
            'level': ['stage', 'map', 'zone'],
            'character': ['avatar', 'player character', 'game hero'],
            'raid': ['group mission', 'team challenge', 'dungeon run']
        },
        'cooking': {
            'recipe': ['cooking instructions', 'food preparation guide', 'dish formula'],
            'ingredient': ['component', 'food item', 'cooking element'],
            'bake': ['cook in oven', 'heat in oven', 'prepare in oven'],
            'simmer': ['cook slowly', 'heat gently', 'low heat cook']
        },
        'outdoors': {
            'hike': ['trek', 'trail walk', 'mountain walk'],
            'trail': ['path', 'route', 'outdoor way'],
            'camping': ['outdoor stay', 'tent trip', 'wilderness lodging'],
            'backpack': ['rucksack', 'hiking pack', 'trail bag']
        },
        'programming': {
            'code': ['program', 'script', 'software'],
            'function': ['method', 'routine', 'procedure'],
            'debug': ['fix issues', 'solve problems', 'find errors'],
            'api': ['interface', 'programming interface', 'service connection']
        },
        'music': {
            'guitar': ['stringed instrument', 'musical instrument', 'acoustic instrument'],
            'band': ['music group', 'musical ensemble', 'performers'],
            'chord': ['note combination', 'musical harmonic', 'tone group'],
            'melody': ['tune', 'musical phrase', 'song line']
        }
    }
    
    for _, row in df.iterrows():
        message = row['message']
        category = row['category']
        user = row['user']
        guild = row['guild']
        timestamp = row['timestamp']
        
        words = message.split()
        if len(words) > 3:
            # Apply multiple augmentation techniques based on category
            for _ in range(augment_factor):
                augmented_message = message
                
                # 1. Category-specific keyword replacement (higher probability)
                if category in category_keywords and random.random() < 0.8:
                    for keyword, replacements in category_keywords[category].items():
                        if keyword in augmented_message.lower() and random.random() < 0.7:
                            replacement = random.choice(replacements)
                            augmented_message = re.sub(r'\b' + keyword + r'\b', replacement, 
                                                     augmented_message, flags=re.IGNORECASE)
                
                # 2. Random word swapping (lower probability)
                if len(words) > 5 and random.random() < 0.3:
                    words_copy = augmented_message.split()
                    idx1, idx2 = random.sample(range(len(words_copy)), 2)
                    words_copy[idx1], words_copy[idx2] = words_copy[idx2], words_copy[idx1]
                    augmented_message = ' '.join(words_copy)
                
                # 3. Add category-specific phrases (high probability for better augmentation)
                if random.random() < 0.5:
                    category_phrases = {
                        'astronomy': [
                            " with my telescope", " under clear skies", " in the night sky",
                            " during the meteor shower", " using star charts"
                        ],
                        'badminton': [
                            " on the court", " with my racket", " during the match",
                            " with proper technique", " at the badminton club"
                        ],
                        'gaming': [
                            " in the game", " on my gaming PC", " with high FPS",
                            " during the raid", " with my gaming group"
                        ],
                        'cooking': [
                            " in the kitchen", " with fresh ingredients", " following the recipe",
                            " at the right temperature", " with proper seasoning"
                        ],
                        'outdoors': [
                            " on the trail", " during my hike", " in the wilderness",
                            " at the campsite", " in the mountains"
                        ],
                        'programming': [
                            " in my code", " while debugging", " with efficient algorithms",
                            " in the development environment", " using the API"
                        ],
                        'music': [
                            " during band practice", " with perfect pitch", " in the studio",
                            " with my guitar", " at the concert"
                        ]
                    }
                    
                    if category in category_phrases and not augmented_message.endswith('.'):
                        phrase = random.choice(category_phrases[category])
                        if not any(phrase.strip() in augmented_message.lower() for phrase in category_phrases[category]):
                            augmented_message += phrase
                
                if augmented_message != message:
                    new_timestamp = str(int(timestamp) + random.randint(1, 1000))
                    
                    # Create a new row with the same structure as the original dataframe
                    new_row = [None] * len(original_columns)
                    new_row[original_columns.index('message')] = augmented_message
                    new_row[original_columns.index('category')] = category
                    new_row[original_columns.index('user')] = user
                    new_row[original_columns.index('guild')] = guild
                    new_row[original_columns.index('timestamp')] = new_timestamp
                    
                    augmented_data.append(new_row)
    
    # Convert augmented data to DataFrame and concatenate with original
    if augmented_data:
        augmented_df = pd.DataFrame(augmented_data, columns=original_columns)
        df_combined = pd.concat([df, augmented_df], ignore_index=True)
        return df_combined
    return df

def targeted_augmentation(df, confusion_pairs):
    """
    Add more examples for categories that get confused
    confusion_pairs: list of tuples with (category1, category2) pairs
    """
    print("Performing targeted augmentation for commonly confused categories...")
    augmented_data = []
    
    original_columns = df.columns.tolist()
    
    # More distinctive examples for all commonly confused categories
    category_distinctive_examples = {
        'astronomy': [
            "My telescope captured Saturn's rings in amazing detail last night",
            "The astronomy lecture covered neutron stars and pulsars in depth",
            "Setting up my equatorial mount for astrophotography tonight",
            "The planetarium show displayed the life cycle of massive stars",
            "Using star charts to identify constellations in the summer sky",
            "The observatory's 20-inch reflector telescope is amazing for deep sky objects",
            "Learning about spectroscopy to analyze stellar compositions",
            "The red shift of distant galaxies proves the universe is expanding",
            "My CCD camera captures much better nebula details than my DSLR",
            "The solar filter allows safe viewing of sunspots and prominences"
        ],
        'outdoors': [
            "The hiking trail has several stream crossings requiring waterproof boots",
            "Setting up our tent on a high spot to avoid water if it rains",
            "My backpacking gear weighs under 20 pounds for multi-day trips",
            "The mountain summit offers panoramic views of three states",
            "Rock climbing requires careful route planning and proper protection",
            "The forest trail is marked with blue blazes for navigation",
            "Kayaking through the rapids requires proper paddling technique",
            "My hiking poles reduce impact on knee joints during steep descents",
            "The campsite has designated fire rings and bear-proof containers",
            "The waterfall at mile 7 is the highlight of this wilderness trail"
        ],
        'music': [
            "The bass guitar riff drives the rhythm section in this song",
            "Our band rehearses twice weekly before recording sessions",
            "The chorus has four-part harmony with complex voice leading",
            "My new amplifier gives a much cleaner tone for jazz playing",
            "The melody modulates from C major to A minor in the bridge",
            "Recording multiple vocal takes helps find the best performance",
            "The symphonic arrangement features woodwinds and brass sections",
            "Learning chord theory helped improve my songwriting skills",
            "The acoustic treatment in the studio eliminated unwanted reverb",
            "The guitar solo uses pentatonic scales over the chord progression"
        ],
        'cooking': [
            "The reduction sauce needs to simmer for at least 30 minutes",
            "Kneading the bread dough develops gluten structure for proper rise",
            "My chef's knife needs sharpening for clean vegetable cuts",
            "Fermenting the kimchi in clay pots enhances flavor development",
            "The sourdough starter needs feeding every 12 hours at room temperature",
            "Sous vide cooking gives precise temperature control for perfect steaks",
            "My cast iron skillet requires proper seasoning for non-stick cooking",
            "The roux needs to reach a dark brown color for authentic gumbo",
            "Blanching vegetables preserves color and texture before freezing",
            "The pastry dough needs to rest in the refrigerator before rolling"
        ],
        'gaming': [
            "The raid boss has a two-phase battle with different mechanics",
            "My character build focuses on critical damage and attack speed",
            "The game's latest patch nerfed several overpowered abilities",
            "Our guild organizes weekly events for team coordination practice",
            "My gaming setup includes a mechanical keyboard and high-refresh monitor",
            "The speedrun requires precise movement and glitch exploitation",
            "Farming legendary items requires efficient dungeon route planning",
            "The game's matchmaking system pairs players of similar skill levels",
            "The open world has dynamic events that change based on player actions",
            "My mod loadout improves inventory management and user interface"
        ],
        'programming': [
            "My recursive function optimizes path finding through memoization",
            "The API documentation lacks examples for authentication methods",
            "Refactoring the codebase reduced technical debt significantly",
            "My pull request implements a more efficient search algorithm",
            "The Docker container ensures consistent development environments",
            "Debugging multithreaded code requires careful race condition analysis",
            "The database query performance improved with proper indexing",
            "Unit tests caught several edge cases before production deployment",
            "My CI/CD pipeline automates testing and deployment workflows",
            "Learning design patterns helped improve code maintainability"
        ],
        'badminton': [
            "My backhand slice creates deceptive shots near the net",
            "The tournament uses rally scoring with 21 points per game",
            "Proper footwork is essential for court coverage in singles",
            "My racket string tension is set higher for better control",
            "The defensive clear shot should reach the opponent's back line",
            "Mixed doubles requires good communication between partners",
            "The jump smash generates more power but requires precise timing",
            "Training includes shadow badminton for movement patterns",
            "The shuttle's flight path changes drastically with different shots",
            "Grip changing technique is essential for versatile shot making"
        ]
    }
    
    # Add examples for all categories that appear in confusion pairs
    confusion_categories = set()
    for cat1, cat2 in confusion_pairs:
        confusion_categories.add(cat1)
        confusion_categories.add(cat2)
    
    for category in confusion_categories:
        if category in category_distinctive_examples:
            for example in category_distinctive_examples[category]:
                timestamp = str(int(time.time() * 1000) + random.randint(1, 1000))
                user = df['user'].sample(1).iloc[0]
                guild = df['guild'].sample(1).iloc[0]
                
                new_row = [None] * len(original_columns)
                new_row[original_columns.index('message')] = example
                new_row[original_columns.index('category')] = category
                new_row[original_columns.index('user')] = user
                new_row[original_columns.index('guild')] = guild
                new_row[original_columns.index('timestamp')] = timestamp
                
                augmented_data.append(new_row)
    
    # Convert augmented data to DataFrame and concatenate with original
    if augmented_data:
        augmented_df = pd.DataFrame(augmented_data, columns=original_columns)
        return pd.concat([df, augmented_df], ignore_index=True)
    return df

def extract_advanced_features(df):
    """Extract rich features to improve classification accuracy"""
    print("Extracting advanced features...")
    
    # Basic features
    df['message_length'] = df['message'].apply(len)
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))
    df['avg_word_length'] = df['message'].apply(lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0)
    
    # Create specialized feature extractors
    def count_punctuation(text):
        return sum(1 for c in text if c in '.,!?;:')
    
    def count_special_chars(text):
        return sum(1 for c in text if c in '@#$%^&*()-_=+[]{}|\\/<>')
    
    def count_uppercase(text):
        return sum(1 for c in text if c.isupper())
    
    def count_digits(text):
        return sum(1 for c in text if c.isdigit())
    
    def extract_bigrams(text):
        words = text.lower().split()
        return ['_'.join(words[i:i+2]) for i in range(len(words)-1)]
    
    # Add character-level features
    df['punctuation_count'] = df['message'].apply(count_punctuation)
    df['special_char_count'] = df['message'].apply(count_special_chars)
    df['uppercase_count'] = df['message'].apply(count_uppercase)
    df['digit_count'] = df['message'].apply(count_digits)
    
    # Ratio features (normalize by message length)
    df['punctuation_ratio'] = df['punctuation_count'] / df['message_length'].apply(lambda x: max(x, 1))
    df['special_char_ratio'] = df['special_char_count'] / df['message_length'].apply(lambda x: max(x, 1))
    df['uppercase_ratio'] = df['uppercase_count'] / df['message_length'].apply(lambda x: max(x, 1))
    df['digit_ratio'] = df['digit_count'] / df['message_length'].apply(lambda x: max(x, 1))
    
    # POS features if nltk is available
    try:
        import nltk
        from nltk.corpus import stopwords
        
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        # More detailed POS features
        def get_pos_counts(text):
            tokens = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            
            pos_counts = {
                'nouns': 0,      # NN, NNS, NNP, NNPS
                'verbs': 0,      # VB, VBD, VBG, VBN, VBP, VBZ
                'adjectives': 0, # JJ, JJR, JJS
                'adverbs': 0,    # RB, RBR, RBS
                'pronouns': 0,   # PRP, PRP$, WP, WP$
                'prepositions': 0, # IN
                'determiners': 0,  # DT, PDT, WDT
                'conjunctions': 0,  # CC
                'interjections': 0, # UH
                'numbers': 0      # CD
            }
            
            for word, tag in pos_tags:
                if tag.startswith('NN'):
                    pos_counts['nouns'] += 1
                elif tag.startswith('VB'):
                    pos_counts['verbs'] += 1
                elif tag.startswith('JJ'):
                    pos_counts['adjectives'] += 1
                elif tag.startswith('RB'):
                    pos_counts['adverbs'] += 1
                elif tag in ('PRP', 'PRP$', 'WP', 'WP$'):
                    pos_counts['pronouns'] += 1
                elif tag == 'IN':
                    pos_counts['prepositions'] += 1
                elif tag in ('DT', 'PDT', 'WDT'):
                    pos_counts['determiners'] += 1
                elif tag == 'CC':
                    pos_counts['conjunctions'] += 1
                elif tag == 'UH':
                    pos_counts['interjections'] += 1
                elif tag == 'CD':
                    pos_counts['numbers'] += 1
            
            return pos_counts
        
        # Process each message
        pos_features = df['message'].apply(get_pos_counts)
        
        # Extract each POS feature
        for pos_category in ['nouns', 'verbs', 'adjectives', 'adverbs', 'pronouns', 
                            'prepositions', 'determiners', 'conjunctions', 'interjections', 'numbers']:
            df[f'{pos_category}_count'] = pos_features.apply(lambda x: x[pos_category])
            df[f'{pos_category}_ratio'] = df[f'{pos_category}_count'] / df['word_count'].apply(lambda x: max(x, 1))
        
        # Stopword features
        stop_words = set(stopwords.words('english'))
        df['stopword_count'] = df['message'].apply(
            lambda x: sum(1 for word in nltk.word_tokenize(x.lower()) if word in stop_words)
        )
        df['stopword_ratio'] = df['stopword_count'] / df['word_count'].apply(lambda x: max(x, 1))
        
    except Exception as e:
        print(f"NLTK processing error: {e}. Skipping advanced POS features.")
    
    # N-gram features (using simple word presence)
    category_keywords = {
        'badminton': ['racket', 'shuttle', 'court', 'smash', 'net', 'serve', 'backhand', 'forehand'],
        'astronomy': ['telescope', 'star', 'planet', 'galaxy', 'moon', 'constellation', 'orbit', 'cosmic'],
        'gaming': ['game', 'level', 'character', 'quest', 'multiplayer', 'raid', 'guild', 'loot'],
        'cooking': ['recipe', 'ingredient', 'bake', 'kitchen', 'cook', 'flavor', 'dish', 'oven'],
        'outdoors': ['hike', 'trail', 'mountain', 'camping', 'outdoor', 'wilderness', 'tent', 'forest'],
        'programming': ['code', 'function', 'bug', 'api', 'algorithm', 'database', 'debug', 'framework'],
        'music': ['guitar', 'band', 'song', 'chord', 'melody', 'concert', 'rhythm', 'instrument']
    }
    
    # Create keyword presence features
    for category, keywords in category_keywords.items():
        feature_name = f'{category}_keyword_count'
        df[feature_name] = df['message'].apply(
            lambda x: sum(1 for keyword in keywords if keyword.lower() in x.lower().split())
        )
    
    # Add bigram features for each category
    all_messages = ' '.join(df['message']).lower()
    for category in df['category'].unique():
        category_messages = ' '.join(df[df['category'] == category]['message']).lower()
        
        # Get top bigrams for this category
        category_bigrams = []
        for msg in df[df['category'] == category]['message']:
            category_bigrams.extend(extract_bigrams(msg))
        
        # Count frequency
        bigram_counter = Counter(category_bigrams)
        
        # Get top 5 distinctive bigrams (that appear mostly in this category)
        distinctive_bigrams = []
        for bigram, count in bigram_counter.most_common(15):
            # Check if this bigram is more common in this category than others
            if bigram in all_messages:
                category_freq = category_messages.count(bigram.replace('_', ' ')) / len(category_messages) if len(category_messages) > 0 else 0
                all_freq = all_messages.count(bigram.replace('_', ' ')) / len(all_messages) if len(all_messages) > 0 else 0
                
                if category_freq > 2 * all_freq and len(distinctive_bigrams) < 5:
                    distinctive_bigrams.append(bigram)
        
        # Create feature for presence of distinctive bigrams
        for i, bigram in enumerate(distinctive_bigrams):
            feature_name = f'{category}_bigram_{i}'
            bigram_words = bigram.split('_')
            df[feature_name] = df['message'].apply(
                lambda x: 1 if ' '.join(bigram_words) in x.lower() else 0
            )
    
    return df

def remove_near_duplicates(df, threshold=0.9):
    """Remove messages that are too similar to avoid memorization"""
    print("Removing near-duplicate messages...")
    messages = df['message'].tolist()
    
    try:
        # Create TF-IDF vectors
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(messages)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find duplicates, but preserve category balance
        duplicates = set()
        category_counts = df['category'].value_counts()
        min_category_count = category_counts.min()
        
        for i in range(len(similarity_matrix)):
            category_i = df.iloc[i]['category']
            
            # Only consider removing if this category has more than minimum examples
            if category_counts[category_i] > min_category_count:
                for j in range(i+1, len(similarity_matrix)):
                    if j not in duplicates and similarity_matrix[i, j] > threshold:
                        category_j = df.iloc[j]['category']
                        
                        # If both are from same category and this category has extras
                        if category_i == category_j and category_counts[category_i] > min_category_count:
                            duplicates.add(j)
                            category_counts[category_i] -= 1
        
        # Filter dataframe
        print(f"Found {len(duplicates)} near-duplicate messages to remove.")
        keep_indices = [i for i in range(len(df)) if i not in duplicates]
        return df.iloc[keep_indices].reset_index(drop=True)
    except Exception as e:
        print(f"Error in duplicate removal: {e}. Returning original dataset.")
        return df

def build_advanced_model(vocab_size, embedding_dim, max_sequence_length, num_classes, embedding_matrix=None):
    """
    Build an advanced model with parallel convolutional layers for different n-gram sizes
    """
    # Input layer
    input_layer = Input(shape=(max_sequence_length,))
    
    # Embedding layer
    if embedding_matrix is not None:
        embedding = Embedding(
            vocab_size, 
            embedding_dim,
            weights=[embedding_matrix],
            trainable=True,
            embeddings_regularizer=l2(0.0001)
        )(input_layer)
    else:
        embedding = Embedding(
            vocab_size,
            embedding_dim,
            embeddings_regularizer=l2(0.0001)
        )(input_layer)
    
    # Apply spatial dropout to embeddings
    embedding_dropout = SpatialDropout1D(0.2)(embedding)
    
    # Parallel convolutional layers for capturing different n-gram patterns
    conv1 = Conv1D(filters=64, kernel_size=2, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(embedding_dropout)
    conv2 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(embedding_dropout)
    conv3 = Conv1D(filters=64, kernel_size=4, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(embedding_dropout)
    
    # Max pooling for each convolutional layer
    pool1 = GlobalMaxPooling1D()(conv1)
    pool2 = GlobalMaxPooling1D()(conv2)
    pool3 = GlobalMaxPooling1D()(conv3)
    
    # Bi-directional LSTM for sequence understanding
    lstm = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.0001)))(embedding_dropout)
    lstm_pool = GlobalMaxPooling1D()(lstm)
    
    # Concatenate all features
    concatenated = Concatenate()([pool1, pool2, pool3, lstm_pool])
    
    # Add batch normalization
    batch_norm = BatchNormalization()(concatenated)
    
    # Dense layers for classification
    dense1 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(batch_norm)
    dropout1 = Dropout(0.3)(dense1)
    
    # Second dense layer
    dense2 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(dropout1)
    dropout2 = Dropout(0.3)(dense2)
    
    # Output layer
    output_layer = Dense(num_classes, activation='softmax')(dropout2)
    
    # Create and compile the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Use Adam optimizer with learning rate and clipnorm
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model

def custom_lr_schedule(epoch):
    """Learning rate schedule with warmup and decay phases"""
    if epoch < 3:
        return 0.0001 * (epoch + 1) / 3  # Gradual warmup
    elif epoch < 10:
        return 0.001  # Maintain initial rate
    elif epoch < 20:
        return 0.0005  # First step down
    elif epoch < 30:
        return 0.0001  # Second step down
    else:
        return 0.00005  # Final low rate for fine tuning

def weighted_categorical_crossentropy(class_weights):
    """Custom loss function to apply class weights"""
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Scale predictions so they don't underflow
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # Apply class weights
        loss = y_true * tf.math.log(y_pred) * class_weights
        return -tf.reduce_sum(loss, axis=-1)
    return loss

def create_ensemble(X_padded, y_one_hot, embedding_matrix, vocab_size, embedding_dim, 
                   max_sequence_length, num_classes, X_val=None, y_val=None, num_models=3):
    """Create an ensemble of models with different architectures"""
    print(f"Training ensemble of {num_models} models...")
    models = []
    
    for i in range(num_models):
        print(f"\nTraining ensemble model {i+1}/{num_models}")
        
        # Set different random seed for each model
        set_seed(42 + i*10)
        
        # Create models with different architectures
        if i == 0:
            # Model 1: CNN-focused with multiple kernel sizes
            input_layer = Input(shape=(max_sequence_length,))
            
            if embedding_matrix is not None:
                embedding = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=True)(input_layer)
            else:
                embedding = Embedding(vocab_size, embedding_dim)(input_layer)
                
            x = SpatialDropout1D(0.2)(embedding)
            
            # Multiple kernel sizes to capture different n-gram patterns
            conv1 = Conv1D(filters=64, kernel_size=2, padding='same', activation='relu')(x)
            conv2 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
            conv3 = Conv1D(filters=64, kernel_size=4, padding='same', activation='relu')(x)
            
            # Global max pooling
            pool1 = GlobalMaxPooling1D()(conv1)
            pool2 = GlobalMaxPooling1D()(conv2)
            pool3 = GlobalMaxPooling1D()(conv3)
            
            # Concatenate
            concat = Concatenate()([pool1, pool2, pool3])
            
            # Dense layers
            x = Dense(128, activation='relu')(concat)
            x = Dropout(0.3)(x)
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.3)(x)
            output_layer = Dense(num_classes, activation='softmax')(x)
            
            model = Model(inputs=input_layer, outputs=output_layer)
        
        elif i == 1:
            # Model 2: LSTM-focused
            input_layer = Input(shape=(max_sequence_length,))
            
            if embedding_matrix is not None:
                embedding = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=True)(input_layer)
            else:
                embedding = Embedding(vocab_size, embedding_dim)(input_layer)
                
            x = SpatialDropout1D(0.3)(embedding)
            
            # Stacked BiLSTM
            x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
            x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)
            
            # Dense layers
            x = Dense(128, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.4)(x)
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.3)(x)
            output_layer = Dense(num_classes, activation='softmax')(x)
            
            model = Model(inputs=input_layer, outputs=output_layer)
        
        else:
            # Model 3: Hybrid CNN-LSTM with attention-like mechanism
            input_layer = Input(shape=(max_sequence_length,))
            
            if embedding_matrix is not None:
                embedding = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=True)(input_layer)
            else:
                embedding = Embedding(vocab_size, embedding_dim)(input_layer)
                
            x = SpatialDropout1D(0.25)(embedding)
            
            # CNN branch
            conv = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
            
            # LSTM branch with residual connection
            lstm = Bidirectional(LSTM(48, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
            
            # Combine
            combined = Concatenate()([conv, lstm])
            
            # Global max pooling
            pooled = GlobalMaxPooling1D()(combined)
            
            # Dense layers with batch normalization
            x = Dense(128, activation='relu')(pooled)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            x = Dense(64, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            output_layer = Dense(num_classes, activation='softmax')(x)
            
            model = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile model with slightly different learning rates
        lr = 0.001 - (i * 0.0002)  # Slightly different learning rate for each model
        opt = Adam(learning_rate=lr)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=10,
                restore_best_weights=True,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5, 
                min_lr=0.00001,
                verbose=1
            ),
            LearningRateScheduler(custom_lr_schedule)
        ]
        
        # Train model
        batch_size = 16
        
        if X_val is not None and y_val is not None:
            history = model.fit(
                X_padded, y_one_hot,
                epochs=100,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )
        else:
            # Use a validation split if validation data not provided
            history = model.fit(
                X_padded, y_one_hot,
                epochs=100,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )
        
        models.append(model)
    
    return models

def ensemble_predict(models, X_test):
    """Combine predictions from multiple models with soft voting"""
    predictions = []
    
    for i, model in enumerate(models):
        print(f"Getting predictions from model {i+1}/{len(models)}")
        pred = model.predict(X_test)
        predictions.append(pred)
    
    # Weighted average predictions (give slightly more weight to better models)
    weights = [1.2, 1.0, 0.8]  # Assuming first model is best
    weighted_predictions = []
    
    for i, pred in enumerate(predictions):
        weight = weights[i] if i < len(weights) else 1.0
        weighted_predictions.append(pred * weight)
    
    # Compute weighted average
    ensemble_pred = np.sum(weighted_predictions, axis=0) / np.sum(weights[:len(predictions)])
    return ensemble_pred

def train_cross_validation(guild_id):
    """
    Train a highly accurate model using cross-validation to ensure robustness
    """
    # Guild ID for the dataset
    model_save_path = f"{DATA_FOLDER}/{guild_id}"
    pathlib.Path(model_save_path).mkdir(parents=True, exist_ok=True)
    
    # Check if data file exists
    data_file = f"{DATA_FOLDER}/{guild_id}.csv"
    if not os.path.exists(data_file):
        print("Data file not found. Creating test data...")
        # Note: create_test_csv() function should be defined elsewhere in your code
        data_file = create_test_csv()
    
    # Load the data
    print("Loading and preprocessing data...")
    df = pd.read_csv(data_file)
    print(f"Original dataset size: {len(df)} examples")
    
    # Display initial category distribution
    print("\nCategory distribution in original dataset:")
    category_counts = df['category'].value_counts()
    for category, count in category_counts.items():
        print(f"{category}: {count} examples ({count/len(df)*100:.1f}%)")
    
    # Add advanced features
    df = extract_advanced_features(df)
    
    # Apply data augmentation to increase dataset size and balance classes
    augment_factor = 2
    df = data_augmentation(df, augment_factor=augment_factor)
    print(f"Dataset size after basic augmentation: {len(df)} examples")
    
    # Apply targeted augmentation for commonly confused categories
    confusion_pairs = [
        ('music', 'astronomy'), 
        ('cooking', 'music'),
        ('astronomy', 'outdoors'),
        ('programming', 'gaming'),
        ('badminton', 'outdoors')
    ]
    df = targeted_augmentation(df, confusion_pairs)
    print(f"Dataset size after targeted augmentation: {len(df)} examples")
    
    # Display augmented category distribution
    print("\nCategory distribution after augmentation:")
    category_counts = df['category'].value_counts()
    for category, count in category_counts.items():
        print(f"{category}: {count} examples ({count/len(df)*100:.1f}%)")
    
    # Remove near-duplicates to prevent memorization
    df = remove_near_duplicates(df, threshold=0.92)  # Higher threshold for less aggressive filtering
    print(f"Dataset size after removing near-duplicates: {len(df)} examples")
    
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
    max_words = 20000  # Increased vocabulary size
    max_sequence_length = 80  # Adjusted sequence length
    embedding_dim = 100
    num_classes = len(categories)
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
    print(f"Vocabulary size: {vocab_size}")
    
    # Try to load pre-trained embeddings (if available)
    embedding_matrix = None
    try:
        # Load GloVe embeddings
        print("Trying to load GloVe embeddings...")
        embeddings_index = {}
        glove_path = 'glove.6B.100d.txt'
        
        if os.path.exists(glove_path):
            print(f"Loading embeddings from {glove_path}")
            with open(glove_path, encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
                    
            print(f"Found {len(embeddings_index)} word vectors in GloVe file.")
            
            # Prepare embedding matrix
            embedding_matrix = np.zeros((vocab_size, embedding_dim))
            hits = 0
            misses = 0
            
            for word, i in tokenizer.word_index.items():
                if i >= vocab_size:
                    continue
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # Words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
                    hits += 1
                else:
                    misses += 1
                    
            print(f"Converted {hits} words ({hits / min(vocab_size, len(tokenizer.word_index)) * 100:.1f}%).")
            print(f"Missing {misses} words.")
        else:
            print("GloVe embeddings file not found. Training with random embeddings.")
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        print("Proceeding with random initialization for embeddings.")
    
    # Convert to one-hot encoding for categorical loss
    y_one_hot = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)
    
    # Compute class weights to handle imbalanced data
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_encoded),
        y=y_encoded
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    print("\nClass weights to handle imbalance:")
    for i, weight in class_weight_dict.items():
        print(f"Class {label_encoder.inverse_transform([i])[0]}: {weight:.2f}")
    
    # Use stratified k-fold to maintain category balance
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Start K-fold cross-validation
    print(f"\nStarting {n_splits}-fold cross-validation...")
    fold_no = 1
    fold_accuracies = []
    fold_val_losses = []
    
    best_accuracy = 0
    best_val_loss = float('inf')
    best_model_path = f"{model_save_path}/best_model"
    
    # For ensemble modeling
    ensemble_models = []
    all_X_val = []
    all_y_val = []
    all_val_indices = []
    
    for train_idx, val_idx in stratified_kfold.split(X_padded, y_encoded):
        print(f"\n----- Training Fold {fold_no}/{n_splits} -----")
        
        # Split data
        X_train, X_val = X_padded[train_idx], X_padded[val_idx]
        y_train, y_val = y_one_hot[train_idx], y_one_hot[val_idx]
        
        # Store validation data for later ensemble evaluation
        all_X_val.append(X_val)
        all_y_val.append(y_val)
        all_val_indices.append(val_idx)
        
        # Build advanced model
        model = build_advanced_model(vocab_size, embedding_dim, max_sequence_length, num_classes, embedding_matrix)
        
        if fold_no == 1:
            print("Model Architecture:")
            model.summary()
        
        # Define callbacks with model checkpoint
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=12,  # Increased patience
                restore_best_weights=True,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5, 
                min_lr=0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f"{model_save_path}/fold_{fold_no}_best.keras",
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            LearningRateScheduler(custom_lr_schedule)
        ]
        
        # Train the model
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            epochs=100,  # Maximum epochs
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            class_weight=class_weight_dict,
            shuffle=True
        )
        training_time = time.time() - start_time       
        
        # Evaluate on validation set
        val_loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"Fold {fold_no} - Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        
        # Save accuracy and loss
        fold_accuracies.append(accuracy)
        fold_val_losses.append(val_loss)
        
        # Save the model if it's the best so far (by validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_accuracy = accuracy
            model.save(best_model_path + '.keras')
            print(f"New best model saved with val_loss: {best_val_loss:.4f}, accuracy: {best_accuracy:.4f}")
        
        # Save the model for ensemble
        model.save(f"{model_save_path}/fold_{fold_no}.keras")
        ensemble_models.append(model)
        
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
    print(f"Validation loss for all folds: {[f'{loss:.4f}' for loss in fold_val_losses]}")
    print(f"Average accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Best fold accuracy: {best_accuracy:.4f}")
    
    # Create ensemble prediction model
    print("\nTraining ensemble model...")
    # Split data for ensemble training
    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
        X_padded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    y_train_one_hot = tf.keras.utils.to_categorical(y_train_encoded, num_classes=num_classes)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test_encoded, num_classes=num_classes)
    
    # Train ensemble of diverse models
    ensemble_models = create_ensemble(
        X_train, y_train_one_hot, embedding_matrix, vocab_size, embedding_dim,
        max_sequence_length, num_classes, X_test, y_test_one_hot, num_models=3
    )
    
    # Save ensemble models
    for i, model in enumerate(ensemble_models):
        model.save(f"{model_save_path}/ensemble_model_{i+1}.keras", overwrite=True)
    
    # Evaluate ensemble on test data
    print("\nEvaluating ensemble model on test data...")
    ensemble_predictions = ensemble_predict(ensemble_models, X_test)
    ensemble_pred_classes = np.argmax(ensemble_predictions, axis=1)
    
    # Calculate accuracy
    ensemble_accuracy = accuracy_score(y_test_encoded, ensemble_pred_classes)
    print(f"Ensemble Test Accuracy: {ensemble_accuracy:.4f}")
    
    # Classification report
    print("\nEnsemble Classification Report:")
    class_names = label_encoder.classes_
    report = classification_report(y_test_encoded, ensemble_pred_classes, target_names=class_names)
    print(report)
    
    # Confusion matrix
    print("\nEnsemble Confusion Matrix:")
    cm = confusion_matrix(y_test_encoded, ensemble_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Ensemble Confusion Matrix')
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
    plt.savefig(f"{model_save_path}/ensemble_confusion_matrix.png")
    
    # Calculate additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_encoded, ensemble_pred_classes, average='weighted')
    
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    
    # Test predictions on challenging examples
    print("\nTesting predictions on challenging examples:")
    test_messages = [
        # Clear category examples
        "Anyone want to play badminton tomorrow evening?",
        "I'm looking for a good telescope for beginners",
        "The new game update has amazing graphics",
        "I need a good recipe for chocolate chip cookies",
        "Going hiking in the mountains next weekend",
        "Having trouble with my JavaScript code",
        "The concert yesterday was amazing",
        
        # More ambiguous examples
        "Just got some new equipment for my hobby",
        "Working on my technique every day",
        "Looking for recommendations from experienced people",
        "Setting up for practice later tonight",
        
        # Examples with mixed signals
        "Looking at my racket while listening to music",
        "Cooking dinner after my programming class",
        "Taking pictures of stars during my camping trip"
    ]
    
    # Preprocess test messages
    test_messages_preprocessed = [text_preprocessing(msg) for msg in test_messages]
    test_sequences = tokenizer.texts_to_sequences(test_messages_preprocessed)
    test_padded = pad_sequences(test_sequences, maxlen=max_sequence_length, padding='post')
    
    # Get ensemble predictions
    ensemble_test_predictions = ensemble_predict(ensemble_models, test_padded)
    ensemble_test_classes = np.argmax(ensemble_test_predictions, axis=1)
    predicted_categories = label_encoder.inverse_transform(ensemble_test_classes)
    
    print("\nEnsemble Prediction Examples:")
    for msg, category, confidence in zip(test_messages, predicted_categories, np.max(ensemble_test_predictions, axis=1)):
        print(f"Message: {msg}")
        print(f"Predicted Category: {category}")
        print(f"Confidence: {confidence:.4f}")
        print()
    
    # Create and save the predictor
    predictor = TextCategoryPredictor(
        ensemble_models, tokenizer, label_encoder, max_sequence_length
    )
    
    # Save the predictor
    with open(f"{model_save_path}/text_category_predictor.pkl", "wb") as f:
        pickle.dump(predictor, f)
    
    print(f"\nSaved text category predictor to {model_save_path}/text_category_predictor.pkl")
    print("Usage example:")
    print("  with open('path/to/text_category_predictor.pkl', 'rb') as f:")
    print("      predictor = pickle.load(f)")
    print("  categories, confidences = predictor.predict(['Your message here'], return_confidence=True)")
    
    return ensemble_models, ensemble_accuracy

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
    max_sequence_length = 80  # Max length of each message
    
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
    model_path = f"{os.getenv('DATA_FOLDER')}/{guild_id}_model"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found for guild {guild_id}. Run train({guild_id}) first.")
    
    # Load the model, tokenizer, and label encoder
    if not os.path.exists(f"{model_path}/best_model.keras"): return None, None

    model = load_model(f"{os.path.abspath(model_path)}/best_model.keras")
    
    with open(f"{model_path}/tokenizer.pickle", 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    with open(f"{model_path}/label_encoder.pickle", 'rb') as handle:
        label_encoder = pickle.load(handle)
    
    # Prepare the message
    max_sequence_length = 80
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
