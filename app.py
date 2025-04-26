from flask import Flask, render_template, request, session, redirect, url_for
import os
import random
from deepface import DeepFace
import pandas as pd
import io
import base64
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import csv
import numpy as np
import base64
import  uuid
from datetime import datetime
from google.cloud import storage


app = Flask(__name__)
app.secret_key = os.urandom(24)

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'static', 'dataset', 'FER-2013')
TEST_DIR = os.path.join(DATASET_DIR, 'test')

# Load dataset
def create_dataframe(directory):
    image_paths, labels = [], []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            for img_name in os.listdir(label_dir):
                image_paths.append(f"static/dataset/FER-2013/test/{label}/{img_name}")
                labels.append(label)
    return pd.DataFrame({'image': image_paths, 'label': labels})

test = create_dataframe(TEST_DIR)

# DeepFace prediction function
def predict_emotion_deepface(image_path):
    try:
        result = DeepFace.analyze(image_path, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except:
        return "unknown"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start-game', methods=['POST'])
def start_game():
    session.clear()  # Ensure a fresh start

    # Safely retrieve form data
    session['username'] = request.form.get('username', 'Unknown')  
    session['gender'] = request.form.get('gender', 'Unknown')
    session['age'] = request.form.get('age', 'Unknown')
    session['city'] = request.form.get('city', 'Unknown')
    session['education'] = request.form.get('education', 'Unknown')

    # Initialise game session
    session['round'] = 0
    session['user_guesses'] = []
    session['ai_predictions'] = []
    session['actual_labels'] = []
    session['images_shown'] = []

    # Generate unique User ID
    session['user_id'] = str(uuid.uuid4())

    return redirect(url_for('game'))  # Start the game


@app.route('/game', methods=['GET', 'POST'])
def game():
    total_rounds = 5

    if session['round'] >= total_rounds:
        return redirect(url_for('results'))

    if request.method == 'POST':
        user_guess = request.form.get('user_guess')
        image_path = session['current_image']['image']
        actual_label = session['current_image']['label']

        ai_prediction = predict_emotion_deepface(image_path)

        session['user_guesses'].append(user_guess)
        session['actual_labels'].append(actual_label)
        session['ai_predictions'].append(ai_prediction)
        session['images_shown'].append(image_path)

        session['round'] += 1

        if session['round'] < total_rounds:
            return redirect(url_for('game'))
        else:
            return redirect(url_for('results'))

    session['current_image'] = test.sample(n=1).iloc[0].to_dict()
    return render_template('game.html', image=session['current_image'])

@app.route('/results')
def results():
    total_rounds = 5
    
    if 'user_guesses' not in session or 'ai_predictions' not in session or 'actual_labels' not in session:
        return redirect(url_for('index'))

    # Calculate scores
    user_score = sum(1 for guess, label in zip(session['user_guesses'], session['actual_labels']) if guess.lower() == label.lower())
    ai_score = sum(1 for ai_pred, label in zip(session['ai_predictions'], session['actual_labels']) if ai_pred.lower() == label.lower())

    # Prepare results data
    results_data = list(zip(session['images_shown'], session['actual_labels'], session['user_guesses'], session['ai_predictions']))

    # Store user information
    user_info = {
        "user_id": session.get('user_id', 'Unknown'),
        "username": session.get('username', 'Unknown'),
        "gender": session.get('gender', 'Unknown'),
        "age": session.get('age', 'Unknown'),
        "city": session.get('city', 'Unknown'),
        "education": session.get('education', 'Unknown'),
        "user_score": user_score,
        "ai_score": ai_score
    }

    # Define CSV file path
    csv_file = 'game_results.csv'
    file_exists = os.path.isfile(csv_file)

    # Write data to CSV
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write header only if file is newly created
        if not file_exists:
            writer.writerow(["User ID", "Username", "Gender", "Age", "City", "Education", "Image", "Actual Label", "User Guess", "AI Prediction", "User Score", "AI Score", "Date", "Time"])

        # Write user data and game results
        for image, actual_label, user_guess, ai_prediction in results_data:
            now=datetime.now()
            date_str = now.strftime("%m-%d-%y") #Format: Year-Month-Day
            time_str = now.strftime("%H:%M:%S") # Format: Hour: Minutes:Seconds

            writer.writerow([user_info["user_id"], user_info["username"], user_info["gender"], user_info["age"], user_info["city"], user_info["education"], image, actual_label, user_guess, ai_prediction, user_score, ai_score, date_str, time_str])

    return render_template(
        'results.html',
        user_score=user_score,
        ai_score=ai_score,
        total_rounds=total_rounds,
        results_data=results_data,
        user_id=user_info["user_id"],
        username=user_info["username"],
        gender=user_info["gender"],
        age=user_info["age"],
        city=user_info["city"],
        education=user_info["education"],
        time=time_str,
        date=date_str
    )

def process_csv():
    """Processes the raw CSV and creates a cleaned version with one row per user."""
    raw_csv = 'game_results.csv'
    cleaned_csv = 'cleaned_game_results.csv'

    if not os.path.exists(raw_csv) or os.stat(raw_csv).st_size == 0:
        print("[DEBUG] Raw CSV does not exist or is empty.")
        return False  # No data available

    try:
        # Read CSV without headers
        column_names = ['user_id', 'name', 'gender', 'age', 'city', 'education', 'image', 'emotion', 'user_response', 'ai_prediction', 'user_score', 'ai_score','date_str','time_str']
        df = pd.read_csv(raw_csv, names=column_names, header=None)

        df_debug=pd.read_csv(raw_csv)
        print("[DEBUG] First 5 rows:\n", df_debug.head())
        print("[DEBUG] Raw CSV shape:", df.shape)  
        

        # Strip spaces from string columns
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # Convert numeric columns
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['user_score'] = pd.to_numeric(df['user_score'], errors='coerce')
        df['ai_score'] = pd.to_numeric(df['ai_score'], errors='coerce')

        # Drop rows with missing values in key columns
        df = df.dropna(subset=['user_id', 'age', 'education', 'user_score', 'ai_score'])

        # Aggregate by user, keeping one row per unique 'name'
        df_cleaned = df.drop_duplicates(subset=['user_id'], keep='first')

        print("[DEBUG] Cleaned CSV preview:")
        print(df_cleaned.head())

        # Save cleaned data to a new CSV file
        df_cleaned.to_csv(cleaned_csv, index=False)
        print("[DEBUG] Cleaned CSV saved successfully.")
        return True  # Successfully processed

    except Exception as e:
        print(f"[ERROR] Error processing CSV: {e}")
        return False  # Failed processing

@app.route('/dashboard')
def dashboard():
    cleaned_csv = 'cleaned_game_results.csv'

    # Process the raw CSV before generating plots
    if not process_csv() or not os.path.exists(cleaned_csv):
        return "No data available yet."

    try:
        # Load cleaned CSV
        df = pd.read_csv(cleaned_csv)
        print("[DEBUG] Loaded cleaned CSV successfully.")
        print(df.head())

        # Education colors
        education_colors = {
            "Primary": "blue",
            "Secondary": "green",
            "Undergraduate": "orange",
            "Higher education": "purple"
        }

        # Assign colors (default gray for unknown values)
        df['color'] = df['education'].map(education_colors).fillna("black")

        # Scatter plot
        fig, ax1 = plt.subplots(figsize=(8, 5))
        for edu_level, group in df.groupby('education'):
            ax1.scatter(group['user_score'], group['age'], 
                        label=edu_level, color=education_colors.get(edu_level, 'gray'), 
                        s=100, alpha=0.7)

        ax1.set_xlim(0 , df['user_score'].max() + 1)
        ax1.set_ylim(df['age'].min()-1, df['age'].max() + 1)
        ax1.set_xlabel("User Score")
        ax1.set_ylabel("Age")
        ax1.set_title("User Score by Education Level")
        ax1.legend(title="Education Level")

        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url_1 = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)

        # Histogram for score distribution
        fig, ax2 = plt.subplots(figsize=(8, 5))
        
        print("[DEBUG] User scores before conversion:", df["user_score"].head())
        print("[DEBUG] AI scores before conversion:", df["ai_score"].head())
        
        user_scores = df["user_score"].dropna().astype(int)
        ai_scores = df["ai_score"].dropna().astype(int)
        
        print("[DEBUG] Processed User Scores:", user_scores.head())
        print("[DEBUG] Processed AI Scores:", ai_scores.head())
        
       # Ensure bin range includes both user & AI scores
        min_bin = min(user_scores.min(), ai_scores.min()) - 0.25
        max_bin = max(user_scores.max(), ai_scores.max()) + 1.5
        bins = np.arange(min_bin, max_bin, 1)

        ax2.hist(user_scores, bins=bins, align='mid',color='coral', edgecolor='coral', width=0.5, label= "User score")
        ax2.hist(ai_scores, bins=bins, align='mid',color= 'darkcyan', edgecolor='darkcyan', width=0.5, label= "AI score")

        ax2.set_xticks(np.arange(0, 7, 1))
        ax2.legend(title="Score Type", loc="upper right", frameon=True, fancybox=True, edgecolor="lightgray")

        ax2.set_xlabel("Scores")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of User Scores")

        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url_2 = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)

        return render_template('dashboard.html', plot_url_1=plot_url_1, plot_url_2=plot_url_2)

    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        return f"Error loading data: {e}"

if __name__ == '__main__':
    app.run(debug=True)