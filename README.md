The Emotion Face-Off Game is a web-based application developed using the Flask framework. The project aims to compare human and artificial intelligence (AI) performance in recognizing facial emotions. Participants are tasked with identifying emotions from images, and their selections are compared against AI model predictions. The game also serves as a tool for conducting a pilot study into human perception of emotions versus machine learning outputs.

Features
Interactive emotion recognition game.

Comparison between user guesses and AI predictions.

Dashboard for viewing individual and aggregated results.

Data collection for further research analysis.

Analytical scripts for assessing human and AI accuracy.

Emotions Included
The game evaluates recognition across the following emotions:

Angry

Disgust

Fear

Happy

Sad

Surprise

Neutral

Project Structure
Flask_Project/
│
├── app.py                 # Main Flask application 
├── static/                # Static files (CSS, JavaScript, images)
│   ├── dataset/           # (Dataset images are stored externally in cloud storage.)
├── templates/             # HTML templates for rendering the web pages
│   ├── index.html         # Home page
│   ├── game.html          # Game interface
│   ├── results.html       # Individual result display
│   ├── dashboard.html     # Dashboard for overall statistics
│
├── requirements.txt       # List of Python dependencies
#Note: The dataset used for the game is not included in the repository and is hosted on an external cloud service.

Installation and Running the Application
Clone the repository:
git clone https://github.com/your-username/Emotion-Face-Off-Game.git
cd Emotion-Face-Off-Game

Create a virtual environment:
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

Install the required dependencies:
pip install -r requirements.txt

Run the Flask application:
python app.py

License
This project is currently shared for educational and research purposes.
License information will be provided in future updates.

Acknowledgments
This work was developed as part of a data science bootcamp project under the guidance of Tech Cornwall. Special thanks to all participants who contributed to the pilot study.
