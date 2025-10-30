from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy # Reverting to SQLAlchemy/SQLite
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import joblib
import numpy as np
import os
import random
from datetime import datetime

# --- CONFIGURATION & SETUP ---
app = Flask(__name__)

# CONFIGURATION: REVERTING TO LOCAL SQLITE DATABASE FOR STABILITY
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
# ----------------------------------------------------------------

app.config['SECRET_KEY'] = 'your_super_secret_key_for_national_hackathon'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- ML ARTIFACTS LOADING ---
try:
    model = joblib.load('risk_prediction_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError as e:
    print(f"ERROR: ML artifact not found: {e}. Please run model_training.py first.")
    exit()

FEATURE_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# --- USER MODEL DEFINITION (SQLAlchemy) ---
class User(db.Model):
    """Database model for storing user credentials, biometrics, and analysis."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    # BIOMETRIC FIELDS (Onboarding)
    age = db.Column(db.Integer, default=None)
    height_cm = db.Column(db.Float, default=None)
    weight_kg = db.Column(db.Float, default=None)
    bmi = db.Column(db.Float, default=None)

    # Persistent Analysis Fields
    last_risk_percent = db.Column(db.Float, default=None)
    last_risk_level = db.Column(db.String(50), default=None)
    last_analysis_data = db.Column(db.Text, default=None)

    # NEW: History data stored as JSON string in a Text field
    analysis_history_json = db.Column(db.Text, default='[]')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def calculate_bmi(self):
        """Calculates BMI and updates the instance."""
        if self.height_cm and self.weight_kg:
            height_m = self.height_cm / 100
            self.bmi = round(self.weight_kg / (height_m ** 2), 1)

# --- HELPER FUNCTIONS ---

def calculate_bmi(height_cm, weight_kg):
    if height_cm and weight_kg:
        height_m = height_cm / 100
        return round(weight_kg / (height_m ** 2), 1)
    return None

def extract_data_from_pdf(filepath, risk_type='medium'):
    try:
        os.remove(filepath)
    except OSError:
        pass

    base_data = {
        'Pregnancies': 1, 'Glucose': 120, 'BloodPressure': 72,
        'SkinThickness': 30, 'Insulin': 100, 'BMI': 30.0,
        'DiabetesPedigreeFunction': 0.45, 'Age': 35
    }

    if risk_type == 'high':
        base_data.update({
            'Pregnancies': random.randint(3, 8), 'Glucose': random.randint(150, 200),
            'BloodPressure': random.randint(90, 100), 'BMI': round(random.uniform(35.0, 45.0), 1),
            'Age': random.randint(50, 75)
        })
    elif risk_type == 'low':
        base_data.update({
            'Pregnancies': random.randint(0, 1), 'Glucose': random.randint(80, 105),
            'BloodPressure': random.randint(60, 75), 'BMI': round(random.uniform(20.0, 25.0), 1),
            'Age': random.randint(22, 30)
        })

    return base_data

def generate_recommendations(risk_level, bmi):
    tips = []
    if risk_level == "HIGH RISK":
        tips.append("🚨 Consult your physician immediately to validate this high-risk profile.")
        tips.append("📉 Focus on reducing plasma glucose levels through controlled diet and exercise.")
    elif risk_level == "Low Risk":
        tips.append("✅ Excellent progress! Maintain your current healthy lifestyle and monitoring.")

    if bmi >= 30:
        tips.append("🏃 Increase physical activity (30+ minutes daily) to target weight reduction (Obese category).")
    elif 25 <= bmi < 30:
        tips.append("⚖️ Monitor calorie intake and aim for moderate weight loss to move out of the overweight category.")
    else:
        tips.append("🍎 Maintain your balanced diet and aim for a minimum of 150 minutes of moderate weekly exercise.")
    return tips


# --- ROUTES ---

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user:
            return render_template('register.html', error="Username already taken.")

        new_user = User(username=username)
        new_user.set_password(password)

        try:
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login', registered=True))
        except Exception:
            db.session.rollback()
            return render_template('register.html', error="Registration failed due to database error.")

    return render_template('register.html')


@app.route('/onboarding', methods=['GET', 'POST'])
def onboarding():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    username = session['username']
    user = User.query.filter_by(username=username).first()

    if not user:
        session.pop('logged_in', None)
        return redirect(url_for('login'))

    if user.height_cm and user.weight_kg and user.age:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        try:
            user.age = int(request.form.get('age'))
            user.height_cm = float(request.form.get('height_cm'))
            user.weight_kg = float(request.form.get('weight_kg'))

            user.calculate_bmi()
            db.session.commit()

            return redirect(url_for('dashboard'))
        except Exception as e:
            db.session.rollback()
            return render_template('onboarding.html', error=f"Invalid input provided: {e}")

    return render_template('onboarding.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            session['logged_in'] = True
            session['username'] = username

            if not (user.height_cm and user.weight_kg and user.age):
                return redirect(url_for('onboarding'))

            return redirect(url_for('dashboard'))

        error_message = "Invalid Username or Password."
        if not user:
            error_message = "User not found. Please register a new account."

        return render_template('login.html', error=error_message)

    if request.args.get('registered'):
        return render_template('login.html', success="Registration successful! Please log in.")

    if 'logged_in' in session:
        return redirect(url_for('dashboard'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/')
@app.route('/dashboard')
def dashboard():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    username = session['username']
    user = User.query.filter_by(username=username).first()

    if not user:
        session.pop('logged_in', None)
        return redirect(url_for('login'))

    onboarding_complete = user.height_cm and user.weight_kg and user.age
    if not onboarding_complete:
        return redirect(url_for('onboarding'))

    has_analysis = user.last_risk_percent is not None

    previous_report = None
    if has_analysis:
        previous_report = {
            'risk_level': user.last_risk_level,
            'risk_percent': user.last_risk_percent,
        }

    biometrics = {
        'age': user.age,
        'height': user.height_cm,
        'weight': user.weight_kg,
        'bmi': user.bmi
    }

    return render_template('index.html',
                           has_analysis=has_analysis,
                           previous_report=previous_report,
                           biometrics=biometrics,
                           username=username)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    username = session['username']
    user = User.query.filter_by(username=username).first()

    if not (user.height_cm and user.weight_kg and user.age):
        return redirect(url_for('onboarding'))

    if request.method == 'POST':
        # FIX: HARDCODED 'low' risk simulation
        selected_risk_type = 'low'

        if 'health_data_file' not in request.files:
            return render_template('upload.html', error="No file part in the request.")

        file = request.files['health_data_file']

        if file.filename == '':
            return render_template('upload.html', error="No file selected.")

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 1. EXTRACT DATA (SIMULATED)
            extracted_data = extract_data_from_pdf(filepath, risk_type=selected_risk_type)

            # 2. PERFORM PREDICTION
            input_list = [extracted_data[name] for name in FEATURE_NAMES]
            input_array = np.array([input_list])
            scaled_input = scaler.transform(input_array)

            risk_prob = model.predict_proba(scaled_input)[0][1]
            prediction = model.predict(scaled_input)[0]

            # FIX: Convert NumPy float to standard Python float before saving
            risk_percent_python_float = float(round(risk_prob * 100, 2))

            risk_level = "HIGH RISK" if prediction == 1 else "Low Risk"

            # Record for history array
            analysis_record = {
                'timestamp': datetime.now().isoformat(),
                'risk_percent': risk_percent_python_float,
                'risk_level': risk_level,
                'data_snapshot': extracted_data
            }

            # Get existing history, append new record, and dump back to DB
            history_json = user.analysis_history_json if user.analysis_history_json else '[]'
            history = eval(history_json) # Use eval() on trusted internal string
            history.append(analysis_record)

            # 3. SAVE TO DATABASE (SQLAlchemy Update)
            user.last_risk_percent = risk_percent_python_float
            user.last_risk_level = risk_level
            user.last_analysis_data = str(extracted_data)
            user.analysis_history_json = str(history) # Store updated history
            db.session.commit()

            return redirect(url_for('analysis'))

    return render_template('upload.html')


@app.route('/analysis')
def analysis():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    username = session['username']
    user = User.query.filter_by(username=username).first()

    if not user or user.last_risk_percent is None:
        return redirect(url_for('upload'))

    try:
        user_data = eval(user.last_analysis_data) if user.last_analysis_data else {}
    except:
        user_data = {}

    risk_level = user.last_risk_level
    risk_percent = user.last_risk_percent

    bmi = user.bmi if user.bmi else 25.0
    personalized_tips = generate_recommendations(risk_level, bmi)

    return render_template('analysis.html',
                           data=user_data,
                           risk_level=risk_level,
                           risk_percent=risk_percent,
                           bmi=bmi,
                           tips=personalized_tips)

# --- API ROUTE FOR CHARTING (Reads from local DB) ---
@app.route('/api/history')
def api_history():
    if 'logged_in' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    username = session['username']
    user = User.query.filter_by(username=username).first()

    if not user:
        return jsonify({'labels': [], 'data': []})

    history = eval(user.analysis_history_json) if user.analysis_history_json else []

    # Process history for Chart.js
    labels = [datetime.fromisoformat(record['timestamp']).strftime('%b %d - %H:%M') for record in history]
    data = [record['risk_percent'] for record in history]

    return jsonify({'labels': labels, 'data': data})
# ----------------------------------------------------


if __name__ == '__main__':
    # Initialize the database file if it doesn't exist when starting the app
    with app.app_context():
        # This will create users.db and the necessary tables
        db.create_all()
        print("Database initialized (users.db created).")

    app.run(debug=True)