import os
from flask import Flask, render_template, request, redirect, flash, session, send_file, jsonify
import pymysql
import joblib
import pandas as pd
from dotenv import load_dotenv
import openai

# ------------------ Load environment variables ------------------
load_dotenv()  # must come before accessing os.getenv

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

if not OPENAI_API_KEY or not SECRET_KEY:
    raise ValueError("Missing OPENAI_API_KEY or SECRET_KEY in .env")

openai.api_key = OPENAI_API_KEY

# ------------------ Flask app ------------------
app = Flask(__name__)
app.secret_key = SECRET_KEY

# ------------------ MySQL connection ------------------
db = pymysql.connect(
    host=os.getenv("MYSQLHOST"),
    user=os.getenv("MYSQLUSER"),
    password=os.getenv("MYSQLPASSWORD"),
    database=os.getenv("MYSQLDATABASE"),
    port=int(os.getenv("MYSQLPORT"))
)

cursor = db.cursor()

# ------------------ Load ML models ------------------
diabetes_model = joblib.load("models/rf_Diabetes.joblib")
diabetes_scaler = joblib.load("models/scaler_Diabetes.joblib")

heart_model = joblib.load("models/lr_heart.joblib")
heart_scaler = joblib.load("models/scaler_heart.joblib")

bodyfat_model = joblib.load("models/Body_fat.joblib")
bodyfat_scaler = joblib.load("models/scaler_bodyfat.joblib")


# ----------------------------Signup Route-------------------------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        first_name = request.form["first_name"]
        last_name = request.form["last_name"]
        email = request.form["email"]
        password = request.form["password"]
        dob = request.form["dob"]
        phone = request.form["phone"]
        recovery_password = request.form["recovery_password"]

        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            flash("User already exists. Please login.", "warning")
            return redirect("/")

        cursor.execute("INSERT INTO users (first_name, last_name, email, phone, dob, password, recovery_password) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                       (first_name, last_name, email, phone, dob, password, recovery_password))
        db.commit()
        flash("Registration successful. Please login.", "success")
        return redirect("/")
    return render_template("signup.html")

# -----------------------------------Signin Route----------------------------------------------
@app.route("/", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        cursor.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
        user = cursor.fetchone()

        if user:
            session["user"] = email
            return redirect("/main")
        else:
            flash("User not found. Please register first.", "danger")
            return redirect("/")
    return render_template("signin.html")



@app.route("/forgot", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form["email"]
        dob = request.form["dob"]
        recovery_password = request.form["recovery_password"]
        new_password = request.form["new_password"]

        cursor.execute(
            "SELECT * FROM users WHERE email = %s AND dob = %s AND recovery_password = %s",
            (email, dob, recovery_password)
        )
        user = cursor.fetchone()

        if user:
            cursor.execute("UPDATE users SET password = %s WHERE email = %s", (new_password, email))
            db.commit()
            flash("Password updated successfully! You can now login.", "success")
            return redirect("/")
        else:
            flash("Invalid credentials. Please try again.", "danger")
            return redirect("/forgot")

    return render_template("forgot.html")










# --------------------------------Main Dashboard Route------------------------------------
@app.route("/main")
def main_page():
    if "user" in session:
        return render_template("main.html", user=session["user"])
    else:
        flash("Please login first.", "warning")
        return redirect("/")

# Contact form
@app.route("/contact", methods=["POST"])
def contact():
    name = request.form.get("name")
    email = request.form.get("email")
    message = request.form.get("message")
    latitude = request.form.get("latitude")
    longitude = request.form.get("longitude")
    address = request.form.get("address")

    cursor.execute(
        "INSERT INTO contact_messages (name, email, message, latitude, longitude, address) VALUES (%s, %s, %s, %s, %s, %s)",
        (name, email, message, latitude, longitude, address)
    )
    db.commit()
    flash("Your message and full address were received. Thank you!", "success")
    return redirect("/main")

# ------------------------------Heart Prediction Route---------------------------------
@app.route("/heart", methods=["GET", "POST"])
def heart_prediction():
    if "user" not in session:
        flash("Please login first.", "warning")
        return redirect("/")

    prediction = None
    probability = None

    if request.method == "POST":
        try:
            # Step 1: Extract form inputs
            input_fields = [
                "Chest_Pain", "Shortness_of_Breath", "Fatigue", "Palpitations",
                "Dizziness", "Swelling", "Pain_Arms_Jaw_Back", "Cold_Sweats_Nausea",
                "High_BP", "High_Cholesterol", "Diabetes", "Smoking", "Obesity",
                "Sedentary_Lifestyle", "Family_History", "Chronic_Stress", "Gender", "Age"
            ]
            inputs = {field: int(request.form[field]) for field in input_fields}
            print("Form Inputs:", inputs)

            # Step 2: Prepare DataFrame
            df = pd.DataFrame([inputs])
            print("DataFrame:\n", df)

            # Step 3: Scale inputs
            scaled = heart_scaler.transform(df)
            print("Scaled Input:", scaled)

            # Step 4: Predict
            prediction = int(heart_model.predict(scaled)[0])
            print("Prediction:", prediction)

            # Step 5: Get probability
            if hasattr(heart_model, "predict_proba"):
                prob_values = heart_model.predict_proba(scaled)[0]
                print("Probabilities:", prob_values)
                probability = float(prob_values[prediction]) * 100
            else:
                flash("Model does not support predict_proba", "danger")
                probability = None

            # Step 6: Fetch user info
            user_email = session["user"]
            cursor.execute("SELECT first_name, last_name FROM users WHERE email = %s", (user_email,))
            user = cursor.fetchone()
            user_name = f"{user[0]} {user[1]}" if user else "Unknown"

            # Step 7: Store prediction to DB
            cursor.execute("""
                INSERT INTO heart_predictions (
                    user_email, user_name, Chest_Pain, Shortness_of_Breath, Fatigue,
                    Palpitations, Dizziness, Swelling, Pain_Arms_Jaw_Back, Cold_Sweats_Nausea,
                    High_BP, High_Cholesterol, Diabetes, Smoking, Obesity,
                    Sedentary_Lifestyle, Family_History, Chronic_Stress,
                    Gender, Age, prediction, probability
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                user_email, user_name, *[inputs[field] for field in input_fields],
                prediction, probability
            ))
            db.commit()

        except Exception as e:
            print("💥 Prediction Error:", e)
            flash(f"Prediction Error: {str(e)}", "danger")
            prediction = None
            probability = None

    return render_template("heart.html", prediction=prediction, probability=probability)





#-----------------------------Diabetes Prediction-----------------------------------
@app.route("/diabetes", methods=["GET", "POST"])
def diabetes_prediction():
    if "user" not in session:
        flash("Please login first.", "warning")
        return redirect("/")

    prediction = None
    probability = None

    if request.method == "POST":
        try:
            inputs = {
                'age': int(request.form["age"]),
                'gender': int(request.form["gender"]),
                'polyuria': int(request.form["polyuria"]),
                'polydipsia': int(request.form["polydipsia"]),
                'sudden_weight_loss': int(request.form["sudden_weight_loss"]),
                'weakness': int(request.form["weakness"]),
                'polyphagia': int(request.form["polyphagia"]),
                'genital_thrush': int(request.form["genital_thrush"]),
                'visual_blurring': int(request.form["visual_blurring"]),
                'itching': int(request.form["itching"]),
                'irritability': int(request.form["irritability"]),
                'delayed_healing': int(request.form["delayed_healing"]),
                'partial_paresis': int(request.form["partial_paresis"]),
                'muscle_stiffness': int(request.form["muscle_stiffness"]),
                'alopecia': int(request.form["alopecia"]),
                'obesity': int(request.form["obesity"])
            }

            df = pd.DataFrame([inputs])
            scaled = diabetes_scaler.transform(df)
            prediction = diabetes_model.predict(scaled)[0]
            prob_values = diabetes_model.predict_proba(scaled)[0]
            probability = float(prob_values[prediction]) * 100  # convert to percent

            # Fetch user name
            user_email = session["user"]
            cursor.execute("SELECT first_name, last_name FROM users WHERE email = %s", (user_email,))
            user = cursor.fetchone()
            user_name = f"{user[0]} {user[1]}" if user else "Unknown"

            # Store into DB
            cursor.execute("""
                INSERT INTO diabetes_predictions (
                    user_email, user_name, age, gender, polyuria, polydipsia,
                    sudden_weight_loss, weakness, polyphagia, genital_thrush,
                    visual_blurring, itching, irritability, delayed_healing,
                    partial_paresis, muscle_stiffness, alopecia, obesity,
                    prediction, probability
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                user_email, user_name, *inputs.values(), int(prediction), probability
            ))
            db.commit()

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")

    return render_template("diabetes.html", prediction=prediction, probability=probability)


#---------------------------Body_Fat Prediciton-----------------------------------
@app.route("/bodyfat", methods=["GET", "POST"])
def bodyfat_prediction():
    if "user" not in session:
        flash("Please login first.", "warning")
        return redirect("/")

    prediction = None
    lower = None
    upper = None

    if request.method == "POST":
        try:
            fields = [
                "Age", "Neck", "Chest", "Abdomen", "Hip", "Thigh", "Knee",
                "Ankle", "Biceps", "Forearm", "Wrist", "Weight"
            ]

            # Extract and scale input
            inputs = {field: float(request.form[field]) for field in fields}
            df = pd.DataFrame([inputs])
            scaled = bodyfat_scaler.transform(df)

            # Predict body fat
            prediction = float(bodyfat_model.predict(scaled)[0])

            # Manually set interval ±3.10
            lower = round(prediction - 3.10, 2)
            upper = round(prediction + 3.10, 2)

            # Get user details
            user_email = session["user"]
            cursor.execute("SELECT first_name, last_name FROM users WHERE email = %s", (user_email,))
            user = cursor.fetchone()
            user_name = f"{user[0]} {user[1]}" if user else "Unknown"

            # Save to DB
            cursor.execute("""
                INSERT INTO bodyfat_predictions (
                    user_email, user_name, Age, Neck, Chest, Abdomen, Hip, Thigh,
                    Knee, Ankle, Biceps, Forearm, Wrist, Weight,
                    prediction, lower_bound, upper_bound
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                user_email, user_name, *[inputs[f] for f in fields],
                prediction, lower, upper
            ))
            db.commit()

        except Exception as e:
            print("💥 BodyFat Prediction Error:", e)
            flash(f"Prediction Error: {str(e)}", "danger")

    return render_template("BodyFat.html", prediction=prediction, lower=lower, upper=upper)

#-------------------------Report Generate --------------------------------------
@app.route("/report", methods=["GET", "POST"])
def report():
    user_email = None
    heart = diabetes = bodyfat = None

    if request.method == "POST":
        user_email = request.form.get("email")

        def fetch_latest(table, time_col):
            cursor.execute(f"SELECT * FROM {table} WHERE user_email = %s ORDER BY {time_col} DESC LIMIT 1", (user_email,))
            row = cursor.fetchone()
            if row:
                columns = [col[0] for col in cursor.description]
                return dict(zip(columns, row))
            return None

        heart = fetch_latest("heart_predictions", "created_at")
        diabetes = fetch_latest("diabetes_predictions", "created_at")
        bodyfat = fetch_latest("bodyfat_predictions", "timestamp")

    return render_template("report.html",
                           user_email=user_email,
                           heart=heart,
                           diabetes=diabetes,
                           bodyfat=bodyfat,
                           pdf_export=False)


@app.route("/download-report", methods=["POST"])
def download_report():
    user_email = request.form.get("email")

    def fetch_latest(table, time_col):
        cursor.execute(f"SELECT * FROM {table} WHERE user_email = %s ORDER BY {time_col} DESC LIMIT 1", (user_email,))
        row = cursor.fetchone()
        if row:
            columns = [col[0] for col in cursor.description]
            return dict(zip(columns, row))
        return None

    heart = fetch_latest("heart_predictions", "created_at")
    diabetes = fetch_latest("diabetes_predictions", "created_at")
    bodyfat = fetch_latest("bodyfat_predictions", "timestamp")

    rendered_html = render_template("report.html",
                                    user_email=user_email,
                                    heart=heart,
                                    diabetes=diabetes,
                                    bodyfat=bodyfat,
                                    pdf_export=True)

    from io import BytesIO
    from xhtml2pdf import pisa

    pdf = BytesIO()
    pisa_status = pisa.CreatePDF(rendered_html, dest=pdf)
    pdf.seek(0)

    if pisa_status.err:
        flash("Error generating PDF report", "danger")
        return redirect("/report")

    return send_file(pdf, download_name="Health_Report.pdf", as_attachment=True)


#--------------------Tips Section -----------------------------------
# -------------------- Tips Section -----------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import base64
import pandas as pd
import os
import fitz  # PyMuPDF
import re
from docx import Document
import random

# ------------------ Load and Standardize Datasets ------------------
def load_and_standardize_data(filepath, target_col, age_col='age'):
    try:
        df = pd.read_csv(filepath)
        df.columns = [c.lower() for c in df.columns]
        if age_col in df.columns:
            df.rename(columns={age_col: 'Age'}, inplace=True)
        if target_col in df.columns:
            df.rename(columns={target_col: 'Risk'}, inplace=True)
        else:
            return pd.DataFrame()
        return df
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

heart_data = load_and_standardize_data("heart_disease_risk_dataset_earlymed.csv", target_col='heart_risk', age_col='age')
diabetes_data = load_and_standardize_data("diabetes_data_cleaned.csv", target_col='class', age_col='age')
try:
    bodyfat_data = pd.read_csv("bodyfat_modifed.csv")
except FileNotFoundError:
    bodyfat_data = pd.DataFrame()

# -------- Helper Functions --------
def get_age_group(age):
    if age < 40:
        return "Young (<40)"
    elif age < 60:
        return "Middle (40–59)"
    else:
        return "Senior (60+)"

def parse_docx(path):
    try:
        if not os.path.exists(path):
            return {}
        doc = Document(path)
        tips_data = {}
        current_section = current_age_group = current_risk_type = current_tip_type = None
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            if "HEART HEALTH" in text:
                current_section = "heart"; tips_data[current_section] = {}
            elif "DIABETES MANAGEMENT" in text:
                current_section = "diabetes"; tips_data[current_section] = {}
            elif "BODY FAT CONTROL" in text:
                current_section = "bodyfat"; tips_data[current_section] = {}
            elif any(k in text for k in ["Young", "Middle", "Senior"]):
                current_age_group = text; tips_data[current_section].setdefault(current_age_group, {})
            elif text in ["Risky", "Safe", "High Body Fat", "Low Body Fat"]:
                current_risk_type = text; tips_data[current_section][current_age_group].setdefault(current_risk_type, {})
            elif text in ["Diet", "Exercise"]:
                current_tip_type = text.lower(); tips_data[current_section][current_age_group][current_risk_type].setdefault(current_tip_type, [])
            elif current_tip_type:
                tips_data[current_section][current_age_group][current_risk_type][current_tip_type].append(text.lstrip("- "))
        return tips_data
    except Exception:
        return {}

TIPS = parse_docx("health_tips_full.docx")

def get_tips(section, age_group, risk_type, n=2):
    pool = TIPS.get(section, {}).get(age_group, {}).get(risk_type, {})
    diet_list, exercise_list = pool.get("diet", []), pool.get("exercise", [])
    num_tips = n if risk_type in ["Risky", "High Body Fat"] else 1
    return {
        "diet": random.sample(diet_list, min(num_tips, len(diet_list))) if diet_list else [],
        "exercise": random.sample(exercise_list, min(num_tips, len(exercise_list))) if exercise_list else []
    }

# -------- Graph Functions --------

def create_heart_age_distribution_chart(dataset, user_age):
    fig, ax = plt.subplots(figsize=(8, 5))
    if not dataset.empty and 'Age' in dataset.columns:
        sns.histplot(data=dataset[dataset['Risk'] == 0], x='Age', ax=ax, color='lightgreen', label='Safe', kde=True)
        sns.histplot(data=dataset[dataset['Risk'] == 1], x='Age', ax=ax, color='salmon', label='At Risk', kde=True)
        ax.axvline(user_age, color='black', linestyle='--', linewidth=2.5, label=f'Your Age: {user_age}')
    else:
        ax.text(0.5, 0.5, 'Heart Age data not found', ha='center')
    ax.set_title('Age Distribution by Heart Risk', fontweight='bold')
    ax.set_xlabel('Age'); ax.set_ylabel('Number of Cases'); ax.legend()
    plt.tight_layout()
    buf = BytesIO(); plt.savefig(buf, format='png'); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def create_chest_pain_chart(dataset, user_chest_pain_type):
    fig, ax = plt.subplots(figsize=(8, 5))
    if not dataset.empty and 'chest_pain' in dataset.columns:
        counts = dataset.groupby(['chest_pain', 'Risk']).size().unstack(fill_value=0)
        safe_bars = ax.bar(counts.index, counts[0], label='Safe', color='#2ecc71', edgecolor='black')
        risk_bars = ax.bar(counts.index, counts[1], bottom=counts[0], label='At Risk', color='#e74c3c', edgecolor='black')
        if user_chest_pain_type in counts.index:
            idx = counts.index.get_loc(user_chest_pain_type)
            risk_bars[idx].set_edgecolor('blue'); risk_bars[idx].set_linewidth(2.5)
            safe_bars[idx].set_edgecolor('blue'); safe_bars[idx].set_linewidth(2.5)
            ax.text(safe_bars[idx].get_x() + safe_bars[idx].get_width() / 2, counts.loc[user_chest_pain_type].sum(),
                    'Your Type', ha='center', va='bottom', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Chest Pain data not found', ha='center')
    ax.set_title('Heart Risk by Chest Pain Type', fontweight='bold')
    ax.set_xlabel('Type of Chest Pain (0=None)')
    ax.set_ylabel('Case Count')
    ax.legend()
    plt.tight_layout()
    buf = BytesIO(); plt.savefig(buf, format='png'); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def create_cholesterol_chart(dataset, user_cholesterol_status):
    fig, ax = plt.subplots(figsize=(8, 5))
    if not dataset.empty and 'high_cholesterol' in dataset.columns:
        sns.countplot(data=dataset, x='high_cholesterol', hue='Risk', ax=ax,
                      palette=['#2ecc71', '#e74c3c'], edgecolor='black')
        if user_cholesterol_status in [0, 1]:
            for patch in ax.patches:
                patch.set_edgecolor('black')
    else:
        ax.text(0.5, 0.5, 'Cholesterol data not found', ha='center')
    ax.set_title('Heart Risk by Cholesterol Status', fontweight='bold')
    ax.set_xlabel('Cholesterol Status'); ax.set_ylabel('Case Count')
    ax.set_xticklabels(['Normal', 'High']); ax.legend(title='Risk', labels=['Safe', 'At Risk'])
    plt.tight_layout()
    buf = BytesIO(); plt.savefig(buf, format='png'); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def create_age_bodyfat_scatter(dataset, user_age, user_bodyfat):
    fig, ax = plt.subplots(figsize=(8, 5))
    if not dataset.empty and 'Age' in dataset.columns and 'BodyFat' in dataset.columns:
        sns.scatterplot(data=dataset, x='Age', y='BodyFat', ax=ax, color='purple', alpha=0.5, label='Population Data')
        if user_age and user_bodyfat:
            ax.plot(user_age, user_bodyfat, 'X', color='red', markersize=15,
                    label='You Are Here', markeredgecolor='black')
    else:
        ax.text(0.5, 0.5, 'Age or BodyFat data not found', ha='center')
    ax.set_title('Age vs. Body Fat Percentage', fontweight='bold')
    ax.set_xlabel('Age'); ax.set_ylabel('Body Fat %'); ax.legend()
    plt.tight_layout()
    buf = BytesIO(); plt.savefig(buf, format='png'); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def create_age_group_risk_chart(dataset, user_age, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    if not dataset.empty and 'Age' in dataset.columns:
        bins = list(range(20, 81, 10))
        labels = [f'{i}-{i+9}' for i in bins[:-1]]
        dataset['Age Group'] = pd.cut(dataset['Age'], bins=bins, labels=labels, right=False)
        risk_counts = dataset[dataset['Risk'] == 1].groupby('Age Group', observed=False).size().reindex(labels).fillna(0)
        bars = ax.bar(risk_counts.index, risk_counts.values, color='skyblue', edgecolor='black')
        user_age_group = pd.cut([user_age], bins=bins, labels=labels, right=False)[0]
        if user_age_group in risk_counts.index:
            user_index = risk_counts.index.get_loc(user_age_group)
            bars[user_index].set_color('salmon'); bars[user_index].set_edgecolor('red')
            ax.text(bars[user_index].get_x() + bars[user_index].get_width()/2.0,
                    bars[user_index].get_height(), 'Your Group', ha='center', va='bottom', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Data not available', ha='center')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Age Group'); ax.set_ylabel('Number of At-Risk Cases')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    buf = BytesIO(); plt.savefig(buf, format='png', dpi=100); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def create_bodyfat_gauge_chart(user_bodyfat):
    fig, ax = plt.subplots(figsize=(8, 3))
    ranges = {
        'Healthy': (0, 25, '#2ecc71'),
        'Overweight': (25, 30, '#f39c12'),
        'Obese': (30, 50, '#e74c3c')
    }
    for label, (start, end, color) in ranges.items():
        ax.axvspan(start, end, alpha=0.6, color=color)
        ax.text((start + end) / 2, 0.5, label, ha='center', va='center',
                color='white', fontsize=12, fontweight='bold')
    if user_bodyfat is not None:
        ax.axvline(user_bodyfat, color='black', ymin=0.2, ymax=0.8, linewidth=3)
        ax.text(user_bodyfat, 0.85, f'You: {user_bodyfat:.1f}%',
                ha='center', va='bottom', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))
    ax.set_title('Body Fat Percentage Guide', fontsize=14, fontweight='bold')
    ax.set_xlabel('Body Fat (%)'); ax.set_yticks([]); ax.set_xlim(0, 50)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False)
    plt.tight_layout()
    buf = BytesIO(); plt.savefig(buf, format='png', dpi=100); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# -------- Extract data from PDF --------
def extract_user_data_from_report(text):
    data = {}
    heart_match = re.search(r"Heart Prediction.*?prediction\s+(\d).*?probability\s+([\d.]+).*?Gender\s+(\d).*?Age\s+(\d+)", text, re.DOTALL)
    if heart_match:
        data["heart_prediction"] = int(heart_match.group(1))
        data["heart_probability"] = float(heart_match.group(2))
        data["gender"] = "Male" if heart_match.group(3) == "1" else "Female"
        data["age"] = int(heart_match.group(4))

    diab_match = re.search(r"Diabetes Prediction.*?prediction\s+(\d).*?probability\s+([\d.]+).*?age\s+(\d+).*?gender\s+(\d)", text, re.DOTALL)
    if diab_match:
        data["diabetes_prediction"] = int(diab_match.group(1))
        data["diabetes_probability"] = float(diab_match.group(2))

    fat_match = re.search(r"Body Fat Prediction.*?prediction\s+(-?[\d.]+).*?Age\s+([\d.]+)", text, re.DOTALL)
    if fat_match:
        data["bodyfat_prediction"] = float(fat_match.group(1))
        data["bodyfat_age"] = float(fat_match.group(2))

    chest_match = re.search(r"Chest Pain\s+(\d+)", text)
    chol_match = re.search(r"High Cholesterol\s+(\d+)", text)
    data["chest_pain"] = int(chest_match.group(1)) if chest_match else 0
    data["cholesterol_status"] = int(chol_match.group(1)) if chol_match else 0
    return data

# -------- /tips Route --------
@app.route("/tips", methods=["GET", "POST"])
def tips():
    data = None
    if request.method == "POST":
        pdf = request.files.get("pdf")
        if not pdf or not pdf.filename.endswith(".pdf"):
            flash("Please upload a valid PDF report", "danger")
            return redirect("/tips")

        pdf_path = os.path.join("temp_uploads", "report.pdf")
        os.makedirs("temp_uploads", exist_ok=True)
        pdf.save(pdf_path)

        try:
            with fitz.open(pdf_path) as doc:
                text = "".join(page.get_text() for page in doc)

            user_data = extract_user_data_from_report(text)
            age = user_data.get("age")
            gender = user_data.get("gender")
            heart_risk = user_data.get("heart_prediction", 0)
            diabetes_risk = user_data.get("diabetes_prediction", 0)
            bodyfat_percent = user_data.get("bodyfat_prediction", None)
            chest_pain = user_data.get("chest_pain", 0)
            cholesterol_status = user_data.get("cholesterol_status", 0)

            bodyfat_risk = 1 if bodyfat_percent and bodyfat_percent > 25 else 0
            age_group = get_age_group(age)
            total_risks = sum([heart_risk, diabetes_risk, bodyfat_risk])
            health_status = "Excellent" if total_risks == 0 else "Good" if total_risks == 1 else "Fair" if total_risks == 2 else "Needs Attention"

            data = {
                "age": age,
                "gender": gender,
                "health_status": health_status,
                "heart": {"risk": heart_risk, "tips": get_tips("heart", age_group, "Risky" if heart_risk else "Safe")},
                "diabetes": {"risk": diabetes_risk, "tips": get_tips("diabetes", age_group, "Risky" if diabetes_risk else "Safe")},
                "bodyfat": {"risk": bodyfat_risk, "percent": bodyfat_percent, "tips": get_tips("bodyfat", age_group, "Risky" if bodyfat_risk else "Safe")},
                "heart_age_chart": create_heart_age_distribution_chart(heart_data, age),
                "chest_pain_chart": create_chest_pain_chart(heart_data, chest_pain),
                "cholesterol_chart": create_cholesterol_chart(heart_data, cholesterol_status),
                "age_bodyfat_chart": create_age_bodyfat_scatter(bodyfat_data, age, bodyfat_percent),
                "diabetes_age_chart": create_age_group_risk_chart(diabetes_data, age, 'Diabetes Cases by Age Group'),
                "bodyfat_gauge_chart": create_bodyfat_gauge_chart(bodyfat_percent)
            }

        except Exception as e:
            flash(f"PDF read error: {e}", "danger")
            return redirect("/tips")
        finally:
            try: os.remove(pdf_path)
            except OSError: pass

    return render_template("tips.html", data=data)



#-----------------------------Chatbot Section -----------------------------------
# from flask import Flask, render_template, request, jsonify
# import openai
# import os
# from dotenv import load_dotenv





@app.route("/chatbot")
def home():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    user_message = request.form["msg"]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a friendly chatbot."},
                {"role": "user", "content": user_message}
            ]
        )

        bot_reply = response.choices[0].message.content
        return jsonify({"reply": bot_reply})

    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})





#---------------------------Run app-----------------------------------
if __name__ == "__main__":
    app.run(debug=True)
