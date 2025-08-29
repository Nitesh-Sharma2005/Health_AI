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
from flask import Flask, render_template, request
from docx import Document
import random
import base64
import matplotlib.pyplot as plt
from io import BytesIO



DOCX_PATH = "health_tips_full.docx"  # file ka path

# -------- Helper: Age group from age --------
def get_age_group(age):
    if age < 40:
        return "Young (<40)"
    elif age < 60:
        return "Middle (40–59)"
    else:
        return "Senior (60+)"

# -------- Helper: Bodyfat category --------
def get_bodyfat_category(percent, risk):
    if risk == 0:
        return "Safe"
    if percent is None:
        return "High Body Fat"  # default if risky but % missing
    return "High Body Fat" if percent > 25 else "Low Body Fat"

# -------- Parse docx into structured dict --------
def parse_docx(path):
    doc = Document(path)
    tips_data = {}
    current_section = None
    current_age_group = None
    current_risk_type = None
    current_tip_type = None

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Detect section
        if text.startswith("🫀 SECTION 1: HEART HEALTH"):
            current_section = "heart"
            tips_data[current_section] = {}
            continue
        elif text.startswith("🩸 SECTION 2: DIABETES MANAGEMENT"):
            current_section = "diabetes"
            tips_data[current_section] = {}
            continue
        elif text.startswith("⚖️ SECTION 3: BODY FAT CONTROL"):
            current_section = "bodyfat"
            tips_data[current_section] = {}
            continue

        # Detect age group
        if any(k in text for k in ["Young", "Middle", "Senior"]):
            current_age_group = text
            tips_data[current_section].setdefault(current_age_group, {})
            continue

        # Detect risk type
        if text in ["Risky", "Safe", "High Body Fat", "Low Body Fat"]:
            current_risk_type = text
            tips_data[current_section][current_age_group].setdefault(current_risk_type, {})
            continue

        # Detect tip type
        if text in ["Diet", "Exercise"]:
            current_tip_type = text.lower()
            tips_data[current_section][current_age_group][current_risk_type].setdefault(current_tip_type, [])
            continue

        # Otherwise: it's a tip line
        if current_tip_type:
            tips_data[current_section][current_age_group][current_risk_type][current_tip_type].append(text.lstrip("- "))

    return tips_data

# Load tips at startup
TIPS = parse_docx(DOCX_PATH)

# -------- Chart function --------
def create_chart(percent):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie([percent, 100 - percent], labels=[f"{percent}%", ""], startangle=90, wedgeprops={'width':0.4})
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# -------- Pick tips --------
def get_tips(section, age_group, risk_type, n=2):
    pool = TIPS.get(section, {}).get(age_group, {}).get(risk_type, {})
    diet_list = pool.get("diet", [])
    exercise_list = pool.get("exercise", [])
    return {
        "diet": random.sample(diet_list, min(n, len(diet_list))),
        "exercise": random.sample(exercise_list, min(n, len(exercise_list)))
    }

# -------- /tips route --------
import fitz  # pip install pymupdf
import re
import os

@app.route("/tips", methods=["GET", "POST"])
def tips():
    data = None
    if request.method == "POST":
        pdf = request.files.get("pdf")
        if not pdf or not pdf.filename.endswith(".pdf"):
            flash("Please upload a valid PDF report", "danger")
            return redirect("/tips")

        # Save temp file
        pdf_path = "temp_uploads/report.pdf"
        os.makedirs("temp_uploads", exist_ok=True)
        pdf.save(pdf_path)

        # ---- Extract text from PDF ----
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            flash(f"PDF read error: {e}", "danger")
            return redirect("/tips")

        # ---- Extract Age (optional) ----
        age_match = re.search(r"Age\s*:\s*(\d+)", text)
        age = int(age_match.group(1)) if age_match else 30  # default

        # ---- Heart prediction ----
        heart_match = re.search(r"Heart Prediction.*?prediction\s+(\d)", text, re.DOTALL)
        heart_risk = int(heart_match.group(1)) if heart_match else 0

        # ---- Diabetes prediction ----
        diab_match = re.search(r"Diabetes Prediction.*?prediction\s+(\d)", text, re.DOTALL)
        diabetes_risk = int(diab_match.group(1)) if diab_match else 0

        # ---- Body fat prediction ----
        fat_match = re.search(r"Body Fat Prediction.*?prediction\s+([\d.]+)", text, re.DOTALL)
        bodyfat_percent = float(fat_match.group(1)) if fat_match else None
        bodyfat_risk = 1 if bodyfat_percent and bodyfat_percent > 25 else 0

        # ---- Age group + types ----
        age_group = get_age_group(age)
        heart_type = "Risky" if heart_risk else "Safe"
        diabetes_type = "Risky" if diabetes_risk else "Safe"
        bodyfat_type = get_bodyfat_category(bodyfat_percent, bodyfat_risk)

        # ---- Prepare data for HTML ----
        data = {
            "age": age,
            "heart": {
                "risk": heart_risk,
                "tips": get_tips("heart", age_group, heart_type, n=2 if heart_risk else 1)
            },
            "diabetes": {
                "risk": diabetes_risk,
                "tips": get_tips("diabetes", age_group, diabetes_type, n=2 if diabetes_risk else 1)
            },
            "bodyfat": {
                "risk": bodyfat_risk,
                "percent": bodyfat_percent,
                "tips": get_tips("bodyfat", age_group, bodyfat_type, n=2 if bodyfat_risk else 1),
                "chart": create_chart(bodyfat_percent) if bodyfat_percent else None
            }
        }

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