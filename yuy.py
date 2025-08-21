from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
import numpy as np
import joblib
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg')   # ✅ Use non-GUI backend to avoid Tkinter threading errors
import matplotlib.pyplot as plt

import io
import base64
import logging
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
import datetime
import openai
from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify


# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Precautions and Do's/Don'ts per CKD Stage


app = Flask(__name__)
app.secret_key = os.urandom(24)




import os
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify


# Load API Key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# Use the correct model path
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")




@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message", "")
    try:
        response = model.generate_content(user_input)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})





@app.route('/predict_clinical', methods=['POST'])
def predict_clinical():
    # your existing prediction logic
    ckd_presence = True  # example output
    ckd_stage = "Stage 2"  # example stage


    stage_precautions = {
        "Stage 1": {
            "precautions": [
                "Maintain healthy diet",
                "Control blood pressure",
                "Exercise regularly",
                "Avoid smoking",
                "Limit salt intake"
            ],
            "dos": [
                "Drink 2-3 litres water daily",
                "Monitor BP monthly",
                "Get yearly kidney tests",
                "Eat balanced meals",
                "Manage stress"
            ],
            "donts": [
                "Avoid NSAIDs unnecessarily",
                "Don't ignore urinary symptoms",
                "Avoid dehydration",
                "Do not self-medicate",
                "Avoid excessive protein intake"
            ]
        },
        "Stage 2": {
            "precautions": [
                "Control blood pressure and sugar",
                "Avoid nephrotoxic drugs",
                "Maintain healthy weight",
                "Eat kidney-friendly diet",
                "Regular nephrologist visits"
            ],
            "dos": [
                "Reduce salt intake",
                "Hydrate adequately",
                "Exercise as advised",
                "Monitor kidney function every 6 months",
                "Continue prescribed medications"
            ],
            "donts": [
                "Avoid painkillers like NSAIDs",
                "Don’t consume high phosphate foods",
                "Avoid herbal supplements without advice",
                "Don’t miss follow-up appointments",
                "Avoid smoking and alcohol excess"
            ]
        },
        "Stage 3": {
            "precautions": [
                "Follow renal diet strictly",
                "Monitor BP & sugar regularly",
                "Consult nephrologist quarterly",
                "Limit protein intake",
                "Prepare for possible dialysis discussion"
            ],
            "dos": [
                "Eat low potassium foods",
                "Take medications as prescribed",
                "Stay active as per capacity",
                "Monitor creatinine & GFR",
                "Reduce processed food consumption"
            ],
            "donts": [
                "Avoid salt substitutes",
                "No over-the-counter NSAIDs",
                "Don’t ignore swelling or fatigue",
                "Avoid dehydration",
                "Don’t delay medical visits"
            ]
        },
        "Stage 4": {
            "precautions": [
                "Strict diet & fluid management",
                "Plan for dialysis or transplant",
                "Regular labs and follow-up",
                "Consult renal dietitian",
                "Limit phosphorus and potassium intake"
            ],
            "dos": [
                "Discuss dialysis options early",
                "Follow fluid restrictions",
                "Take phosphate binders if prescribed",
                "Report symptoms promptly",
                "Attend all scheduled appointments"
            ],
            "donts": [
                "Avoid missed dialysis sessions (if started)",
                "Don’t eat processed meats",
                "Avoid excess fluid intake",
                "No OTC medications without approval",
                "Avoid magnesium-containing antacids"
            ]
        },
        "Stage 5": {
            "precautions": [
                "Dialysis or transplant required",
                "Follow nephrologist plan strictly",
                "Manage fluid & diet intake",
                "Prevent infections",
                "Maintain mental health support"
            ],
            "dos": [
                "Attend all dialysis sessions",
                "Follow strict fluid restrictions",
                "Maintain catheter hygiene",
                "Take medications regularly",
                "Stay connected with care team"
            ],
            "donts": [
                "Don’t skip dialysis",
                "Avoid fluid overload",
                "No herbal remedies without approval",
                "Don’t miss medication doses",
                "Avoid contact with infections"
            ]
        }
    }


    if prediction['CKD_Presence'] and f"Stage {prediction['CKD_Stage']}" in stage_precautions:
        precautions_info = stage_precautions[f"Stage {prediction['CKD_Stage']}"]
    else:
        precautions_info = None


    return render_template(
        'clinical_output.html',
        ckd_presence=prediction['CKD_Presence'],
        ckd_stage=prediction['CKD_Stage'],
        precautions_info=precautions_info,
        pdf_filename=pdf_filename
    )






########################
# --- Normal Ranges Dictionary ---
########################


normal_ranges = {
    "sg": "1.005 - 1.030",  # Specific Gravity
    "al": "0",  # Albumin normally absent
    "su": "0",  # Sugar normally absent
    "bgr": "< 140 mg/dl",  # Blood Glucose Random
    "bu": "7 - 20 mg/dl",  # Blood Urea
    "sc": "0.6 - 1.2 mg/dl (M), 0.5 - 1.1 mg/dl (F)",  # Serum Creatinine
    "sod": "135 - 145 mEq/L",  # Sodium
    "pot": "3.5 - 5.0 mEq/L",  # Potassium
    "hemo": "13.8 - 17.2 g/dl (M), 12.1 - 15.1 g/dl (F)",  # Hemoglobin
    "pcv": "40 - 54% (M), 36 - 48% (F)",  # Packed Cell Volume
    "wc": "4,500 - 11,000 cells/cu mm",  # White Blood Cell Count
    "rc": "4.7 - 6.1 million/cu mm (M), 4.2 - 5.4 million/cu mm (F)",  # Red Blood Cell Count
    "htn": "No (Normal BP)",  # Hypertension
    "dm": "No",  # Diabetes Mellitus
    "cad": "No",  # Coronary Artery Disease
    "appet": "Good",  # Appetite
    "pe": "No",  # Pedal Edema
    "ane": "No"  # Anemia
}



########################
# --- Stage Precautions Dictionary ---
########################


stage_precautions = {
    "Stage 1": { ... },
    "Stage 2": { ... },
    "Stage 3": { ... },
    "Stage 4": { ... },
    "Stage 5": { ... }
}




########################
# --- Clinical Model Setup ---
########################


MODELS_DIR = 'trained_models'
CKD_MODEL_PATH = os.path.join(MODELS_DIR, 'ckd_presence_model.joblib')
STAGE_MODEL_PATH = os.path.join(MODELS_DIR, 'ckd_stage_model.joblib')


try:
    ckd_model = joblib.load(CKD_MODEL_PATH)
    stage_model = joblib.load(STAGE_MODEL_PATH)
    print("Clinical models loaded successfully!")
except FileNotFoundError:
    print(f"Clinical model files not found. Please check your '{MODELS_DIR}' folder.")
    ckd_model = None
    stage_model = None


try:
    df_original = pd.read_csv('ckd_dataset.csv')
    df_original = df_original.iloc[2:].copy()
    df_original.reset_index(drop=True, inplace=True)
    df_original.columns = df_original.columns.str.lower().str.replace(r'[^a-zA-Z0-9_]', '', regex=True).str.replace(' ', '_')


    def convert_to_numeric_initial_robust(value):
        value_str = str(value).strip().lower()
        if value_str in ['nan', '?', ' p ']:
            return np.nan
        if isinstance(value, (int, float)):
            return value
        try:
            if ' - ' in value_str:
                lower, upper = map(float, value_str.split(' - '))
                return (lower + upper) / 2
            elif value_str.startswith('<'):
                return float(value_str[1:]) - 0.01
            elif value_str.startswith('≥'):
                return float(value_str[1:]) + 0.01
            else:
                return float(value_str)
        except ValueError:
            return np.nan


    for col in df_original.columns:
        df_original[col] = df_original[col].apply(convert_to_numeric_initial_robust)
        if df_original[col].dtype == 'object':
            df_original[col] = pd.to_numeric(df_original[col], errors='coerce')


    if 'class' in df_original.columns and df_original['class'].dtype == 'object':
        df_original['class'] = df_original['class'].map({'ckd': 1, 'notckd': 0})
    if 'stage' in df_original.columns and df_original['stage'].dtype == 'object':
        stage_mapping = {'s1': 1, 's2': 2, 's3': 3, 's4': 4, 's5': 5}
        df_original['stage'] = df_original['stage'].map(stage_mapping)


    df_original['grf'] = pd.to_numeric(df_original['grf'], errors='coerce')
    GLOBAL_GRF_MEDIAN = df_original['grf'].median() if 'grf' in df_original.columns else None
    if GLOBAL_GRF_MEDIAN:
        df_original['grf'].fillna(GLOBAL_GRF_MEDIAN, inplace=True)


    feature_columns = [col for col in df_original.columns if col not in ['class', 'stage', 'affected']]
    GLOBAL_TRAINING_COLUMNS = pd.Index(feature_columns)


except Exception as e:
    print(f"Error loading/preprocessing original clinical dataset: {e}")
    GLOBAL_GRF_MEDIAN = None
    GLOBAL_TRAINING_COLUMNS = None


def preprocess_clinical_input(input_data):
    processed_input = pd.Series(input_data)


    def convert_to_numeric_single(value):
        value_str = str(value).strip().lower()
        if value_str in ['nan', '?', ' p ']:
            return np.nan
        if isinstance(value, (int, float)):
            return value
        try:
            if ' - ' in value_str:
                lower, upper = map(float, value_str.split(' - '))
                return (lower + upper) / 2
            elif value_str.startswith('<'):
                return float(value_str[1:]) - 0.01
            elif value_str.startswith('≥'):
                return float(value_str[1:]) + 0.01
            else:
                return float(value_str)
        except ValueError:
            return np.nan


    for col in processed_input.index:
        processed_input[col] = convert_to_numeric_single(processed_input[col])


    processed_df = pd.DataFrame([processed_input.reindex(GLOBAL_TRAINING_COLUMNS)], columns=GLOBAL_TRAINING_COLUMNS)
    if 'grf' in processed_df.columns and processed_df['grf'].isnull().any() and GLOBAL_GRF_MEDIAN is not None:
        processed_df['grf'].fillna(GLOBAL_GRF_MEDIAN, inplace=True)
    processed_df = processed_df.apply(pd.to_numeric, errors='coerce')


    return processed_df


def clinical_predict_ckd_and_stage(input_data_dict):
    full_input_data = {col: np.nan for col in GLOBAL_TRAINING_COLUMNS}
    full_input_data.update(input_data_dict)
    processed_df = preprocess_clinical_input(full_input_data)
    if ckd_model is None or stage_model is None:
        return {'CKD_Presence': None, 'CKD_Stage': None}


    ckd_pred = ckd_model.predict(processed_df)[0]
    result = {'CKD_Presence': bool(ckd_pred), 'CKD_Stage': None}
    if ckd_pred == 1:
        stage_pred = stage_model.predict(processed_df)[0]
        result['CKD_Stage'] = int(stage_pred)
    return result


def generate_clinical_report_pdf(patient_data, prediction_result):
    reports_dir = os.path.join('static', 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    pdf_filename = f"clinical_report_{timestamp}.pdf"
    pdf_filepath = os.path.join(reports_dir, pdf_filename)

    c = canvas.Canvas(pdf_filepath, pagesize=letter)
    width, height = letter

    # Header bar and logo
    c.setFillColorRGB(0.20, 0.45, 0.75)  # blue color
    c.rect(0, height - 60, width, 60, fill=True, stroke=False)
    try:
        c.drawImage('static/images/logo.png', 30, height - 55, width=50, height=50, mask='auto')
    except Exception:
        pass
    c.setFont("Helvetica-Bold", 18)
    c.setFillColorRGB(1, 1, 1)  # white text
    c.drawString(90, height - 40, "Clinical CKD Prediction Report")
    c.setFillColorRGB(0, 0, 0)  # reset color

    # Timestamp below header
    c.setFont("Helvetica", 12)
    c.drawString(400, height - 40, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y = height - 90

    # Section title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Patient Details with Normal Ranges (Tabular):")
    y -= 30

    # Prepare table data
    table_data = [['Parameter', 'Entered Value', 'Normal Range']]
    for key, value in patient_data.items():
        range_str = normal_ranges.get(key, 'N/A')
        table_data.append([key, str(value), range_str])

    # Create table
    table = Table(table_data, colWidths=[150, 150, 180])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    table.wrapOn(c, width, height)
    table.drawOn(c, 50, y - (20 * len(table_data)))

    y -= (20 * len(table_data)) + 20

    # Prediction results section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Prediction Result:")
    y -= 20
    c.setFont("Helvetica", 12)
    c.drawString(60, y, f"CKD Presence: {'Yes' if prediction_result['CKD_Presence'] else 'No'}")
    y -= 20
    c.drawString(60, y, f"CKD Stage: {prediction_result['CKD_Stage'] if prediction_result['CKD_Stage'] else 'Not applicable'}")

    c.save()
    return f"reports/{pdf_filename}"



########################
# --- Image Model Setup ---
########################


UPLOAD_FOLDER = os.path.join('static', 'uploads')
REPORTS_FOLDER = os.path.join('static', 'reports')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    logger.info("Loading image model...")
    image_model = load_model('kidney_model2.h5')
    logger.info("Image model loaded successfully")
except Exception as e:
    logger.error(f"Error loading image model: {e}")
    image_model = None


def preprocess_nii_image(filepath):
    try:
        img = nib.load(filepath).get_fdata()
        if len(img.shape) == 3:
            img = img[:, :, img.shape[2] // 2]
        img = img.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        img = tf.image.resize(img[..., np.newaxis], [256, 256])
        return np.expand_dims(img, axis=0)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return None


def generate_image_report_pdf(filename, result):
    reports_dir = os.path.join('static', 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    pdf_filename = f"image_report_{timestamp}.pdf"
    pdf_filepath = os.path.join(reports_dir, pdf_filename)

    c = canvas.Canvas(pdf_filepath, pagesize=letter)
    width, height = letter

    # Header bar and logo
    c.setFillColorRGB(0.20, 0.45, 0.75)
    c.rect(0, height-60, width, 60, fill=True, stroke=False)
    try:
        c.drawImage('static/images/logo.png', 30, height-55, width=50, height=50, mask='auto')
    except Exception:
        pass
    c.setFont("Helvetica-Bold", 18)
    c.setFillColorRGB(1,1,1)
    c.drawString(90, height-40, "Imaging CKD Prediction Report")
    c.setFillColorRGB(0,0,0)
    c.setFont("Helvetica", 11)
    c.drawString(420, height-40, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y = height - 90

    # Section divider
    c.setStrokeColorRGB(0.7,0.7,0.7)
    c.line(50, y, width-50, y)
    y -= 25

    # Prediction status (colored)
    c.setFont("Helvetica-Bold", 14)
    c.setFillColorRGB(0.25,0.31,0.68)
    c.drawString(50, y, "Prediction Summary")
    c.setFillColorRGB(0,0,0)
    y -= 25

    # Show prediction details
    c.setFont("Helvetica", 12)
    c.drawString(60, y, f"File Name: {filename}")
    y -= 20
    # Color for CKD status
    if result["ckd_status"].lower() == "ckd detected":
        c.setFillColorRGB(0.92,0.16,0.16)
    else:
        c.setFillColorRGB(0.18,0.7,0.33)
    c.drawString(60, y, f"CKD Status: {result['ckd_status']}")
    c.setFillColorRGB(0,0,0)
    y -= 20
    c.drawString(60, y, f"Confidence: {result['confidence']:.2f}")
    y -= 20
    stage_txt = str(result['stage']) if result['stage'] else "Not applicable"
    c.drawString(60, y, f"Predicted Stage: {stage_txt}")
    y -= 30

    # Optional: Add advice message
    advice = ""
    if result["ckd_status"].lower() == "ckd detected":
        advice = "Consult a nephrologist. Early intervention is crucial for CKD management."
    else:
        advice = "Maintain a healthy lifestyle and annual check-ups."
    c.setFont("Helvetica-Oblique", 11)
    c.drawString(60, y, f"Note: {advice}")
    y -= 25

    # Footer disclaimer
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.45,0.45,0.45)
    c.drawString(60, 30, "This is an AI-generated report. For investigational use only. Not a substitute for clinical diagnosis.")

    c.save()
    return f"reports/{pdf_filename}"


########################
# --- Routes ---
########################


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/clinical_input')
def clinical_input():
    return render_template('clinical_input.html')


@app.route('/clinical_predict', methods=['POST'])
def clinical_predict():
    patient_data = {}
    for key, value in request.form.items():
        patient_data[key] = value


    prediction = clinical_predict_ckd_and_stage(patient_data)
    pdf_filename = generate_clinical_report_pdf(patient_data, prediction)


    return render_template('clinical_output.html',
                          ckd_presence=prediction['CKD_Presence'],
                          ckd_stage=prediction['CKD_Stage'],
                          pdf_filename=pdf_filename)


@app.route('/image_input')
def image_input():
    return render_template('image_input.html')


@app.route('/image_predict', methods=['POST'])
def image_predict():
    if 'nii_file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('image_input'))
    file = request.files['nii_file']
    if not file or file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('image_input'))
    if not file.filename.lower().endswith('.nii'):
        flash('Only .nii files are allowed', 'error')
        return redirect(url_for('image_input'))
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        processed_img = preprocess_nii_image(filepath)
        if processed_img is None or image_model is None:
            raise ValueError("Image processing or model error")


        predictions = image_model.predict(processed_img)
        ckd_prob = predictions[0][0]
        stage_probs = predictions[1][0]
        ckd_status = 'CKD' if ckd_prob > 0.5 else 'Non-CKD'
        predicted_stage = np.argmax(stage_probs) + 1


        img_slice = nib.load(filepath).get_fdata()
        if len(img_slice.shape) == 4:
            img_slice = img_slice[:, :, :, 0]
        slice_idx = img_slice.shape[2] // 2 if len(img_slice.shape) == 3 else 0
        plt.figure(figsize=(6, 6))
        plt.imshow(img_slice[:, :, slice_idx], cmap='gray')
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()


        result = {
            'ckd_status': ckd_status,
            'confidence': float(ckd_prob if ckd_status == 'CKD' else 1 - ckd_prob),
            'stage': predicted_stage if ckd_status == 'CKD' else None,
            'image_data': img_base64,
            'filename': filename
        }

        # Generate PDF report for image prediction
        pdf_filename = generate_image_report_pdf(filename, result)

        return render_template('image_result.html', result=result, pdf_filename=pdf_filename)


    except Exception as e:
        logger.error(f"Image processing error: {e}")
        flash(f'Error processing image: {str(e)}', 'error')
        return redirect(url_for('image_input'))


########################
# Run app
########################


if __name__ == '__main__':
    app.run(debug=True)
