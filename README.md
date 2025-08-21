# Kidney Disease Classification using DL

A user-friendly, AI-powered web app to detect and stage Chronic Kidney Disease (CKD) using deep learning models on both clinical data and kidney MRI images.

---

## Features

- **Dual Input:** Predict CKD from either clinical test results or kidney MRI scans (`.nii` format).
- **Deep Learning Models:** Uses pre-trained clinical and image-based models for accurate diagnosis and staging.
- **Professional Reports:** Downloads detailed PDF reports (with logo, prediction details, and summary table).
- **Visualization:** Embedded MRI slice preview in imaging reports.
- **CKD Chatbot:** Integrated chatbot for CKD education and support (powered by LLM APIs).
- **Dark/Light Mode:** Toggle for user comfort and accessibility.
- **Accessible UI:** Clean, mobile-friendly, responsive design (built with Tailwind CSS & Flask).
- **User Prediction History:** (Optional) Store and revisit recent predictions locally or per user.

---

## Demo

> **Add screenshots or animated GIFs here showing the UI, prediction, and report download pages.**

---

## Getting Started

### 1. **Clone the Repo**
 git clone https://github.com/saicharanreddynarra/Kidney-Disease-Classification-using-DL.git
 cd Kidney-Disease-Classification-using-DL


### 2. **Set Up Python Environment**

It is recommended to use a virtual environment.


### 3. **Install Dependencies**

pip install -r requirements.txt

## Usage

- **Home:** Start from the landing page, select prediction via clinical data or via MRI image.
- **Prediction:** Fill in the form (or upload MRI), click "Predict," then view your prediction/stage and download the PDF report.
- **Chatbot:** Use the chat icon for AI-powered CKD education and support.
- **Theme:** Use the toggle to switch between dark and light mode for easier viewing.

## File Structure

 project/
├── app.py
├── ckd_dataset.csv
├── requirements.txt
├── trained_models/
│ ├── ckd_presence_model.joblib
│ ├── ckd_stage_model.joblib
│ └── kidney_model2.h5
├── static/
│ ├── images/
│ ├── uploads/
│ └── reports/
├── templates/
│ ├── index.html
│ ├── clinical_input.html
│ ├── clinical_output.html
│ ├── image_input.html
│ └── image_result.html
└── README.md


## License

This project is licensed under the MIT License.
