print("Hello, this is app.py")
import gradio as gr
import numpy as np
import joblib
import pandas as pd

# Load the trained model and scaler
try:
    model = joblib.load("stress_nn_model.pkl")
    scaler = joblib.load("model.pkl")
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None


def stress_category(score):
    """Categorize stress level based on score"""
    # If model outputs very small values (0-1 range after scaling)
    if score < 1:
        # Treat as normalized 0-1 scale
        if score < 0.35:
            return "Low Stress"
        elif score < 0.65:
            return "Moderate Stress"
        else:
            return "High Stress"
    else:
        # Treat as actual stress level (dataset range: -3 to 40, mostly 2.5-8)
        if score < 4.0:
            return "Low Stress"
        elif score < 6.5:
            return "Moderate Stress"
        else:
            return "High Stress"


def generate_report(level):
    """Generate stress management report based on level"""
    if level == "Low Stress":
        return (
            "✅ Stress Level: Low\n\n"
            "• You are managing stress well.\n"
            "• Maintain 7–8 hours of sleep.\n"
            "• Keep balanced study and social life.\n"
            "• Continue physical activity."
        )

    elif level == "Moderate Stress":
        return (
            "⚠️ Stress Level: Moderate\n\n"
            "• Take short study breaks.\n"
            "• Reduce screen time before sleep.\n"
            "• Practice meditation or deep breathing.\n"
            "• Increase light physical activity."
        )

    else:
        return (
            "🚨 Stress Level: High\n\n"
            "• High stress detected.\n"
            "• Prioritize rest and sleep.\n"
            "• Reduce workload temporarily.\n"
            "• Talk to a counselor or trusted person.\n"
            "• Daily exercise and relaxation is advised."
        )


def predict_stress(study_hours, sleep_hours, anxiety_level, exam_pressure, 
                   breaks_per_day, cgpa, gender, college_name):
    """Predict stress level based on user inputs"""
    
    if model is None or scaler is None:
        return "❌ Error: Model not loaded", "", "Please ensure model files are in the directory."
    
    try:
        # Encode gender (Male=1, Female=0)
        gender_encoded = 1 if gender == "Male" else 0
        
        # Create college one-hot encoding matching EXACT training features
        # Order: "UET", COMSATS, FAST, LUMS, NUST, PU, UET
        college_mapping = {
            "COMSATS": [0, 1, 0, 0, 0, 0, 0],
            "FAST":    [0, 0, 1, 0, 0, 0, 0],
            "LUMS":    [0, 0, 0, 1, 0, 0, 0],
            "NUST":    [0, 0, 0, 0, 1, 0, 0],
            "PU":      [0, 0, 0, 0, 0, 1, 0],
            "UET":     [0, 0, 0, 0, 0, 0, 1]
        }
        college_features = college_mapping.get(college_name, [0, 0, 1, 0, 0, 0, 0])  # Default to FAST
        
        # Create feature array matching EXACT training order
        # ['study_hours', 'sleep_hours', 'anxiety_level', 'exam_pressure', 
        #  'breaks_per_day', 'gender', 'cgpa', 
        #  'college_name_"UET"', 'college_name_COMSATS', 'college_name_FAST',
        #  'college_name_LUMS', 'college_name_NUST', 'college_name_PU', 'college_name_UET']
        
        features = [
            study_hours,
            sleep_hours,
            anxiety_level,
            exam_pressure,
            breaks_per_day,
            gender_encoded,
            cgpa
        ] + college_features
        
        # Convert to DataFrame with exact feature names
        feature_names = [
            'study_hours', 'sleep_hours', 'anxiety_level', 'exam_pressure',
            'breaks_per_day', 'gender', 'cgpa',
            'college_name_"UET"', 'college_name_COMSATS', 'college_name_FAST',
            'college_name_LUMS', 'college_name_NUST', 'college_name_PU', 'college_name_UET'
        ]
        
        input_df = pd.DataFrame([features], columns=feature_names)
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        
        # Get stress score (model directly outputs stress level, not normalized)
        stress_score = float(prediction)
        
        # Debug: Print raw prediction
        print(f"Raw prediction: {stress_score}")
        
        # Clamp to reasonable range based on dataset (mostly 2-8)
        stress_score = max(-3, min(40, stress_score))
        
        # Get stress category using actual score
        level = stress_category(stress_score)
        
        # Generate report using your function
        report = generate_report(level)
        
        # Format output - show actual stress score
        score_display = f"{stress_score:.2f}"
        
        # Generate report using your function
        report = generate_report(level)
        
        # Format output - show actual stress score
        score_display = f"{stress_score:.2f}"
        
        return score_display, level, report
        
    except Exception as e:
        return "Error", "Error", f"Prediction failed: {str(e)}"


# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Student Stress Detector") as demo:
    
    gr.Markdown(
        """
        # 🎓 Student Stress Level Detector
        ### AI-Powered Stress Analysis with Personalized Recommendations
        Enter your information to predict your stress level and receive tailored advice.
        """
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📚 Academic & Lifestyle Inputs")
            
            study_hours = gr.Number(
                label="Study Hours (per day)",
                value=5.0,
                minimum=0,
                maximum=24
            )
            
            sleep_hours = gr.Number(
                label="Sleep Hours (per day)",
                value=6.0,
                minimum=0,
                maximum=24
            )
            
            anxiety_level = gr.Slider(
                minimum=0,
                maximum=10,
                value=5,
                step=1,
                label="Anxiety Level (0-10)"
            )
            
            exam_pressure = gr.Slider(
                minimum=0,
                maximum=10,
                value=5,
                step=1,
                label="Exam Pressure (0-10)"
            )
        
        with gr.Column():
            gr.Markdown("### 👤 Personal Information")
            
            breaks_per_day = gr.Number(
                label="Breaks Per Day",
                value=2,
                minimum=0,
                maximum=20
            )
            
            cgpa = gr.Number(
                label="CGPA",
                value=3.0,
                minimum=0.0,
                maximum=4.0,
                step=0.01
            )
            
            gender = gr.Radio(
                choices=["Male", "Female"],
                label="Gender",
                value="Male"
            )
            
            college_name = gr.Dropdown(
                choices=["COMSATS", "FAST", "LUMS", "NUST", "PU", "UET"],
                label="College Name",
                value="FAST"
            )
    
    predict_btn = gr.Button("🔍 Predict Stress Level", variant="primary", size="lg")
    
    gr.Markdown("---")
    gr.Markdown("### 📊 Results")
    
    with gr.Row():
        with gr.Column(scale=1):
            stress_score = gr.Textbox(
                label="Stress Score",
                interactive=False
            )
        
        with gr.Column(scale=1):
            stress_level = gr.Textbox(
                label="Stress Level",
                interactive=False
            )
    
    advice_output = gr.Textbox(
        label="Personalized Advice & Recommendations",
        lines=10,
        interactive=False
    )
    
    # Set up the prediction
    predict_btn.click(
        fn=predict_stress,
        inputs=[
            study_hours,
            sleep_hours,
            anxiety_level,
            exam_pressure,
            breaks_per_day,
            cgpa,
            gender,
            college_name
        ],
        outputs=[stress_score, stress_level, advice_output]
    )
    
    gr.Markdown(
        """
        ---
        ### ℹ️ Note
        This tool uses machine learning to predict stress levels based on your inputs.
        It should be used for informational purposes and does not replace professional mental health advice.
        """
    )
)