
# Student Stress Level Detector

This project is an AI-based tool that helps students understand their stress levels using everyday academic and lifestyle information. It also gives simple and practical suggestions to manage stress better.

## What This App Does

* Predicts a student’s stress level using a trained machine learning model
* Gives personalized tips based on the predicted stress level
* Looks at common student habits and pressures such as:

  * Daily study hours
  * Sleep routine
  * Anxiety and exam pressure
  * Number of breaks
  * CGPA
  * Gender
  * Educational institution

## How to Use the App

1. Enter how many hours you study and sleep daily
2. Rate your anxiety level and exam pressure on a scale of 0 to 10
3. Add how many breaks you take per day and your CGPA
4. Choose your gender and college or university
5. Click on **“Predict Stress Level”**
6. View your stress score along with helpful suggestions

## Stress Levels Explained

* **Low Stress**
  You are managing your stress well. Keep following healthy habits.

* **Moderate Stress**
  Some stress is present. Small changes in routine can help improve balance.

* **High Stress**
  Stress levels are high and should be addressed seriously. Proper rest and support are recommended.

## Important Note

This application is meant for learning and awareness only. It is not a medical or psychological diagnosis. If you feel overwhelmed or stressed for a long time, please talk to a qualified counselor or mental health professional.

## Technology Used

* **Interface**: Gradio
* **Machine Learning Model**: Neural Network (MLPRegressor)
* **Libraries**: scikit-learn, pandas, numpy

## About the Model

The model is trained on student-related data that includes academic workload and lifestyle patterns. All inputs are standardized before prediction, and the model outputs a stress score that helps categorize stress levels.

