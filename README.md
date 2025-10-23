# HAPIVET_Assessment (Team-EZ)

**CNN-based Veterinary Animal Disease Detection**



1
**1.Problem Statement**

Animals can suffer from various diseases that are hard to identify visually by owners.
This project builds a CNN-based system to detect animal diseases from images and provide advice
for treatment or precaution.

**2. Objectives**
- Detect diseases in animal images using CNN.
- Provide treatment suggestions based on predicted disease.
- Evaluate model performance using metrics like accuracy, F1-score.
- Help owners take timely action for animal care.


**3. Dataset Description**
- 6 classes: Demodicosis, Dermatitis, Healthy, Hypersensitivity, Ringworm, Fungal_infections
- Images split into training   (~70-80%),  validation (~10-15%), and test (~10-15%)
- Folder structure example:
dataset/
train/
validation/
test/


**4. Solution Approach**
- Preprocess images: resize to 128x128, normalize pixel values.
- Train CNN with Conv2D, MaxPooling2D, BatchNormalization, Dropout.
- Use softmax output for 6 classes.
- Map predictions to suggestion dictionary for owner advice.
- Evaluate using classification report and confusion matrix.


**5. CNN Code**
(Training, saving, loading model, prediction with advice.)
6. Results
- Accuracy on validation: 0.67
- Weighted F1-score: 0.66
- Best performing classes: Healthy, Demodicosis, Ringworm
- Weak classes: Hypersensitivity, Fungal_infections
- Sample prediction: 'Image: 1000010494.jpg -> Predicted: Fungal_infections -> Suggestion: Consult
a vet for antifungal treatment.'

**7. Conclusion**
- The CNN-based system can detect animal diseases and provide actionable advice.
- Accuracy is good but can improve with more data, augmentation, and fine-tuning.
- Ready for deployment in a simple web/mobile interface for animal owners.




2


DOG CONDITION ANALYZER 
1. Project Description

The Dog Condition Analyzer is a machine learning-based application designed to predict potential health conditions in dogs based on textual clinical notes or observed symptoms. This tool can assist veterinarians, pet owners, or animal care professionals in quickly identifying likely conditions, enabling timely intervention and care.

2. Problem Statement

Veterinary diagnosis often relies on the observation of symptoms and laboratory results. However:

Manual diagnosis can be time-consuming and error-prone.

Pet owners may struggle to interpret symptoms or decide when to consult a vet.

Problem: Automate the process of predicting possible dog health conditions from textual symptom descriptions using machine learning.

3. Objective

Build a text-based classifier that can predict dog health conditions from symptom descriptions.

Provide confidence scores for predictions.

Enable fast and reliable preliminary diagnosis support.

4. Tools & Libraries Used
Tool / Library	Purpose
Python	Programming language for implementation
pandas	Data loading and manipulation
numpy	Numerical operations
re	Text cleaning using regular expressions
scikit-learn	Machine learning library for model training, evaluation, and TF-IDF vectorization
RandomForestClassifier	Ensemble classifier used for prediction
TF-IDF Vectorizer	Converts text into numerical features for the model
accuracy_score, classification_report	Evaluate model performance
5. Solution Approach

Data Collection:
Load a dataset containing dog symptoms (text) and corresponding health conditions (condition).

Data Cleaning & Preprocessing:

Convert text to lowercase.

Remove special characters, numbers, and extra spaces.

Handle missing values.

Feature Extraction:

Use TF-IDF Vectorization to transform text into numerical vectors suitable for machine learning.

Include unigrams and bigrams for better context understanding.

Model Training:

Use Random Forest Classifier to learn patterns from symptom descriptions.

Train on 80% of data and validate on 20% of data.

Model Evaluation:

Check accuracy and other metrics using classification_report.

Prediction Function:

Clean input text.

Transform it into TF-IDF vectors.

Predict the condition and provide confidence score.

Usage:

Input: Text describing dog symptoms.

Output: Predicted condition with probability.

6. Advantages

Fast preliminary diagnosis support for dog owners and veterinarians.

Can handle multiple symptoms and long text descriptions.

Uses a robust ensemble model (Random Forest) to improve prediction accuracy.

