import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Anxiety Level Predictor", layout="wide")
st.title("Anxiety Level Predictor")
st.markdown("---")

try:
    # Load the model and files
    model = joblib.load('models/anxiety_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    columns_info = joblib.load('models/columns.pkl')
    
    # Extract information from the files
    numerical_features = columns_info['num_cols']
    categorical_features = columns_info['cat_cols']
    model_columns = columns_info['model_columns']
    
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.error("Did you run main.py first to generate the model files?")
    model_loaded = False

# navigation section in the sidebar
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Select Section", options=["Prediction", "Data Analysis Visualizations"])
st.sidebar.markdown("---")

if model_loaded:

    st.sidebar.header("Model Information")
    st.sidebar.info("Model loaded successfully")

# Add instructions for using the app
st.sidebar.header("How to Use This App")
st.sidebar.write("""
1. Enter your personal information in all fields
2. Click the 'Predict Anxiety Level' button
3. View your predicted anxiety level and contributing factors
""")
st.sidebar.markdown("---")


# Main content based on navigation selection
if app_mode == "Prediction":
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    # Get user inputs
    with col1:
        st.subheader("Personal Information")
        age = st.number_input("Age", min_value=16, max_value=80, value=30)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
        occupation = st.selectbox("Occupation", 
                                options=["Student", "Engineer", "Doctor", "Nurse", 
                                        "Teacher", "Artist", "Athlete", "Lawyer", 
                                        "Scientist", "Chef", "Musician", "Freelancer", "Other"])
        sleep_hours = st.slider("Sleep Hours", min_value=3.0, max_value=12.0, value=7.0, step=0.1)
        physical_activity = st.slider("Physical Activity (hours/week)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    
    with col2:
        st.subheader("Health & Lifestyle")
        caffeine_intake = st.slider("Caffeine Intake (mg/day)", min_value=0, max_value=1200, value=200)
        alcohol_consumption = st.slider("Alcohol Consumption (drinks/week)", min_value=0, max_value=20, value=5)
        smoking = st.selectbox("Smoking", options=["Yes", "No"])
        family_history = st.selectbox("Family History of Anxiety", options=["Yes", "No"])
        stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
    
    # More inputs
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Physical Symptoms")
        heart_rate = st.slider("Heart Rate (bpm)", min_value=60, max_value=120, value=80)
        breathing_rate = st.slider("Breathing Rate (breaths/min)", min_value=10, max_value=30, value=15)
        sweating_level = st.slider("Sweating Level (1-5)", min_value=1, max_value=5, value=2)
        dizziness = st.selectbox("Dizziness", options=["Yes", "No"])
    
    with col4:
        st.subheader("Treatment & Support")
        medication = st.selectbox("Taking Medication", options=["Yes", "No"])
        therapy_sessions = st.slider("Therapy Sessions (per month)", min_value=0, max_value=10, value=2)
        recent_life_event = st.selectbox("Recent Major Life Event", options=["Yes", "No"])
        diet_quality = st.slider("Diet Quality (1-10)", min_value=1, max_value=10, value=6)
    
    # Predict button
    if st.button("Predict Anxiety Level"):
        if not model_loaded:
            st.error("Model not loaded. Please ensure model files are in the 'models' directory.") # Handle model loading error
        else:
            # Create a DataFrame with user inputs (matching the dataset's columns)
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender],
                'Occupation': [occupation],
                'Sleep Hours': [sleep_hours],
                'Physical Activity (hrs/week)': [physical_activity],
                'Caffeine Intake (mg/day)': [caffeine_intake],
                'Alcohol Consumption (drinks/week)': [alcohol_consumption],
                'Smoking': [smoking],
                'Family History of Anxiety': [family_history],
                'Stress Level (1-10)': [stress_level],
                'Heart Rate (bpm)': [heart_rate],
                'Breathing Rate (breaths/min)': [breathing_rate],
                'Sweating Level (1-5)': [sweating_level],
                'Dizziness': [dizziness],
                'Medication': [medication],
                'Therapy Sessions (per month)': [therapy_sessions],
                'Recent Major Life Event': [recent_life_event],
                'Diet Quality (1-10)': [diet_quality]
            })
            
            try:
                # Handle categorical features manually to ensure correct columns
                input_encoded = pd.DataFrame(index=[0]) # Create a DataFrame with a single row
                
                # Add numerical features directly
                for col in numerical_features:
                    input_encoded[col] = input_data[col].values
                
                # Handle categorical features manually
                for col in categorical_features:
                    # Get all possible values for this categorical feature from model_columns
                    feature_cols = [c for c in model_columns if c.startswith(f"{col}_")]
                    
                    # For each possible value, check if the input matches and set accordingly
                    for feature_col in feature_cols:
                        # Extract the value part after the underscore
                        value = feature_col.split('_', 1)[1]
                        # Set 1 if input matches this value, 0 otherwise
                        input_encoded[feature_col] = 1 if input_data[col].values[0] == value else 0
                
                # Ensure all columns match the training data
                for col in model_columns:
                    if col not in input_encoded.columns:
                        input_encoded[col] = 0
                input_encoded = input_encoded[model_columns]
                
                # Scale numerical features
                if len(numerical_features) > 0:
                    input_encoded[numerical_features] = scaler.transform(input_encoded[numerical_features])
                
                # Make prediction
                prediction = model.predict(input_encoded)[0]
                prediction = max(1, min(10, prediction))  # Ensure prediction is within 1-10 range
                
                # Display prediction
                st.success(f"Predicted Anxiety Level: {prediction:.2f}/10")
                
                # Visualization - anxiety scale
                fig, ax = plt.subplots(figsize=(10, 2)) # fix = figure size, ax = axes
                cmap = plt.cm.RdYlGn_r
                norm = plt.Normalize(1, 10)
                
                plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                            cax=ax, orientation='horizontal',
                            ticks=range(1, 11))
                
                ax.axvline(prediction, color='black', linewidth=4)
                ax.set_title('Anxiety Level Scale')
                ax.set_xticks(range(1, 11)) # set x axis ticks to 1-10 in order to align with scale
                ax.set_xticklabels(['1\nVery Low', '2', '3', '4', '5\nModerate', 
                                    '6', '7', '8', '9', '10\nVery High'])
                
                st.pyplot(fig)
                
                # Interpretation
                st.subheader("Interpretation")
                if prediction <= 3:
                    st.info("Low anxiety level. Generally indicates good mental well-being.")
                elif prediction <= 6:
                    st.warning("Moderate anxiety level. Consider some stress management techniques.")
                else:
                    st.error("High anxiety level. Consider consulting with a mental health professional.")
                
            except Exception as e:
                st.error("Error making prediction")
                st.write("Please ensure all input fields are correctly filled and try again.")

elif app_mode == "Data Analysis Visualizations":
    st.header("Data Analysis Visualizations")
    st.markdown("These visualizations show the analysis performed on the anxiety level dataset.")
    
    # Create tabs for different visualization categories
    viz_tabs = st.tabs(["Target Distribution", "Correlations", "PCA", "Model Performance", "Diagnostics"])
    
    # Tab 1: Target Distribution
    with viz_tabs[0]:
        st.subheader("Anxiety Level Distribution")
        try:
            anxiety_dist = plt.imread("plots/anxiety_distribution.png")
            st.image(anxiety_dist, caption="Distribution and boxplot of anxiety levels")
            st.markdown("""
            This plot shows:
            - The distribution of anxiety levels in the dataset
            - The boxplot showing median, quartiles, and potential outliers
            """)
        except Exception:
            st.warning("Plot not found. Run main.py to generate all visualizations.")
    
    # Tab 2: Correlations
    with viz_tabs[1]:
        st.subheader("Feature Correlations")
        
        # Top correlations
        try:
            top_corr = plt.imread("plots/top_correlations.png")
            st.image(top_corr, caption="Top features correlated with anxiety level")
            st.markdown("""
            This plot shows the features that have the strongest correlation with anxiety levels.
            - Positive correlation (to the right): As this factor increases, anxiety tends to increase
            - Negative correlation (to the left): As this factor increases, anxiety tends to decrease
            """)
        except Exception:
            st.info("Top correlations plot not found.")
            
        # Correlation heatmap
        try:
            corr_heatmap = plt.imread("plots/correlation_heatmap.png")
            st.image(corr_heatmap, caption="Correlation matrix between all numerical features")
            st.markdown("""
            The correlation heatmap shows how all numerical features relate to each other:
            - Red indicates positive correlation
            - Blue indicates negative correlation
            - The intensity of color shows the strength of correlation
            """)
        except Exception:
            st.info("Correlation heatmap not found.")
    
    # Tab 3: PCA
    with viz_tabs[2]:
        st.subheader("Principal Component Analysis")
        try:
            pca_plot = plt.imread("plots/pca_variance.png")
            st.image(pca_plot, caption="Explained variance by PCA components")
            st.markdown("""
            This plot shows:
            - How many principal components are needed to explain the variance in the data
            - The red dashed line indicates the 95% variance threshold
            - The cumulative explained variance increases with each additional component
            """)
        except Exception:
            st.warning("PCA plot not found. Run main.py to generate all visualizations.")
    
    # Tab 4: Model Performance
    with viz_tabs[3]:
        st.subheader("Model Performance")
        try:
            model_perf = plt.imread("plots/actual_vs_predicted.png")
            st.image(model_perf, caption="Actual vs Predicted Anxiety Levels")
            st.markdown("""
            This plot shows how well the model predictions match the actual anxiety levels:
            - The red dashed line represents perfect prediction
            - Points close to the line indicate accurate predictions
            - Scattered points indicate prediction errors
            """)
        except Exception:
            st.warning("Model performance plot not found. Run main.py to generate all visualizations.")
    
    # Tab 5: Diagnostics
    with viz_tabs[4]:
        st.subheader("Regression Diagnostics")
        try:
            diagnostics = plt.imread("plots/regression_diagnostics.png")
            st.image(diagnostics, caption="Regression diagnostic plots")
            st.markdown("""
            These diagnostic plots validate the model assumptions:
            - Residuals vs Fitted: Check for non-linearity and heteroscedasticity
            - Normal Q-Q: Check if residuals are normally distributed
            - Scale-Location: Check for constant variance of residuals
            - Residuals Distribution: Check the overall shape of residuals
            """)
        except Exception:
            st.warning("Diagnostic plots not found. Run main.py to generate all visualizations.")
