import streamlit as st
import pandas as pd
import numpy as np
import os
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.pipeline.train_pipeline import run_train_pipeline
from src.exception import CustomException
from src.logger import logging


# Configure page
st.set_page_config(
    page_title="AI Impact on Jobs - Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .sub-header {
        color: #555;
        font-size: 1.2em;
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    """Main function to run the Streamlit app"""
    
    # Sidebar navigation
    st.sidebar.markdown("### 💼 AI Impact on Jobs")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "🤖 Train Model", "💰 Predict Salary", "📊 Project Info"]
    )
    
    if page == "🏠 Home":
        show_home()
    elif page == "🤖 Train Model":
        show_train_model()
    elif page == "💰 Predict Salary":
        show_predict_salary()
    elif page == "📊 Project Info":
        show_project_info()


def show_home():
    """Display home page"""
    st.markdown('<h1 class="main-header">🌍 AI Impact on Jobs Salary Predictor</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the AI Impact on Jobs Salary Prediction Application! This application 
    uses machine learning to predict job salaries based on various factors related to 
    AI and job market dynamics.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📌 Features")
        st.markdown("""
        - **Train Model**: Train machine learning models on historical salary data
        - **Predict Salary**: Predict salary for specific job profiles
        - **Multiple Models**: Compare performance of various ML algorithms
        - **Data Transformation**: Automated preprocessing and feature engineering
        """)
    
    with col2:
        st.markdown("### 🔧 Technologies Used")
        st.markdown("""
        - **Python**: Core language
        - **Scikit-learn**: ML algorithms
        - **XGBoost & CatBoost**: Advanced boosting models
        - **Streamlit**: Web interface
        - **Pandas & NumPy**: Data manipulation
        """)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 📊 How it Works
    1. **Data Ingestion**: Load and prepare the dataset
    2. **Data Transformation**: Preprocess and scale features
    3. **Model Training**: Train multiple models and select the best one
    4. **Predictions**: Use the trained model to predict salaries
    """)
    st.markdown('</div>', unsafe_allow_html=True)


def show_train_model():
    """Display model training page"""
    st.markdown('<h1 class="main-header">🤖 Train Machine Learning Model</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This section allows you to train the machine learning model on the entire dataset.
    The pipeline includes:
    - Data ingestion from CSV
    - Preprocessing (scaling and encoding)
    - Training multiple algorithms
    - Selecting the best performing model
    """)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Start Training Pipeline", use_container_width=True):
            with st.spinner("⏳ Training model... This may take a few minutes..."):
                try:
                    result = run_train_pipeline()
                    
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.success("✅ Model training completed successfully!")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display results
                    st.markdown("### 📈 Training Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Best R² Score", f"{result['best_r2_score']:.4f}")
                    with col2:
                        st.metric("Train Data Path", result['train_data_path'][-20:])
                    with col3:
                        st.metric("Test Data Path", result['test_data_path'][-20:])
                    
                    st.markdown("### 📁 Model Artifacts")
                    st.markdown(f"""
                    - **Preprocessor**: `{result['preprocessor_path']}`
                    - **Train Data**: `{result['train_data_path']}`
                    - **Test Data**: `{result['test_data_path']}`
                    """)
                    
                    logging.info("Training completed through Streamlit app")
                    
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")
                    logging.error(f"Training error: {str(e)}")


def show_predict_salary():
    """Display salary prediction page"""
    st.markdown('<h1 class="main-header">💰 Predict Salary</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Enter job profile details to predict the expected salary using the trained model.
    """)
    
    # Check if model files exist
    preprocessor_path = 'artifacts/preprocessor.pkl'
    model_path = 'artifacts/model.pkl'
    
    if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
        st.warning("⚠️ Model not found. Please train the model first using the 'Train Model' section.")
        return
    
    # Create input form
    with st.form("prediction_form", clear_on_submit=False):
        st.markdown("### 📋 Job Profile Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            posting_year = st.slider(
                "Posting Year",
                min_value=2010,
                max_value=2025,
                value=2023,
                help="Year job was posted"
            )
            
            country = st.text_input(
                "Country",
                value="United States",
                help="Country of employment"
            )
            
            region = st.text_input(
                "Region",
                value="North America",
                help="Region/Continent"
            )
            
            city = st.text_input(
                "City",
                value="New York",
                help="City of employment"
            )
            
            company_name = st.text_input(
                "Company Name",
                value="Tech Company",
                help="Name of the company"
            )
        
        with col2:
            company_size = st.selectbox(
                "Company Size",
                options=["S", "M", "L"],
                help="S=Small, M=Medium, L=Large"
            )
            
            industry = st.text_input(
                "Industry",
                value="Technology",
                help="Industry sector"
            )
            
            job_title = st.text_input(
                "Job Title",
                value="Data Scientist",
                help="Your job title"
            )
            
            seniority_level = st.text_input(
                "Seniority Level",
                value="Senior",
                help="Your seniority level"
            )
            
            ai_mentioned = st.selectbox(
                "AI Mentioned in Job",
                options=[True, False],
                format_func=lambda x: "Yes" if x else "No",
                help="Was AI mentioned in the job posting?"
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            ai_keywords = st.text_input(
                "AI Keywords",
                value="machine learning, deep learning",
                help="AI-related keywords in job description"
            )
            
            ai_intensity_score = st.slider(
                "AI Intensity Score",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="How AI-intensive is the role (0-1)"
            )
            
            core_skills = st.text_input(
                "Core Skills",
                value="Python, SQL",
                help="Core technical skills required"
            )
            
            ai_skills = st.text_input(
                "AI Skills",
                value="TensorFlow, PyTorch",
                help="AI-specific skills required"
            )
        
        with col4:
            salary_usd = st.number_input(
                "Salary (USD)",
                min_value=10000,
                max_value=500000,
                value=120000,
                step=5000,
                help="Annual salary in USD"
            )
            
            salary_change_vs_prev_year_percent = st.slider(
                "Salary Change vs Prev Year (%)",
                min_value=-50.0,
                max_value=50.0,
                value=0.0,
                step=1.0,
                help="Salary change percentage"
            )
            
            automation_risk_score = st.slider(
                "Automation Risk Score",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help="Risk of job automation (0-1)"
            )
            
            reskilling_required = st.selectbox(
                "Reskilling Required",
                options=[True, False],
                format_func=lambda x: "Yes" if x else "No",
                help="Is reskilling required?"
            )
        
        col5, col6 = st.columns(2)
        
        with col5:
            ai_job_displacement_risk = st.selectbox(
                "AI Job Displacement Risk",
                options=["Low", "Medium", "High", "Very High"],
                help="Risk of job displacement by AI"
            )
            
            job_description_embedding_cluster = st.number_input(
                "Job Embedding Cluster",
                min_value=0,
                max_value=50,
                value=0,
                step=1,
                help="Job description embedding cluster ID"
            )
        
        with col6:
            industry_ai_adoption_stage = st.selectbox(
                "Industry AI Adoption Stage",
                options=["Early", "Growth", "Mature", "Advanced"],
                help="Stage of AI adoption in the industry"
            )
        
        # Prediction button
        submit_button = st.form_submit_button(
            "🔮 Predict Salary",
            use_container_width=True
        )
        
        if submit_button:
            try:
                # Create custom data object
                custom_data = CustomData(
                    posting_year=posting_year,
                    country=country,
                    region=region,
                    city=city,
                    company_name=company_name,
                    company_size=company_size,
                    industry=industry,
                    job_title=job_title,
                    seniority_level=seniority_level,
                    ai_mentioned=ai_mentioned,
                    ai_keywords=ai_keywords,
                    ai_intensity_score=ai_intensity_score,
                    core_skills=core_skills,
                    ai_skills=ai_skills,
                    salary_usd=salary_usd,
                    salary_change_vs_prev_year_percent=salary_change_vs_prev_year_percent,
                    automation_risk_score=automation_risk_score,
                    reskilling_required=reskilling_required,
                    ai_job_displacement_risk=ai_job_displacement_risk,
                    job_description_embedding_cluster=job_description_embedding_cluster,
                    industry_ai_adoption_stage=industry_ai_adoption_stage
                )
                
                # Get data as dataframe
                pred_df = custom_data.get_data_as_dataframe()
                
                # Load pipeline and make prediction
                predict_pipeline = PredictPipeline()
                predict_pipeline.load_models(preprocessor_path, model_path)
                
                with st.spinner("🔄 Processing prediction..."):
                    predicted_salary = predict_pipeline.predict_salary(pred_df)[0]
                
                # Display prediction
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown(f"""
                ### 💵 Predicted Salary
                
                #### ${predicted_salary:,.2f}
                
                Based on your profile information, the estimated annual salary is **${predicted_salary:,.2f}**
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Additional insights
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Monthly Salary", f"${predicted_salary/12:,.2f}")
                with col2:
                    st.metric("Current Annual", f"${salary_usd:,.2f}")
                with col3:
                    difference = predicted_salary - salary_usd
                    st.metric(
                        "Difference", 
                        f"${difference:,.2f}",
                        delta=f"{(difference/salary_usd)*100:.1f}%"
                    )
                
                logging.info(f"Prediction made: Input job_title={job_title}, Predicted salary=${predicted_salary:,.2f}")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                logging.error(f"Prediction error: {str(e)}")


def show_project_info():
    """Display project information page"""
    st.markdown('<h1 class="main-header">📊 Project Information</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("### 📖 About This Project")
    st.markdown("""
    This project aims to predict job salaries considering the impact of AI on the job market 
    from 2010 to 2025. The application uses various machine learning models to provide 
    accurate salary predictions based on multiple factors.
    """)
    
    st.markdown("### 🎯 Objectives")
    st.markdown("""
    1. **Data Collection**: Gather comprehensive job salary data from 2010-2025
    2. **Analysis**: Understand factors affecting salaries in AI-impacted roles
    3. **Modeling**: Build predictive models using multiple algorithms
    4. **Prediction**: Provide accurate salary predictions for job seekers
    5. **Insight**: Offer insights about salary trends and market dynamics
    """)
    
    st.markdown("### 🛠️ Technical Stack")
    
    tech_cols = st.columns(3)
    with tech_cols[0]:
        st.markdown("**Backend**")
        st.markdown("""
        - Python 3.8+
        - Pandas
        - NumPy
        - Scikit-learn
        """)
    
    with tech_cols[1]:
        st.markdown("**ML Models**")
        st.markdown("""
        - Linear Regression
        - Ridge/Lasso
        - Random Forest
        - XGBoost
        - CatBoost
        - SVM
        """)
    
    with tech_cols[2]:
        st.markdown("**Frontend**")
        st.markdown("""
        - Streamlit
        - HTML/CSS
        - Interactive Forms
        """)
    
    st.markdown("### 📁 Project Structure")
    st.markdown("""
    ```
    ai_impact_on_jobs/
    ├── src/
    │   ├── components/
    │   │   ├── data_ingestion.py
    │   │   ├── data_transformation.py
    │   │   └── model_trainer.py
    │   ├── pipeline/
    │   │   ├── train_pipeline.py
    │   │   └── predict_pipeline.py
    │   ├── exception.py
    │   ├── logger.py
    │   └── utils.py
    ├── notebook/
    │   ├── 01_EDA.ipynb
    │   ├── 02_model_trainer.ipynb
    │   └── data/
    │       └── ai_impact_jobs_2010_2025.csv
    ├── app.py
    ├── requirements.txt
    └── README.md
    ```
    """)
    
    st.markdown("### 📚 Data Information")
    st.markdown("""
    **Dataset**: AI Impact Jobs (2010-2025)
    
    **Features Used**:
    - work_year
    - experience_level
    - employment_type
    - job_title
    - salary
    - employee_residence
    - remote_ratio
    - company_location
    - company_size
    """)
    
    st.markdown("### 👨‍💻 Development Notes")
    st.markdown("""
    - Following modular Python programming practices
    - Custom exception handling with detailed error tracking
    - Comprehensive logging for debugging and monitoring
    - Pickle-based model persistence for easy deployment
    - Pipeline architecture for easy extension
    """)


if __name__ == "__main__":
    main()
