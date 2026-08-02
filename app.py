import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Student Dropout Risk Detection System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { background: linear-gradient(135deg, #1a237e, #1565c0); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; }
    .risk-high { background: #d32f2f; color: white; padding: 1.5rem; border-radius: 12px; text-align: center; }
    .risk-moderate { background: #f57c00; color: white; padding: 1.5rem; border-radius: 12px; text-align: center; }
    .risk-low { background: #2e7d32; color: white; padding: 1.5rem; border-radius: 12px; text-align: center; }
    .intervention-high { background: #ffebee; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #d32f2f; }
    .intervention-medium { background: #fff3e0; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #f57c00; }
    .intervention-low { background: #e8f5e9; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #2e7d32; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================
if 'history' not in st.session_state:
    st.session_state.history = []

# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/best_model.pkl')
        le = joblib.load('models/label_encoder.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        explainer = joblib.load('models/shap_explainer.pkl')
        feature_names = [name.strip() for name in feature_names]
        return model, le, feature_names, explainer
    except FileNotFoundError:
        st.error("🚨 Model files not found. Please ensure all model files are in the 'models/' directory.")
        st.stop()

model, le, feature_names, explainer = load_model()

# ============================================================================
# FUNCTIONS
# ============================================================================
def get_interventions(student_data):
    interventions = []
    grade_1 = student_data.get('Curricular units 1st sem (grade)', 0)
    grade_2 = student_data.get('Curricular units 2nd sem (grade)', 0)
    approved_1 = student_data.get('Curricular units 1st sem (approved)', 0)
    approved_2 = student_data.get('Curricular units 2nd sem (approved)', 0)
    tuition = student_data.get('Tuition fees up to date', 1)
    scholarship = student_data.get('Scholarship holder', 0)
    admission_grade = student_data.get('Admission grade', 0)
    
    if grade_1 < 8:
        interventions.append({'priority': 'High', 'icon': '🔴', 'issue': 'Very Low 1st Semester Grades', 'action': 'Immediate academic intervention required. Schedule tutoring for foundational CS courses.'})
    elif grade_1 < 10:
        interventions.append({'priority': 'Medium', 'icon': '🟡', 'issue': 'Low 1st Semester Grades', 'action': 'Recommend tutoring support and study skills workshop.'})
    
    if grade_2 < 8:
        interventions.append({'priority': 'High', 'icon': '🔴', 'issue': 'Very Low 2nd Semester Grades', 'action': 'Urgent academic support needed. Connect with academic advisor immediately.'})
    elif grade_2 < 10:
        interventions.append({'priority': 'Medium', 'icon': '🟡', 'issue': 'Low 2nd Semester Grades', 'action': 'Recommend academic counseling and peer mentoring.'})
    
    if approved_1 < 4:
        interventions.append({'priority': 'High', 'icon': '🔴', 'issue': 'Few Approved Units in 1st Semester', 'action': 'Schedule academic advising session to review course load and identify challenges.'})
    
    if approved_2 < 4:
        interventions.append({'priority': 'High', 'icon': '🔴', 'issue': 'Few Approved Units in 2nd Semester', 'action': 'Urgent academic review needed. Consider reducing course load.'})
    
    if tuition == 0:
        interventions.append({'priority': 'High', 'icon': '💰', 'issue': 'Tuition Fees Not Up to Date', 'action': 'Refer to financial aid office for payment plan options and financial counseling.'})
    
    if scholarship == 0 and tuition == 0:
        interventions.append({'priority': 'Medium', 'icon': '🎓', 'issue': 'No Scholarship + Financial Difficulty', 'action': 'Provide scholarship application information and financial assistance resources.'})
    
    if admission_grade < 110:
        interventions.append({'priority': 'Low', 'icon': '📉', 'issue': 'Low Admission Grade', 'action': 'Consider foundational support programs and bridging courses.'})
    
    if len(interventions) == 0:
        interventions.append({'priority': 'Low', 'icon': '✅', 'issue': 'No Major Risk Factors Detected', 'action': 'Continue regular monitoring and maintain current support.'})
    
    return interventions

def validate_input(student_data):
    warnings_list = []
    if student_data.get('Curricular units 1st sem (approved)', 0) > 6:
        warnings_list.append("Approved units exceed enrolled units in 1st semester")
    if student_data.get('Curricular units 2nd sem (approved)', 0) > 6:
        warnings_list.append("Approved units exceed enrolled units in 2nd semester")
    return warnings_list

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0; font-size: 2rem;">🎓 Student Dropout Risk Detection</h1>
    <p style="color: #90caf9; margin: 0.2rem 0 0 0;">Computer Science Department - University of Maiduguri</p>
    <p style="color: #b3d4fc; margin: 0.1rem 0 0 0; font-size: 0.85rem;">Early Warning System for At-Risk Students</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 📋 Student Information")
    student_name = st.text_input("👤 Student Name", placeholder="Enter student name")
    
    st.markdown("---")
    st.markdown("### 📚 Academic Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        units_1_approved = st.number_input("1st Sem Approved", min_value=0, max_value=10, value=5)
        units_2_approved = st.number_input("2nd Sem Approved", min_value=0, max_value=10, value=5)
    with col2:
        grade_1 = st.number_input("1st Sem Grade", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
        grade_2 = st.number_input("2nd Sem Grade", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
    
    st.markdown("### 👤 Demographic & Financial")
    
    admission_grade = st.number_input("📈 Admission Grade", min_value=50.0, max_value=200.0, value=130.0, step=0.1)
    age = st.number_input("🎂 Age at Enrollment", min_value=17, max_value=70, value=20)
    tuition = st.selectbox("💰 Tuition Fees", options=[1, 0], format_func=lambda x: "✅ Up to Date" if x == 1 else "❌ Not Up to Date")
    scholarship = st.selectbox("🎓 Scholarship Holder", options=[0, 1], format_func=lambda x: "✅ Yes" if x == 1 else "❌ No")
    
    st.markdown("---")
    predict_clicked = st.button("🔍 Predict Dropout Risk", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📤 Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV of Students", type=["csv"])
    
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.metric("📈 Accuracy", "88.7%", delta="Good", delta_color="normal")
    st.metric("🎯 F1-Score", "81.8%", delta="Good", delta_color="normal")
    st.caption("Model: Random Forest | AUC-ROC: 0.923")
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    
    st.markdown("---")
    st.caption("Developed by Mohammed Baba Grema")
    st.caption("University of Maiduguri, 2026")

# ============================================================================
# BATCH PREDICTION
# ============================================================================
if uploaded_file:
    st.subheader("📊 Batch Prediction Results")
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        st.info(f"📁 Loaded {len(df)} student records")
        
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            st.error(f"Missing columns: {missing_features[:5]}...")
            st.stop()
        
        with st.spinner("Processing batch predictions..."):
            X_batch = df[feature_names].values
            predictions = model.predict(X_batch)
            probabilities = model.predict_proba(X_batch)[:, 1]
            
            results = df.copy()
            results['Dropout Risk'] = ['🔴 High Risk' if p == 1 else '🟢 Low Risk' for p in predictions]
            results['Risk Probability'] = probabilities
            
            st.dataframe(results.style.background_gradient(subset=['Risk Probability'], cmap='RdYlGn_r'))
            
            csv = results.to_csv(index=False)
            st.download_button("📥 Download Results", data=csv, file_name=f"predictions_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
            
            high_risk = results[results['Dropout Risk'] == '🔴 High Risk']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Students", len(results))
            with col2:
                st.metric("🚨 At Risk", len(high_risk), delta=f"{(len(high_risk)/len(results)*100):.1f}%")
            with col3:
                st.metric("✅ Safe", len(results) - len(high_risk))
                
    except Exception as e:
        st.error(f"Error processing file: {e}")

# ============================================================================
# PREDICTION
# ============================================================================
if predict_clicked:
    # Prepare input
    input_data = np.zeros((1, len(feature_names)))
    
    feature_map = {
        'Curricular units 1st sem (approved)': units_1_approved,
        'Curricular units 1st sem (grade)': grade_1,
        'Curricular units 2nd sem (approved)': units_2_approved,
        'Curricular units 2nd sem (grade)': grade_2,
        'Admission grade': admission_grade,
        'Age at enrollment': age,
        'Tuition fees up to date': tuition,
        'Scholarship holder': scholarship
    }
    
    for i, feature in enumerate(feature_names):
        if feature in feature_map:
            input_data[0, i] = feature_map[feature]
    
    # Validate
    warnings_list = validate_input(feature_map)
    for w in warnings_list:
        st.warning(f"⚠️ {w}")
    
    # Progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Processing input...")
    progress_bar.progress(30)
    
    status_text.text("Generating prediction...")
    progress_bar.progress(60)
    
    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    risk_prob = probability[1] * 100
    
    status_text.text("Analyzing risk factors...")
    progress_bar.progress(90)
    
    status_text.empty()
    progress_bar.progress(100)
    
    student_data = feature_map
    interventions = get_interventions(student_data)
    
    # ========================================================================
    # RESULT CARD
    # ========================================================================
    st.subheader("📊 Prediction Results")
    
    risk_text = "HIGH RISK" if risk_prob >= 60 else "MODERATE RISK" if risk_prob >= 30 else "LOW RISK"
    risk_color = "🔴" if risk_prob >= 60 else "🟡" if risk_prob >= 30 else "🟢"
    risk_class = "risk-high" if risk_prob >= 60 else "risk-moderate" if risk_prob >= 30 else "risk-low"
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"""
        <div class="{risk_class}">
            <div style="font-size: 0.85rem; opacity: 0.8;">{risk_color} {risk_text}</div>
            <div style="font-size: 3rem; font-weight: 700;">{risk_prob:.1f}%</div>
            <div style="font-size: 0.85rem; opacity: 0.8;">Dropout Risk Probability</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.metric("Prediction Confidence", f"{max(probability)*100:.1f}%")
    with col3:
        st.metric("Student Status", risk_text)
    
    # Save to history
    st.session_state.history.append({
        'name': student_name or "Unnamed",
        'risk': risk_prob,
        'status': risk_text,
        'time': datetime.now().strftime("%H:%M")
    })
    
    st.markdown("---")
    
    # ========================================================================
    # SHAP EXPLANATION
    # ========================================================================
    st.subheader("📊 Why This Student is At Risk (SHAP Analysis)")
    
    try:
        shap_values = explainer.shap_values(input_data)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(shap_values.shape) == 3:
            shap_vals = shap_values[0, :, 1]
        elif len(shap_values.shape) == 2:
            shap_vals = shap_values[0, :]
        else:
            shap_vals = shap_values
        
        if len(shap_vals) == len(feature_names):
            explanation_df = pd.DataFrame({
                'Feature': feature_names,
                'Contribution': shap_vals
            }).sort_values('Contribution', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                risk_factors = explanation_df[explanation_df['Contribution'] > 0].head(5)
                if not risk_factors.empty:
                    st.markdown("#### 🔴 Factors Increasing Risk")
                    for _, row in risk_factors.iterrows():
                        st.markdown(f"- **{row['Feature']}**: +{row['Contribution']:.3f}")
            
            with col2:
                protective_factors = explanation_df[explanation_df['Contribution'] < 0].head(5)
                if not protective_factors.empty:
                    st.markdown("#### 🟢 Protective Factors")
                    for _, row in protective_factors.iterrows():
                        st.markdown(f"- **{row['Feature']}**: {row['Contribution']:.3f}")
            
            try:
                st.markdown("#### 📈 Feature Contribution Chart")
                fig, ax = plt.subplots(figsize=(10, 6))
                top_features = explanation_df.head(10).copy()
                colors = ['#d32f2f' if x > 0 else '#2e7d32' for x in top_features['Contribution']]
                ax.barh(top_features['Feature'], top_features['Contribution'], color=colors)
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                ax.set_xlabel('SHAP Contribution (Impact on Dropout Risk)')
                ax.set_title('Top Features Contributing to Dropout Risk', fontsize=12)
                ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig)
                plt.close()
                st.caption("🔴 Red = Increases risk | 🟢 Green = Decreases risk")
            except Exception as e:
                st.warning(f"SHAP plot could not be displayed: {e}")
        else:
            st.warning("SHAP values shape mismatch.")
            
    except Exception as e:
        st.warning(f"SHAP explanation could not be generated: {e}")
    
    st.markdown("---")
    
    # ========================================================================
    # INTERVENTIONS
    # ========================================================================
    st.subheader("💡 Recommended Interventions")
    
    for intervention in interventions:
        class_name = f"intervention-{intervention['priority'].lower()}"
        st.markdown(f"""
        <div class="{class_name}">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 1.2rem; margin-right: 0.5rem;">{intervention['icon']}</span>
                <div>
                    <strong>{intervention['issue']}</strong>
                    <div style="font-size: 0.9rem; color: #555; margin-top: 0.2rem;">{intervention['action']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("💡 These recommendations are generated based on the student's risk factors and should be used as a guide for academic intervention decisions.")
    
    # ========================================================================
    # HISTORY
    # ========================================================================
    if st.session_state.history:
        with st.expander("📋 Prediction History", expanded=False):
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df.tail(10))