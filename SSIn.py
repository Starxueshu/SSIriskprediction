# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st
import shap

st.header("Establishment of an artificial intelligence-based assessment application for prediction of surgical site infection among patients with metastatic spinal disease: a prospective multicenter validated cohort study")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")

Tumortype = st.sidebar.selectbox("Tumor type", ("Thyroid cancer", "Prostate cancer", "Breast cancer", "Renal cancer", "Lung cancer", "Hepatocellular carcinoma", "Gastrointestinal system cancer", "Urogenital cancer", "Others"))
Smoking = st.sidebar.selectbox("Smoking", ("Never", "Previous", "Current"))
Numberofcommorbidity = st.sidebar.selectbox("Number of comorbidity", ("0", "1", "2", "≧3"))
Diabetes = st.sidebar.selectbox("Diabetes", ("No", "Yes"))
Viscerametastases = st.sidebar.selectbox("Visceral metastases", ("No", "Yes"))
Surgicalsegements= st.sidebar.selectbox("Surgical segments", ("One", "Two", "≥Three"))
Surgicaltime= st.sidebar.slider("Surgery time (min)", 150, 400)

if st.button("Submit"):
    rf_clf = jl.load("gbm_clf_final_roundonlineai.pkl")
    x = pd.DataFrame([[Tumortype,Smoking,Numberofcommorbidity, Diabetes, Surgicalsegements,Viscerametastases,Surgicaltime]],
                     columns=["Tumortype", "Smoking", "Numberofcommorbidity", "Diabetes",  "Surgicalsegements","Viscerametastases","Surgicaltime"])
    x = x.replace(["Thyroid cancer", "Prostate cancer", "Breast cancer", "Renal cancer", "Lung cancer", "Hepatocellular carcinoma", "Gastrointestinal system cancer", "Urogenital cancer", "Others"], [1, 2, 3, 4, 5, 6, 7, 8, 9])
    x = x.replace(["Never", "Previous", "Current"], [1, 2, 3])
    x = x.replace(["0", "1", "2", "≧3"], [0, 1, 2, 3])
    x = x.replace(["No", "Yes"], [0, 1])
    x = x.replace(["One", "Two", "≥Three"], [1, 2, 3])

    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.success(f"Risk of postoperative SSI: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.401:
        st.success(f"Risk group: low-risk group")
    else:
        st.error(f"Risk group: High-risk group")
    if prediction < 0.401:
        st.success(f"For patients with metastatic spinal disease who are at low risk for SSI, standard preoperative assessment and antibiotic prophylaxis according to standard guidelines are typically sufficient. Adhering to standard surgical protocols for wound preparation and closure, as well as regular postoperative monitoring for any signs of SSI, is important. While the risk of SSI may be lower in these patients, it is still crucial to provide appropriate care and attention to prevent any potential postoperative complications. Ultimately, both high and low-risk patients should receive individualized management measures based on their specific needs and risk factors, with the guidance of a multidisciplinary team of healthcare professionals.")
    else:
        st.error(f"Patients with metastatic spinal disease who are at high risk for surgical site infection (SSI) require a comprehensive approach to management. This includes preoperative optimization of overall health and nutritional status, as well as the administration of appropriate antibiotic prophylaxis before surgery. Additionally, the surgical technique should aim to minimize invasiveness and ensure meticulous wound closure. Postoperatively, close monitoring of the surgical wound for any signs of infection is essential, and prompt intervention is necessary if SSI is suspected. High-risk patients may also benefit from additional measures such as advanced wound care and closer postoperative follow-up to mitigate the risk of SSI.")

    st.subheader('Model explanation: contribution of each model predictor')
    star = pd.read_csv('X_train.csv', low_memory=False)
    y_trainy = pd.read_csv('y_train.csv', low_memory=False)
    data_train_X = star.loc[:, ["Tumortype", "Smoking", "Numberofcommorbidity", "Diabetes", "Surgicalsegements","Viscerametastases", "Surgicaltime"]]
    y_train = y_trainy.SSI
    model = rf_clf.fit(data_train_X, y_train)
    explainer = shap.Explainer(model)
    shap_value = explainer(x)
    #st.text(shap_value)

    shap.initjs()
    #image = shap.plots.force(shap_value)
    #image = shap.plots.bar(shap_value)

    shap.plots.waterfall(shap_value[0])
    st.pyplot(bbox_inches='tight')
    st.set_option('deprecation.showPyplotGlobalUse', False)


st.subheader('About the model')
st.markdown('The online calculator is accessible at no cost and utilizes the gradient boosting machine algorithm. Model validation has demonstrated exceptional performance, achieving a remarkable Area Under the Curve (AUC) of 0.986 (95% Confidence Interval: 0.972-1.000). However, it is crucial to emphasize that this model was developed specifically for research purposes. Consequently, clinical treatment decisions for metastatic spinal disease should not rely solely on this AI platform. Instead, the predictions should be regarded as an additional resource to aid in decision-making, complementing the expertise and judgment of healthcare professionals.')