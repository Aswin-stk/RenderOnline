import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import os

st.set_page_config(page_title="Model Checker", layout="wide")
st.title("🔎 Model Prediction Checker 🤖")
st.write("✨ Simple Streamlit app to test your saved model (.pkl) interactively and interpret results easily. 🚀")

@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
model_path = "rf_iris_py58.pkl"

if not os.path.exists(model_path):
    st.error(f"❌ File '{model_path}' tidak ditemukan. Pastikan file ada di folder yang sama ya!")
    st.stop()

model = load_model(model_path)
st.success(f" ✅ Model berhasil dimuat: '{model.__class__.__name__}'")

# --- INPUT FEATURE ---
st.subheader("🎛️ Input Feature Values")
# Gantu sesuai fitur model kamu!
feature_names = ["sepal_length", "sepal_width","petal_length","petal_width"]
# buat input box dinamis
input_data = {}
col1, col2 = st.columns(2)

for i, feat in enumerate(feature_names):
    with (col1 if i % 2 == 0 else col2):
        input_data[feat] = st.number_input(f"📏{feat.replace('_',' ').title()}", value = 0.0, step=0.1)
input_df = pd.DataFrame([input_data])
st.write("👀 **Preview Input Data:**")
st.dataframe(input_df)

# --- CLASS LABEL EXPLANATION ---
# kamu bisa ganti sesuai dataset/model kamu
class_labels = {
    0:"Setosa 🌸",
    1: "Versicolor 🌺",
    2: "Virginica 🌻"
}

# --- PREDICTION ---
if st.button("🔮 Predict"):
    try:
        prediction = model.predict(input_df)
        pred_class = prediction[0]

        st.subheader("🎯 Prediction Result")
        st.write(f"Predicted Class: **{pred_class} ➡️ {class_labels.get(pred_class, 'Unknown Class')}**")

        # Kalau model punya predict_proba (klasifikasi)
        if hasattr (model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            proba_df = pd.DataFrame({
                "Class": [class_labels.get(i,str(i)) for i in range(len(proba))],
                "Probability": proba
            })

            st.write("📊**Prediction Probabilities:**")
            st.dataframe(proba_df)

            fig = px.bar(proba_df, x="Class", y="Probability", title="📈 Prediction Probabilities", text="Probability")
            st.plotly_chart(fig, use_container_width=True)
        
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feat_imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }). sort_values(by="Importance", ascending=False)

            st.write(" 🌟 **Feature Importances:**")
            st.dataframe(feat_imp_df)

            fig2 = px.bar(feat_imp_df, 
                          x="Feature", 
                          y="Importance", 
                          title="Feature Importances 📉", 
                          text="Importance",
                          orientation="h")
            st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f" ⚠️ Terjadi kesalahan saat prediksi: {e}")