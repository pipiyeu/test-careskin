import streamlit as st
import joblib
import numpy as np
from scipy.sparse import issparse

# --- Load model, mlb, vectorizer, dan selected features ---
@st.cache_data
def load_model():
    model = joblib.load("fix_classifier_chain.pkl")
    mlb = joblib.load("fix_mlb.pkl")
    vectorizer = joblib.load("fix_tfidf_ing.pkl")
    selected_idx = joblib.load("fix_selected_idx.pkl")
    return model, mlb, vectorizer, selected_idx

model, mlb, vectorizer, selected_idx = load_model()

st.title("Prediksi Efek Samping Kosmetik")

# Input teks
text_input = st.text_area("Masukkan deskripsi produk:")

if st.button("Prediksi"):
    if text_input.strip() == "":
        st.warning("Masukkan teks dulu!")
    else:
        # Transformasi teks ke TF-IDF
        X_new_tfidf = vectorizer.transform([text_input])

        # Seleksi fitur Chi-Square
        X_new_chi = X_new_tfidf[:, selected_idx]

        # Ubah ke array kalau sparse
        if issparse(X_new_chi):
            X_new_array = X_new_chi.toarray()
        else:
            X_new_array = X_new_chi

        # Prediksi
        y_pred = model.predict(X_new_array)

        # Ambil label aktif
        active_idx = np.where(y_pred[0] == 1)[0]
        active_labels = mlb.classes_[active_idx]

        if len(active_labels) == 0:
            st.info("Tidak ada efek samping terdeteksi.")
        else:
            st.success(f"Efek samping terdeteksi: {', '.join(active_labels)}")