import streamlit as st
import joblib
import numpy as np
from scipy.sparse import issparse

# --- Load model dan fitur ---
@st.cache_data
def load_model():
    model = joblib.load("fix_classifier_chain.pkl")
    mlb = joblib.load("fix_mlb.pkl")
    feature_names_ing = joblib.load("fix_selected_features.pkl")  # nama fitur
    return model, mlb, feature_names_ing

model, mlb, feature_names_ing = load_model()

st.title("Prediksi Efek Samping Kosmetik")

# Input teks
text_input = st.text_area("Masukkan deskripsi produk:")

if st.button("Prediksi"):
    if text_input.strip() == "":
        st.warning("Masukkan teks dulu!")
    else:
        # Asumsikan input sudah berupa fitur numerik sesuai feature_names_ing
        # Misal user mengisi 0/1 untuk tiap fitur
        # Contoh sederhana:
        input_vector = np.zeros(len(feature_names_ing))  # default 0 semua
        # Di sini bisa isi sesuai input user (misal cek kata, dll)
        # input_vector[idx] = 1 jika fitur ada

        # Pastikan bentuk 2D untuk predict
        input_vector = input_vector.reshape(1, -1)

        # Prediksi
        y_pred = model.predict(input_vector)

        # Jika hasil sparse, ubah ke array
        if issparse(y_pred):
            y_pred = y_pred.toarray()

        # Ambil label aktif
        active_idx = np.where(y_pred[0] == 1)[0]
        active_labels = mlb.classes_[active_idx]

        if len(active_labels) == 0:
            st.info("Tidak ada efek samping terdeteksi.")
        else:
            st.success(f"Efek samping terdeteksi: {', '.join(active_labels)}")