import joblib
import streamlit as st
from scipy.sparse import issparse

# --- Load model, mlb, vectorizer fit, dan selected fitur Chi-Square ---
model = joblib.load("fix_classifier_chain.pkl")
mlb = joblib.load("fix_mlb.pkl")                # MultiLabelBinarizer dari training
tfidf_ing = joblib.load("fix_tfidf_ing.pkl")   # vectorizer yang sudah fit
selected_idx = joblib.load("fix_selected_idx.pkl") # index fitur hasil Chi-Square

st.title("Prediksi Efek Samping Kosmetik")

text_input = st.text_area("Masukkan deskripsi produk:")

if st.button("Prediksi") and text_input.strip() != "":
    # Transform input teks
    X_new_tfidf = tfidf_ing.transform([text_input])
    X_new_chi = X_new_tfidf[:, selected_idx]
    
    # Pastikan dalam bentuk array agar pemrosesan index lebih stabil
    X_new_array = X_new_chi.toarray() if issparse(X_new_chi) else X_new_chi

    # Prediksi model
    y_pred = model.predict(X_new_array)

    # KONVERSI KE ARRAY BIASA: 
    # Jika y_pred adalah sparse matrix, ubah ke dense array dulu
    if issparse(y_pred):
        y_pred_dense = y_pred.toarray()
    else:
        y_pred_dense = y_pred

    # Ambil index di mana nilainya adalah 1 (aktif) untuk baris pertama [0]
    # Menggunakan np.where jauh lebih aman untuk array 2D
    import numpy as np
    active_idx = np.where(y_pred_dense[0] == 1)[0]
    
    # Ambil label berdasarkan index tersebut
    active_labels = mlb.classes_[active_idx]

    if len(active_labels) == 0:
        st.info("Tidak ada efek samping terdeteksi.")
    else:
        # Gunakan list(set(...)) untuk memastikan tidak ada duplikasi teks
        unique_labels = list(set(active_labels))
        st.success(f"Efek samping terdeteksi: {', '.join(unique_labels)}")    