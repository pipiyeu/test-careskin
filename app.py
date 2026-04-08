import joblib
import streamlit as st
from scipy.sparse import issparse

# Load model + mlb + vectorizer + selected features
model = joblib.load("fix_classifier_chain.pkl")
mlb = joblib.load("fix_mlb.pkl")
tfidf_ing = joblib.load("fix_feature_name_ing.pkl")   # <- vectorizer fit
selected_idx = joblib.load("fix_selected_features.pkl")    # <- fitur Chi-Square

text_input = st.text_area("Masukkan deskripsi produk:")

if st.button("Prediksi") and text_input.strip() != "":
    X_new_tfidf = tfidf_ing.transform([text_input])
    X_new_chi = X_new_tfidf[:, selected_idx]
    X_new_array = X_new_chi.toarray() if issparse(X_new_chi) else X_new_chi
    y_pred = model.predict(X_new_array)

    active_idx = y_pred[0].nonzero()[0]   # cara aman ambil label aktif
    active_labels = mlb.classes_[active_idx]

    if len(active_labels) == 0:
        st.info("Tidak ada efek samping terdeteksi.")
    else:
        st.success(f"Efek samping terdeteksi: {', '.join(active_labels)}")