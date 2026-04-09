import joblib
import streamlit as st
import numpy as np
from scipy.sparse import issparse

# --- Load model, mlb, vectorizer fit, dan selected fitur Chi-Square ---
# Pastikan file-file ini ada di direktori yang sama
model = joblib.load("fix_classifier_chain.pkl")
mlb = joblib.load("fix_mlb.pkl")
tfidf_ing = joblib.load("fix_tfidf_ing.pkl")
selected_idx = joblib.load("fix_selected_idx.pkl")

st.title("✨ Prediksi Efek Samping & Manfaat Kosmetik")
st.write("Masukkan daftar bahan (ingredients) produk Anda di bawah ini.")

text_input = st.text_area("Masukkan deskripsi produk:", placeholder="Contoh: Aqua, Niacinamide, Glycerin, Salicylic Acid...")

if st.button("Analisis Kandungan"):
    if text_input.strip() == "":
        st.warning("Mohon masukkan teks terlebih dahulu.")
    else:
        # --- 1. PROSES TRANSFORMASI ---
        X_new_tfidf = tfidf_ing.transform([text_input])
        X_new_chi = X_new_tfidf[:, selected_idx]
        
        # Pastikan dalam bentuk array agar stabil
        X_new_array = X_new_chi.toarray() if issparse(X_new_chi) else X_new_chi

        # --- 2. PROSES PREDIKSI ---
        y_pred = model.predict(X_new_array)

        # Konversi hasil prediksi ke dense array
        if issparse(y_pred):
            y_pred_dense = y_pred.toarray()
        else:
            y_pred_dense = y_pred

        # Ambil index di mana nilainya adalah 1 (aktif)
        active_idx = np.where(y_pred_dense[0] == 1)[0]
        
        # Ambil label dari MultiLabelBinarizer
        active_labels = mlb.classes_[active_idx]

        # --- 3. DEFINISI KATEGORI ---
        manfaat_labels = {
            "acne fighting", "anti-aging", "brightening", "dark spots", 
            "good for oily skin", "hydrating", "redness reducing", 
            "reduces irritation", "reduces large pores", "scar healing", "skin texture"
        }
        
        efek_samping_labels = {
            "acne trigger", "drying", "eczema", "irritating", 
            "may worsen oily skin", "rosacea"
        }

        # --- 4. TAMPILAN HASIL ---
        if len(active_labels) == 0:
            st.info("Tidak ada indikasi manfaat atau efek samping spesifik yang terdeteksi dari teks tersebut.")
        else:
            st.write("### Hasil Analisis Kandungan:")
            
            # Filter hasil unik dan pisahkan berdasarkan kategori
            unique_found = list(set(active_labels))
            manfaat_found = [l for l in unique_found if l in manfaat_labels]
            efek_found = [l for l in unique_found if l in efek_samping_labels]

            # Tampilan Manfaat (Warna Hijau)
            if manfaat_found:
                st.markdown("**🍀 Manfaat yang ditemukan:**")
                labels_html = "".join([
                    f'<span style="background-color: #d4edda; color: #155724; padding: 4px 10px; border-radius: 15px; margin-right: 5px; border: 1px solid #c3e6cb; display: inline-block; margin-bottom: 5px;">{l}</span>' 
                    for l in manfaat_found
                ])
                st.markdown(labels_html, unsafe_allow_html=True)

            # Tampilan Efek Samping (Warna Merah)
            if efek_found:
                st.markdown("---")
                st.markdown("**⚠️ Perhatian / Efek Samping:**")
                labels_html = "".join([
                    f'<span style="background-color: #f8d7da; color: #721c24; padding: 4px 10px; border-radius: 15px; margin-right: 5px; border: 1px solid #f5c6cb; display: inline-block; margin-bottom: 5px;">{l}</span>' 
                    for l in efek_found
                ])
                st.markdown(labels_html, unsafe_allow_html=True)
                st.warning("Jika Anda memiliki kulit sensitif, harap perhatikan kandungan di atas.")