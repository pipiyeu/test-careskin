import joblib
import streamlit as st
import numpy as np
from scipy.sparse import issparse

# --- Load model, mlb, vectorizer fit, dan selected fitur Chi-Square ---
model = joblib.load("fix_classifier_chain.pkl")
mlb = joblib.load("fix_mlb.pkl")                # MultiLabelBinarizer dari training
tfidf_ing = joblib.load("fix_tfidf_ing.pkl")   # vectorizer yang sudah fit
selected_idx = joblib.load("fix_selected_idx.pkl") # index fitur hasil Chi-Square

import joblib
import streamlit as st
import numpy as np
from scipy.sparse import issparse

# Konfigurasi Halaman (Harus di paling atas)
st.set_page_config(
    page_title="Mandali - Cosmetic Analyzer",
    page_icon="logo.png",
    layout="centered"
)

# --- Custom CSS untuk mempercantik UI ---
st.markdown("""
    <style>
    .main {
        background-color: #f9fbfd;
    }
    .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid #d1d5db;
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .pill {
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.85rem;
        display: inline-block;
        margin: 4px;
        border: 1px solid transparent;
    }
    .pill-manfaat {
        background-color: #e6f4ea;
        color: #1e7e34;
        border-color: #c3e6cb;
    }
    .pill-efek {
        background-color: #fdecea;
        color: #d93025;
        border-color: #f5c6cb;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Fungsi Helper Load Model ---
@st.cache_resource # Gunakan cache agar load model tidak lambat setiap kali klik
def load_assets():
    model = joblib.load("fix_classifier_chain.pkl")
    mlb = joblib.load("fix_mlb.pkl")
    tfidf_ing = joblib.load("fix_tfidf_ing.pkl")
    selected_idx = joblib.load("fix_selected_idx.pkl")
    return model, mlb, tfidf_ing, selected_idx

try:
    model, mlb, tfidf_ing, selected_idx = load_assets()
except Exception as e:
    st.error("Gagal memuat model. Pastikan file .pkl sudah benar.")
    st.stop()

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #2c3e50;'> Mandali Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d;'>Cek manfaat dan risiko kandungan kosmetik Anda dalam satu klik.</p>", unsafe_allow_html=True)
st.divider()

# --- Input Area ---
text_input = st.text_area(
    "Tempel Ingredients (Bahan Produk):", 
    placeholder="Contoh: Aqua, Glycerin, Niacinamide, Salicylic Acid, Phenoxyethanol...",
    height=150
)

# --- Action Button ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    btn_predict = st.button("🚀 Analisis Sekarang", use_container_width=True)

if btn_predict:
    if text_input.strip() == "":
        st.warning("Silakan masukkan teks daftar bahan produk terlebih dahulu.")
    else:
        with st.spinner("Menganalisis bahan..."):
            # 1. Proses Transformasi
            X_new_tfidf = tfidf_ing.transform([text_input])
            X_new_chi = X_new_tfidf[:, selected_idx]
            X_new_array = X_new_chi.toarray() if issparse(X_new_chi) else X_new_chi

            # 2. Proses Prediksi
            y_pred = model.predict(X_new_array)
            y_pred_dense = y_pred.toarray() if issparse(y_pred) else y_pred
            active_idx = np.where(y_pred_dense[0] == 1)[0]
            active_labels = mlb.classes_[active_idx]

        # --- Tampilan Hasil ---
        st.subheader("🔍 Hasil Analisis")
        
        if len(active_labels) == 0:
            st.info("Kami tidak menemukan kecocokan spesifik untuk label manfaat atau risiko yang terdaftar. Produk mungkin bersifat netral.")
        else:
            # Definisi Kategori
            manfaat_labels = {
                "acne fighting", "anti-aging", "brightening", "dark spots", 
                "good for oily skin", "hydrating", "redness reducing", 
                "reduces irritation", "reduces large pores", "scar healing", "skin texture"
            }
            
            efek_samping_labels = {
                "acne trigger", "drying", "eczema", "irritating", 
                "may worsen oily skin", "rosacea"
            }

            unique_found = list(set(active_labels))
            manfaat_found = [l for l in unique_found if l in manfaat_labels]
            efek_found = [l for l in unique_found if l in efek_samping_labels]

            # Container Hasil
            with st.container():
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                
                # Baris Manfaat
                if manfaat_found:
                    st.markdown("**🍀 Potensi Manfaat:**")
                    m_html = "".join([f'<span class="pill pill-manfaat">{l.title()}</span>' for l in manfaat_found])
                    st.markdown(m_html, unsafe_allow_html=True)
                    st.write("") # Spasi

                # Baris Efek Samping
                if efek_found:
                    if manfaat_found: st.divider()
                    st.markdown("**⚠️ Perlu Diperhatikan:**")
                    e_html = "".join([f'<span class="pill pill-efek">{l.title()}</span>' for l in efek_found])
                    st.markdown(e_html, unsafe_allow_html=True)
                    st.caption("_Disarankan untuk melakukan patch test jika Anda memiliki kulit sensitif._")

                st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.8rem; color: #bdc3c7;'>Mandali By Luthfinaf © 2026 - Data dianalisis berdasarkan algoritma Machine Learning</p>", unsafe_allow_html=True)