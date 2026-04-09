import joblib
import streamlit as st
import numpy as np
import time
from scipy.sparse import issparse

# --- 1. Konfigurasi Halaman ---
st.set_page_config(
    page_title="Mandali - Cosmetic Analyzer",
    page_icon="🍀", 
    layout="centered"
)

# --- 2. Custom CSS Modern ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background-color: #FDF5F6; } 

    .main-title { color: #72102C; font-weight: 700; font-size: 32px; margin-bottom: 0px; text-align: center; }
    .sub-title { color: #A64452; font-weight: 400; font-size: 16px; margin-bottom: 20px; text-align: center; }

    /* Card Styling */
    .glass-card {
        background: white;
        padding: 25px;
        border-radius: 20px;
        border: 1px solid #FADADD;
        box-shadow: 0 10px 25px rgba(144, 12, 63, 0.05);
        margin-bottom: 20px;
    }

    /* Button Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #900C3F 0%, #C70039 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        border: none !important;
        font-weight: 600 !important;
        width: 100%;
        transition: all 0.3s ease;
    }

    .pill {
        padding: 8px 16px; border-radius: 50px; margin-right: 8px; 
        display: inline-block; margin-bottom: 10px; font-size: 13px;
        font-weight: 600; cursor: help;
    }
    .pill-manfaat { background-color: #E8F5E9; color: #2E7D32; border: 1px solid #C8E6C9; }
    .pill-efek { background-color: #FFEBEE; color: #C62828; border: 1px solid #FFCDD2; }

    .vision-box {
        background-color: #900C3F; padding: 30px; border-radius: 20px;
        text-align: center; color: white; margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. Fungsi Load Model ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("fix_classifier_chain.pkl")
        mlb = joblib.load("fix_mlb.pkl")
        tfidf = joblib.load("fix_tfidf_ing.pkl")
        idx = joblib.load("fix_selected_idx.pkl")
        return model, mlb, tfidf, idx
    except: return None

assets = load_assets()

# --- 4. Inisialisasi Session State ---
if 'analisis_selesai' not in st.session_state: st.session_state.analisis_selesai = False
if 'hasil_prediksi' not in st.session_state: st.session_state.hasil_prediksi = []

# --- 5. Logika Analisis ---
def jalankan_analisis(text):
    if assets:
        model, mlb, tfidf, idx = assets
        X = tfidf.transform([text])
        X_chi = X[:, idx]
        y_pred = model.predict(X_chi)
        y_dense = y_pred.toarray() if issparse(y_pred) else y_pred
        active_idx = np.where(y_dense[0] == 1)[0]
        return list(mlb.classes_[active_idx])
    return []

# --- 6. ALUR TAMPILAN ---

if st.session_state.analisis_selesai:
    # --- HALAMAN HASIL ---
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    try: st.image("logo.png", width=100)
    except: st.write("🍀")
    st.markdown("<h2 style='color: #900C3F;'>Analysis Result</h2></div>", unsafe_allow_html=True)
    
    active_labels = st.session_state.hasil_prediksi
    manfaat_labels = {"acne fighting", "anti-aging", "brightening", "dark spots", "good for oily skin", "hydrating", "redness reducing", "reduces irritation", "reduces large pores", "scar healing", "skin texture"}
    efek_samping_labels = {"acne trigger", "drying", "eczema", "irritating", "may worsen oily skin", "rosacea"}

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    if not active_labels:
        st.info("Tidak ada indikasi spesifik terdeteksi.")
    else:
        manfaat_found = [l for l in active_labels if l.lower() in manfaat_labels]
        efek_found = [l for l in active_labels if l.lower() in efek_samping_labels]

        if manfaat_found:
            st.markdown("<p style='font-weight: 600; color: #2E7D32;'>🍀 Manfaat Terdeteksi:</p>", unsafe_allow_html=True)
            for l in manfaat_found:
                st.markdown(f'<span class="pill pill-manfaat">{l.title()}</span>', unsafe_allow_html=True)
        
        if efek_found:
            st.markdown("<p style='font-weight: 600; color: #C62828; margin-top:15px;'>⚠️ Perhatian Khusus:</p>", unsafe_allow_html=True)
            for l in efek_found:
                st.markdown(f'<span class="pill pill-efek">{l.title()}</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔄 Cek Produk Lain"):
        st.session_state.analisis_selesai = False
        st.rerun()

else:
    # --- HALAMAN UTAMA ---
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    try: st.image("logo.png", width=120)
    except: st.write("🍀")
    st.markdown("<h1 class='main-title'>Mandali Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Your personal cosmetic ingredients expert</p></div>", unsafe_allow_html=True)

    # Input Section
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight: 600; color: #900C3F;'>Side Effect Checker</p>", unsafe_allow_html=True)
    text_input = st.text_area("", height=150, placeholder="Salicylic Acid, Niacinamide...", label_visibility="collapsed")
    
    if st.button("MULAI ANALISIS SEKARANG"):
        if text_input.strip():
            with st.status("Menganalisis...", expanded=True):
                time.sleep(1.5)
                st.session_state.hasil_prediksi = jalankan_analisis(text_input)
                st.session_state.analisis_selesai = True
            st.rerun()
        else: st.warning("Masukkan daftar bahan.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Info Section - DIPERBAIKI (Hapus kotak kosong)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class='glass-card' style='min-height: 150px;'>
                <p style='font-weight: 700; color: #900C3F; margin-bottom: 5px;'>About Mandali</p>
                <p style='font-size: 13px; line-height: 1.4;'>Mandali membantu Anda memahami isi produk skincare agar keputusan belanja lebih cerdas dan aman.</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class='glass-card' style='min-height: 150px;'>
                <p style='font-weight: 700; color: #900C3F; margin-bottom: 5px;'>Why Choose Us?</p>
                <p style='font-size: 13px; line-height: 1.4;'>
                    🔬 Berbasis Sains<br>
                    ⚡ Hasil Instan<br>
                    👤 Personalisasi Kulit
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Vision Section
    st.markdown("<div class='vision-box'><h3>Our Vision</h3><p>\"To make every skincare decision safer, smarter, and more personal\"</p></div>", unsafe_allow_html=True)

# Footer
st.markdown("<div style='text-align: center; font-size: 12px; color: #A64452; margin-top: 30px;'>Mandali AI By Luthfinaf © 2026</div>", unsafe_allow_html=True)