import joblib
import streamlit as st
import numpy as np
import time
from scipy.sparse import issparse

# --- 1. Konfigurasi Halaman (WAJIB PALING ATAS) ---
st.set_page_config(
    page_title="Mandali - Cosmetic Analyzer",
    page_icon="favicon.png", 
    layout="centered"
)

# --- 2. Custom CSS (Tema Maroon & Pink) ---
st.markdown("""
    <style>
    /* Background Utama */
    .stApp { background-color: #F8E1E5; } 
    
    /* Tombol Utama Maroon */
    div.stButton > button {
        background-color: #900C3F !important;
        color: white !important;
        border-radius: 25px;
        padding: 10px 25px;
        border: none;
        font-weight: bold;
    }
    
    /* Input Box */
    .stTextArea textarea { 
        border-radius: 15px; 
        border: 1px solid #EBBAB9;
    }

    /* Card Hasil Prediksi */
    .result-card {
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .card-manfaat { background-color: #D4EDDA; color: #155724; border: 1px solid #C3E6CB; }
    .card-resiko { background-color: #F8D7DA; color: #721C24; border: 1px solid #F5C6CB; }
    
    /* Section Box Footer */
    .section-box {
        padding: 30px;
        background-color: #EBBAB9;
        border-radius: 15px;
        text-align: center;
        color: #900C3F;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. Fungsi Load Model (Cached) ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("fix_classifier_chain.pkl")
        mlb = joblib.load("fix_mlb.pkl")
        tfidf = joblib.load("fix_tfidf_ing.pkl")
        idx = joblib.load("fix_selected_idx.pkl")
        return model, mlb, tfidf, idx
    except Exception as e:
        return None

assets = load_assets()

# --- 4. Inisialisasi Session State ---
if 'analisis_selesai' not in st.session_state:
    st.session_state.analisis_selesai = False
if 'hasil_prediksi' not in st.session_state:
    st.session_state.hasil_prediksi = None

# --- 5. Logika Prediksi ---
def jalankan_analisis(text):
    if assets:
        model, mlb, tfidf, idx = assets
        X = tfidf.transform([text])
        X_chi = X[:, idx]
        y_pred = model.predict(X_chi)
        y_dense = y_pred.toarray() if issparse(y_pred) else y_pred
        active_idx = np.where(y_dense[0] == 1)[0]
        return mlb.classes_[active_idx]
    return []

# --- 6. ALUR TAMPILAN ---

# A. HALAMAN HASIL (Setelah Klik Prediksi)
if st.session_state.analisis_selesai:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image("logo.png", width=120) 
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center; color: #900C3F;'>Inilah risiko/manfaat yang mungkin ditimbulkan:</h3>", unsafe_allow_html=True)
    
    active_labels = st.session_state.hasil_prediksi
    
    manfaat_labels = {
        "acne fighting", "anti-aging", "brightening", "dark spots", 
        "good for oily skin", "hydrating", "redness reducing", 
        "reduces irritation", "reduces large pores", "scar healing", "skin texture"
    }
    
    efek_samping_labels = {
        "acne trigger", "drying", "eczema", "irritating", 
        "may worsen oily skin", "rosacea"
    }

    if len(active_labels) == 0:
        st.info("Tidak ada indikasi manfaat atau risiko spesifik yang terdeteksi.")
    else:
        for l in active_labels:
            is_manfaat = l.lower() in manfaat_labels
            bg_class = "card-manfaat" if is_manfaat else "card-resiko"
            emoji = "🍀" if is_manfaat else "⚠️"
            
            st.markdown(f"""
                <div class="result-card {bg_class}">
                    <h4 style="margin:0;">{emoji} {l.title()}</h4>
                    <p style="margin:0; opacity:0.8;">Berdasarkan analisis kandungan bahan</p>
                </div>
            """, unsafe_allow_html=True)

    st.write("")
    if st.button("🔄 Analyze Again"):
        st.session_state.analisis_selesai = False
        st.rerun()

# B. HALAMAN UTAMA (Landing Page)
else:
    # Header Section
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.image("logo.png", use_container_width=True)
    
    st.markdown("<h4 style='text-align: center; background-color: #EBBAB9; padding: 15px; border-radius: 15px; color: #900C3F;'>Discover what's really in your skincare</h4>", unsafe_allow_html=True)
    
    st.write("")
    st.markdown("<h3 style='color: #900C3F;'>Side Effect Checker</h3>", unsafe_allow_html=True)
    text_input = st.text_area("Type or paste the list of ingredients here:", height=150, placeholder="Aqua, Glycerin, Niacinamide...")
    
    if st.button("START ANALYSIS NOW"):
        if text_input.strip() == "":
            st.warning("Silakan masukkan daftar ingredients terlebih dahulu.")
        elif assets is None:
            st.error("Model gagal dimuat. Periksa file .pkl Anda.")
        else:
            with st.status("Please wait a moment...", expanded=True) as status:
                st.write("Sedang membedah kandungan...")
                time.sleep(1.5) # Efek dramatis loading
                st.session_state.hasil_prediksi = jalankan_analisis(text_input)
                st.session_state.analisis_selesai = True
                status.update(label="Analisis Selesai!", state="complete")
            st.rerun()

    # About Section
    st.write("---")
    col_about1, col_about2 = st.columns([1, 1.5])
    with col_about1:
        st.markdown("<h3 style='color: #900C3F;'>About Mandali</h3>", unsafe_allow_html=True)
    with col_about2:
        st.write("Mandali is a smart platform that helps you understand the ingredients in your skincare products. It allows you to check for possible side effects that certain ingredients may have on your skin.")

    # Vision Section
    st.markdown("""
        <div class='section-box'>
            <h3>Our Vision</h3>
            <p>"To make every skincare decision safer, smarter, and more personal"</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Difference Section
    st.write("")
    st.markdown("<h3 style='color: #900C3F;'>What Makes Mandali Different?</h3>", unsafe_allow_html=True)
    st.markdown("""
    * 🔬 **Science-based** ingredient analysis
    * ✨ **Simple & elegant** design for smooth experience
    * ⚡ **Instant** risk assessment
    * 👤 **Personalized** for your unique skin needs
    """)

# --- Footer ---
st.write("---")
st.markdown("<p style='text-align: center; font-size: 0.8rem; color: #7f8c8d;'>Mandali AI By Luthfinaf © 2026 - Data dianalisis berdasarkan algoritma Machine Learning</p>", unsafe_allow_html=True)