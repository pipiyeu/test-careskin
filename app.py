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

# --- 2. Custom CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background-color: #FDF5F6; } 

    /* Card untuk Halaman Utama agar rapi */
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #FADADD;
        height: 100%;
    }

    /* Button Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #900C3F 0%, #C70039 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        border: none !important;
        font-weight: 600 !important;
    }

    /* Tampilan Pill (Hasil) */
    .pill {
        padding: 6px 12px; 
        border-radius: 20px; 
        margin-right: 8px; 
        display: inline-block; 
        margin-bottom: 8px; 
        font-weight: 500; 
        cursor: help;
        border: 1px solid;
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

# --- 5. Logika Prediksi ---
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

deskripsi_label = {
    "acne fighting": "Melawan jerawat dan membantu mencegah munculnya jerawat baru.",
    "acne trigger": "Dapat menyumbat pori-pori atau memicu timbulnya jerawat.",
    "anti-aging": "Menyamarkan garis halus dan membantu kulit tampak lebih muda.",
    "brightening": "Mengembalikan kecerahan pada kulit yang kusam dan tampak lelah.",
    "dark spots": "Memudarkan bintik hitam untuk warna kulit yang lebih merata.",
    "drying": "Dapat menghilangkan kelembapan alami dan memperburuk kekeringan.",
    "eczema": "Dapat memperburuk rasa gatal atau iritasi pada kulit eksim.",
    "good for oily skin": "Menyeimbangkan kadar minyak dan membantu mengurangi kilap.",
    "hydrating": "Meningkatkan hidrasi dan mengatasi kulit kering serta terasa kencang.",
    "irritating": "Potensi menyebabkan iritasi atau ketidaknyamanan pada kulit.",
    "may worsen oily skin": "Dapat meningkatkan kilap atau produksi minyak berlebih.",
    "redness reducing": "Meredakan kemerahan yang terlihat dan menenangkan iritasi.",
    "reduces irritation": "Mengurangi rasa tidak nyaman dan mendukung ketahanan kulit.",
    "reduces large pores": "Meminimalkan tampilan pori-pori yang membesar.",
    "rosacea": "Dapat memicu kekambuhan pada kulit yang rentan rosacea.",
    "scar healing": "Memperbaiki tampilan bekas luka dan noda pada kulit.",
    "skin texture": "Menghaluskan bagian kulit yang kasar dan memperbaiki tekstur."
}

# --- 6. ALUR TAMPILAN ---

if st.session_state.analisis_selesai:
    # --- A. HALAMAN HASIL (Sesuai Struktur Asli Anda) ---
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image("logo.png", width=100) 
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center; color: #900C3F;'>Analysis Result</h3>", unsafe_allow_html=True)
    
    active_labels = st.session_state.hasil_prediksi
    
    manfaat_labels = {"acne fighting", "anti-aging", "brightening", "dark spots", "good for oily skin", "hydrating", "redness reducing", "reduces irritation", "reduces large pores", "scar healing", "skin texture"}
    efek_samping_labels = {"acne trigger", "drying", "eczema", "irritating", "may worsen oily skin", "rosacea"}

    if not active_labels:
        st.info("Tidak ada indikasi manfaat atau risiko spesifik yang terdeteksi.")
    else:
        manfaat_found = [l for l in active_labels if l.lower() in manfaat_labels]
        efek_found = [l for l in active_labels if l.lower() in efek_samping_labels]

        if manfaat_found:
            st.markdown("**🍀 Manfaat yang ditemukan:** (Sentuh pill untuk detail)")
            m_html = ""
            for l in manfaat_found:
                desc = deskripsi_label.get(l.lower(), "Informasi tidak tersedia.")
                m_html += f'<span title="{desc}" class="pill" style="background-color: #d4edda; color: #155724; border-color: #c3e6cb;">{l.title()}</span>'
            st.markdown(m_html, unsafe_allow_html=True)

        if efek_found:
            st.write("") 
            st.markdown("**⚠️ Perhatian / Efek Samping:** (Sentuh pill untuk detail)")
            e_html = ""
            for l in efek_found:
                desc = deskripsi_label.get(l.lower(), "Informasi tidak tersedia.")
                e_html += f'<span title="{desc}" class="pill" style="background-color: #f8d7da; color: #721c24; border-color: #f5c6cb;">{l.title()}</span>'
            st.markdown(e_html, unsafe_allow_html=True)
            st.warning("Jika Anda memiliki kulit sensitif, harap perhatikan kandungan di atas.")

    st.write("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🔄 Analyze Another Product", use_container_width=True):
            st.session_state.analisis_selesai = False
            st.rerun()

else:
    # --- B. HALAMAN UTAMA (Diperbaiki agar tidak ada kotak kosong) ---
    col_logo, col_text = st.columns([1, 3]) 
    with col_logo:
        st.image("logo.png", width=120) 
    with col_text:
        st.markdown("<h1 style='color: #900C3F; margin-top: 20px;'>Mandali Analyzer</h1>", unsafe_allow_html=True)

    st.markdown("<h4 style='text-align: center; background-color: #EBBAB9; padding: 15px; border-radius: 15px; color: #900C3F;'>Discover what's really in your skincare</h4>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<h3 style='color: #900C3F;'>Side Effect Checker</h3>", unsafe_allow_html=True)
    text_input = st.text_area("Type or paste ingredients:", height=150, placeholder="Aqua, Glycerin...", label_visibility="collapsed")
    
    if st.button("START ANALYSIS NOW"):
        if text_input.strip():
            with st.status("Analyzing...", expanded=True):
                time.sleep(1.5)
                st.session_state.hasil_prediksi = jalankan_analisis(text_input)
                st.session_state.analisis_selesai = True
            st.rerun()
        else: st.warning("Silakan masukkan daftar ingredients.")

    # Bagian Info (About & Why) - Diperbaiki tampilannya
    st.write("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="info-card">
            <b style='color:#900C3F'>About Mandali</b><br>
            <small>Mandali membantu Anda memahami isi produk skincare agar keputusan belanja lebih cerdas dan aman.</small>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="info-card">
            <b style='color:#900C3F'>Why Us?</b><br>
            <small>🔬 Berbasis Sains<br>⚡ Hasil Instan<br>👤 Personal</small>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div style='background-color:#900C3F; color:white; padding:20px; border-radius:15px; text-align:center; margin-top:20px;'>
        <h4 style='color:white'>Our Vision</h4>
        <p style='font-style:italic; font-size:14px;'>"To make every skincare decision safer, smarter, and more personal"</p>
    </div>""", unsafe_allow_html=True)

# Footer
st.markdown("<p style='text-align: center; font-size: 11px; color: #A64452; margin-top: 30px;'>Mandali AI By Luthfinaf © 2026</p>", unsafe_allow_html=True)