import joblib
import streamlit as st
import numpy as np
import time
from scipy.sparse import issparse

# --- 1. Konfigurasi Halaman ---
st.set_page_config(
    page_title="Mandali - Cosmetic Analyzer",
    page_icon="Favicon.png", 
    layout="centered"
)

# --- 2. Custom CSS Modern & Friendly ---
st.markdown("""
    <style>
    /* Mengatur Font Utama */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp { background-color: #FDF5F6; } 

    /* Header Styling */
    .main-title {
        color: #72102C;
        font-weight: 700;
        font-size: 32px;
        margin-bottom: 0px;
    }
    
    .sub-title {
        color: #A64452;
        font-weight: 400;
        font-size: 16px;
        margin-bottom: 20px;
    }

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
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(144, 12, 63, 0.3) !important;
    }

    /* Input Area */
    .stTextArea textarea { 
        border-radius: 15px !important; 
        border: 1px solid #FADADD !important;
        background-color: #FFF !important;
    }

    /* Pill Styling */
    .pill {
        padding: 8px 16px; 
        border-radius: 50px; 
        margin-right: 8px; 
        display: inline-block; 
        margin-bottom: 10px; 
        font-size: 13px;
        font-weight: 600; 
        cursor: help;
        transition: all 0.2s ease;
    }
    
    .pill-manfaat {
        background-color: #E8F5E9;
        color: #2E7D32;
        border: 1px solid #C8E6C9;
    }
    
    .pill-efek {
        background-color: #FFEBEE;
        color: #C62828;
        border: 1px solid #FFCDD2;
    }

    /* Vision Box */
    .vision-box {
        background-color: #900C3F;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin: 20px 0;
    }

    .vision-box h3 { color: white !important; margin-bottom: 10px; }
    .vision-box p { font-style: italic; opacity: 0.9; }

    /* Footer */
    .footer-text {
        text-align: center;
        font-size: 12px;
        color: #A64452;
        margin-top: 50px;
        padding-bottom: 20px;
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
    except Exception:
        return None

assets = load_assets()

# --- 4. Inisialisasi Session State ---
if 'analisis_selesai' not in st.session_state:
    st.session_state.analisis_selesai = False
if 'hasil_prediksi' not in st.session_state:
    st.session_state.hasil_prediksi = []

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
    # --- A. HALAMAN HASIL ---
    st.markdown("<div style='text-align: center; margin-bottom: 30px;'>", unsafe_allow_html=True)
    st.image("logo.png", width=100) 
    st.markdown("<h2 style='color: #900C3F;'>Analysis Result</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
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

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    if not active_labels:
        st.info("Kami tidak menemukan klaim spesifik dari daftar bahan ini.")
    else:
        manfaat_found = [l for l in active_labels if l.lower() in manfaat_labels]
        efek_found = [l for l in active_labels if l.lower() in efek_samping_labels]

        if manfaat_found:
            st.markdown("<p style='font-weight: 600; color: #2E7D32; margin-bottom: 10px;'>🍀 Manfaat Terdeteksi:</p>", unsafe_allow_html=True)
            m_html = "".join([
                f'<span title="{deskripsi_label.get(l.lower(), "")}" class="pill pill-manfaat">{l.title()}</span>' 
                for l in manfaat_found
            ])
            st.markdown(m_html, unsafe_allow_html=True)
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        if efek_found:
            st.markdown("<p style='font-weight: 600; color: #C62828; margin-bottom: 10px;'>⚠️ Perhatian Khusus:</p>", unsafe_allow_html=True)
            e_html = "".join([
                f'<span title="{deskripsi_label.get(l.lower(), "")}" class="pill pill-efek">{l.title()}</span>' 
                for l in efek_found
            ])
            st.markdown(e_html, unsafe_allow_html=True)
            st.caption("Arahkan kursor ke label untuk melihat penjelasan lengkap.")
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Cek Produk Lain"):
            st.session_state.analisis_selesai = False
            st.rerun()

else:
    # --- B. HALAMAN UTAMA (Landing Page) ---
    st.markdown("<div style='text-align: center; margin-top: 20px;'>", unsafe_allow_html=True)
    st.image("logo.png", width=120) 
    st.markdown("<h1 class='main-title'>Mandali Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Your personal cosmetic ingredients expert</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Input Section
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight: 600; color: #900C3F; margin-bottom: 5px;'>Side Effect Checker</p>", unsafe_allow_html=True)
    text_input = st.text_area("", height=150, placeholder="Salicylic Acid, Niacinamide, Glycerin...", label_visibility="collapsed")
    
    st.write("")
    if st.button("MULAI ANALISIS SEKARANG"):
        if text_input.strip() == "":
            st.warning("Mohon masukkan daftar bahan produk Anda.")
        elif assets is None:
            st.error("Sistem sedang bermasalah, silakan coba lagi nanti.")
        else:
            with st.status("Sedang memproses data...", expanded=True) as status:
                st.write("Menganalisis profil bahan...")
                time.sleep(1.8)
                st.session_state.hasil_prediksi = jalankan_analisis(text_input)
                st.session_state.analisis_selesai = True
                status.update(label="Selesai!", state="complete")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Info Section
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='glass-card' style='height: 180px;'>", unsafe_allow_html=True)
        st.markdown("<p style='font-weight: 700; color: #900C3F;'>About Mandali</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 14px;'>Mandali membantu Anda memahami isi produk skincare agar keputusan belanja lebih cerdas dan aman.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='glass-card' style='height: 180px;'>", unsafe_allow_html=True)
        st.markdown("<p style='font-weight: 700; color: #900C3F;'>Why Choose Us?</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 14px;'>🔬 Berbasis Sains<br>⚡ Hasil Instan<br>👤 Personalisasi Kulit</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Vision Section
    st.markdown("""
        <div class='vision-box'>
            <h3>Our Vision</h3>
            <p>"To make every skincare decision safer, smarter, and more personal"</p>
        </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown(f"""
    <div class='footer-text'>
        Mandali AI By Luthfinaf © 2026<br>
        <span style='opacity: 0.7;'>Menganalisis ribuan data bahan untuk kesehatan kulitmu.</span>
    </div>
""", unsafe_allow_html=True)