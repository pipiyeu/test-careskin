import joblib
import streamlit as st
import numpy as np
import time
from scipy.sparse import issparse

# --- 1. Konfigurasi Halaman ---
st.set_page_config(
    page_title="Mandali - Cosmetic Analyzer",
    page_icon="favicon.png", 
    layout="centered"
)

# --- 2. Custom CSS (Tema Maroon & Pink) ---
st.markdown("""
    <style>
    .stApp { background-color: #F8E1E5; } 
    
    div.stButton > button {
        background-color: #900C3F !important;
        color: white !important;
        border-radius: 25px;
        padding: 10px 25px;
        border: none;
        font-weight: bold;
    }
    
    .stTextArea textarea { 
        border-radius: 15px; 
        border: 1px solid #EBBAB9;
    }

    .section-box {
        padding: 30px;
        background-color: #EBBAB9;
        border-radius: 15px;
        text-align: center;
        color: #900C3F;
        margin-top: 20px;
    }

    /* Styling untuk Pills */
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

# Kamus Deskripsi untuk Tooltip
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
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image("logo.png", width=100) 
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center; color: #900C3F;'>Analysis Result</h3>", unsafe_allow_html=True)
    
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
    # --- B. HALAMAN UTAMA (Landing Page) ---
    col_logo, col_text = st.columns([1, 3]) 
    with col_logo:
        st.image("logo.png", width=120) 
    with col_text:
        st.markdown("""
            <div style='display: flex; flex-direction: column; justify-content: center; height: 100px;'>
                <h1 style='margin: 0; color: #900C3F; font-size: 28px;'>Mandali Cosmetic Ingredients Analysis</h1>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <h4 style='text-align: center; background-color: #EBBAB9; padding: 15px; 
        border-radius: 15px; color: #900C3F; margin-top: 10px;'>
            Discover what's really in your skincare
        </h4>
    """, unsafe_allow_html=True)

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
                time.sleep(1.5)
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
        st.write("Mandali is a smart platform that helps you understand the ingredients in your skincare products.")

    st.markdown("""
        <div class='section-box'>
            <h3>Our Vision</h3>
            <p>"To make every skincare decision safer, smarter, and more personal"</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    st.markdown("<h3 style='color: #900C3F;'>What Makes Mandali Different?</h3>", unsafe_allow_html=True)
    st.markdown("""
    * 🔬 **Science-based** ingredient analysis
    * ✨ **Simple & elegant** design
    * ⚡ **Instant** risk assessment
    """)

# --- Footer ---
st.write("---")
st.markdown("<p style='text-align: center; font-size: 0.8rem; color: #7f8c8d;'>Mandali AI By Luthfinaf © 2026</p>", unsafe_allow_html=True)