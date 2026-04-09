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

    st.markdown("<h4 style='text-align: center; background-color: #EBBAB9; padding: 15px; border-radius: 15px; color: #900C3F;'>Beauty in Every Ingredient, Clarity in Every Scan</h4>", unsafe_allow_html=True)

    st.write("")
    text_input = st.text_area("Type or paste ingredients:", height=150, placeholder="Aqua, Glycerin...", label_visibility="collapsed")
    
    if st.button("START ANALYSIS NOW"):
        if text_input.strip():
            with st.status("Please wait a moment...", expanded=True):
                time.sleep(1.5)
                st.session_state.hasil_prediksi = jalankan_analisis(text_input)
                st.session_state.analisis_selesai = True
            st.rerun()
        else: st.warning("Please enter a list of ingredients.")

    # Bagian Info (About & Why) - Diperbaiki tampilannya
    # --- About Mandali Section ---
    st.write("---")
    st.markdown("""
        <div class="info-card" style="margin-bottom: 20px;">
            <h3 style='color:#900C3F; margin-top:0;'>About Mandali</h3>
            <p style='font-size: 14px; line-height: 1.6; color: #444;'>
                <b>Mandali</b> is an intelligent cosmetic content analysis platform designed to be a bridge between the complexity of product labels and user understanding. 
                We believe that everyone has the right to know what they are putting on their skin without having to be a chemist.
                In the midst of thousands of choices, Mandali is here to provide transparency. Using data-driven technology, 
                we dissect every component in your ingredients list to uncover hidden benefits and potential risks.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # --- What We Do Section ---
    st.markdown("<h3 style='color: #900C3F; text-align: center; margin-top: 30px;'>What We Do</h3>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
            <div class="info-card">
                <p style='font-weight: 700; color: #900C3F; margin-bottom: 5px;'>⚡ Instant Analysis</p>
                <p style='font-size: 13px; color: #555;'>Simply input your ingredients list, and Mandali will process it in a matter of seconds.</p>
            </div>
            <div style='height: 15px;'></div>
            <div class="info-card">
                <p style='font-weight: 700; color: #900C3F; margin-bottom: 5px;'>🔍 Benefit Identification</p>
                <p style='font-size: 13px; color: #555;'>Discover active ingredients that support skin health, ranging from anti-aging to acne-fighting properties.</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col_b:
        st.markdown("""
            <div class="info-card">
                <p style='font-weight: 700; color: #900C3F; margin-bottom: 5px;'>⚠️ Risk Warnings</p>
                <p style='font-size: 13px; color: #555;'>Identify ingredients that potentially trigger irritation, acne, or flare-ups for sensitive skin conditions.</p>
            </div>
            <div style='height: 15px;'></div>
            <div class="info-card">
                <p style='font-weight: 700; color: #900C3F; margin-bottom: 5px;'>📖 Trusted Education</p>
                <p style='font-size: 13px; color: #555;'>Provide easy-to-understand descriptions for every analysis result, allowing you to learn while you care.</p>
            </div>
        """, unsafe_allow_html=True)

    # --- Why Mandali Section ---
    st.markdown("<h3 style='color: #900C3F; text-align: center; margin-top: 40px;'>Why Mandali?</h3>", unsafe_allow_html=True)
    
    row_why = st.columns(3)
    with row_why[0]:
        st.markdown("""<div class="info-card" style="text-align:center; min-height:180px;">
            <span style='font-size:30px;'>🔬</span><br>
            <b style='color:#900C3F'>Science-Based</b><br>
            <p style='font-size: 12px; color: #555;'>Predictions based on technically tested cosmetic data patterns.</p>
        </div>""", unsafe_allow_html=True)
    with row_why[1]:
        st.markdown("""<div class="info-card" style="text-align:center; min-height:180px;">
            <span style='font-size:30px;'>✨</span><br>
            <b style='color:#900C3F'>Elegant Design</b><br>
            <p style='font-size: 12px; color: #555;'>Prioritizing visual comfort for a luxurious self-care experience.</p>
        </div>""", unsafe_allow_html=True)
    with row_why[2]:
        st.markdown("""<div class="info-card" style="text-align:center; min-height:180px;">
            <span style='font-size:30px;'>👤</span><br>
            <b style='color:#900C3F'>Personalized</b><br>
            <p style='font-size: 12px; color: #555;'>Helping you tailor choices to your unique skin needs.</p>
        </div>""", unsafe_allow_html=True)

    # --- Vision Section ---
    st.markdown("""
        <div style='background-color:#900C3F; color:white; padding:35px; border-radius:20px; text-align:center; margin-top:40px; box-shadow: 0 4px 15px rgba(144, 12, 63, 0.2);'>
            <h4 style='color:white; margin:0; font-size: 20px;'>Our Vision</h4>
            <p style='font-style:italic; font-size:16px; opacity:0.9; margin-top:10px;'>
                "To make every skincare decision safer, smarter, and more personal"
            </p>
        </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("<p style='text-align: center; font-size: 11px; color: #A64452; margin-top: 40px; padding-bottom: 20px;'>Mandali AI By Luthfinaf © 2026</p>", unsafe_allow_html=True)