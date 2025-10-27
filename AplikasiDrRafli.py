# =====================================================
# 🧠 SeizureDetect.AI (Final Clean Version - Fixed Output)
# =====================================================

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="🧠 SeizureDetect.AI", layout="centered")

# -----------------------------------------------------
# 1️⃣ Load Model dan Metadata Otomatis (Jika Ada)
# -----------------------------------------------------
@st.cache_resource
def load_model_and_metadata():
    model, metadata = None, None
    if os.path.exists("bestmodel_xgb_drRafli.pkl"):
        try:
            with open("bestmodel_xgb_drRafli.pkl", "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            st.warning(f"Gagal memuat model: {e}")
    if os.path.exists("bestmodel_xgb_drRafli_metadata.pkl"):
        try:
            with open("bestmodel_xgb_drRafli_metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
        except Exception as e:
            st.warning(f"Gagal memuat metadata: {e}")
    return model, metadata

model_auto, metadata_auto = load_model_and_metadata()

# -----------------------------------------------------
# 2️⃣ Metadata Default (jika file metadata tidak ada)
# -----------------------------------------------------
FEATURE_ORDER = metadata_auto["FEATURE_ORDER"] if metadata_auto else [
    'Jenis Kelamin',
    'Usia saat ini (Kategorik)',
    'Usia Terdiagnosis',
    'Jumlah OAE yang diminum',
    'Golongan Obat yang Dipakai',
    'Jenis Epilepsi',
    'Hasil Pemeriksaan EEG',
    'Hasil Pemeriksaan MRI',
    'OAE Sesuai Protokol'
]

MANUAL_ENCODING = metadata_auto["MANUAL_ENCODING"] if metadata_auto else {
    'Jenis Kelamin': {'Laki-laki': 0, 'Perempuan': 1},
    'Usia saat ini (Kategorik)': {
        '<1 tahun': 0, '1-<5 tahun': 1, '5-<10 tahun': 2, '10-<15 tahun': 3, '15-18 tahun': 4
    },
    'Usia Terdiagnosis': {
        '<1 tahun': 0, '1-<5 tahun': 1, '5-<10 tahun': 2, '10-<15 tahun': 3, '15-18 tahun': 4
    },
    'Jumlah OAE yang diminum': {'2 Obat': 0, '3 Obat': 1},
    'Golongan Obat yang Dipakai': {
        'Golongan 1': 0, 'Golongan 2': 1, 'Golongan 3': 2, 'Golongan 4': 3, 'Golongan 5': 4
    },
    'Jenis Epilepsi': {
        'Fokal': 0, 'Fokal ke umum': 1, 'Sindrom epilepsi': 3, 'Umum': 4
    },
    'Hasil Pemeriksaan EEG': {
        'Normal': 0, 'Abnormal dengan gelombang epileptiform': 1,
        'Abnormal tanpa gelombang epileptiform': 2, 'Sindrom epilepsi': 3
    },
    'Hasil Pemeriksaan MRI': {
        'Normal': 0, 'Abnormal epileptogenik': 1, 'Abnormal non-epileptogenik': 2
    },
    'OAE Sesuai Protokol': {'Tidak': 0, 'Ya': 1}
}

def normalize_manual_encoding(enc):
    norm = {}
    for col, mapping in enc.items():
        new_map = {str(k).strip(): v for k, v in mapping.items()}
        norm[col] = new_map
    return norm

MANUAL_ENCODING = normalize_manual_encoding(MANUAL_ENCODING)

# -----------------------------------------------------
# 3️⃣ Helper Functions
# -----------------------------------------------------
def mapping_choices(mapping):
    return [str(k).strip() for k in mapping.keys()]

def encode_input(input_dict):
    encoded = {}
    for k, v in input_dict.items():
        if k in MANUAL_ENCODING:
            val = MANUAL_ENCODING[k].get(str(v).strip(), 0)
            encoded[k] = val
        else:
            encoded[k] = v
    return encoded

def predict_from_model(model, input_df):
    """Return (pred_label, prob_terkontrol)"""
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            prob_terkontrol = float(proba[1]) if len(proba) > 1 else 0.0
        else:
            prob_terkontrol = 0.0

        pred_raw = model.predict(input_df)[0]
        # Pastikan hasil 0 atau 1
        if isinstance(pred_raw, (int, np.integer, float)):
            pred = int(pred_raw)
        elif str(pred_raw).strip().lower() in ['1', 'terkontrol']:
            pred = 1
        elif str(pred_raw).strip().lower() in ['0', 'tidak terkontrol']:
            pred = 0
        else:
            pred = 1 if prob_terkontrol >= 0.5 else 0

        return pred, prob_terkontrol
    except Exception as e:
        st.error(f"Kesalahan prediksi: {e}")
        return 0, 0.0

# -----------------------------------------------------
# 4️⃣ Session Init
# -----------------------------------------------------
if 'users' not in st.session_state:
    st.session_state['users'] = {
        'drrafli': {
            'name': 'Dr Rafli',
            'instansi': 'RS Contoh',
            'email': 'drrafli@example.com',
            'phone': '08123456789',
            'password': '123456'
        }
    }
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'model' not in st.session_state:
    st.session_state['model'] = model_auto

# -----------------------------------------------------
# 5️⃣ Navigasi
# -----------------------------------------------------
PAGES = ['home', 'auth_choice', 'register', 'login', 'form', 'history']
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
def go_to(page): st.session_state['page'] = page

# -----------------------------------------------------
# 6️⃣ UI
# -----------------------------------------------------
if st.session_state['page'] == 'home':
    st.title("🧠 SeizureDetect.AI")
    st.markdown("### Prediksi Pengendalian Kejang Berdasarkan Karakteristik Pasien")
    if st.button("Mulai Aplikasi"):
        go_to('auth_choice')

elif st.session_state['page'] == 'auth_choice':
    st.header("Apakah Anda sudah punya akun?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Login'): go_to('login')
    with col2:
        if st.button('Register'): go_to('register')
    if st.button('Kembali ke Beranda'): go_to('home')

elif st.session_state['page'] == 'register':
    st.header("Registrasi Akun Baru")
    with st.form("register_form"):
        name = st.text_input("Nama Lengkap")
        instansi = st.text_input("Instansi / Rumah Sakit")
        email = st.text_input("Email")
        phone = st.text_input("Nomor Telepon")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Daftar")
    if submitted:
        if username in st.session_state['users']:
            st.error("Username sudah digunakan.")
        else:
            st.session_state['users'][username] = {
                'name': name, 'instansi': instansi, 'email': email, 'phone': phone, 'password': password
            }
            st.success("Registrasi berhasil! Silakan login.")
            go_to('login')
    if st.button("Kembali"): go_to('auth_choice')

elif st.session_state['page'] == 'login':
    st.header("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Masuk")
    if submitted:
        user = st.session_state['users'].get(username)
        if user and user['password'] == password:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success(f"Selamat datang, {user['name']}!")
            go_to('form')
        else:
            st.error("Username atau password salah.")
    if st.button("Kembali"): go_to('auth_choice')

elif st.session_state['page'] == 'form':
    if not st.session_state['logged_in']:
        st.warning("Silakan login terlebih dahulu.")
        go_to('login')
    else:
        st.subheader("Masukkan Data Pasien")

        # Sidebar
        st.sidebar.header("🔧 Model & Metadata")
        model_status = "✅ Model Terdeteksi" if st.session_state['model'] else "⚠️ Belum Ada Model"
        st.sidebar.write(model_status)

        uploaded_model = st.sidebar.file_uploader("Unggah Model (.pkl)", type=["pkl"])
        uploaded_metadata = st.sidebar.file_uploader("Unggah Metadata (.pkl)", type=["pkl"])
        if uploaded_model:
            try:
                model_obj = pickle.load(uploaded_model)
                st.session_state['model'] = model_obj
                st.sidebar.success("Model berhasil dimuat.")
            except Exception as e:
                st.sidebar.error(f"Gagal memuat model: {e}")

        if uploaded_metadata:
            try:
                metadata_obj = pickle.load(uploaded_metadata)
                if "FEATURE_ORDER" in metadata_obj:
                    FEATURE_ORDER[:] = metadata_obj["FEATURE_ORDER"]
                if "MANUAL_ENCODING" in metadata_obj:
                    MANUAL_ENCODING.update(normalize_manual_encoding(metadata_obj["MANUAL_ENCODING"]))
                st.sidebar.success("Metadata berhasil dimuat.")
            except Exception as e:
                st.sidebar.error(f"Gagal memuat metadata: {e}")

        # Form input
        with st.form("input_form"):
            input_data = {}
            for key in FEATURE_ORDER:
                if key in MANUAL_ENCODING:
                    input_data[key] = st.selectbox(key, mapping_choices(MANUAL_ENCODING[key]))
                else:
                    input_data[key] = st.text_input(f"{key} (manual)")
            submitted = st.form_submit_button("🔍 Prediksi")

        if submitted:
            model_used = st.session_state['model']
            encoded = encode_input(input_data)
            X_input = pd.DataFrame([[encoded.get(c, 0) for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
            prediction, prob_terkontrol = predict_from_model(model_used, X_input)

            st.subheader("📊 Hasil Prediksi:")
            st.write(f"Probabilitas **TERKONTROL** = {prob_terkontrol*100:.2f}%")

            if prediction == 1:
                st.success("✅ Penanganan **TERKONTROL**")
            else:
                st.error("⚠️ Penanganan **TIDAK TERKONTROL**")

            st.write("### 🧾 Data Pasien:")
            st.dataframe(pd.DataFrame([input_data]))

            st.session_state['history'].append({
                **input_data,
                'prediction': 'TERKONTROL' if prediction == 1 else 'TIDAK TERKONTROL',
                'probability': prob_terkontrol
            })

        if st.button("Lihat Riwayat"):
            go_to('history')
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            go_to('login')

elif st.session_state['page'] == 'history':
    st.header("Riwayat Diagnosa")
    if len(st.session_state['history']) == 0:
        st.info("Belum ada riwayat.")
    else:
        df_hist = pd.DataFrame(st.session_state['history'])
        st.dataframe(df_hist)
    if st.button("Kembali"):
        go_to('form')

st.markdown("---")
st.caption("Developed with ❤️ by Dr. Rafli, AISeeyou, & BDC IMERI | Epilepsy Prediction Model (XGBoost)")
