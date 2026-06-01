# ==========================================
# app.py — Sistem Prediksi Curah Hujan BMKG
# Streamlit Web Application
# ==========================================
# Struktur folder yang dibutuhkan:
# project/
# ├── app.py
# ├── requirements.txt
# └── export_assets/
#     ├── lstm_model.keras
#     ├── mlp_model.keras
#     ├── scaler_X.pkl
#     ├── scaler_y.pkl
#     └── metadata.json
# ==========================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Prediksi Curah Hujan BMKG",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CSS KUSTOM
# ==========================================
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    .result-box {
        background: #eaf3fb;
        border-left: 5px solid #185FA5;
        border-radius: 8px;
        padding: 20px 24px;
        margin: 12px 0;
    }
    .result-value { font-size: 3rem; font-weight: 700; color: #185FA5; }
    .result-unit  { font-size: 1.2rem; color: #555; }
    .cat-chip {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin-top: 6px;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 13px;
        margin-top: 8px;
    }
    .danger-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 13px;
        margin-top: 8px;
    }
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 10px 14px;
    }
    .stTabs [data-baseweb="tab"] { font-size: 14px; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL & ASET
# ==========================================
@st.cache_resource(show_spinner="Memuat model deep learning...")
def load_all_assets():
    try:
        from tensorflow.keras.models import load_model
        lstm  = load_model('export_assets/lstm_model.keras')
        mlp   = load_model('export_assets/mlp_model.keras')
        scX   = joblib.load('export_assets/scaler_X.pkl')
        scY   = joblib.load('export_assets/scaler_y.pkl')
        with open('export_assets/metadata.json', 'r') as f:
            meta = json.load(f)
        return lstm, mlp, scX, scY, meta, None
    except Exception as e:
        return None, None, None, None, None, str(e)

lstm_model, mlp_model, scaler_X, scaler_y, meta, load_err = load_all_assets()

if load_err:
    st.error(f"❌ Gagal memuat model: {load_err}")
    st.info("""
    Pastikan folder `export_assets/` berada satu direktori dengan `app.py` dan berisi:
    - `lstm_model.keras`
    - `mlp_model.keras`
    - `scaler_X.pkl`
    - `scaler_y.pkl`
    - `metadata.json`
    """)
    st.stop()

WS_LSTM  = meta['sequence_length_lstm']
WS_MLP   = meta['sequence_length_mlp']
FEATURES = meta['feature_columns']
N_FEAT   = meta['n_features']
ML       = meta['metrics_lstm']
MM       = meta['metrics_mlp']


# ==========================================
# FUNGSI HELPER
# ==========================================
def get_category(val):
    if val < 1:   return "Tidak Hujan",        "#6c757d", "#f8f9fa"
    if val < 5:   return "Hujan Sangat Ringan", "#0d6e56", "#e1f5ee"
    if val < 20:  return "Hujan Ringan",        "#185fa5", "#e6f1fb"
    if val < 50:  return "Hujan Sedang",        "#ba7517", "#fff3cd"
    if val < 100: return "Hujan Lebat",         "#993c1d", "#f8d7da"
    return               "Hujan Sangat Lebat",  "#7b0000", "#f5c6cb"

def get_alert(val):
    if val >= 100: return "danger",  "⚠️ SIAGA MERAH: Potensi banjir dan longsor. Evakuasi jika diperlukan."
    if val >= 50:  return "warning", "⚠️ SIAGA KUNING: Hujan lebat. Waspadai banjir lokal dan pohon tumbang."
    if val >= 20:  return "info",    "ℹ️ PERHATIAN: Hujan sedang. Bawa payung dan waspada genangan."
    return None, None

def predict_from_array(input_arr, model_type='lstm'):
    ws = WS_LSTM if model_type == 'lstm' else WS_MLP
    X_scaled = scaler_X.transform(input_arr.reshape(1, -1))
    X_window = np.repeat(X_scaled, ws, axis=0).reshape(1, ws, N_FEAT)
    if model_type == 'lstm':
        pred_sc = lstm_model.predict(X_window, verbose=0)
    else:
        pred_sc = mlp_model.predict(X_window.reshape(1, -1), verbose=0)
    val = float(scaler_y.inverse_transform(pred_sc)[0][0])
    return max(0.0, round(val, 2))

def predict_from_df(df, model_type='lstm'):
    ws  = WS_LSTM if model_type == 'lstm' else WS_MLP
    miss = [c for c in FEATURES if c not in df.columns]
    if miss:
        return None, f"Kolom tidak lengkap: {miss}"
    if len(df) < ws:
        return None, f"Data minimal {ws} baris. Tersedia: {len(df)}"
    X_sc  = scaler_X.transform(df[FEATURES].tail(ws).values)
    X_win = X_sc.reshape(1, ws, N_FEAT)
    if model_type == 'lstm':
        pred_sc = lstm_model.predict(X_win, verbose=0)
    else:
        pred_sc = mlp_model.predict(X_win.reshape(1, -1), verbose=0)
    val = float(scaler_y.inverse_transform(pred_sc)[0][0])
    return max(0.0, round(val, 2)), None

def save_history(record):
    if 'pred_history' not in st.session_state:
        st.session_state.pred_history = []
    st.session_state.pred_history.insert(0, record)
    if len(st.session_state.pred_history) > 50:
        st.session_state.pred_history.pop()


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## ⚙️ Konfigurasi Model")

    model_choice = st.radio(
        "Arsitektur model",
        ["🧠 LSTM (Direkomendasikan)", "🔗 MLP"],
        help="LSTM unggul secara global. MLP kompetitif di kejadian hujan deras."
    )
    m_type   = 'lstm' if 'LSTM' in model_choice else 'mlp'
    ws_aktif = WS_LSTM if m_type == 'lstm' else WS_MLP
    m_meta   = ML if m_type == 'lstm' else MM

    st.divider()
    st.markdown("**Performa model aktif**")
    c1, c2 = st.columns(2)
    c1.metric("KGE",  f"{m_meta['kge']}")
    c2.metric("NSE",  f"{m_meta['nse']}")
    c1.metric("RMSE", f"{m_meta['rmse']} mm")
    c2.metric("PBIAS",f"{m_meta['pbias']}%")

    st.divider()
    st.markdown(f"""
    **Info pipeline**
    - Window size : `{ws_aktif}` hari
    - Horizon     : `H+1` (1 hari ke depan)
    - Split data  : `{meta.get('split_ratio','70/10/20')}`
    - Periode     : `{meta.get('training_period','1971–2025')}`
    - Fitur input : `{N_FEAT}` kolom
    """)

    st.divider()
    st.caption("Sistem Prediksi CH BMKG · Deep Learning Pipeline · 2025")


# ==========================================
# HEADER
# ==========================================
st.title("🌧️ Sistem Prediksi Curah Hujan Harian")
st.caption(
    f"Prediksi intensitas curah hujan **1 hari ke depan (H+1)** · "
    f"Stasiun BMKG · Data {meta.get('training_period','1971–2025')} · "
    f"Model aktif: **{m_type.upper()}**"
)

# Metrik header
h1, h2, h3, h4, h5 = st.columns(5)
h1.metric("KGE LSTM",   ML['kge'],  f"MLP: {MM['kge']}")
h2.metric("NSE LSTM",   ML['nse'],  f"MLP: {MM['nse']}")
h3.metric("RMSE LSTM",  f"{ML['rmse']} mm", f"MLP: {MM['rmse']} mm")
h4.metric("MAE LSTM",   f"{ML['mae']} mm",  f"MLP: {MM['mae']} mm")
h5.metric("PBIAS LSTM", f"{ML['pbias']}%",  f"MLP: {MM['pbias']}%")

st.divider()


# ==========================================
# TAB UTAMA
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "✏️  Input Manual",
    "📂  Upload CSV",
    "📊  Perbandingan Model",
    "📋  Riwayat Prediksi",
])


# ─────────────────────────────────────────
# TAB 1: INPUT MANUAL
# ─────────────────────────────────────────
with tab1:
    st.subheader("Input Data Meteorologi Hari Ini")
    st.caption("Masukkan observasi hari ini → sistem memprediksi curah hujan H+1")

    with st.form("form_manual"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🌡️ Suhu**")
            t_rata = st.number_input("Suhu rata-rata (°C)",   15.0, 45.0, 27.5, 0.1)
            t_max  = st.number_input("Suhu maksimum (°C)",    15.0, 45.0, 32.0, 0.1)
            t_min  = st.number_input("Suhu minimum (°C)",     10.0, 35.0, 23.0, 0.1)
            t_0700 = st.number_input("Suhu 07:00 WIB (°C)",  15.0, 45.0, 24.5, 0.1)
            t_1300 = st.number_input("Suhu 13:00 WIB (°C)",  15.0, 45.0, 31.0, 0.1)
            t_1800 = st.number_input("Suhu 18:00 WIB (°C)",  15.0, 45.0, 28.0, 0.1)

        with col2:
            st.markdown("**💧 Kelembaban & Tekanan**")
            rh_rata = st.number_input("RH rata-rata (%)",    0, 100, 82, 1)
            rh_0700 = st.number_input("RH 07:00 WIB (%)",   0, 100, 88, 1)
            rh_1300 = st.number_input("RH 13:00 WIB (%)",   0, 100, 75, 1)
            rh_1800 = st.number_input("RH 18:00 WIB (%)",   0, 100, 83, 1)
            qff     = st.number_input("Tekanan QFF (hPa)",  900.0, 1100.0, 1011.5, 0.1)
            qfe     = st.number_input("Tekanan QFE (hPa)",  900.0, 1100.0, 1008.0, 0.1)

        with col3:
            st.markdown("**🌬️ Angin & Lainnya**")
            ff_rata = st.number_input("Kec. angin rata-rata (km/h)", 0.0, 150.0, 12.0, 0.5)
            ff_max  = st.number_input("Kec. angin maks (km/h)",      0.0, 200.0, 25.0, 0.5)
            dd      = st.number_input("Arah angin (°)",     0, 360, 225, 1)
            dd_max  = st.number_input("Arah angin maks (°)",0, 360, 270, 1)
            lpm     = st.number_input("Lama penyinaran (jam)", 0.0, 12.0, 6.5, 0.1)
            ch_hari = st.number_input("CH hari ini (mm)",   0.0, 500.0, 5.0, 0.1)

        submitted = st.form_submit_button(
            "🔮 Prediksi Curah Hujan H+1",
            use_container_width=True,
            type="primary"
        )

    if submitted:
        # Susun input sesuai urutan FEATURES
        inp_dict = {
            'T0700': t_0700, 'T1300': t_1300, 'T1800': t_1800,
            'Trata-rata': t_rata, 'Tmax': t_max, 'Tmin': t_min,
            'LPM': lpm, 'QFF': qff, 'QFE': qfe,
            'RH0700': rh_0700, 'RH1300': rh_1300, 'RH1800': rh_1800,
            'RHrata-rata': rh_rata, 'ffrata-rata': ff_rata,
            'dd': dd, 'ffmax': ff_max, 'ddmax': dd_max,
            'CH': ch_hari,
        }
        inp_arr = np.array([inp_dict.get(f, 0.0) for f in FEATURES], dtype=float)

        with st.spinner(f"Menghitung prediksi dengan {m_type.upper()} (ws={ws_aktif})..."):
            result = predict_from_array(inp_arr, m_type)

        cat_label, cat_color, cat_bg = get_category(result)
        alert_type, alert_msg        = get_alert(result)

        # Tampilkan hasil
        r1, r2 = st.columns([1, 1.6])
        with r1:
            st.markdown(f"""
            <div class="result-box">
                <div style="font-size:12px;color:#555;margin-bottom:6px;">
                    Prediksi H+1 · {m_type.upper()} · ws={ws_aktif} hari
                </div>
                <div class="result-value">{result}</div>
                <div class="result-unit">mm / hari</div>
                <div class="cat-chip"
                     style="background:{cat_bg};color:{cat_color};">
                    {cat_label}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            if alert_msg:
                box_class = "danger-box" if alert_type == "danger" else "warning-box"
                st.markdown(f'<div class="{box_class}">{alert_msg}</div>',
                            unsafe_allow_html=True)

            st.markdown("**Detail prediksi**")
            d1, d2 = st.columns(2)
            d1.metric("Intensitas",  f"{result} mm")
            d1.metric("Kategori",    cat_label)
            d2.metric("Model",       m_type.upper())
            d2.metric("Window size", f"{ws_aktif} hari")

        # Simpan ke riwayat
        save_history({
            'Waktu'            : datetime.now().strftime('%d/%m/%Y %H:%M'),
            'CH Input (mm)'    : ch_hari,
            'RH (%)'           : rh_rata,
            'Suhu (°C)'        : t_rata,
            'Model'            : m_type.upper(),
            'Window (hari)'    : ws_aktif,
            'Prediksi H+1 (mm)': result,
            'Kategori'         : cat_label,
        })
        st.success("✅ Prediksi berhasil dan tersimpan di tab Riwayat.")


# ─────────────────────────────────────────
# TAB 2: UPLOAD CSV
# ─────────────────────────────────────────
with tab2:
    st.subheader("Prediksi Batch dari File CSV")
    st.caption(
        f"Upload CSV dengan kolom sesuai format model. "
        f"Minimal **{max(WS_LSTM, WS_MLP)} baris** data historis dibutuhkan."
    )

    with st.expander("📋 Format kolom CSV yang dibutuhkan"):
        df_fmt = pd.DataFrame({
            'No'         : range(1, len(FEATURES)+1),
            'Nama Kolom' : FEATURES,
            'Tipe Data'  : ['Numerik (float)'] * len(FEATURES),
        })
        st.dataframe(df_fmt, use_container_width=True, hide_index=True)

        template = pd.DataFrame(
            np.zeros((5, len(FEATURES))), columns=FEATURES
        )
        st.download_button(
            "⬇️ Download Template CSV",
            template.to_csv(index=False).encode('utf-8'),
            "template_input.csv", "text/csv",
            use_container_width=True
        )

    uploaded = st.file_uploader(
        "Pilih file CSV", type=['csv'],
        help="File harus memiliki header kolom sesuai format di atas."
    )

    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            st.success(f"✅ File dimuat: **{df_up.shape[0]}** baris × **{df_up.shape[1]}** kolom")

            st.markdown("**Preview 5 baris terakhir:**")
            st.dataframe(df_up.tail(5), use_container_width=True)

            miss_cols = [c for c in FEATURES if c not in df_up.columns]
            if miss_cols:
                st.error(f"❌ Kolom tidak ditemukan: `{miss_cols}`")
                st.stop()

            st.divider()
            st.markdown("**Jalankan prediksi:**")
            b1, b2 = st.columns(2)

            lstm_res, mlp_res = None, None

            with b1:
                if st.button("🧠 Prediksi LSTM", use_container_width=True):
                    with st.spinner("Menghitung..."):
                        lstm_res, err = predict_from_df(df_up, 'lstm')
                    if err:
                        st.error(err)

            with b2:
                if st.button("🔗 Prediksi MLP", use_container_width=True):
                    with st.spinner("Menghitung..."):
                        mlp_res, err = predict_from_df(df_up, 'mlp')
                    if err:
                        st.error(err)

            if lstm_res is not None or mlp_res is not None:
                st.divider()
                st.markdown("**Hasil Prediksi H+1:**")
                rc1, rc2 = st.columns(2)
                if lstm_res is not None:
                    cat_l, col_l, _ = get_category(lstm_res)
                    rc1.metric("🧠 LSTM H+1", f"{lstm_res} mm", cat_l)
                    _, alert_l = get_alert(lstm_res)
                    if alert_l:
                        rc1.warning(alert_l)
                if mlp_res is not None:
                    cat_m, col_m, _ = get_category(mlp_res)
                    rc2.metric("🔗 MLP H+1",  f"{mlp_res} mm",  cat_m)
                    _, alert_m = get_alert(mlp_res)
                    if alert_m:
                        rc2.warning(alert_m)

        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")


# ─────────────────────────────────────────
# TAB 3: PERBANDINGAN MODEL
# ─────────────────────────────────────────
with tab3:
    st.subheader("Perbandingan Performa Model")

    # Tabel metrik
    df_cmp = pd.DataFrame({
        'Metrik'       : ['RMSE (mm)', 'MAE (mm)', 'R²', 'NSE',
                          'KGE', 'PBIAS (%)', 'Window Size (hari)'],
        'LSTM'         : [ML['rmse'], ML['mae'], ML['r2'], ML['nse'],
                          ML['kge'],  ML['pbias'], WS_LSTM],
        'MLP'          : [MM['rmse'], MM['mae'], MM['r2'], MM['nse'],
                          MM['kge'],  MM['pbias'], WS_MLP],
        'Lebih baik'   : ['↓ kecil','↓ kecil','↑ besar','↑ besar',
                          '↑ besar','↓ |kecil|', '—'],
        'Threshold'    : ['< 5 mm','< 3 mm','> 0.90','> 0.75',
                          '> 0.75','< ±10%', '—'],
    })
    st.dataframe(df_cmp, use_container_width=True, hide_index=True)

    st.divider()

    # Radar chart sederhana dengan matplotlib
    st.markdown("**Radar chart multi-metrik**")
    labels = ['NSE', 'KGE', 'R²', '1−|PBIAS/100|', 'r']
    lstm_v = [ML['nse'], ML['kge'], ML['r2'],
              max(0, 1 - abs(ML['pbias'])/100), ML.get('r', ML['kge'])]
    mlp_v  = [MM['nse'], MM['kge'], MM['r2'],
              max(0, 1 - abs(MM['pbias'])/100), MM.get('r', MM['kge'])]

    N      = len(labels)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]
    lv     = lstm_v + [lstm_v[0]]
    mv     = mlp_v  + [mlp_v[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.plot(angles, lv, 'o-', lw=2.5, color='#185FA5', label='LSTM')
    ax.fill(angles, lv, alpha=0.12, color='#185FA5')
    ax.plot(angles, mv, 's-', lw=2.5, color='#E24B4A', label='MLP')
    ax.fill(angles, mv, alpha=0.12, color='#E24B4A')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=10)
    ax.set_title('Radar Chart Metrik Evaluasi', fontsize=12,
                 fontweight='bold', pad=20)
    st.pyplot(fig, use_container_width=False)
    plt.close()

    st.divider()

    # Perbandingan dengan literatur
    st.markdown("**Perbandingan dengan literatur terkini**")
    df_lit = pd.DataFrame({
        'Penelitian'   : ['Markuna et al. (2023)', 'SVMD-MLP (2024)',
                          'MLP-RF Ensemble (2024)',
                          '**Penelitian Ini — LSTM**',
                          '**Penelitian Ini — MLP**'],
        'Model'        : ['Random Forest', 'MLP Hybrid',
                          'Ensemble', 'LSTM', 'MLP'],
        'KGE'          : [0.77, 0.83, 0.985, ML['kge'], MM['kge']],
        'Horizon'      : ['H+1','Bulanan','H+1–15','H+1','H+1'],
        'Data'         : ['Harian','Bulanan','Harian',
                          f"Harian {meta.get('training_period','')}",
                          f"Harian {meta.get('training_period','')}"],
    })
    st.dataframe(df_lit, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────
# TAB 4: RIWAYAT PREDIKSI
# ─────────────────────────────────────────
with tab4:
    st.subheader("Riwayat Prediksi Sesi Ini")

    if 'pred_history' in st.session_state and st.session_state.pred_history:
        df_hist = pd.DataFrame(st.session_state.pred_history)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        col_dl, col_clr = st.columns([3, 1])
        with col_dl:
            st.download_button(
                "⬇️ Download Riwayat CSV",
                df_hist.to_csv(index=False).encode('utf-8'),
                f"riwayat_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                use_container_width=True
            )
        with col_clr:
            if st.button("🗑️ Hapus Riwayat", use_container_width=True):
                st.session_state.pred_history = []
                st.rerun()

        # Mini chart intensitas
        if len(df_hist) >= 2:
            st.markdown("**Grafik intensitas prediksi:**")
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.bar(range(len(df_hist)),
                    df_hist['Prediksi H+1 (mm)'].values[::-1],
                    color='#185FA5', alpha=0.75, edgecolor='white')
            ax2.set_xlabel('Urutan prediksi (terbaru di kanan)')
            ax2.set_ylabel('CH H+1 (mm)')
            ax2.set_title('Riwayat Prediksi Curah Hujan H+1', fontweight='bold')
            ax2.axhline(20,  color='orange', ls='--', lw=1, label='Batas sedang (20 mm)')
            ax2.axhline(50,  color='red',    ls='--', lw=1, label='Batas lebat (50 mm)')
            ax2.legend(fontsize=9); ax2.grid(True, axis='y', alpha=0.3)
            st.pyplot(fig2, use_container_width=True)
            plt.close()
    else:
        st.info("Belum ada riwayat prediksi. Lakukan prediksi di tab **Input Manual** terlebih dahulu.")