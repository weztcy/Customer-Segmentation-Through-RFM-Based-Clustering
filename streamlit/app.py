"""
Customer Segmentation Through RFM-Based Clustering
Interactive Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Mode Matplotlib & Seaborn Styling
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1d24'
plt.rcParams['axes.facecolor'] = '#1a1d24'
plt.rcParams['text.color'] = '#e0e6ed'
plt.rcParams['axes.labelcolor'] = '#e0e6ed'
plt.rcParams['xtick.color'] = '#e0e6ed'
plt.rcParams['ytick.color'] = '#e0e6ed'
plt.rcParams['grid.color'] = '#2f3542'
plt.rcParams['figure.edgecolor'] = '#1a1d24'

# Custom CSS for Dark Mode styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4b7bec;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #a4b0be;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1a1d24;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #4b7bec;
        box-shadow: 0 2px 4px rgba(0,0,0,0.4);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #1e222d;
        color: #c8d6e5;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4b7bec !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    seg_path = 'dataset/customer_segments.csv'
    metrics_path = 'dataset/model_metrics.csv'
    if not os.path.exists(seg_path):
        st.error(f"File {seg_path} tidak ditemukan. Silakan jalankan `python src/customer_segmentation.py` terlebih dahulu.")
        st.stop()
    df = pd.read_csv(seg_path)
    if 'PCA1' not in df.columns or 'PCA2' not in df.columns:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        rfm_log = np.log1p(df[['Recency', 'Frequency', 'Monetary']])
        scaled = StandardScaler().fit_transform(rfm_log)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(scaled)
        df['PCA1'] = coords[:, 0]
        df['PCA2'] = coords[:, 1]
    metrics_df = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else None
    return df, metrics_df

df_seg, metrics_df = load_data()

# Header
st.markdown('<div class="main-title">🛍️ Customer Segmentation Through RFM-Based Clustering</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Dashboard Analisis Segmentasi Pelanggan E-Commerce Berdasarkan Recency, Frequency, dan Monetary (K-Means Clustering)</div>', unsafe_allow_html=True)

# Sidebar Filters
st.sidebar.header("🔍 Filter & Pengaturan")

all_segments = sorted(df_seg['Segment_Name'].unique().tolist())
selected_segments = st.sidebar.multiselect(
    "Pilih Segmen Pelanggan:",
    options=all_segments,
    default=all_segments
)

all_countries = ['Semua Negara'] + sorted(df_seg['Country'].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Filter Negara:", all_countries)

# Apply filters
filtered_df = df_seg[df_seg['Segment_Name'].isin(selected_segments)]
if selected_country != 'Semua Negara':
    filtered_df = filtered_df[filtered_df['Country'] == selected_country]

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Informasi Proyek")
st.sidebar.markdown("""
- **Dataset:** UCI Online Retail (352)
- **Periode:** Des 2010 - Des 2011
- **Algoritma:** K-Means (k=4)
- **Transformasi:** log1p + StandardScaler
- **Metrik Model:** Silhouette = 0.337, Inertia = 3,938.6
""")

# Key Metric Cards
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Pelanggan", f"{len(filtered_df):,}", f"{(len(filtered_df)/len(df_seg)*100):.1f}% dari total")
with col2:
    st.metric("Total Pendapatan", f"£{filtered_df['Monetary'].sum():,.0f}", f"{(filtered_df['Monetary'].sum()/df_seg['Monetary'].sum()*100):.1f}% dari total")
with col3:
    st.metric("Median Recency", f"{filtered_df['Recency'].median():.0f} hari", f"Mean: {filtered_df['Recency'].mean():.1f} hari")
with col4:
    st.metric("Median Frequency", f"{filtered_df['Frequency'].median():.0f} order", f"Mean: {filtered_df['Frequency'].mean():.1f} order")
with col5:
    st.metric("Median Monetary", f"£{filtered_df['Monetary'].median():,.0f}", f"Mean: £{filtered_df['Monetary'].mean():,.0f}")

st.markdown("<br>", unsafe_allow_html=True)

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Ringkasan Profil Segmen",
    "🗺️ Proyeksi PCA & Distribusi",
    "👥 Eksplorasi Data Pelanggan",
    "💡 Rekomendasi Strategis"
])

PALETTE = {
    'Champions / High-Value': '#27ae60',
    'Loyal Customers': '#2980b9',
    'Recent / Promising Customers': '#f39c12',
    'At-Risk / Hibernating': '#e74c3c'
}

with tab1:
    st.subheader("Ringkasan Metrik Aktual per Segmen Pelanggan")
    
    # Aggregated profile table
    summary_table = []
    tot_rev = df_seg['Monetary'].sum()
    tot_cust = len(df_seg)
    
    for seg in all_segments:
        sub = df_seg[df_seg['Segment_Name'] == seg]
        summary_table.append({
            'Segmen Pelanggan': seg,
            'Jumlah Pelanggan': f"{len(sub):,}",
            'Pangsa Pelanggan (%)': f"{(len(sub)/tot_cust*100):.2f}%",
            'Recency (Median)': f"{sub['Recency'].median():.0f} hari",
            'Recency (Mean)': f"{sub['Recency'].mean():.1f} hari",
            'Frequency (Median)': f"{sub['Frequency'].median():.0f} order",
            'Frequency (Mean)': f"{sub['Frequency'].mean():.2f} order",
            'Monetary (Median)': f"£{sub['Monetary'].median():,.2f}",
            'Monetary (Mean)': f"£{sub['Monetary'].mean():,.2f}",
            'Total Pendapatan (£)': f"£{sub['Monetary'].sum():,.2f}",
            'Pangsa Pendapatan (%)': f"{(sub['Monetary'].sum()/tot_rev*100):.2f}%"
        })
    st.dataframe(pd.DataFrame(summary_table), width='stretch')
    
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 👥 Proporsi Jumlah Pelanggan per Segmen")
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#1a1d24')
        ax.set_facecolor('#1a1d24')
        seg_counts = df_seg['Segment_Name'].value_counts()[all_segments]
        colors = [PALETTE[s] for s in seg_counts.index]
        ax.pie(seg_counts, labels=seg_counts.index, autopct='%1.1f%%', colors=colors, startangle=140,
               wedgeprops=dict(width=0.55, edgecolor='#1a1d24', linewidth=2),
               textprops={'color': '#ffffff', 'weight': 'bold'})
        ax.set_title("Distribusi Pelanggan (N=4,338)", fontweight='bold', color='#ffffff', pad=12)
        st.pyplot(fig)
        plt.close()
        
    with c2:
        st.markdown("##### 💰 Proporsi Kontribusi Pendapatan per Segmen")
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#1a1d24')
        ax.set_facecolor('#1a1d24')
        seg_rev = df_seg.groupby('Segment_Name')['Monetary'].sum()[all_segments]
        colors = [PALETTE[s] for s in seg_rev.index]
        ax.pie(seg_rev, labels=seg_rev.index, autopct='%1.1f%%', colors=colors, startangle=140,
               wedgeprops=dict(width=0.55, edgecolor='#1a1d24', linewidth=2),
               textprops={'color': '#ffffff', 'weight': 'bold'})
        ax.set_title("Distribusi Pendapatan (Total £8.89 Juta)", fontweight='bold', color='#ffffff', pad=12)
        st.pyplot(fig)
        plt.close()

with tab2:
    st.subheader("Visualisasi Proyeksi 2D PCA & Karakteristik Fitur")
    
    pcol1, pcol2 = st.columns([1.3, 1.0])
    with pcol1:
        st.markdown("##### 📍 Scatter Plot Proyeksi PCA 2D")
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#1a1d24')
        ax.set_facecolor('#1e222d')
        for seg in selected_segments:
            sub = filtered_df[filtered_df['Segment_Name'] == seg]
            ax.scatter(sub['PCA1'], sub['PCA2'], c=PALETTE[seg], label=f"{seg} (n={len(sub):,})", alpha=0.7, s=32)
        ax.set_xlabel("Principal Component 1 (Engagement / Nilai Pembelian Tinggi)", color='#e0e6ed')
        ax.set_ylabel("Principal Component 2 (Variasi Recency / Transaksi Terakhir)", color='#e0e6ed')
        ax.set_title("Peta Persebaran Segmen Pelanggan dalam Ruang 2D PCA", fontweight='bold', color='#ffffff', pad=12)
        ax.legend(loc='upper right', frameon=True, facecolor='#1e222d', edgecolor='#4b6584', labelcolor='white')
        ax.grid(True, linestyle='--', alpha=0.3, color='#4b6584')
        st.pyplot(fig)
        plt.close()
        
    with pcol2:
        st.markdown("##### 📦 Perbandingan Boxplot RFM (Skala log1p)")
        feature_choice = st.selectbox("Pilih Fitur untuk Boxplot:", ["Monetary", "Frequency", "Recency"])
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor('#1a1d24')
        ax.set_facecolor('#1e222d')
        sns.boxplot(
            data=filtered_df,
            x='Segment_Name',
            y=np.log1p(filtered_df[feature_choice]),
            hue='Segment_Name',
            palette=PALETTE,
            legend=False,
            ax=ax
        )
        ax.tick_params(axis='x', rotation=25, colors='#e0e6ed')
        ax.set_ylabel(f"log1p({feature_choice})", color='#e0e6ed')
        ax.set_xlabel("", color='#e0e6ed')
        ax.set_title(f"Sebaran {feature_choice} per Segmen Terpilih", fontweight='bold', color='#ffffff', pad=12)
        ax.grid(True, linestyle='--', alpha=0.3, color='#4b6584', axis='y')
        st.pyplot(fig)
        plt.close()

with tab3:
    st.subheader("Eksplorasi Data Pelanggan Tersegmentasi")
    st.markdown(f"Menampilkan **{len(filtered_df):,}** pelanggan berdasarkan kriteria filter.")
    
    search_id = st.text_input("Cari Customer ID:", placeholder="Ketik angka ID pelanggan (misal: 12347)...")
    display_df = filtered_df[['CustomerID', 'Country', 'Recency', 'Frequency', 'Monetary', 'Segment_Name']].copy()
    
    if search_id.strip():
        display_df = display_df[display_df['CustomerID'].astype(str).str.contains(search_id.strip())]
        
    st.dataframe(
        display_df.sort_values(by='Monetary', ascending=False),
        column_config={
            "CustomerID": st.column_config.NumberColumn("Customer ID", format="%d"),
            "Recency": st.column_config.NumberColumn("Recency (Hari)", format="%d"),
            "Frequency": st.column_config.NumberColumn("Frequency (Order)", format="%d"),
            "Monetary": st.column_config.NumberColumn("Monetary (Total Belanja)", format="£%.2f"),
        },
        width='stretch',
        height=350
    )
    
    csv_bytes = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data Pelanggan Terfilter (CSV)",
        data=csv_bytes,
        file_name="filtered_customers_rfm.csv",
        mime="text/csv"
    )

with tab4:
    st.subheader("Rekomendasi Bisnis & Strategi Tindakan per Segmen")
    
    rcol1, rcol2 = st.columns(2)
    with rcol1:
        st.markdown("""
        ### 🌟 1. Champions / High-Value (16.3% Pelanggan | 64.7% Pendapatan)
        - **Karakteristik:** Pembeli paling aktif (median order: 10x), belanja sangat tinggi (median: £3,723), dan bertransaksi sangat baru (median: 8 hari).
        - **Strategi Utama (Retention & Advocacy):**
          - Program VIP Eksklusif dengan Dedicated Account Manager / concierge service.
          - Early access untuk peluncuran lini produk dan edisi terbatas musiman.
          - Program apresiasi non-moneter (hadiah personal, ucapan khusus, packaging premium).
          - Hindari diskon masif yang menurunkan perceived value; fokus pada kemudahan transaksi B2B/grosir.
        
        ---
        
        ### 💎 2. Loyal Customers (27.0% Pelanggan | 23.8% Pendapatan)
        - **Karakteristik:** Pelanggan setia dengan pembelian stabil berkala (median order: 4x, belanja: £1,342), recency wajar (median: 55 hari).
        - **Strategi Utama (Upselling & Loyalty Building):**
          - Program loyalitas berjenjang (tier-based rewards) untuk mendorong frekuensi belanja.
          - Rekomendasi produk berbasis riwayat belanja (Cross-selling produk pelengkap).
          - Insentif bundle produk atau minimal transaksi untuk menaikkan basket size.
        """)
        
    with rcol2:
        st.markdown("""
        ### 🌱 3. Recent / Promising Customers (19.3% Pelanggan | 5.1% Pendapatan)
        - **Karakteristik:** Baru bertransaksi (median: 17 hari), frekuensi belanja masih awal (median: 2x, belanja: £467).
        - **Strategi Utama (Nurturing & Habituation):**
          - Kampanye onboarding dan edukasi katalog lengkap produk kado/hadiah.
          - Voucher "Second Purchase" atau "Next Order" berbatas waktu (misal berlaku 30 hari).
          - Survei kepuasan pelanggan pertama dan layanan purna jual yang proaktif.
        
        ---
        
        ### ⚠️ 4. At-Risk / Hibernating (37.3% Pelanggan | 6.3% Pendapatan)
        - **Karakteristik:** Sudah sangat lama tidak aktif (median: 177 hari / ~6 bulan), frekuensi rendah (median: 1x, belanja: £298).
        - **Strategi Utama (Win-Back & Cost-Efficient Engagement):**
          - Kampanye reaktivasi selektif: Email "We Miss You" dengan penawaran diskon khusus.
          - Survei kendala untuk memahami alasan churn (harga, pengiriman, ketersediaan produk).
          - Jika tidak merespons setelah 2-3 kali kontak, kurangi frekuensi promosi langsung untuk menghemat biaya pemasaran.
        """)

st.markdown("---")
st.caption("Customer Segmentation Dashboard | Dibangun menggunakan Streamlit & Scikit-Learn | Dataset UCI Online Retail (DOI: 10.24432/C5BW33)")
