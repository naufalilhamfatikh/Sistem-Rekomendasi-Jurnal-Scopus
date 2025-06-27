import streamlit as st
from utils import load_model_safely, preprocess_text
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Rekomendasi Jurnal Scopus Bidang Computer Scince ",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS kustom untuk tampilan yang lebih baik
st.markdown("""
<style>
    .journal-card {
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4f46e5;
        background-color: #ffffff;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    .journal-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    .journal-header {
        color: #4f46e5;
        margin-top: 0;
        font-size: 1.25rem;
        font-weight: 600;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 0.75rem;
        margin-bottom: 1rem;
    }
    .journal-detail {
        margin-bottom: 0.75rem;
        font-size: 0.95rem;
        color: #4b5563;
        display: flex;
    }
    .journal-detail-label {
        font-weight: 600;
        min-width: 150px;
        color: #1f2937;
    }
    .journal-detail-value {
        flex: 1;
    }
    .similarity-high {
        font-weight: bold;
        color: #dc2626;
    }
    .apc-value {
        font-weight: bold;
        background-color: #d1fae5;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        color: #065f46;
    }
    .apc-free {
        font-weight: bold;
        background-color: #dbeafe;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        color: #1e40af;
    }
    .unknown-apc {
        font-weight: bold;
        background-color: #f3f4f6;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        color: #6b7280;
    }
    .journal-detail-section {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #f9fafb;
        border-radius: 8px;
    }
    .section-title {
        color: #4f46e5;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 0.5rem;
    }
    .stButton>button {
        background-color: #4f46e5;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #4338ca;
        transform: translateY(-1px);
    }
    .stTextArea>div>div>textarea {
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .stSelectbox>div>div>select {
        border-radius: 0.375rem;
    }
    .journal-detail-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
    }
    @media (max-width: 768px) {
        .journal-detail-grid {
            grid-template-columns: 1fr;
        }
    }
    .expandable-content {
        margin-top: 1rem;
        padding: 1.5rem;
        background-color: #f9fafb;
        border-radius: 8px;
        border-left: 4px solid #4f46e5;
    }
    .graph-container {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
        .status-active {
        font-weight: bold;
        background-color: #d1fae5;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        color: #065f46;
    }
    .status-inactive {
        font-weight: bold;
        background-color: #fee2e2;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        color: #b91c1c;
    }
        .status-badge {
        font-weight: 600;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        display: inline-block;
        margin-left: 0.5rem;
    }
    .status-active {
        background-color: #d1fae5;
        color: #065f46;
    }
    .status-inactive {
        background-color: #fee2e2;
        color: #b91c1c;
    }
</style>
""", unsafe_allow_html=True)

def bersihkan_cite_score(x):
    try:
        if pd.isna(x):
            return np.nan
        x = str(x).strip()
        if x in ['-', '', 'nan', 'NaN']:
            return np.nan
        if ',' in x:
            return float(x.replace(',', '.'))
        return float(x)
    except:
        return np.nan

def bersihkan_apc(x):
    x_str = str(x).strip()
    if x_str == '-':
        return 0
    elif x_str in ['', 'nan', 'NaN']:
        return np.nan
    try:
        return float(x_str.replace(',', ''))
    except:
        return np.nan

@st.cache_resource
def muat_model():
    model_path = "models/model_final_TF-IDF&SVD_rev.pkl"
    model_data = load_model_safely(model_path)
    
    model_data['df']['APC_asli'] = model_data['df'][' APC (Biaya Publikasi)'].copy()
    model_data['df'][' APC (Biaya Publikasi)'] = model_data['df'][' APC (Biaya Publikasi)'].apply(bersihkan_apc)
    model_data['df']['Cite Score'] = model_data['df']['Cite Score'].apply(bersihkan_cite_score)
    
    return model_data

model_data = muat_model()
model = model_data['model']
df = model_data['df']

def dapatkan_rekomendasi(abstrak_input, top_n=15, harga_min=None, harga_max=None, indeks_scopus=None):
    query_processed = preprocess_text(abstrak_input)
    query_vec = model.tfidf.transform([query_processed])
    query_svd = model.svd.transform(query_vec)

    doc_vectors = model.transform(df['processed_focus_scope'])
    similarities = cosine_similarity(query_svd, doc_vectors).flatten()
    
    rekomendasi = df.copy()
    rekomendasi['similarity_score'] = similarities
    
    if indeks_scopus is not None:
        rekomendasi = rekomendasi[rekomendasi['Index Scopus'].isin(indeks_scopus)]
    if harga_min is not None:
        rekomendasi = rekomendasi[rekomendasi[' APC (Biaya Publikasi)'].notna() & (rekomendasi[' APC (Biaya Publikasi)'] >= harga_min)]
    if harga_max is not None:
        rekomendasi = rekomendasi[rekomendasi[' APC (Biaya Publikasi)'].notna() & (rekomendasi[' APC (Biaya Publikasi)'] <= harga_max)]
    
    return rekomendasi.sort_values('similarity_score', ascending=False).head(top_n)

def format_metric_value(value):
    if pd.isna(value):
        return "Tidak tersedia"
    try:
        if isinstance(value, (int, float)):
            if value == int(value):
                return str(int(value))
            return f"{float(value):.3f}".replace('.000', '').rstrip('0').rstrip('.') if '.' in f"{float(value):.3f}" else f"{float(value):.3f}"
        return str(value)
    except:
        return str(value)

def tampilkan_detail_jurnal(row):
    with st.expander(f"📖 Detail Lengkap: {row['Source Title']}", expanded=False):
        # Format status
        status = row.get('Active or Inactive', 'Tidak tersedia')
        if status == 'Active':
            status_display = "<span class='status-active'>Aktif</span>"
        elif status == 'Inactive':
            status_display = "<span class='status-inactive'>Tidak Aktif</span>"
        else:
            status_display = f"<span>{status}</span>"
            
        st.markdown(f"""
        <div class="expandable-content">
            <div class="journal-detail-grid">
                <div>
                    <div class="section-title">Informasi Dasar</div>
                    <div class="journal-detail">
                        <div class="journal-detail-label">Status:</div>
                        <div class="journal-detail-value">{status_display}</div>
                    </div>
                    <div class="journal-detail">
                        <div class="journal-detail-label">Journal ID:</div>
                        <div class="journal-detail-value">{row.get('Journal ID', 'Tidak tersedia')}</div>
                    </div>
                    <div class="journal-detail">
                        <div class="journal-detail-label">Nama Jurnal:</div>
                        <div class="journal-detail-value">{row.get('Source Title', 'Tidak tersedia')}</div>
                    </div>
                    <div class="journal-detail">
                        <div class="journal-detail-label">Publisher:</div>
                        <div class="journal-detail-value">{row.get('Publisher', 'Tidak tersedia')}</div>
                    </div>
                    <div class="journal-detail">
                        <div class="journal-detail-label">Kuartil Scopus:</div>
                        <div class="journal-detail-value">{row.get('Index Scopus', 'Tidak tersedia')}</div>
                    </div>
                    <div class="journal-detail">
                        <div class="journal-detail-label">URL Jurnal:</div>
                        <div class="journal-detail-value"><a href="{row.get('URL Journal', '#')}" target="_blank">Kunjungi Jurnal</a></div>
                    </div>
                </div>
                <div>
                    <div class="section-title">Statistik Publikasi</div>
                    <div class="journal-detail">
                        <div class="journal-detail-label">2020-23 Citations:</div>
                        <div class="journal-detail-value">{format_metric_value(row.get('2020 - 23 Citations'))}</div>
                    </div>
                    <div class="journal-detail">
                        <div class="journal-detail-label">2020-23 Documents:</div>
                        <div class="journal-detail-value">{format_metric_value(row.get('2020 - 23 Documents'))}</div>
                    </div>
                    <div class="journal-detail">
                        <div class="journal-detail-label">% Cited:</div>
                        <div class="journal-detail-value">{format_metric_value(row.get('% Cited'))}</div>
                    </div>
                </div>
                <div>
                    <div class="section-title">Metrik Kualitas</div>
                    <div class="journal-detail">
                        <div class="journal-detail-label">Cite Score:</div>
                        <div class="journal-detail-value">{format_metric_value(row.get('Cite Score'))}</div>
                    </div>
                    <div class="journal-detail">
                        <div class="journal-detail-label">Faktor Dampak:</div>
                        <div class="journal-detail-value">{format_metric_value(row.get('Impact Factor'))}</div>
                    </div>
                    <div class="journal-detail">
                        <div class="journal-detail-label">SNIP:</div>
                        <div class="journal-detail-value">{format_metric_value(row.get('SNIP'))}</div>
                    </div>
                    <div class="journal-detail">
                        <div class="journal-detail-label">SJR:</div>
                        <div class="journal-detail-value">{format_metric_value(row.get('SJR'))}</div>
                    </div>
                    <div class="journal-detail">
                        <div class="journal-detail-label">Highest Precentile:</div>
                        <div class="journal-detail-value">{format_metric_value(row.get('Highest Precentile'))}</div>
                    </div>
                </div>
                <div>
                    <div class="section-title">Proses Editorial </div>
                    <div class="journal-detail">
                        <div class="journal-detail-label">Tingkat Penerimaan:</div>
                        <div class="journal-detail-value">{format_metric_value(row.get('Acceptance Rate'))}</div>
                    </div>
                    <div class="journal-detail">
                        <div class="journal-detail-label">Accept to Publish:</div>
                        <div class="journal-detail-value">{format_metric_value(row.get('Accept to Publish'))}</div>
                    </div>
                    <div class="journal-detail">
                        <div class="journal-detail-label">Submission to Acceptance:</div>
                        <div class="journal-detail-value">{format_metric_value(row.get('Submission to Acceptance'))}</div>
                    </div>
                    <div class="journal-detail">
                        <div class="journal-detail-label">APC:</div>
                        <div class="journal-detail-value">{
                            f"<span class='unknown-apc'>Tidak diketahui</span>" if str(row['APC_asli']).strip() == '-' 
                            else f"<span class='apc-free'>Gratis (Rp0)</span>" if row[' APC (Biaya Publikasi)'] == 0 
                            else f"<span class='unknown-apc'>Tidak tersedia</span>" if pd.isna(row[' APC (Biaya Publikasi)']) 
                            else f"<span class='apc-value'>${row[' APC (Biaya Publikasi)']:,.2f}</span>"
                        }</div>
                    </div>
                </div>
            </div>
            <div class="section-title">Fokus dan Ruang Lingkup</div>
            <div style="line-height: 1.6; text-align: justify;">
                {row.get('focus_scope', 'Tidak tersedia')}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Inisialisasi session state
if 'rekomendasi' not in st.session_state:
    st.session_state.rekomendasi = None
if 'abstrak_submitted' not in st.session_state:
    st.session_state.abstrak_submitted = ""

# Antarmuka Pengguna
st.title("📚 Sistem Rekomendasi Jurnal Scopus Bidang Computer Scince ")
st.markdown("""
<div style="color: #4b5563; margin-bottom: 1.5rem;">
    Masukkan abstrak dan keyword penelitian Anda untuk mendapatkan rekomendasi jurnal Scopus bidang computer scince yang relevan berdasarkan kesesuaian topik.
</div>
""", unsafe_allow_html=True)

# Filter
with st.sidebar:
    st.header("⚙️ Filter Pencarian")
    filter_harga = st.checkbox("Filter Biaya Publikasi (APC)", key="filter_harga")
    filter_scopus = st.checkbox("Filter Index Scopus", key="filter_scopus")
    
    if st.session_state.filter_harga:
        harga_min, harga_max = st.slider(
            "Rentang Biaya Publikasi (USD)",
            0, 10000, (0, 5000),
            50
        )
    else:
        harga_min, harga_max = None, None
    
    if st.session_state.filter_scopus:
        scopus_terpilih = st.multiselect(
            "Pilih Index Scopus:", 
            options=sorted(df['Index Scopus'].dropna().unique().tolist()),
            default=['Q1', 'Q2']
        )
    else:
        scopus_terpilih = None

# Form input
with st.form("form_rekomendasi"):
    col1, col2 = st.columns([3, 1])
    with col1:
        abstrak = st.text_area(
            "Abstrak dan Keyword Penelitian Anda", 
            height=200, 
            placeholder="Masukkan abstrak dan keyword penelitian Anda dalam bahasa Inggris...",
            help="Semakin lengkap abstrak dan keyword yang Anda masukkan, semakin akurat rekomendasi yang diberikan"
        )
    with col2:
        top_n = st.slider("Jumlah Rekomendasi", 5, 30, 15, 1)
        submitted = st.form_submit_button("🔍 Dapatkan Rekomendasi")

# Contoh abstrak
with st.expander("📋 Contoh Abstrak", expanded=False):
    contoh = [
        "Machine learning algorithms for natural language processing have shown remarkable progress in recent years...",
        "Renewable energy solutions and their impact on climate change mitigation strategies...",
        "Advanced techniques in cancer immunotherapy focusing on checkpoint inhibitors and CAR-T cell therapy..."
    ]
    contoh_terpilih = st.selectbox("Pilih contoh abstrak:", contoh)
    if st.button("Gunakan Contoh", key="contoh_btn"):
        abstrak = contoh_terpilih

# Proses rekomendasi
if submitted and abstrak:
    st.session_state.abstrak_submitted = abstrak
    with st.spinner('🔍 Mencari rekomendasi jurnal yang sesuai...'):
        st.session_state.rekomendasi = dapatkan_rekomendasi(
            abstrak_input=abstrak,
            top_n=top_n,
            harga_min=harga_min if st.session_state.filter_harga else None,
            harga_max=harga_max if st.session_state.filter_harga else None,
            indeks_scopus=scopus_terpilih if st.session_state.filter_scopus else None
        )

# Tampilkan hasil rekomendasi
if st.session_state.rekomendasi is not None:
    rekomendasi = st.session_state.rekomendasi.copy()
    
    if not rekomendasi.empty:
        st.success(f"🎉 Menemukan {len(rekomendasi)} rekomendasi jurnal yang relevan!")
        
        # Hitung metrik rata-rata
        avg_similarity = rekomendasi['similarity_score'].mean()
        avg_cite_score = rekomendasi['Cite Score'].replace(0, np.nan).mean(skipna=True)
        avg_apc = rekomendasi[' APC (Biaya Publikasi)'].replace(0, np.nan).mean(skipna=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rata-rata Kemiripan", f"{avg_similarity:.4f}")
        with col2:
            st.metric("Rata-rata Cite Score", f"{avg_cite_score:.1f}" if not pd.isna(avg_cite_score) else "Tidak tersedia")
        with col3:
            st.metric("Rata-rata APC (USD)", f"${avg_apc:,.2f}" if not pd.isna(avg_apc) else "Tidak tersedia")
        
        # Tambahkan visualisasi distribusi
        st.subheader("📊 Distribusi Rekomendasi")
        
        # Buat tab untuk grafik
        tab1, tab2 = st.tabs(["Distribusi Cite Score", "Distribusi APC"])
        
        with tab1:
            st.markdown('<div class="graph-container">', unsafe_allow_html=True)
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.histplot(rekomendasi['Cite Score'].dropna(), bins=10, kde=True, ax=ax1, color='#4f46e5')
            ax1.set_title('Distribusi Cite Score pada Rekomendasi', pad=20)
            ax1.set_xlabel('Cite Score')
            ax1.set_ylabel('Jumlah Jurnal')
            ax1.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig1)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Box plot untuk Cite Score per Quartile
            if 'Index Scopus' in rekomendasi.columns:
                st.markdown('<div class="graph-container">', unsafe_allow_html=True)
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=rekomendasi, x='Index Scopus', y='Cite Score', order=['Q1', 'Q2', 'Q3', 'Q4'], 
                            palette='viridis', ax=ax2)
                ax2.set_title('Distribusi Cite Score per Kuartil Scopus', pad=20)
                ax2.set_xlabel('Kuartil Scopus')
                ax2.set_ylabel('Cite Score')
                ax2.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig2)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            # Filter out free journals for APC visualization
            apc_data = rekomendasi[rekomendasi[' APC (Biaya Publikasi)'] > 0]
            
            if not apc_data.empty:
                st.markdown('<div class="graph-container">', unsafe_allow_html=True)
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                sns.histplot(apc_data[' APC (Biaya Publikasi)'], bins=10, kde=True, ax=ax3, color='#10b981')
                ax3.set_title('Distribusi APC (Biaya Publikasi) pada Rekomendasi', pad=20)
                ax3.set_xlabel('APC (USD)')
                ax3.set_ylabel('Jumlah Jurnal')
                ax3.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig3)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Box plot untuk APC per Quartile
                if 'Index Scopus' in apc_data.columns:
                    st.markdown('<div class="graph-container">', unsafe_allow_html=True)
                    fig4, ax4 = plt.subplots(figsize=(10, 6))
                    sns.boxplot(data=apc_data, x='Index Scopus', y=' APC (Biaya Publikasi)', 
                                order=['Q1', 'Q2', 'Q3', 'Q4'], palette='viridis', ax=ax4)
                    ax4.set_title('Distribusi APC per Kuartil Scopus', pad=20)
                    ax4.set_xlabel('Kuartil Scopus')
                    ax4.set_ylabel('APC (USD)')
                    ax4.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig4)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Tidak ada data APC yang tersedia untuk divisualisasikan (kecuali jurnal gratis)")
        
        # Opsi pengurutan
        opsi_pengurutan = st.selectbox(
            "Urutkan berdasarkan:",
            ["Similarity Tertinggi", "Cite Score Tertinggi", "Cite Score Terendah", "Biaya Publikasi Terendah"],
            key="opsi_pengurutan"
        )
        
        if opsi_pengurutan == "Cite Score Tertinggi":
            rekomendasi = rekomendasi.sort_values('Cite Score', ascending=False)
        elif opsi_pengurutan == "Cite Score Terendah":
            rekomendasi = rekomendasi.sort_values('Cite Score', ascending=True)
        elif opsi_pengurutan == "Biaya Publikasi Terendah":
            rekomendasi = rekomendasi.sort_values(' APC (Biaya Publikasi)', ascending=True)
        else:
            rekomendasi = rekomendasi.sort_values('similarity_score', ascending=False)
        
        st.subheader("📋 Hasil Rekomendasi")
        st.markdown(f"<div style='color: #4b5563; margin-bottom: 1rem;'>Menampilkan {len(rekomendasi)} jurnal terbaik berdasarkan kriteria Anda</div>", unsafe_allow_html=True)
        
        # Tampilkan setiap rekomendasi
        for idx, (_, row) in enumerate(rekomendasi.iterrows(), 1):
            similarity_class = "similarity-high" if row['similarity_score'] > 0.8 else ""
            
            # Format data
            cite_score = format_metric_value(row.get('Cite Score'))
            apc_asli = row['APC_asli']
            apc_hitung = row[' APC (Biaya Publikasi)']
            
            if str(apc_asli).strip() == '-':
                apc_display = "<span class='unknown-apc'>Tidak diketahui</span>"
            elif apc_hitung == 0:
                apc_display = "<span class='apc-free'>Gratis</span>"
            else:
                apc_display = f"<span class='apc-value'>${apc_hitung:,.2f}</span>"
            
            # Format status untuk kartu
            status = row.get('Active or Inactive', 'Unknown')
            if status == 'Active':
                status_badge = "<span class='status-badge status-active'>Aktif</span>"
            elif status == 'Inactive':
                status_badge = "<span class='status-badge status-inactive'>Tidak Aktif</span>"
            else:
                status_badge = f"<span class='status-badge unknown-apc'>{status}</span>"
            
            # Tampilkan kartu jurnal
            st.markdown(f"""
            <div class="journal-card">
                <h4 class="journal-header">
                    <span style="color: #4f46e5;">#{idx}</span> {row.get('Source Title', 'Tidak tersedia')} {status_badge}
                </h4>
                <div class="journal-detail">
                    <div class="journal-detail-label">Kuartil Scopus:</div>
                    <div class="journal-detail-value">{row.get('Index Scopus', 'Tidak tersedia')}</div>
                </div>
                <div class="journal-detail">
                    <div class="journal-detail-label">Similarity:</div>
                    <div class="journal-detail-value"><span class="{similarity_class}">{row['similarity_score']:.4f}</span></div>
                </div>
                <div class="journal-detail">
                    <div class="journal-detail-label">Cite Score:</div>
                    <div class="journal-detail-value">{cite_score}</div>
                </div>
                <div class="journal-detail">
                    <div class="journal-detail-label">APC:</div>
                    <div class="journal-detail-value">{apc_display}</div>
                </div>
                <div class="journal-detail">
                    <div class="journal-detail-label">Publisher:</div>
                    <div class="journal-detail-value">{row.get('Publisher', 'Tidak tersedia')}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Tampilkan detail dalam expander yang terintegrasi dengan card
            tampilkan_detail_jurnal(row)
            
            # Tutup card
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Tombol unduh
        st.download_button(
            "📥 Unduh Hasil Rekomendasi (CSV)",
            rekomendasi.to_csv(index=False).encode('utf-8'),
            "rekomendasi_jurnal_scopus.csv",
            "text/csv",
            help="Unduh seluruh hasil rekomendasi dalam format CSV"
        )
    else:
        st.error("⚠️ Tidak ditemukan jurnal yang sesuai dengan kriteria pencarian. Coba perlebar filter Anda.")
elif not submitted and st.session_state.abstrak_submitted:
    st.info("ℹ️ Silakan klik tombol 'Dapatkan Rekomendasi' untuk memproses abstrak Anda")
else:
    st.info("ℹ️ Masukkan abstrak penelitian Anda dan klik tombol 'Dapatkan Rekomendasi' untuk memulai")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; font-size: 0.9em; padding: 1rem;'>
    Sistem Rekomendasi Jurnal Scopus Bidang Computer Scince © 2025 | Dibangun oleh Naufal Ilham Fatikh
</div>
""", unsafe_allow_html=True)