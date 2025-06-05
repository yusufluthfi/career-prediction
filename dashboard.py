import streamlit as st
import mysql.connector

# Fungsi koneksi database
def connect_to_db():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="data_mining_system"
        )
    except mysql.connector.Error as e:
        st.error(f"Gagal terhubung ke database: {e}")
        return None

# Fungsi untuk mengambil jumlah data dari tabel
def get_count(table_name):
    db = connect_to_db()
    if not db:
        return 0
    cursor = db.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    cursor.close()
    db.close()
    return count

# Fungsi utama dashboard
def show_dashboard():
    st.write("")
    st.subheader("📊 Dashboard Utama")
    st.markdown("Berikut statistik penggunaan sistem prediksi karier")

    jumlah_user = get_count("pengguna")
    jumlah_riwayat = get_count("riwayat_prediksi")  # Sesuaikan dengan nama tabel Anda

    # st.markdown("### 🔍 Ringkasan Data")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("#### 👥 Jumlah Pengguna")
            st.metric(label="Total User", value=f"{jumlah_user} Akun",label_visibility="hidden")
            st.progress(min(jumlah_user, 100) / 100)  # animasi progres jika ingin

    with col2:
        with st.container(border=True):
            st.markdown("#### 📁 Telah Memprediksi")
            st.metric(label="Total Riwayat", value=f"{jumlah_riwayat} Prediksi",label_visibility="hidden")
            st.progress(min(jumlah_riwayat, 100) / 100)

    st.markdown("---")

    st.info("Gunakan sidebar untuk mulai mengunggah data dan menjalankan model prediksi.")

