import streamlit as st
import mysql.connector
import bcrypt
import uuid
import CSS


def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="data_mining_system"
    )

def add_user(nama, username, password, hak_akses):
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    user_id = str(uuid.uuid4())

    db = connect_to_db()
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO pengguna (id_pengguna, nama_lengkap, nama_pengguna, kata_sandi, hak_akses) VALUES (%s, %s, %s, %s, %s)",
        (user_id, nama, username, hashed, hak_akses)
    )
    db.commit()
    cursor.close()
    db.close()

def form_tambah_pengguna():
    CSS.csstambahpengguna()
    st.subheader("Tambah Pengguna Baru")
    st.markdown("Silakan lengkapi form berikut untuk menambahkan pengguna baru.")

    if "new_nama" not in st.session_state:
        st.session_state.new_nama = ""
    if "new_nama_pengguna" not in st.session_state:
        st.session_state.new_nama_pengguna = ""
    if "new_kata_sandi" not in st.session_state:
        st.session_state.new_kata_sandi = ""
    if "new_hak_akses" not in st.session_state:
        st.session_state.new_hak_akses = "user"

    st.session_state.new_nama = st.text_input("Nama Lengkap", value=st.session_state.new_nama, key="form_nama", placeholder="Masukkan nama lengkap")
    st.session_state.new_nama_pengguna = st.text_input("Username", value=st.session_state.new_nama_pengguna, key="form_nama_pengguna", placeholder="Buat Username Akun")
    st.session_state.new_kata_sandi = st.text_input("Kata Sandi", type="password", value=st.session_state.new_kata_sandi, key="form_kata_sandi", placeholder="Buat Kata Sandi")
    st.session_state.new_hak_akses = st.selectbox("Hak Akses", ["user", "admin"], index=0 if st.session_state.new_hak_akses == "user" else 1, key="form_hak_akses")

    col1, col2, col3 = st.columns([6.5,4,2])
    with col1:
        if st.button("Simpan Pengguna"):
            if st.session_state.new_nama and st.session_state.new_nama_pengguna and st.session_state.new_kata_sandi:
                add_user(
                    st.session_state.new_nama,
                    st.session_state.new_nama_pengguna,
                    st.session_state.new_kata_sandi,
                    st.session_state.new_hak_akses
                )
                st.success("✅ Pengguna baru berhasil ditambahkan.")
                # Reset & arahkan kembali
                st.session_state.new_nama = ""
                st.session_state.new_nama_pengguna = ""
                st.session_state.new_kata_sandi = ""
                st.session_state.new_hak_akses = "user"
                st.session_state.halaman = "kelola_pengguna"
                st.rerun()
            else:
                st.error("❗ Semua field wajib diisi.")

    with col3:
        if st.button("↩ Kembali"):
            st.session_state.halaman = "kelola_pengguna"
            st.rerun()
