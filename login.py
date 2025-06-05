import streamlit as st
import mysql.connector
import uuid
import bcrypt
import os
import app
import CSS
import time
# Fungsi untuk koneksi ke database
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

# Hashing password
def hash_password(password):
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt)

# Verifikasi password
def check_password(stored_password, input_password):
    return bcrypt.checkpw(input_password.encode(), stored_password.encode() if isinstance(stored_password, str) else stored_password)

# Fungsi login
def login_user(username, password):
    db = connect_to_db()
    if not db:
        return None
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM pengguna WHERE nama_pengguna = %s", (username,))
    user = cursor.fetchone()
    cursor.close()
    db.close()

    if user and check_password(user['kata_sandi'], password):
        return user
    return None

# Fungsi register
def register_user(namalengkap, username, password):
    db = connect_to_db()
    if not db:
        return
    cursor = db.cursor()
    user_id = str(uuid.uuid4())
    hak_akses="user"
    hashed = hash_password(password)
    namalengkap = namalengkap.upper()
    cursor.execute(
        "INSERT INTO pengguna (id_pengguna,nama_lengkap, nama_pengguna, kata_sandi, hak_akses) VALUES (%s, %s, %s, %s, %s)",
        (user_id, namalengkap, username, hashed,hak_akses)
    )
    db.commit()
    cursor.close()
    db.close()

# Halaman login
def login_page():
    CSS.cssLogin()
    st.title("Login")

    with st.form(key="login_form"):
        username = st.text_input("Username")
        password = st.text_input("Kata Sandi", type="password")
        login_button = st.form_submit_button("Login")

        if login_button:
            user = login_user(username, password)
            if user:
                st.session_state.user = user
                st.session_state.page = "main"
                st.rerun()
            else:
                st.error("Username atau password salah.")

    if st.button("Daftar Akun"):
        st.session_state.page = "register"
        st.rerun()

# Halaman register
def register_page():
    CSS.cssLogin()
    st.title("Daftar Akun")

    with st.form(key="register_form"):
        namalengkap = st.text_input("Nama Lengkap", placeholder="Masukkan nama lengkap Anda")
        username = st.text_input("Username", placeholder="Buat username Anda")
        password1 = st.text_input("Kata Sandi", type="password", placeholder= "Buat kata sandi Anda")
        password2 = st.text_input("Konfirmasi Kata Sandi", type="password", placeholder="Ulangi kata sandi Anda")
        daftar_button = st.form_submit_button("Daftar")

        if daftar_button:
            if password1 != password2:
                st.error("Konfirmasi kata sandi tidak cocok.")
            else:
                db = connect_to_db()
                if not db:
                    return
                cursor = db.cursor(dictionary=True)
                cursor.execute("SELECT * FROM pengguna WHERE nama_pengguna = %s", (username,))
                if cursor.fetchone():
                    st.error("Upss.. Username sudah digunakan!")
                else:
                    register_user(namalengkap, username, password1)
                    st.session_state.page = "login"
                    st.success("Pendaftaran berhasil! Silakan Login sesaat lagi...")
                    time.sleep(3)
                    st.rerun()
                cursor.close()
                db.close()

    if st.button("Sudah Punya Akun?"):
        st.session_state.page = "login"
        st.rerun()

# def logout():
#     for key in list(st.session_state.keys()):
#         del st.session_state[key]
#     st.rerun()

# def main_page():
#     st.title("Selamat datang di Aplikasi")
#     st.write(f"Hai, {st.session_state.user['nama_pengguna']}!")
#     st.button("Logout", on_click=lambda: logout())


# Routing

if "page" not in st.session_state:
    st.session_state.page = "register"  # Halaman awal adalah register

if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "register":
    register_page()
elif st.session_state.page == "main":
    if "user" in st.session_state:
        app.main()
    else:
        st.session_state.page = "login"
        st.rerun()
