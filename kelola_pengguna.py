import streamlit as st
import mysql.connector
import time
import streamlit.components.v1 as components
import CSS


def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="data_mining_system"
    )

def main():

    st.subheader("Kelola Pengguna")
    st.markdown("Berikut data pengguna sistem prediksi karier")

    db = connect_to_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM pengguna")
    data = cursor.fetchall()
    cursor.close()
    db.close()

    col5, col6, col7 = st.columns([6, 5, 3])
    with col5:
        search_query = st.text_input("🔍 Cari Nama Pengguna", placeholder="Cari nama pengguna...", label_visibility="hidden")
    with col7 :
        st.markdown("<div style='padding-top: 1.8rem;'>", unsafe_allow_html=True)
        if st.button("Buat User Baru"):
            st.session_state.halaman = "form_tambah_pengguna"
            st.rerun()

    st.markdown("---")

    if search_query:
        data = [user for user in data if search_query.lower() in user['nama_pengguna'].lower()]

    if "delete_confirm" not in st.session_state:
        st.session_state.delete_confirm = None  # Menyimpan ID pengguna yang ingin dihapus

    if "delete_success" not in st.session_state:
        st.session_state.delete_success = None  # Menyimpan ID pengguna yang berhasil dihapus


    if data:
        # Header
        col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns([1.5, 2, 1.5, 3, 1.6])
        with col_h1:
            st.markdown("**ID**")
        with col_h2:
            st.markdown("**Nama**")
        with col_h3:
            st.markdown("**Username**")
        with col_h4:
            st.markdown("**Hak Akses**")
        with col_h5:
            st.markdown("**Aksi**")

        st.markdown("")

        for user in data:
            col1, col2, col3, col4,col5 = st.columns([1.5, 2, 1.5, 3, 1.6])
            with col1:
                st.markdown("<div style='padding-top: 1.9rem;'>", unsafe_allow_html=True)
                components.html(f"""
                    <button onclick="navigator.clipboard.writeText('{user['id_pengguna']}'); alert('ID Pengguna: {user['id_pengguna']} berhasil disalin!')" style="
                        padding: 6px 12px;
                        background-color: #4CAF50;
                        color: white;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                    ">
                        Lihat
                    </button>
                """, height=40)
            with col2:
                st.markdown("<div style='padding-top: 1.9rem;'>", unsafe_allow_html=True)
                st.write(user["nama_lengkap"])
            with col3:
                st.markdown("<div style='padding-top: 1.9rem;'>", unsafe_allow_html=True)
                st.write(user["nama_pengguna"])
            with col4:
                col11, col12 = st.columns([2.2, 2])
                with col11:
                    new_access = st.selectbox(
                        f"Hak Akses", 
                        ["user", "admin"], 
                        index=["user", "admin"].index(user["hak_akses"]),
                        key=f"select_{user['id_pengguna']}",label_visibility="hidden"
                    )
                with col12:
                    st.markdown("<div style='padding-bottom: 1.7rem;'>", unsafe_allow_html=True)
                    if st.button("Ubah", key=f"update_{user['id_pengguna']}"):
                        update_access(user["id_pengguna"], new_access)
                        st.success(f"Hak akses untuk {user['nama_pengguna']} diperbarui.")
                        time.sleep(5)
                        st.rerun()

            with col5:
                st.markdown("<div style='padding-top: 1.7rem;'>", unsafe_allow_html=True)
                if st.button("🗑️ Hapus", key=f"delete_{user['id_pengguna']}"):
                    st.session_state.delete_confirm = user  # Simpan user yang akan dihapus

        # Pop-up konfirmasi penghapusan
        if st.session_state.delete_confirm:
            user = st.session_state.delete_confirm
            st.warning(f"⚠️ Apakah Anda yakin ingin menghapus akun **{user['nama_pengguna']}**?")
            colC1, colC2 = st.columns(2)
            with colC1:
                if st.button("✅ Ya, Hapus"):
                    delete_user(user["id_pengguna"])
                    st.session_state.delete_success = user['nama_pengguna']
                    st.session_state.delete_confirm = None
                    st.rerun()
            with colC2:
                if st.button("❌ Batal"):
                    st.session_state.delete_confirm = None
                    st.rerun()

        # Notifikasi sukses penghapusan
        if st.session_state.delete_success:
            st.success(f"Akun **{st.session_state.delete_success}** berhasil dihapus.")
            time.sleep(2)
            st.session_state.delete_success = None
            st.rerun()

        # Inisialisasi nilai default di session_state
    if "new_nama" not in st.session_state:
        st.session_state.new_nama = ""
    if "new_nama_pengguna" not in st.session_state:
        st.session_state.new_nama_pengguna = ""
    if "new_kata_sandi" not in st.session_state:
        st.session_state.new_kata_sandi = ""
    if "new_hak_akses" not in st.session_state:
        st.session_state.new_hak_akses = "user"

    st.markdown("---")
    st.markdown("")


def update_access(user_id, hak_akses):
    db = connect_to_db()
    cursor = db.cursor()
    cursor.execute("UPDATE pengguna SET hak_akses = %s WHERE id_pengguna = %s", (hak_akses, user_id))
    db.commit()
    cursor.close()
    db.close()

def delete_user(user_id):
    db = connect_to_db()
    cursor = db.cursor()
    cursor.execute("DELETE FROM pengguna WHERE id_pengguna = %s", (user_id,))
    db.commit()
    cursor.close()
    db.close()

def add_user(nama,username, password, hak_akses):
    import bcrypt, uuid
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    user_id = str(uuid.uuid4())

    db = connect_to_db()
    cursor = db.cursor()
    cursor.execute("INSERT INTO pengguna (id_pengguna, nama_lengkap, nama_pengguna, kata_sandi, hak_akses) VALUES (%s, %s, %s, %s, %s)", (user_id, nama, username, hashed, hak_akses))
    db.commit()
    cursor.close()
    db.close()
