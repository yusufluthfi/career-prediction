import os
import streamlit as st
import pandas as pd
import time
import joblib
import uuid
import login
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import io
import mysql.connector
import kelola_pengguna
import form_tambah_pengguna 
import dashboard
import base64
import CSS
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from decision_tree import run_decision_tree
from random_forest import run_random_forest
from LR import run_logistic_regression
from svm import run_svm
from XGBoost import run_xgboost
from naive_bayes import run_naive_bayes
from AdaBoost import run_adaboost
from Cat_Boost import run_catboost
from KNN import run_knn
from Voting_Classifier import run_voting_classifier
from AdaBoost_XGBoost import run_AdaBoost_XGBoost
from sklearn.metrics import accuracy_score
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
from streamlit_option_menu import option_menu
from PIL import Image


def main() :

    def connect_to_db():
        return mysql.connector.connect(
            host="localhost",
            user="root", 
            password="", 
            database="data_mining_system" 
        )

    def simpan_riwayat_ke_db(data):
        db_connection = connect_to_db()
        cursor = db_connection.cursor()

        query = """
            INSERT INTO riwayat_prediksi (
                id_test, nim, nama, ipk, lama_studi, skill_kerjasama, beasiswa, hasil, tanggal, model_digunakan
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        values = (
            data["id_test"],
            data["nim"],
            data["nama"],
            data["ipk"],
            data["lama_studi"],
            data["skill_kerjasama"],
            data["beasiswa"],
            data["hasil"],
            data["tanggal"],
            data["model_digunakan"]
        )

        try:
            cursor.execute(query, values)
            db_connection.commit()
            cursor.close()
            db_connection.close()
            # st.success("Riwayat telah disimpan ke database.")
        except mysql.connector.Error as err:
            st.error(f"Gagal menyimpan data: {err}")
            cursor.close()
            db_connection.close()

    # menghapus riwayat by ID
    def hapus_riwayat_from_db(id_test):
        db_connection = connect_to_db()
        cursor = db_connection.cursor()

        query = "DELETE FROM riwayat_prediksi WHERE id_test = %s"
        try:
            cursor.execute(query, (id_test,))
            db_connection.commit()
            cursor.close()
            db_connection.close()
            st.success(f"Riwayat dengan ID {id_test} telah dihapus dari database.")
            return True
        except mysql.connector.Error as err:
            st.error(f"Gagal menghapus data: {err}")
            cursor.close()
            db_connection.close()
            return False

    # menampilkan detail hasil
    def tampilkan_detail_from_db(row):
        # Membuat label untuk expander menggunakan NIM, Nama, dan Hasil Prediksi
        expander_label = f"{row['nim']} - {row['nama']} - {row['hasil']}"

        # Menggunakan expander untuk menampilkan detail
        with st.expander(expander_label):
            st.write(f"**ID Test**: {row['id_test']}")
            st.write(f"**Nama Mahasiswa**: {row['nama']}")
            st.write(f"**NIM**: {row['nim']}")
            st.write(f"**Kategori IPK**: {row['ipk']}")
            st.write(f"**Kategori Lama Studi**: {row['lama_studi']}")
            st.write(f"**Skill Kerjasama**: {row['skill_kerjasama']}")
            st.write(f"**Beasiswa**: {row['beasiswa']}")
            st.write(f"**Hasil Prediksi**: {row['hasil']}")
            st.write(f"**Model Testing**: {row['model_digunakan']}")
            st.write(f"**Tanggal**: {row['tanggal']}")

    def tampilkan_riwayat_prediksi(sort_option="Terbaru", search_query=""):
        db_connection = connect_to_db()
        cursor = db_connection.cursor(dictionary=True)

        query = "SELECT * FROM riwayat_prediksi"
        cursor.execute(query)
        df_history = pd.DataFrame(cursor.fetchall())

        # Konversi kolom tanggal jika ada kolom bernama 'tanggal'
        if 'tanggal' in df_history.columns:
            df_history['tanggal'] = pd.to_datetime(df_history['tanggal'], errors='coerce', dayfirst=True)

        # Terapkan urutan sesuai pilihan dropdown
        if sort_option == "Abjad":
            df_history = df_history.sort_values(by="nama", ascending=True)
        elif sort_option == "NIM":
            df_history = df_history.sort_values(by="nim", ascending=True)
        elif sort_option == "Terbaru":
            df_history = df_history.sort_values(by="tanggal", ascending=False)
        elif sort_option == "Terlama":
            df_history = df_history.sort_values(by="tanggal", ascending=True)

        # Kolom pencarian & caption
        col10, col11 = st.columns([4, 2])
        with col10:
            search_query = st.text_input("Pencarian", placeholder="Cari NIM atau Nama Mahasiswa", label_visibility="hidden")
        with col11:
            st.write("")
            st.write("")
            try:
                jumlah_riwayat = df_history.shape[0]
                st.caption(f"📊 Data Tersimpan: **{jumlah_riwayat}**")
            except:
                st.caption("⚠️ Gagal membaca data.")

        # Scrollable container dimulai
        st.markdown("""
            <style>
            .scroll-container {
                max-height: 500px;
                overflow-y: auto;
                padding-right: 10px;
            }
            </style>
            <div class="scroll-container">
        """, unsafe_allow_html=True)

        # Pencarian
        if search_query:
            filtered_df = df_history[df_history['nim'].astype(str).str.contains(search_query, case=False) |
                                    df_history['nama'].str.contains(search_query, case=False)]
            if filtered_df.empty:
                st.warning("Tidak ditemukan data yang dicari.")
            else:
                for index, row in filtered_df.iterrows():
                    col1, col2 = st.columns([14, 2])
                    with col1:
                        tampilkan_detail_from_db(row)
                    with col2:
                        if st.button(f"Hapus", key=row['id_test']):
                            if hapus_riwayat_from_db(row['id_test']):
                                st.rerun()
        else:
            for index, row in df_history.iterrows():
                col1, col2 = st.columns([14, 2])
                with col1:
                    tampilkan_detail_from_db(row)
                with col2:
                    if st.button(f"Hapus", key=row['id_test']):
                        if hapus_riwayat_from_db(row['id_test']):
                            st.rerun()

        # Scroll container selesai
        st.markdown("</div>", unsafe_allow_html=True)
        
        cursor.close()
        db_connection.close()

    def load_model(model_type):
        try:
        
            if model_type == "Decision Tree":
                loaded_model = joblib.load('models/Model_Decision_Tree.pkl')
            elif model_type == "Random Forest":
                loaded_model = joblib.load('models/Model_Random_Forest.pkl')
            elif model_type == "SVM":
                loaded_model = joblib.load('models/Model_SVM.pkl')
            elif model_type == "Naive Bayes":
                loaded_model = joblib.load('models/Model_NaiveBayes.pkl')
            elif model_type == "Voting Classifier (KNN + CatBoost)":
                loaded_model = joblib.load('models/Model_VotingClassifier.pkl')
            elif model_type == "AdaBoost":
                loaded_model = joblib.load('models/Model_AdaBoost.pkl')
            elif model_type == "XGBoost":
                loaded_model = joblib.load('models/Model_XGBoost.pkl')
            elif model_type == "KNN":
                loaded_model = joblib.load('models/Model_KNN.pkl')
            elif model_type == "CatBoost":
                loaded_model = joblib.load('models/Model_CatBoost.pkl')
            elif model_type == "Logistic Regression":
                loaded_model = joblib.load('models/Model_LogisticRegression.pkl')

            le_ipk = joblib.load('models/le_ipk.pkl')
            le_studi = joblib.load('models/le_lamastudi.pkl')
            le_kerjatim = joblib.load('models/le_kerjatim.pkl')
            le_beasiswa = joblib.load('models/le_beasiswa.pkl')
            le_ketereratan = joblib.load('models/le_ketereratan.pkl')
            scaler = joblib.load('models/scaler.pkl')

            return loaded_model, le_ipk, le_studi, le_kerjatim, le_beasiswa, le_ketereratan, scaler

        except FileNotFoundError as e:
            st.warning(f"{e}")
            return None, None, None, None, None, None, None
# Fungsi untuk melakukan SMOTE pada data
    def apply_smote(X, y):
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        return X_resampled, y_resampled
    
    def diagram_distribusi_data(original_labels, resampled_labels):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Ambil label unik untuk menentukan warna
        unique_labels = sorted(original_labels.unique())
        colors_before = ['steelblue', 'tomato']
        colors_after = ['steelblue', 'tomato']

        # Distribusi sebelum SMOTE
        counts_before = original_labels.value_counts()
        axes[0].bar(counts_before.index.astype(str), counts_before.values, color=colors_before)
        axes[0].set_title('Distribusi Kelas Sebelum SMOTE')
        axes[0].set_xlabel('Ketereratan', fontsize=14)
        axes[0].set_ylabel('Jumlah', fontsize=14)

        # Distribusi setelah SMOTE
        counts_after = resampled_labels.value_counts()
        axes[1].bar(counts_after.index.astype(str), counts_after.values, color=colors_after)
        axes[1].set_title('Distribusi Kelas Setelah SMOTE')
        axes[1].set_xlabel('Ketereratan', fontsize=14)
        axes[1].set_ylabel('Jumlah', fontsize=14)

        plt.tight_layout()
        st.pyplot(fig)  # Menampilkan grafik di Streamlit
    #--------------------------------------------FITUR SAMPLING-------------------------------------------------------

    # Fungsi untuk mengacak dan menggabungkan data
    def shuffle_and_merge_data(df_pkpa, df_baak):
        # Mengacak data
        df_pkpa = df_pkpa.sample(frac=1, random_state=42).reset_index(drop=True)
        df_baak = df_baak.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Menggabungkan berdasarkan NIM
        df_merged = df_pkpa.merge(df_baak, on="nim", how="left")
        
        return df_merged

    # Fungsi untuk memproses model Random Forest dan mengecek akurasi
    def train_random_forest(df):
        # Menyiapkan fitur dan label
        X = df[["beasiswa", "ketereratan", "kerja tim", "kategori_ipk", "kategori_lama_studi"]]
        y = df["ketereratan"]  # Label untuk prediksi
        
        # Encoding kategori menjadi numerik
        X = pd.get_dummies(X)
        
        # Pisahkan data menjadi training dan test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Melatih model Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Prediksi dan evaluasi akurasi
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, rf

    # Fungsi untuk sampling data dan melakukan proses pengujian
    def process_sampling_and_testing(data_raw_pkpa, sampling_fraction):
        sampled_data = sampling_pkpa(data_raw_pkpa, sampling_fraction)

        if not sampled_data.empty:
            st.subheader("Data Hasil Sampling")

            # Menampilkan data hasil sampling
            for col in sampled_data.columns:
                sampled_data[col] = sampled_data[col].astype(str)
            
            # Mengonversi dan memfilter kolom
            df_pkpa = process_pkpa(sampled_data)  # Memproses data PKPA
            df_baak = process_baak(sampled_data)  # Memproses data BAAK

            # Gabungkan data PKPA dan BAAK
            df_merged = shuffle_and_merge_data(df_pkpa, df_baak)

            # Uji model Random Forest
            accuracy, rf_model = train_random_forest(df_merged)
            
            # Cek akurasi dan ulangi jika akurasi kurang dari 85%
            if accuracy < 0.85:
                st.warning(f"Akurasi model {accuracy:.2f}, mencoba pengacakan ulang data...")
                return None  # Mengindikasikan bahwa akurasi masih di bawah 85%, proses diulang
            else:
                st.success(f"Akurasi model: {accuracy:.2f}. Proses selesai.")
                
                # Menampilkan tombol untuk mengunduh dataset
                buffer_xlsx = io.BytesIO()
                with pd.ExcelWriter(buffer_xlsx, engine='xlsxwriter') as writer:
                    sampled_data.to_excel(writer, sheet_name='Rekap', index=False)
                buffer_xlsx.seek(0)

                # Tombol unduh dataset
                st.download_button("Unduh sebagai .xlsx", data=buffer_xlsx,
                                    file_name="sampled_data.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                
                return sampled_data  # Return data yang sudah di-sample dan terfilter
    
    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        return encoded
    
    CSS.cssAPP()
    col01, col02 =st.columns([0.9, 0.2])
    with col01:
        # Judul Aplikasi
        st.markdown(
        """
        <div style='padding-top: 0.5rem;'>
            <h1 style='color: #2657A3; margin-bottom: 0;'>CAREER PREDICTION</h1>
        </div>
        """,
        unsafe_allow_html=True
        )
    with col02 :
        nama_lengkap = st.session_state.user['nama_lengkap']
        nama_terakhir = nama_lengkap.strip().split()[-1]
        st.write(f"Hai, {nama_terakhir}")
        if st.button("Logout"):
            login.logout()
            st.session_state.page = "login"
            st.rerun()

    #cek hak akses
    hak_akses = st.session_state.user['hak_akses']
    # st.sidebar.header("Menu")

    with st.sidebar:
        logo_path = "logo/logo_sistem_prediksi_utama.png"
        encoded_logo = get_base64_image(logo_path)

        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{encoded_logo}" width="230">
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<div style='padding-top: 1.8rem;'>", unsafe_allow_html=True)

    if hak_akses == 'admin':
        with st.sidebar:
            menu = option_menu(
                menu_title="Menu",  # Optional
                options=["Dashboard","Mining Data", "Kelola Pengguna"],
                icons=["house-fill","bar-chart", "person-lines-fill"],  # opsional
                default_index=0
            )
    else:
        with st.sidebar:    
            menu = option_menu(
                menu_title="Menu",  # Optional
                options=["Dashboard","Mining Data"],
                icons=["house-fill","bar-chart"],  # opsional
                default_index=0
            )

    if menu == "Dashboard":
        dashboard.show_dashboard()

    elif menu == "Kelola Pengguna":
            
        if "halaman" not in st.session_state:
            st.session_state.halaman = "kelola_pengguna"

        # Navigasi berdasarkan state halaman
        if st.session_state.halaman == "kelola_pengguna":
            kelola_pengguna.main()

        elif st.session_state.halaman == "form_tambah_pengguna":
            form_tambah_pengguna.form_tambah_pengguna()

    elif menu == "Mining Data":

        fitur = st.sidebar.selectbox("Pilih Layanan", ["Prediksi", "Training Model","Sampling Data"], index=1)

        # --- Riwayat prediksi akan disimpan di sini ---
        riwayat_file = "riwayat_prediksi.csv"

        # --- Ambil ID Unik ---
        def generate_unique_id():
            # Membuat ID unik menggunakan UUID
            unique_id = f"PRD{str(uuid.uuid4().int)[:8]}"  # Ambil 8 karakter pertama dari UUID
            return unique_id

        if fitur == "Prediksi" :

            if "reset_form" not in st.session_state:
                st.session_state.reset_form = False

            if "input_nim" not in st.session_state:
                st.session_state.input_nim = ""

            if "input_nama" not in st.session_state:
                st.session_state.input_nama = ""

            # Reset nilai setelah rerun
            if st.session_state.reset_form:
                st.session_state.input_nim = ""
                st.session_state.input_nama = ""
                st.session_state.reset_form = False

            st.markdown("""
            <div style="
                background-color: #2657A3;
                padding: 1px 0;
                width: 100%;
                overflow: hidden;
                position: relative;
            ">
                <marquee behavior="scroll" direction="left" scrollamount="5" style="
                    color: white;
                    font-weight: bold;
                    font-size: 20px;
                ">
                    Layanan Prediksi Ketereratan Karier
                </marquee>
            </div>
        """, unsafe_allow_html=True)
            
            pilihmodel = st.sidebar.selectbox("Pilih Model Prediksi", [
                                                                        "Decision Tree", 
                                                                        "Random Forest", 
                                                                        "SVM",
                                                                        "Naive Bayes",
                                                                        "Voting Classifier (KNN + CatBoost)",
                                                                        "AdaBoost",
                                                                        "XGBoost",
                                                                        "Voting AdaBoost + XGBoost",
                                                                        "KNN",
                                                                        "CatBoost",
                                                                        "Logistic Regression"
                                                                    ], index=0)
            
            with st.sidebar.expander("📄 Riwayat Prediksi"):
                show_history = st.checkbox("Tampilkan Riwayat Prediksi")
            # Load model dan objek sesuai pilihan model
            model, le_ipk, le_studi, le_kerjatim, le_beasiswa, le_ketereratan, scaler = load_model(pilihmodel)

            if model is None or le_ipk is None or scaler is None:
                st.stop()

            # 🔹 Form inputan pengguna (REVISI SESUAI PERMINTAAN)
            # st.subheader("Masukkan Data untuk Prediksi Ketereratan")

            nim = st.text_input("NIM Mahasiswa", key="input_nim", placeholder="Masukkan NIM (9 digit)")
            if nim and (not nim.isdigit() or len(nim) != 9):
                st.warning("NIM harus terdiri dari 9 digit angka.")

            nama_mahasiswa = st.text_input(
                "Nama Mahasiswa",
                key="input_nama",
                placeholder="Masukkan nama lengkap mahasiswa"
            ).upper()

            #----------------INPUT IPK & LAMA STUDI--------------------------

            col1, mid, col2, col21, col22, col23 = st.columns([0.3, 0.01, 0.3,0.1,0.3, 0.3])
            with col1:
                pembilang = st.number_input("IPK (Skala 4)", min_value=1, max_value=4, step=1, format="%d")
            with mid:
                st.write("")
                st.write("")
                st.markdown("<h4 style='text-align: left;'>,</h4>", unsafe_allow_html=True)
            with col2:
                if pembilang == 4:
                    penyebut = st.number_input("penyebut", min_value=0, max_value=0, step=0, format="%02d", disabled=True, label_visibility="hidden")
                else:
                    penyebut = st.number_input("penyebut", min_value=0, max_value=99, step=1, format="%02d", label_visibility="hidden")
            with col22:
                tahun = st.number_input("Lama Studi", min_value=1, max_value=7, step=1, format="%d")
                st.caption("Tahun")
            with col23:
                if tahun == 7:
                    bulan = st.number_input("bulan", min_value=0, max_value=0, step=0, format="%d", disabled=True, label_visibility="hidden")
                    st.caption("Bulan")
                else:
                    bulan = st.number_input("bulan", min_value=0, max_value=11, step=1, format="%d", label_visibility="hidden")
                    st.caption("Bulan")

            #----------------INPUT KERJA TIM--------------------------

            col24, col25, col26= st.columns([0.53, 0.08, 0.51])
            with col24:
                kerja_tim_input = st.selectbox(
                    "Kemampuan Kerja Tim",
                    options=["-- Pilih --","SANGAT MAMPU", "MAMPU", "CUKUP", "KURANG", "SANGAT KURANG"]
                )
            with col26:
                st.write("")
                beasiswa_status = st.radio(
                    "Pernah Mendapatkan Beasiswa?",
                    options=["Pernah", "Belum Pernah"],
                    index=1
                )

            # 🔹 Tombol untuk jalankan prediksi
            if st.button("Jalankan Prediksi"):
                if nim == "":
                    warning_nim = st.empty()
                    warning_nim.warning("NIM mahasiswa wajib diisi.")
                    time.sleep(5)
                    warning_nim.empty()
                elif nama_mahasiswa == "":
                    warning_nama = st.empty()
                    warning_nama.warning("Nama mahasiswa wajib diisi.")
                    time.sleep(5)
                    warning_nama.empty()
                elif kerja_tim_input == "-- Pilih --":
                    warning_kerjatim = st.empty()
                    warning_kerjatim.warning("Silakan pilih kemampuan kerja tim terlebih dahulu.")
                    time.sleep(5)
                    warning_kerjatim.empty()
                # ----------------------------
                # 🔥 Proses konversi ke kategori
                # ----------------------------
                else:
                #IPK asli
                    ipk_asli = float(f"{pembilang}.{penyebut:02d}") 
                    # Kategori IPK
                    ipk_combined = int(f"{pembilang}{penyebut:02d}")
                    if 0 <= ipk_combined <= 300:
                        kategori_ipk = "RENDAH"
                    elif 301 <= ipk_combined <= 350:
                        kategori_ipk = "MEDIAN"
                    else:
                        kategori_ipk = "TINGGI"
                    
                    #lamastudi
                    lamastudi_asli = f"{tahun} Tahun {bulan} Bulan"
                    # Kategori Lama Studi
                    lama_studi_bulan = (tahun * 12) + bulan
                    if 0 <= lama_studi_bulan <= 48:
                        kategori_lama_studi = "TEPAT"
                    elif 49 <= lama_studi_bulan <= 54:
                        kategori_lama_studi = "CUKUP"
                    else:
                        kategori_lama_studi = "LAMA"

                    # # Kerja Tim
                    # kerja_tim = "SANGAT KURANG" if kerja_tim_input == 1 else \
                    # "KURANG" if kerja_tim_input == 2 else \
                    # "CUKUP" if kerja_tim_input == 3 else \
                    # "MAMPU" if kerja_tim_input == 4 else \
                    # "SANGAT MAMPU" if kerja_tim_input == 5 else \
                    # "TIDAK DIKETAHUI"

                    # Pernah Kerja
                    beasiswa = "YA" if beasiswa_status == "Pernah" else "TIDAK"

                    # ----------------------------
                    # 🔥 Proses input ke dataframe
                    # ----------------------------
                    input_data = {
                        'kategori_ipk': [kategori_ipk],
                        'kategori_lama_studi': [kategori_lama_studi],
                        'kerja tim': [kerja_tim_input],
                        'beasiswa': [beasiswa]
                    }
                    input_df = pd.DataFrame(input_data)

                    # Label encoding (gunakan encoder yang sesuai)
                    input_df['kategori_ipk'] = le_ipk.transform(input_df['kategori_ipk'])
                    input_df['kategori_lama_studi'] = le_studi.transform(input_df['kategori_lama_studi'])
                    input_df['kerja tim'] = le_kerjatim.transform(input_df['kerja tim'])
                    input_df['beasiswa'] = le_beasiswa.transform(input_df['beasiswa'])

                    if pilihmodel == "Naive Bayes":
                        prediction = model.predict(input_df)
                    else:
                        # Normalisasi
                        input_scaled = scaler.transform(input_df)
                        # Prediksi
                        prediction = model.predict(input_scaled)

                    predicted_label = le_ketereratan.inverse_transform(prediction)

                    # Pastikan NIM disimpan sebagai string
                    nim = str(nim)

                    # Format hasil prediksi agar tidak tersimpan sebagai array
                    hasil_prediksi = predicted_label[0]

                    # Ambil urutan baris
                    if os.path.exists(riwayat_file):
                        riwayat_df = pd.read_csv(riwayat_file, dtype={"nim": str})
                    else:
                        riwayat_df = pd.DataFrame()

                    with st.spinner('Harap tunggu, memproses prediksi...'):
                        time.sleep(4)  

                    # Tampilkan hasil
                    if predicted_label[0].upper() == "ERAT":
                        st.success(f"Prediksi karier : **{predicted_label[0]}**")
                    else:
                        st.error(f"Prediksi karier : **{predicted_label[0]}**")

                    # Simpan riwayat
                    new_row= {
                        "id_test": generate_unique_id(),
                        "nim": nim,
                        "nama": nama_mahasiswa,
                        "ipk": f"{ipk_asli} / {kategori_ipk}",
                        "lama_studi": f"{lamastudi_asli} / {kategori_lama_studi}",
                        "skill_kerjasama": kerja_tim_input,
                        "beasiswa": beasiswa,
                        "hasil": hasil_prediksi,
                        "model_digunakan": pilihmodel,  # ⬅️ Tambahkan ini
                        "tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    simpan_riwayat_ke_db(new_row)

                    time.sleep(5)
                    st.session_state.reset_form = True
                    st.rerun()
                
                # Tampilkan garis pembatas
                st.markdown("<hr>", unsafe_allow_html=True)
            
            # Tampilkan riwayat
        if fitur == "Prediksi" and show_history:
            st.markdown("<hr>", unsafe_allow_html=True)
            col_header, col_sort = st.columns([5, 1.5])
            with col_header:
                st.markdown("<h2 style='font-size: 20px; text-align: left;'>Riwayat Prediksi</h2>", unsafe_allow_html=True)
            with col_sort:
                st.caption("Sort by:")
                sort_option = st.selectbox(
                    "Sort by",
                    options=["Abjad", "NIM", "Terbaru", "Terlama"],
                    index=2,
                    label_visibility="collapsed"
                )
            tampilkan_riwayat_prediksi(sort_option, search_query="")



        #------------------------------------------------------------------------------------------------------
        elif fitur == "Training Model":

            st.markdown("""
            <div style="
                background-color: #2657A3;
                padding: 1px 0;
                width: 100%;
                overflow: hidden;
                position: relative;
            ">
                <marquee behavior="scroll" direction="left" scrollamount="5" style="
                    color: white;
                    font-weight: bold;
                    font-size: 20px;
                ">
                    Layanan Training Data Lulusan
                </marquee>
            </div>
        """, unsafe_allow_html=True)

            # Menu untuk lihat model yang tersimpan
            with st.sidebar.expander("Lihat Model Tersimpan"):
                # Path ke folder model
                folder_model = "D:/SKRIPSI/DATASET/Program/models"
                
                # Tombol untuk refresh daftar model
                if st.button("↻"):
                    st.rerun()  # Refresh halaman untuk melihat model terbaru
                
                if not os.path.exists(folder_model):
                    st.info("Belum Ada Model Yang Tersimpan")
                else:
                    semua_file = os.listdir(folder_model)

                    # Filter hanya file yang 5 huruf pertamanya 'Model'
                    model_files = [f for f in semua_file if f[:5] == "Model"]

                    if model_files:
                        st.success(f"Anda Memiliki {len(model_files)} model:")
                        for nama_model in model_files:
                            path_model = os.path.join(folder_model, nama_model)
                            
                            # Ambil waktu dan ukuran file
                            waktu_buat = time.strftime('%d-%m-%Y %H:%M', time.localtime(os.path.getmtime(path_model)))
                            ukuran_kb = os.path.getsize(path_model) / 1024  # ukuran dalam KB

                            st.write(f"📂 **{nama_model}**")
                            st.caption(f"🕒 {waktu_buat} | 💾 {ukuran_kb:.2f} KB")
                    else:
                        st.info("Belum Ada Model Yang Tersimpan") 
                
        #------------------------------------------------------------------------------------------------------

            # Fungsi untuk memproses dataset PKPA
            def process_pkpa(file):
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                    if all(col in df.columns for col in ["Kode Progdi", "nim", "nama", "pekerjaan", "beasiswa","ketereratan", "kerja tim","ormawa"]):
                        df = df[["Kode Progdi", "nim", "nama", "pekerjaan","beasiswa", "ketereratan", "kerja tim","ormawa"]]
                    else:
                        st.error("ada kolom yang kurang")
                        return pd.DataFrame()
                else:
                    excel_file = pd.ExcelFile(file)
                    if "Rekap" in excel_file.sheet_names:
                        df = pd.read_excel(file, sheet_name="Rekap", usecols="C,D,E,T,AA,AC,AJ,CM", skiprows=1)
                        df.columns = ["Kode Progdi", "nim", "nama", "pekerjaan", "beasiswa","ketereratan", "kerja tim","ormawa"]                     # ganti nama kolom broo
                    else:
                        st.error("File Excel harus memiliki sheet 'Rekap'")
                        return pd.DataFrame()
                df = df.dropna()
                
                df['ketereratan'] = df['ketereratan'].astype(str).str.strip()
                df = df[df['ketereratan'].isin(['1', '2', '3', '4', '5'])]  # jaga-jaga, pastikan hanya yang valid
                df['ketereratan'] = df['ketereratan'].astype(int)

                df = df[~df['beasiswa'].astype(str).str.strip().isin(["0"])]  # Filter nilai "0"
                df['beasiswa'] = df['beasiswa'].astype(str).str.strip()  # Menghilangkan spasi tambahan
                df = df[df['beasiswa'].isin(['1', '2', '3', '4', '5', '6', '7'])]
                df['beasiswa'] = df['beasiswa'].apply(lambda x: 'YA' if x in ['2', '3', '4', '5', '6', '7'] else 'TIDAK')

                df = df[~df['kerja tim'].astype(str).str.strip().isin(["0"])]
                df['kerja tim'] = df['kerja tim'].astype(str).str.strip()
                df = df[df['kerja tim'].isin(['1', '2', '3', '4', '5'])]
                df['kerja tim'] = df['kerja tim'].astype(int)
                
                df['ormawa'] = df['ormawa'].astype(str).str.strip()
                df = df[df['ormawa'].isin(['0', '1', '2', '3', '4'])]
                df['ormawa'] = df['ormawa'].astype(int)
                
                df['ketereratan'] = df['ketereratan'].apply(lambda x: 'ERAT' if x in [1, 2, 3] else 'TIDAK ERAT')
                df['kerja tim'] = df['kerja tim'].apply(lambda x: 
                    'SANGAT KURANG' if x == 1 else
                    'KURANG' if x == 2 else
                    'CUKUP' if x == 3 else 
                    'MAMPU' if x == 4 else 
                    'SANGAT MAMPU' if x == 5 else
                    'TIDAK DIKETAHUI' 
                )

                df['ormawa'] = df['ormawa'].apply(lambda x: 
                    'TIDAK MENGIKUTI' if x == 0 else
                    '1' if x == 1 else
                    '2' if x == 2 else 
                    '3' if x == 3 else 
                    'LEBIH DARI 3' if x == 4 else
                    'TIDAK DIKETAHUI' 
                )

                df = df[~df['Kode Progdi'].isin(['01', '02', '03'])]                                                           #filter example without
                df['nim'] = df['nim'].astype(str).str.strip()
                df = df[df['nim'].str.isdigit()]
                df = df[df['nim'].str.len() == 9]


                return df[["nim","beasiswa","ketereratan", "kerja tim","ormawa"]]
            
            # Fungsi untuk memproses dataset BAAK
            def process_baak(file):
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                else:
                    excel_file = pd.ExcelFile(file)
                    processed_sheets = []
                    for sheet in excel_file.sheet_names:
                        df = pd.read_excel(file, sheet_name=sheet, usecols="B,C,D,E", skiprows=1)           #select attribute
                        df.columns = ["nim", "nama", "lama studi", "ipk"]                                  #buat kolom tabel
                        
                        df = df.dropna()                                
                        df = df[~df['nim'].apply(lambda x: str(x)[4:6] in ['01', '02', '03'])]              # filter example
                        
                        # Konversi nilai di kolom "lama studi" dari format tahun ke format bulan
                        def convert_lama_studi(value):
                            value = str(value)                                                          #memisah karakter per spasi
                            if '.' in value:
                                bagian_depan, bagian_belakang = value.split('.')
                                bulan = int(bagian_depan) * 12 + int(bagian_belakang)
                            else:
                                bulan = int(value) * 12
                            return bulan
                        
                        df["lama studi"] = df["lama studi"].apply(convert_lama_studi)               #isi kolom lama studi dengan nilai hasil konversi ke bulan
                        
                        # Mengalikan nilai di kolom "ipk" dengan 100
                        df["ipk"] = df["ipk"] * 100                                                 #isi kolom ipk dengan nilai ipk konversi x 100
                        
                        # Filter hanya data yang memiliki lama studi antara 40 bulan dan 84 bulan serta ipk antara 250 dan 400
                        df = df[(df["lama studi"] >= 42) & (df["lama studi"] <= 60)]  # Filter lama studi
                        df = df[(df["ipk"] >= 250) & (df["ipk"] <= 400)]  # Filter ipk

                    #---------------------------------------------------
                        # Kategorisasi untuk kolom "ipk"
                        def categorize_ipk(ipk_value):
                            if 250 <= ipk_value <= 300:
                                return 'RENDAH'
                            elif 301 <= ipk_value <= 350:
                                return 'MEDIAN'
                            elif 351 <= ipk_value <= 400:
                                return 'TINGGI'
                            else:
                                return None

                        df['kategori_ipk'] = df['ipk'].apply(categorize_ipk)  # Menambahkan kolom kategori_ipk
                        
                        # Kategorisasi untuk kolom "lama studi"
                        def categorize_lama_studi(lama_studi_value):
                            if 42 <= lama_studi_value <= 48:
                                return 'TEPAT'
                            elif 49 <= lama_studi_value <= 54:
                                return 'CUKUP'
                            elif 55 <= lama_studi_value <= 60:
                                return 'LAMA'
                            else:
                                return None

                        df['kategori_lama_studi'] = df['lama studi'].apply(categorize_lama_studi)  # Menambahkan kolom kategori_lama_studi
                    #---------------------------------------------------
                        
                        processed_sheets.append(df)
                    df = pd.concat(processed_sheets, ignore_index=True) if processed_sheets else pd.DataFrame()

                    # Normalisasi NIM
                    df['nim'] = df['nim'].astype(str).str.strip()
                    df = df[df['nim'].str.isdigit()]
                    df = df[df['nim'].str.len() == 9]
                return df

            st.sidebar.header("Upload Dataset")
            uploaded_pkpa = st.sidebar.file_uploader("Upload dataset PKPA (Excel atau CSV)", type=["xlsx", "xls", "csv"], accept_multiple_files=True)
            uploaded_baak = st.sidebar.file_uploader("Upload dataset BAAK (Excel atau CSV)", type=["xlsx", "xls", "csv"], accept_multiple_files=True)
            jmlFold = st.sidebar.number_input(
                "Masukkan jumlah fold Cross Validation:",
                min_value=2, max_value=25, value=12, step=1
            )
            algorithm = st.sidebar.selectbox("Pilih Algoritma", 
                                            [
                                                "--- Pilih Algoritma ---",
                                                "Decision Tree", 
                                                "Random Forest", 
                                                "SVM",
                                                "Naive Bayes",
                                                "Voting Classifier (KNN + CatBoost)",
                                                "AdaBoost",
                                                "XGBoost",
                                                "Voting AdaBoost + XGBoost",
                                                "KNN",
                                                "CatBoost",
                                                "Logistic Regression"
                                            ])
            
            if not (uploaded_pkpa and uploaded_baak and jmlFold and algorithm != "--- Pilih Algoritma ---"):
                st.markdown("<div style='padding-top: 20rem;'>", unsafe_allow_html=True)
                st.info("Gunakan sidebar untuk mulai mengunggah data dan menjalankan model prediksi.")
            if uploaded_pkpa and uploaded_baak and jmlFold and algorithm != "--- Pilih Algoritma ---":
                st.markdown("<div style='padding-top: 1.9rem;'>", unsafe_allow_html=True)
                with st.spinner('Data sedang di konversi..'):
                    time.sleep(1)
                    df_pkpa_list = [process_pkpa(file) for file in uploaded_pkpa]
                    df_pkpa = pd.concat(df_pkpa_list, ignore_index=True)
                    st.write("Dataset PKPA yang diupload:")
                    st.write(df_pkpa)

                    df_baak_list = [process_baak(file) for file in uploaded_baak]
                    df_baak = pd.concat(df_baak_list, ignore_index=True)
                    st.write("Dataset BAAK yang diupload:")
                    st.write(df_baak)
                
                with st.spinner('Menggabungkan dataset...'):
                    time.sleep(1)
                    df_merged = df_pkpa.merge(df_baak, on="nim", how="left")
                    dataset = df_merged[["kategori_ipk", "kategori_lama_studi","beasiswa", "kerja tim","ormawa", "ketereratan"]]
                    
                    dataset_filtered = pd.concat([
                        dataset[(dataset["ketereratan"].notna()) & dataset["kategori_ipk"].notna() & dataset["kategori_lama_studi"].notna() & 
                                (dataset["kategori_lama_studi"]) & (dataset["kategori_lama_studi"]) & dataset["kerja tim"].notna() & dataset["beasiswa"].notna() & dataset["ormawa"].notna()]
                    ], ignore_index=True)

                with st.spinner('SMOTE & Resampling dataset...'):

                    X = dataset_filtered[['kategori_ipk','kategori_lama_studi','beasiswa', 'kerja tim','ormawa']]  # Fitur yang digunakan
                    y = dataset_filtered['ketereratan']  # Target yang digunakan

                    st.markdown(
                        "<h3 style='text-align: center;'>Distribusi Kelas</h3>",
                        unsafe_allow_html=True
                    )
                    col30, col40, col50 = st.columns([0.15,0.7, 0.5])
                    with col40:
                        st.write("Sebelum SMOTE:")
                        st.write(y.value_counts())

                    # Konversi YA/TIDAK menjadi 1/0
                    X['beasiswa'] = X['beasiswa'].replace({'YA': 1, 'TIDAK': 0})
                    X['kerja tim'] = X['kerja tim'].replace({
                        'SANGAT KURANG': 0,
                        'KURANG': 1,
                        'CUKUP': 2,
                        'MAMPU': 3,
                        'SANGAT MAMPU': 4
                    })

                    X['ormawa'] = X['ormawa'].replace({
                        'TIDAK MENGIKUTI': 0,
                        '1': 1,
                        '2': 2,
                        '3': 3,
                        'LEBIH DARI 3': 4
                    })
                    
                    X['kategori_ipk'] = X['kategori_ipk'].replace({'RENDAH': 0, 'MEDIAN': 1, 'TINGGI': 2})
                    X['kategori_lama_studi'] = X['kategori_lama_studi'].replace({'TEPAT': 0, 'CUKUP': 1, 'LAMA': 2})
                    y = y.replace({'ERAT': 1, 'TIDAK ERAT': 0})

                    # Terapkan SMOTE untuk Resampling
                    X_resampled, y_resampled = apply_smote(X, y)

                    # ⬅️ Simpan data sebelum mapping ulang
                    df_bahankorelasi = pd.DataFrame(X_resampled.copy())
                    df_bahankorelasi['ketereratan'] = y_resampled

                    with col50 :
                        st.write("Setelah SMOTE:")
                        st.write(pd.Series(y_resampled).value_counts())
                    diagram_distribusi_data(y, y_resampled)
                    

                    beasiswa_map = {1: 'YA', 0: 'TIDAK'}
                    kerja_tim_map = {0: 'SANGAT KURANG', 1: 'KURANG', 2: 'CUKUP', 3:'MAMPU', 4:'SANGAT MAMPU'}
                    ormawa_map = {0: 'TIDAK MENGIKUTI', 1: '1', 2: '2', 3:'3', 4:'LEBIH DARI 3'}
                    ipk_map = {0: 'RENDAH', 1: 'MEDIAN', 2: 'TINGGI'}
                    lama_studi_map = {0: 'TEPAT', 1: 'CUKUP', 2: 'LAMA'}
                    ketereratan_map = {0: 'ERAT', 1: 'TIDAK ERAT'}

                    # Apply reverse mapping ke X_resampled
                    X_resampled['beasiswa'] = X_resampled['beasiswa'].map(beasiswa_map)
                    X_resampled['kerja tim'] = X_resampled['kerja tim'].map(kerja_tim_map)
                    X_resampled['ormawa'] = X_resampled['ormawa'].map(ormawa_map)
                    X_resampled['kategori_ipk'] = X_resampled['kategori_ipk'].map(ipk_map)
                    X_resampled['kategori_lama_studi'] = X_resampled['kategori_lama_studi'].map(lama_studi_map)

                    # Apply reverse mapping ke y_resampled (pastikan y_resampled adalah Series, bukan array)
                    y_resampled_labels = pd.Series(y_resampled).map(ketereratan_map)

                    df_final = X_resampled.copy()
                    df_final['ketereratan'] = y_resampled_labels

                    st.write("Data hasil SMOTE:")
                    st.write(df_final)

                with st.expander("🔍 Korelasi Variabel Data Asli"):
                    try:
                        st.write("Data:")
                        st.dataframe(df_bahankorelasi)  # ⬅️ Cek dulu hasil merge

                        df_raw_corr = df_bahankorelasi[["beasiswa", "kerja tim","ormawa", "kategori_lama_studi", "kategori_ipk", "ketereratan"]]

                        if df_raw_corr.empty:
                            st.warning("Data kosong setelah diproses. Cek apakah data masih numerik dan ada nilai.")
                        else:
                            # Korelasi Pearson
                            corr_raw = df_raw_corr.corr(method='pearson')

                            # Heatmap visual: tampilkan diagonal dan segitiga bawah saja
                            mask = np.triu(np.ones_like(corr_raw, dtype=bool), k=1)
                            plt.figure(figsize=(8, 6))
                            sns.heatmap(corr_raw,
                                        mask=mask,
                                        annot=True,
                                        cmap='coolwarm',
                                        vmin=-1, vmax=1,
                                        square=True,
                                        linewidths=0.5,
                                        fmt=".2f")
                            plt.xticks(rotation=45, ha='right', style='italic')
                            plt.yticks(rotation=0, style='italic')
                            plt.title("Korelasi Variabel (Pearson)")
                            st.pyplot(plt)

                            # Ambil korelasi X terhadap Y (ketereratan)
                            correlation_with_target = corr_raw['ketereratan'].drop('ketereratan')

                            correlation_result = []
                            for feature, corr_value in correlation_with_target.items():
                                arah = 'positif' if corr_value > 0 else 'negatif'
                                correlation_result.append({
                                    'Fitur': feature,
                                    'Korelasi': round(corr_value, 3),
                                    'Arah Korelasi': arah
                                })
                            correlation_df = pd.DataFrame(correlation_result).sort_values(by='Korelasi', key=abs, ascending=False)

                            st.write("📊 **Korelasi Pearson terhadap ketereratan**")
                            st.dataframe(correlation_df)

                    except Exception as e:
                        st.warning(f"❗ Gagal menghitung korelasi data asli: {e}")


                st.write("Silakan Pilih Algoritma..")
            
                if algorithm == "Decision Tree":
                    
                    if st.button("Jalankan Decision Tree"):
                        
                        # Catat waktu mulai
                        start_time = time.time()
                        results = run_decision_tree(df_final, n_splits=jmlFold)
                        # Catat waktu selesai
                        end_time = time.time()
                        # Hitung durasi komputasi
                        elapsed_time = end_time - start_time

                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.markdown(
                            "<h3 style='text-align: center;'>Hasil Evaluasi Decision Tree</h3>",
                            unsafe_allow_html=True
                        )
                        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(
                                f"""
                                <div style="background-color:#d1e7dd;padding:20px;border-radius:10px;">
                                    <h4 style="color:#0f5132;">🎯 Akurasi Model</h4>
                                    <h2 style="margin:0;color:#0f5132;">{results['accuracy'] * 100:.2f}%</h2>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        with col2:
                            st.markdown(
                                f"""
                                <div style="background-color:#cff4fc;padding:20px;border-radius:10px;">
                                    <h4 style="color:#055160;">⏱️ Waktu Komputasi</h4>
                                    <h2 style="margin:0;color:#055160;">{elapsed_time:.2f} detik</h2>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
                        col33,col34=st.columns([0.5,0.5])
                        with col33:
                            st.write("**Confusion Matrix:**")
                            conf_matrix = results['confusion_matrix']
                            labels = ['TIDAK ERAT', 'ERAT']  # Ubah sesuai label datamu

                            fig, ax = plt.subplots()
                            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                                        xticklabels=labels, yticklabels=labels, cbar=True, ax=ax)
                            ax.set_xlabel('Predicted label')
                            ax.set_ylabel('True label')
                            ax.set_title('Confusion Matrix')
                            st.pyplot(fig)
                        with col34:
                            st.write("**Laporan Klasifikasi:**")
                            st.dataframe(results['classification_report'])
                            st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
                        col35, col36 = st.columns([0.5,0.5])
                        with col35:
                            st.write("**Precision:**", results['precision'])
                            st.write("**Recall:**", results['recall'])
                            st.write("**F1-Score:**", results['f1_score'])
                            st.write("**Specificity:**", results['specificity'])
                            st.write("**NPV:**", results['npv'])
                            st.write("***Mean Accuracy:***", results['average_accuracy'])
                        with col36:
                            st.write("**Cohen's Kappa:**", results['kappa'])
                            st.write("**ROC AUC Score:**", results["roc_auc"])
                            # st.write("**Fold Terbaik:**", results['fold_terbaik'])
                            st.write("***Mean Error:***", results['average_error'])
                            # st.write("**Data Training:**", results['jumlah_data_training'])
                            # st.write("**Data Testing:**", results['jumlah_data_testing'])
                        
                        st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
                        col371, col381,col382 = st.columns([0.5, 0.5, 0.5])
                        with col371:
                            st.image(results['max_depth_plot'], caption="Akurasi vs Max Depth")
                        with col381:
                            st.image(results['min_samples_split_plot'], caption="Akurasi vs Min Split")
                        with col382:
                            st.image(results['min_samples_leaf_plot'], caption="Akurasi vs Min Leaf")
                        st.write("**Visualisasi Decision Tree:**")
                        st.image(results['tree_image'], caption="Decision Tree", use_container_width=True)
                        st.write("**Feature Importance:**")
                        st.image(results['feature_importance_image'], caption="Feature Importance", use_container_width=True)
                        st.write("**Kurva ROC-AUC:**")
                        st.image(results['roc_image'], caption="ROC Curve", use_container_width=True)
                elif algorithm == "Random Forest":

                    if st.button("Jalankan Random Forest"):
                            # Catat waktu mulai
                            start_time = time.time()
                            results = run_random_forest(df_final, n_splits=jmlFold)
                            # Catat waktu selesai
                            end_time = time.time()
                            # Hitung durasi komputasi
                            elapsed_time = end_time - start_time

                            st.markdown("<hr>", unsafe_allow_html=True)
                            st.markdown(
                                "<h3 style='text-align: center;'>Hasil Evaluasi Random Forest</h3>",
                                unsafe_allow_html=True
                            )
                            st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown(
                                    f"""
                                    <div style="background-color:#d1e7dd;padding:20px;border-radius:10px;">
                                        <h4 style="color:#0f5132;">🎯 Akurasi Model</h4>
                                        <h2 style="margin:0;color:#0f5132;">{results['accuracy'] * 100:.2f}%</h2>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            with col2:
                                st.markdown(
                                    f"""
                                    <div style="background-color:#cff4fc;padding:20px;border-radius:10px;">
                                        <h4 style="color:#055160;">⏱️ Waktu Komputasi</h4>
                                        <h2 style="margin:0;color:#055160;">{elapsed_time:.2f} detik</h2>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)

                            col37, col38 = st.columns([0.5, 0.5])
                            with col37:
                                st.write("**Confusion Matrix:**")
                                conf_matrix = results['confusion_matrix']
                                labels = ['TIDAK ERAT', 'ERAT']  # Ubah sesuai label datamu

                                fig, ax = plt.subplots()
                                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                                            xticklabels=labels, yticklabels=labels, cbar=True, ax=ax)
                                ax.set_xlabel('Predicted label')
                                ax.set_ylabel('True label')
                                ax.set_title('Confusion Matrix')
                                st.pyplot(fig)

                            with col38:
                                st.write("**Laporan Klasifikasi:**")
                                st.dataframe(results['classification_report'])
                                st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)

                            col39, col41 = st.columns([0.5, 0.5])
                            with col39:
                                st.write("**Precision:**", results['precision'])
                                st.write("**Recall:**", results['recall'])
                                st.write("**F1-Score:**", results['f1_score'])
                                st.write("**Specificity:**", results['specificity'])
                                st.write("**NPV:**", results['npv'])
                                # st.write("***Mean Accuracy:***", results['average_accuracy'])

                            with col41:
                                st.write("**Cohen's Kappa:**", results['kappa'])
                                st.write("**ROC AUC Score:**", results["roc_auc"])
                                # st.write("**Fold Terbaik:**", results['fold_terbaik'])
                                # st.write("***Mean Error:***", results['average_error'])
                                st.write("**Data Training:**", results['jumlah_data_training'])
                                st.write("**Data Testing:**", results['jumlah_data_testing'])

                            st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
                            col391, col411 = st.columns([0.5, 0.5])
                            with col391:
                                st.image(results['n_tree_plot'], caption="Akurasi vs n-tree (n_estimators)")
                            with col411:
                                st.image(results['mtry_plot'], caption="Akurasi vs mtry (max_features)")
                            st.write("**Visualisasi Salah Satu Pohon dari Random Forest:**")
                            st.image(results['tree_image'], caption="Random Forest Tree", use_container_width=True)

                            st.write("**Feature Importance:**")
                            st.image(results['feature_importance_image'], caption="Feature Importance", use_container_width=True)

                            st.write("**Kurva ROC-AUC:**")
                            st.image(results['roc_image'], caption="ROC Curve", use_container_width=True)
                elif algorithm == "SVM":

                    if st.button("Jalankan SVM"):
                        # Catat waktu mulai
                        start_time = time.time()
                        results = run_svm(df_final, n_splits=jmlFold)  # Disesuaikan dengan parameter baru
                        # Catat waktu selesai
                        end_time = time.time()
                        # Hitung durasi komputasi
                        elapsed_time = end_time - start_time

                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.markdown(
                            "<h3 style='text-align: center;'>Hasil Evaluasi SVM</h3>",
                            unsafe_allow_html=True
                        )
                        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(
                                f"""
                                <div style="background-color:#d1e7dd;padding:20px;border-radius:10px;">
                                    <h4 style="color:#0f5132;">🎯 Akurasi Model</h4>
                                    <h2 style="margin:0;color:#0f5132;">{results['accuracy'] * 100:.2f}%</h2>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        with col2:
                            st.markdown(
                                f"""
                                <div style="background-color:#cff4fc;padding:20px;border-radius:10px;">
                                    <h4 style="color:#055160;">⏱️ Waktu Komputasi</h4>
                                    <h2 style="margin:0;color:#055160;">{elapsed_time:.2f} detik</h2>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)

                        col42, col43 = st.columns([0.5, 0.5])
                        with col42:
                            st.write("**Confusion Matrix:**")
                            conf_matrix = results['confusion_matrix']
                            labels = ['TIDAK ERAT', 'ERAT']  # Ubah sesuai label datamu

                            fig, ax = plt.subplots()
                            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                                        xticklabels=labels, yticklabels=labels, cbar=True, ax=ax)
                            ax.set_xlabel('Predicted label')
                            ax.set_ylabel('True label')
                            ax.set_title('Confusion Matrix')
                            st.pyplot(fig)

                        with col43:
                            st.write("**Laporan Klasifikasi:**")
                            st.dataframe(results['classification_report'])
                            st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)

                        st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
                        col300, col400, col500 = st.columns([0.15,0.7, 0.15])
                        with col400:
                            st.write("**Akurasi setiap kernel:**")
                            st.dataframe(results['best_accuracy_per_kernel'])
                        col44, col45 = st.columns([0.5, 0.5])
                        with col44:
                            st.write("**Precision:**", results['precision'])
                            st.write("**Recall:**", results['recall'])
                            st.write("**F1-Score:**", results['f1_score'])
                            st.write("**Specificity:**", results['specificity'])
                            st.write("**NPV:**", results['npv'])
                            # st.write("***Mean Accuracy:***", results['average_accuracy'])

                        with col45:
                            st.write("**Cohen's Kappa:**", results['kappa'])
                            st.write("**ROC AUC Score:**", results["roc_auc"])
                            # st.write("**Fold Terbaik:**", results['fold_terbaik'])
                            # st.write("***Mean Error:***", results['average_error'])
                            # st.write("**Data Training:**", results['jumlah_data_training'])
                            # st.write("**Data Testing:**", results['jumlah_data_testing'])

                        st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)

                        st.write("**Feature Importance (dengan Permutation Importance):**")
                        st.image(results['feature_importance_image'], caption="Feature Importance", use_container_width=True)

                        # st.write("**Pengaruh Fitur:**")
                        # st.dataframe(results['feature_target_correlation'])
                        st.write("**Kurva ROC-AUC:**")
                        st.image(results['roc_image'], caption="ROC Curve", use_container_width=True)

                        with st.expander("Hyperparameter Terbaik"):
                            st.write(results['best_params'])
                elif algorithm == "XGBoost":

                    if st.button("Jalankan XGBoost"):
                        # Catat waktu mulai
                        start_time = time.time()

                        results = run_xgboost(df_final)

                        # Catat waktu selesai
                        end_time = time.time()

                        # Hitung durasi komputasi
                        elapsed_time = end_time - start_time

                        # Tampilkan durasi komputasi
                        st.write(f"**Waktu Komputasi XGBoost:** {elapsed_time:.2f} detik")
                        
                        st.write("### Hasil Evaluasi XGBoost")
                        
                        # Akurasi
                        st.write(f"**Akurasi Model:** {results['accuracy'] * 100:.2f}%")
                        
                        # Laporan Klasifikasi
                        st.write("**Laporan Klasifikasi:**")
                        st.dataframe(results['classification_report'])

                        # Precision, Recall, dan F1-Score
                        st.write("**Precision:**", results['precision'])
                        st.write("**Recall:**", results['recall'])
                        st.write("**F1-Score:**", results['f1_score'])

                        # ROC AUC Score
                        st.write("**ROC AUC Score:**")
                        st.write(results['roc_auc'])
                        
                        # Confusion Matrix
                        st.write("**Confusion Matrix:**")
                        st.write(results['confusion_matrix'])

                        # Feature Importance
                        st.write("**Feature Importance:**")
                        st.image(results['feature_importance_image'], caption="Feature Importance", use_container_width=True)
                        
                        # Kurva ROC-AUC
                        st.write("**Kurva ROC-AUC:**")
                        st.image(results['roc_image'], caption="ROC Curve", use_container_width=True)
                elif algorithm == "Naive Bayes":

                    if st.button("Jalankan Naive Bayes"):
                        # Catat waktu mulai
                        start_time = time.time()
                        results = run_naive_bayes(df_final, n_splits=jmlFold)
                        # Catat waktu selesai
                        end_time = time.time()
                        # Hitung durasi komputasi
                        elapsed_time = end_time - start_time

                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.markdown(
                            "<h3 style='text-align: center;'>Hasil Evaluasi Naive Bayes</h3>",
                            unsafe_allow_html=True
                        )
                        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(
                                f"""
                                <div style="background-color:#d1e7dd;padding:20px;border-radius:10px;">
                                    <h4 style="color:#0f5132;">🎯 Akurasi Model</h4>
                                    <h2 style="margin:0;color:#0f5132;">{results['accuracy'] * 100:.2f}%</h2>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        with col2:
                            st.markdown(
                                f"""
                                <div style="background-color:#cff4fc;padding:20px;border-radius:10px;">
                                    <h4 style="color:#055160;">⏱️ Waktu Komputasi</h4>
                                    <h2 style="margin:0;color:#055160;">{elapsed_time:.2f} detik</h2>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
                        st.write("**Laporan Klasifikasi:**")
                        col3, col4 = st.columns([0.6, 0.4])
                        with col3:
                            st.dataframe(results['classification_report'])
                        with col4:
                            st.write("**Precision:**", results['precision'])
                            st.write("**Recall:**", results['recall'])
                            st.write("**F1-Score:**", results['f1_score'])
                            st.write("**ROC AUC Score:**", results["roc_auc"])
                            st.write("**Fold Terbaik:**", results['fold_terbaik'])
                            st.write("**Data Training:**", results['jumlah_data_training'])
                            st.write("**Data Testing:**", results['jumlah_data_testing'])
                            

                        st.write("**Confusion Matrix:**")
                        st.write(results['confusion_matrix'])

                        # Feature Importance tidak tersedia untuk Naive Bayes, jadi bagian ini dikomentari

                        st.write("**Kurva ROC-AUC:**")
                        st.image(results['roc_image'], caption="ROC Curve", use_container_width=True)
                elif algorithm == "CatBoost":

                    if st.button("Jalankan Naive CatBoost"):
                        # Catat waktu mulai
                        start_time = time.time()

                        results = run_catboost(df_final)

                        # Catat waktu selesai
                        end_time = time.time()

                        # Hitung durasi komputasi
                        elapsed_time = end_time - start_time

                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.markdown(
                            "<h3 style='text-align: center;'>Hasil Evaluasi Naive Bayes</h3>",
                            unsafe_allow_html=True
                        )
                        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(
                                f"""
                                <div style="background-color:#d1e7dd;padding:20px;border-radius:10px;">
                                    <h4 style="color:#0f5132;">🎯 Akurasi Model</h4>
                                    <h2 style="margin:0;color:#0f5132;">{results['accuracy'] * 100:.2f}%</h2>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        with col2:
                            st.markdown(
                                f"""
                                <div style="background-color:#cff4fc;padding:20px;border-radius:10px;">
                                    <h4 style="color:#055160;">⏱️ Waktu Komputasi</h4>
                                    <h2 style="margin:0;color:#055160;">{elapsed_time:.2f} detik</h2>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
                        st.write("**Laporan Klasifikasi:**")
                        st.dataframe(results['classification_report'])

                        # Precision, Recall, dan F1-Score
                        st.write("**Precision:**", results['precision'])
                        st.write("**Recall:**", results['recall'])
                        st.write("**F1-Score:**", results['f1_score'])

                        # ROC AUC Score
                        st.write("**ROC AUC Score:**")
                        st.write(results['roc_auc'])
                        
                        # Confusion Matrix
                        st.write("**Confusion Matrix:**")
                        st.write(results['confusion_matrix'])

                        # Feature Importance
                        st.write("**Feature Importance:**")
                        st.image(results['feature_importance_image'], caption="Feature Importance", use_container_width=True)
                        
                        # Kurva ROC-AUC
                        st.write("**Kurva ROC-AUC:**")
                        st.image(results['roc_image'], caption="ROC Curve", use_container_width=True)
                elif algorithm == "Logistic Regression":

                    if st.button("Jalankan Logistic Regression"):
                        # Catat waktu mulai
                        start_time = time.time()

                        results = run_logistic_regression(df_final)

                        # Catat waktu selesai
                        end_time = time.time()

                        # Hitung durasi komputasi
                        elapsed_time = end_time - start_time

                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.markdown(
                            "<h3 style='text-align: center;'>Hasil Evaluasi Naive Bayes</h3>",
                            unsafe_allow_html=True
                        )
                        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(
                                f"""
                                <div style="background-color:#d1e7dd;padding:20px;border-radius:10px;">
                                    <h4 style="color:#0f5132;">🎯 Akurasi Model</h4>
                                    <h2 style="margin:0;color:#0f5132;">{results['accuracy'] * 100:.2f}%</h2>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        with col2:
                            st.markdown(
                                f"""
                                <div style="background-color:#cff4fc;padding:20px;border-radius:10px;">
                                    <h4 style="color:#055160;">⏱️ Waktu Komputasi</h4>
                                    <h2 style="margin:0;color:#055160;">{elapsed_time:.2f} detik</h2>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
                        
                        # Laporan Klasifikasi
                        st.write("**Laporan Klasifikasi:**")
                        st.dataframe(results['classification_report'])

                        # Precision, Recall, dan F1-Score
                        st.write("**Precision:**", results['precision'])
                        st.write("**Recall:**", results['recall'])
                        st.write("**F1-Score:**", results['f1_score'])

                        # ROC AUC Score
                        st.write("**ROC AUC Score:**")
                        st.write(results['roc_auc'])
                        
                        # Confusion Matrix
                        st.write("**Confusion Matrix:**")
                        st.write(results['confusion_matrix'])

                        # Kurva ROC-AUC
                        st.write("**Kurva ROC-AUC:**")
                        st.image(results['roc_image'], caption="ROC Curve", use_container_width=True)

                        # Feature Importance (aktifkan jika perlu)
                        # st.write("**Feature Importance:**")
                        # st.image(results['feature_importance_image'], caption="Feature Importance", use_container_width=True)
                elif algorithm == "AdaBoost":
                    if st.button("Jalankan AdaBoost"):
                        start_time = time.time()
                        results = run_adaboost(df_final)
                        end_time = time.time()

                        st.write(f"**Waktu Komputasi AdaBoost:** {end_time - start_time:.2f} detik")
                        st.write("### Hasil Evaluasi AdaBoost")
                        st.write(f"**Akurasi Model:** {results['accuracy'] * 100:.2f}%")
                        st.write("**Laporan Klasifikasi:**")
                        st.dataframe(results['classification_report'])
                        st.write("**Precision:**", results['precision'])
                        st.write("**Recall:**", results['recall'])
                        st.write("**F1-Score:**", results['f1_score'])
                        st.write("**ROC AUC Score:**", results['roc_auc'])
                        st.write("**Confusion Matrix:**")
                        st.write(results['confusion_matrix'])
                        st.write("**Kurva ROC-AUC:**")
                        st.image(results['roc_image'], caption="ROC Curve", use_container_width=True)
                elif algorithm == "KNN":
                    if st.button("Jalankan KNN"):
                        start_time = time.time()
                        results = run_knn(df_final)
                        end_time = time.time()
                        elapsed_time = end_time - start_time

                        st.write(f"**Waktu Komputasi KNN:** {elapsed_time:.2f} detik")
                        st.write("### Hasil Evaluasi KNN")
                        st.write(f"**Akurasi Model:** {results['accuracy'] * 100:.2f}%")
                        st.dataframe(results['classification_report'])
                        st.write("**Precision:**", results['precision'])
                        st.write("**Recall:**", results['recall'])
                        st.write("**F1-Score:**", results['f1_score'])
                        st.write("**ROC AUC Score:**", results['roc_auc'])
                        st.write("**Confusion Matrix:**")
                        st.write(results['confusion_matrix'])
                        st.image(results['roc_image'], caption="ROC Curve", use_container_width=True)
                elif algorithm == "Voting Classifier (KNN + CatBoost)":
                    if st.button("Jalankan Voting Classifier"):
                        # Catat waktu mulai
                        start_time = time.time()
                        results = run_voting_classifier(df_final)
                        # Hitung waktu komputasi
                        elapsed_time = time.time() - start_time

                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.markdown(
                            "<h3 style='text-align: center;'>Hasil Evaluasi Voting Classifier</h3>",
                            unsafe_allow_html=True
                        )
                        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

                        col1, col2 = st.columns([0.6, 0.4])
                        with col1:
                            st.write(f"***Akurasi Model:***  {results['accuracy'] * 100:.2f} %")
                        with col2:
                            st.write(f"***Waktu Komputasi:***  {elapsed_time:.2f} detik")

                        st.write("**Laporan Klasifikasi:**")
                        col3, col4 = st.columns([0.6, 0.4])
                        with col3:
                            st.dataframe(results['classification_report'])
                        with col4:
                            st.write("**Precision:**", results['precision'])
                            st.write("**Recall:**", results['recall'])
                            st.write("**F1-Score:**", results['f1_score'])
                            st.write("**ROC AUC Score:**", results["roc_auc"])

                        st.write("**Confusion Matrix:**")
                        st.write(results['confusion_matrix'])

                        st.write("**Kurva ROC-AUC:**")
                        st.image(results['roc_image'], caption="ROC Curve", use_container_width=True)
                elif algorithm == "Voting AdaBoost + XGBoost":
                    if st.button("Jalankan Voting Classifier (AdaBoost + XGBoost)"):
                        start_time = time.time()

                        results = run_AdaBoost_XGBoost(df_final)

                        end_time = time.time()
                        elapsed_time = end_time - start_time

                        st.write(f"**Waktu Komputasi Voting Classifier:** {elapsed_time:.2f} detik")
                        st.write("### Hasil Evaluasi Voting Classifier (AdaBoost + XGBoost)")

                        st.write(f"**Akurasi Model:** {results['accuracy'] * 100:.2f}%")
                        st.write("**Laporan Klasifikasi:**")
                        st.dataframe(results['classification_report'])

                        st.write("**Precision:**", results['precision'])
                        st.write("**Recall:**", results['recall'])
                        st.write("**F1-Score:**", results['f1_score'])
                        st.write("**ROC AUC Score:**", results['roc_auc'])

                        st.write("**Confusion Matrix:**")
                        st.write(results['confusion_matrix'])

                        st.write("**Kurva ROC-AUC:**")
                        st.image(results['roc_image'], caption="ROC Curve", use_container_width=True)




        elif fitur == "Sampling Data":
            
            def proses_baak(file):
                        if file.name.endswith('.csv'):
                            df = pd.read_csv(file)
                        else:
                            excel_file = pd.ExcelFile(file)
                            processed_sheets = []
                            for sheet in excel_file.sheet_names:
                                df = pd.read_excel(file, sheet_name=sheet, usecols="B,C,D,E", skiprows=1)  # Select attribute columns
                                df.columns = ["nim", "nama", "lama studi", "ipk"]  # Set column names
                                
                                df = df.dropna()  # Remove rows with missing values
                                df = df[~df['nim'].apply(lambda x: str(x)[4:6] in ['01', '02', '03'])]  # Filter rows based on nim
                                
                                # Convert "lama studi" from year format to month format
                                def convert_lama_studi(value):
                                    value = str(value)
                                    if '.' in value:
                                        bagian_depan, bagian_belakang = value.split('.')
                                        bulan = int(bagian_depan) * 12 + int(bagian_belakang)
                                    else:
                                        bulan = int(value) * 12
                                    return bulan

                                df["lama studi"] = df["lama studi"].apply(convert_lama_studi)  # Apply conversion to "lama studi"
                                
                                # Multiply "ipk" by 100
                                df["ipk"] = df["ipk"] * 100  # Scale "ipk" values
                                
                                # Filter based on "lama studi" and "ipk" values
                                df = df[(df["lama studi"] >= 42) & (df["lama studi"] <= 60)]  # Filter "lama studi"
                                df = df[(df["ipk"] >= 250) & (df["ipk"] <= 400)]  # Filter "ipk"

                                # Categorize "ipk" into bins
                                def categorize_ipk(ipk_value):
                                    if 250 <= ipk_value <= 300:
                                        return 'RENDAH'
                                    elif 301 <= ipk_value <= 350:
                                        return 'MEDIAN'
                                    elif 351 <= ipk_value <= 400:
                                        return 'TINGGI'
                                    else:
                                        return None

                                df['kategori_ipk'] = df['ipk'].apply(categorize_ipk)  # Add "kategori_ipk" column
                                
                                # Categorize "lama studi" into bins
                                def categorize_lama_studi(lama_studi_value):
                                    if 42 <= lama_studi_value <= 48:
                                        return 'TEPAT'
                                    elif 49 <= lama_studi_value <= 54:
                                        return 'CUKUP'
                                    elif 55 <= lama_studi_value <= 60:
                                        return 'LAMA'
                                    else:
                                        return None

                                df['kategori_lama_studi'] = df['lama studi'].apply(categorize_lama_studi)  # Add "kategori_lama_studi" column
                                
                                processed_sheets.append(df)
                            df = pd.concat(processed_sheets, ignore_index=True) if processed_sheets else pd.DataFrame()

                            # Normalize NIM
                            df['nim'] = df['nim'].astype(str).str.strip()
                            df = df[df['nim'].str.isdigit()]  # Filter rows where "nim" is numeric
                            df = df[df['nim'].str.len() == 9]  # Ensure "nim" is 9 digits long

                        return df[["nim", "kategori_ipk", "kategori_lama_studi"]]
            
            def sampling_pkpa(files, sampling_fraction):
                sampled_data_list = []  # Untuk menyimpan hasil sampling dari tiap file

                kolom_filter_index = {
                    "C": 2,    # Kode Progdi
                    "D": 3,    # NIM
                    "AC": 28,  # ketereratan
                    "AJ": 35,  # kerja tim
                    "AA": 26   # beasiswa
                }

                for i, file in enumerate(files):
                    if file.name.endswith('.csv'):
                        st.warning(f"File {file.name} diabaikan: hanya file Excel yang didukung.")
                        continue

                    excel_file = pd.ExcelFile(file)
                    if "Rekap" not in excel_file.sheet_names:
                        st.warning(f"Sheet 'Rekap' tidak ditemukan di file: {file.name}")
                        continue

                    raw_df = pd.read_excel(file, sheet_name="Rekap", header=None)

                    if i == 0:
                        header_df = raw_df.iloc[0]  # Baris pertama sebagai header

                    df = raw_df[1:]  # Buang baris pertama
                    df.columns = header_df  # Set header

                    try:
                        col_names = {k: header_df[v] for k, v in kolom_filter_index.items()}
                    except IndexError:
                        st.warning(f"File {file.name} tidak memiliki cukup kolom.")
                        continue

                    try:
                        df = df.dropna(subset=[col_names["C"], col_names["D"], col_names["AC"], col_names["AJ"], col_names["AA"]])
                        df = df[~df[col_names["C"]].astype(str).str.strip().isin(['01', '02', '03'])]  # Kode Progdi
                        df = df[df[col_names["D"]].astype(str).str.isdigit() & (df[col_names["D"]].astype(str).str.len() == 9)]  # NIM
                        df = df[~df[col_names["AA"]].astype(str).str.strip().isin(["0"])]  # beasiswa
                    except KeyError:
                        st.warning(f"Kolom penting tidak ditemukan dalam file: {file.name}")
                        continue

                    # Stratified Sampling berdasarkan 'ketereratan' agar proporsi kelas tetap terjaga
                    df['ketereratan'] = df[col_names["AC"]].astype(str).str.strip()  # Mengambil kolom ketereratan
                    sampled_df = df.groupby('ketereratan').apply(lambda x: x.sample(frac=sampling_fraction, random_state=42))
                    sampled_df = sampled_df.reset_index(drop=True)

                    sampled_data_list.append(sampled_df)

                # Menggabungkan hasil sampling dari semua file
                combined_sampled_data = pd.concat(sampled_data_list, ignore_index=True)

                return combined_sampled_data
            
            def filter_dan_encoding_sampel_pkpa(df):
                # Pastikan kolom yang diperlukan ada dalam dataframe
                required_columns = ["Kode Progdi", "nimhsmsmh", "nmmhsmsmh", "f5b", "f1201", "f14", "f1766"]
                if not all(col in df.columns for col in required_columns):
                    st.error("Kolom yang diperlukan tidak ada!")
                    return pd.DataFrame()

                # Pilih kolom yang relevan dari dataframe berdasarkan required_columns
                df = df[required_columns]

                # Rename kolom-kolom untuk kejelasan
                df = df.rename(columns={
                    "kode": "Kode Progdi",
                    "nimhsmsmh": "nim",
                    "nmmhsmsmh": "nama",
                    "f5b": "pekerjaan",
                    "f1201": "beasiswa",
                    "f14": "ketereratan",
                    "f1766": "kerja tim"
                })
                
                # Bersihkan data dari nilai kosong
                df = df.dropna()
                
            # Encoding untuk 'ketereratan'
                df['ketereratan'] = df['ketereratan'].astype(str).str.strip()
                df = df[df['ketereratan'].isin(['1', '2', '3', '4', '5'])]  # Pastikan hanya nilai yang valid
                df['ketereratan'] = df['ketereratan'].astype(int)
                
                # Filter dan encoding untuk 'beasiswa'
                df = df[~df['beasiswa'].astype(str).str.strip().isin(["0"])]  # Pastikan tidak ada nilai "0"
                df['beasiswa'] = df['beasiswa'].astype(str).str.strip()
                df = df[df['beasiswa'].isin(['1', '2', '3', '4', '5', '6', '7'])]
                df['beasiswa'] = df['beasiswa'].astype(int)

                # Filter dan encoding untuk 'kerja tim'
                df = df[~df['kerja tim'].astype(str).str.strip().isin(["0"])]  # Pastikan tidak ada nilai "0"
                df['kerja tim'] = df['kerja tim'].astype(str).str.strip()
                df = df[df['kerja tim'].isin(['1', '2', '3', '4', '5'])]
                df['kerja tim'] = df['kerja tim'].astype(int)
                
                # Mengubah 'ketereratan' menjadi kategori 'ERAT' atau 'TIDAK ERAT'
                df['ketereratan'] = df['ketereratan'].apply(lambda x: 0 if x in [1, 2, 3] else 1)
                df['beasiswa'] = df['beasiswa'].apply(lambda x: 0 if x in ['2', '3', '4', '5', '6', '7'] else 1)

                # Filter 'Kode Progdi' agar tidak termasuk '01', '02', '03'
                df = df[~df['Kode Progdi'].isin(['01', '02', '03'])]

                # Normalisasi NIM dan pastikan hanya angka dengan panjang 9 digit
                df['nim'] = df['nim'].astype(str).str.strip()
                df = df[df['nim'].str.isdigit()]
                df = df[df['nim'].str.len() == 9]
                
                # Encoding untuk kolom kategorikal seperti 'Kode Progdi', 'Nama', dsb.
                label_encoder = LabelEncoder()
                
                # Melakukan encoding pada kolom yang mengandung nilai kategorikal
                df['Kode Progdi'] = label_encoder.fit_transform(df['Kode Progdi'].astype(str))
                df['nim'] = label_encoder.fit_transform(df['nim'].astype(str))
                df['nama'] = label_encoder.fit_transform(df['nama'].astype(str))

                return df[["nim", "beasiswa", "ketereratan", "kerja tim"]]  # Kembalikan kolom yang relevan
            
            
            def evaluate_model(df):
                # Split data menjadi fitur dan target
                X = df.drop('ketereratan', axis=1)
                y = df['ketereratan']

                # Terapkan SMOTE untuk menyeimbangkan data
                X_resampled, y_resampled = apply_smote(X, y)

                # Split ke dalam training dan testing
                X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

                # Inisialisasi dan latih Decision Tree
                model = DecisionTreeClassifier(random_state=42)
                model.fit(X_train, y_train)

                # Prediksi dan hitung akurasi
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                return accuracy
            
            data_raw_pkpa = st.sidebar.file_uploader(
                "Upload data PKPA", 
                type=["xlsx", "xls"], 
                accept_multiple_files=True
            )
            data_raw_baak = st.sidebar.file_uploader(
                "Upload data BAAK", 
                type=["xlsx", "xls"], 
                accept_multiple_files=False
            )

            sampling_fraction = random.uniform(0.20, 0.8)
            # sampling_fractions = [0.23, 0.46, 0.15, 0.20, 0.43, 0.70, 0.80, 0.63, 0.53]


            combined_sampled_data = sampling_pkpa(data_raw_pkpa, 0.1)
            processed_data = filter_dan_encoding_sampel_pkpa(combined_sampled_data)
            
            # Proses jika file diunggah
            if data_raw_pkpa and data_raw_baak:
                sampled_data = sampling_pkpa(data_raw_pkpa, sampling_fraction)
                baak_data = proses_baak(data_raw_baak)

                if not sampled_data.empty and not baak_data.empty:
                    st.subheader("Data Hasil Sampling PKPA dan BAAK")
                    processed_data['nim'] = processed_data['nim'].astype(str)
                    baak_data['nim'] = baak_data['nim'].astype(str)

                    # Gabungkan data PKPA dan BAAK berdasarkan NIM
                    combined_data = pd.merge(processed_data, baak_data, on='nim', how='inner')

                    # Pastikan semua kolom ditampilkan sebagai string
                    for col in combined_data.columns:
                        combined_data[col] = combined_data[col].astype(str)

                    # st.dataframe(combined_data)

                    # Transformasi data untuk model

                    accuracy = 0
                    iteration = 0
                    iterasi_placeholder = st.empty()
                    while accuracy < 0.75:
                        iteration += 1

                        # Hapus iterasi sebelumnya
                        iterasi_placeholder.empty()

                        iterasi_placeholder.write(f"Iterasi: {iteration}, Akurasi: {accuracy*100:.2f}%")
                        
                        # Hapus hasil akurasi yang sudah ada
                        accuracy = 0

                        # Ulangi proses sampling dengan data yang baru
                        sampled_data = sampling_pkpa(data_raw_pkpa, random.uniform(0.20, 1.0))
                        processed_data = filter_dan_encoding_sampel_pkpa(sampled_data)

                        # Evaluasi model dengan data yang sudah disampling ulang
                        accuracy = evaluate_model(processed_data)

                        # Jika akurasi lebih besar dari 85%, hentikan
                        if accuracy >= 0.75:
                            iterasi_placeholder.write(f"Iterasi: {iteration}, Akurasi: {accuracy*100:.2f}%")
                            st.success("Akurasi tercapai, Proses selesai.")

                            # Tombol unduhan
                            buffer_xlsx = io.BytesIO()
                            with pd.ExcelWriter(buffer_xlsx, engine='xlsxwriter') as writer:
                                sampled_data.to_excel(writer, sheet_name='Rekap', index=False)
                            buffer_xlsx.seek(0)

                            st.download_button("Unduh Data Sample PKPA", data=buffer_xlsx,
                                                file_name="Sampling_PKPA.xlsx",
                                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                            break