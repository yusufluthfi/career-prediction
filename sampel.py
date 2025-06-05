import pandas as pd
import streamlit as st
import io

# Kolom upload untuk multiple file Excel
uploaded_files = st.sidebar.file_uploader("Upload file Excel", type=["xlsx", "xls"], accept_multiple_files=True)

if uploaded_files:
    # Membuat DataFrame kosong untuk menggabungkan hasil filter dari semua file
    combined_filtered_df = pd.DataFrame()

    for uploaded_file in uploaded_files:
        # Membaca file Excel yang diunggah
        df = pd.read_excel(uploaded_file)

        # Menampilkan data yang diunggah
        st.write(f"Data yang diunggah dari file: {uploaded_file.name}")
        st.write(df)

        # Mendapatkan kolom berdasarkan urutan (AA, D, C)
        col_aa = df.columns[26]  # Kolom AA (indeks ke-26)
        col_d = df.columns[3]    # Kolom D (indeks ke-3)
        col_c = df.columns[2]    # Kolom C (indeks ke-2)

        # Filter data dengan kondisi yang diberikan
        filtered_df = df[
            (df[col_aa] != 1) &  # Kolom 'AA' selain 1
            (df[col_aa].notnull()) &  # Kolom 'AA' tidak null
            (df[col_d].astype(str).str.len() == 9) &  # Kolom 'D' berisi 9 angka
            (~df[col_c].isin(['01', '02', '03']))  # Kolom 'C' selain yang berisi '01', '02', atau '03'
        ]

        # Gabungkan hasil filter dari setiap file
        combined_filtered_df = pd.concat([combined_filtered_df, filtered_df], ignore_index=True)

    # Menampilkan data yang sudah difilter
    st.write("Data yang sudah difilter:")
    st.write(combined_filtered_df)

    # Menyimpan data yang difilter sebagai Excel dengan sheet "Rekap"
    buffer_xlsx = io.BytesIO()
    with pd.ExcelWriter(buffer_xlsx, engine='xlsxwriter') as writer:
        combined_filtered_df.to_excel(writer, sheet_name='Rekap', index=False)
    buffer_xlsx.seek(0)

    # Tombol unduhan untuk file Excel
    st.download_button(
        label="Unduh Data sebagai .xlsx",
        data=buffer_xlsx,
        file_name="filtered_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Silakan unggah file Excel untuk memulai.")
