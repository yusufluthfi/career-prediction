import streamlit as st

def cssAPP():
        st.markdown(
        """
        <style>
        /* Warna latar belakang sidebar */
        [data-testid="stSidebar"] {
            background-color: #4076C7; /* Contoh warna sidebar */
        }

        /* Container input (file uploader, input text, selectbox) */
        [data-testid="stSidebar"] .stTextInput > div,
        [data-testid="stSidebar"] .stFileUploader,
        [data-testid="stSidebar"] .stSelectbox,
        [data-testid="stSidebar"] .stNumberInput {
            background-color: #81A6DE; /* Warna background elemen */
            border-radius: 12px;
            padding: 10px;
            margin-bottom: 10px;
            box-shadow: 0 0 4px rgba(0,0,0,0.3);
        }

        /* Label teks */
        [data-testid="stSidebar"] label {
            color: #ffffff !important;
            font-weight: 600;
        }

        /* Placeholder dan input teks */
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] select {
            background-color: #f0f0f0;
            color: #000000;
        }

        /* Tombol browse file */
        [data-testid="stSidebar"] button {
            background-color: #ffffff !important;
            color: #000000 !important;
        }

        /* Ubah warna latar belakang expander */
        [data-testid="stSidebar"] .stExpander {
            background-color: #ffffff; /* Ganti sesuai keinginan */
            border-radius: 8px;
            padding: 5px;
            border: 5px solid #81A6DE;
        }

        /* Ubah warna teks judul expander */
        [data-testid="stSidebar"] .stExpander > summary {
            color: white !important;
            font-weight: bold;
            font-size: 14px;
        }

        /* Ubah warna isi dalam expander */
        [data-testid="stSidebar"] .stExpanderContent {
            background-color: #5b7bb8;  /* Warna isi expander */
            color: white;
            padding: 10px;
            border-radius: 8px;
        }

        /* Optional: ubah ikon panah */
        [data-testid="stSidebar"] .stExpander summary::marker {
            color: white;
        }
        
        </style>
        """,
        unsafe_allow_html=True
    )
        
def cssLogin():
    st.markdown("""
    <style>
    /* Latar belakang halaman */
    body {
        background-color: #f4f6f8;
    }

    /* Judul halaman */
    h1 {
        color: #364761;
        text-align: center;
        font-weight: bold;
    }

    /* Kotak form */
    div[data-testid="stForm"] {
        background-color: #ffffff;
        padding: 2rem 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        margin: auto;
        width: 100%;
        max-width: 400px;
    }

    /* Label input */
    label {
        color: #364761;
        font-weight: 600;
    }

    /* Input field */
    input {
        border-radius: 8px !important;
        padding: 8px 12px;
        border: 1px solid #cbd5e0;
        background-color: #f9fafb;
    }

    /* Tombol utama */
    button[kind="primary"], div.stButton>button {
        background-color: #364761;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        transition: background-color 0.3s ease;
        margin-top: 10px;
        width: 57%;
    }

    /* Hover tombol */
    button[kind="primary"]:hover, div.stButton>button:hover {
        background-color: #2a3b54;
        color: white;
    }

    /* Error dan Success Message */
    div.stAlert {
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }

    /* Centered toggle button */
    div.stButton {
        display: flex;
        justify-content: center;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
def csstambahpengguna():
    st.markdown("""
    <style>

    /* Input Fields */
    input[type="text"], input[type="password"] {
        padding: 12px 14px;
        border: 1px solid #d0d0d0;
        border-radius: 5px;
        background-color: #FAFAFA;
        font-size: 15px;
        transition: border 0.3s ease;
    }

    </style>
    """, unsafe_allow_html=True)
