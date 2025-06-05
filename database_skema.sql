-- Tabel untuk menyimpan informasi pengguna
CREATE TABLE Pengguna (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nama_pengguna VARCHAR(255) NOT NULL UNIQUE,
    kata_sandi VARCHAR(255) NOT NULL,
    tanggal_dibuat TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabel untuk menyimpan hasil pengujian
CREATE TABLE Hasil_Pengujian (
    id INT AUTO_INCREMENT PRIMARY KEY,
    id_pengguna INT,
    jenis_pengujian VARCHAR(255),
    hasil TEXT,
    tanggal_dibuat TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (id_pengguna) REFERENCES Pengguna(id)
);

-- Tabel untuk menyimpan jenis pengujian
CREATE TABLE Jenis_Pengujian (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nama VARCHAR(255) NOT NULL
);

-- Tabel untuk menyimpan sesi otentikasi
CREATE TABLE Sesi (
    id INT AUTO_INCREMENT PRIMARY KEY,
    id_pengguna INT,
    token_sesi VARCHAR(255) NOT NULL,
    tanggal_dibuat TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tanggal_kedaluwarsa TIMESTAMP,
    FOREIGN KEY (id_pengguna) REFERENCES Pengguna(id)
);

-- Tabel untuk menyimpan log aktivitas pengguna
CREATE TABLE Log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    id_pengguna INT,
    tindakan VARCHAR(255),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (id_pengguna) REFERENCES Pengguna(id)
);

-- Tabel untuk menyimpan konfigurasi sistem
CREATE TABLE Konfigurasi (
    id INT AUTO_INCREMENT PRIMARY KEY,
    kunci VARCHAR(255) NOT NULL,
    nilai TEXT
);
