import joblib
import numpy as np

class ModelPredictor:
    def __init__(self):
        # Memuat model dan objek yang telah disimpan
        self.model = joblib.load('best_svm_model.pkl')
        self.label_encoder = joblib.load('label_encoder.pkl')
        self.scaler = joblib.load('scaler.pkl')

    def predict(self, new_data):
        # Pastikan new_data adalah DataFrame yang memiliki kolom yang sesuai
        new_data_scaled = self.scaler.transform(new_data)  # Normalisasi data baru

        # Prediksi dengan model yang dimuat
        prediction = self.model.predict(new_data_scaled)

        # Decode hasil prediksi kembali ke label asli
        predicted_label = self.label_encoder.inverse_transform(prediction)
        return predicted_label
