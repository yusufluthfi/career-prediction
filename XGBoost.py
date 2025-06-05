# Import Library
import streamlit as st
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, auc)
from sklearn.tree import plot_tree

def safe_save_model(model, filename):
    # Hapus file lama jika sudah ada
    if os.path.exists(filename):
        os.remove(filename)
    # Simpan model baru
    joblib.dump(model, filename)

def safe_save_encoder(encoder, filename):
    # Hapus file lama jika sudah ada
    if os.path.exists(filename):
        os.remove(filename)
    # Simpan encoder baru
    joblib.dump(encoder, filename)

# Fungsi utama
def run_xgboost(df_final, binary_class=True):
    """Train XGBoost, Hyperparameter Tuning, dan Evaluasi."""

    # Menambahkan spinner sebelum proses dimulai
    with st.spinner('Harap tunggu, sedang memproses prediksi...'):
        time.sleep(1)
        # 🔹 1. Siapkan dataset
        data_uji = df_final[['kategori_ipk', 'kategori_lama_studi', 'kerja tim', 'beasiswa', 'ketereratan']].copy()
        
        if not all(col in data_uji.columns for col in ["kategori_ipk", "kategori_lama_studi", "kerja tim", "beasiswa", "ketereratan"]):
            raise ValueError("Ada kolom yang kurang pada dataset.")

        # 🔹 2. Label Encoding
        le_ipk = LabelEncoder()
        le_studi = LabelEncoder()
        le_kerjatim = LabelEncoder()
        le_beasiswa = LabelEncoder()
        le_ketereratan = LabelEncoder()

        data_uji['kategori_ipk'] = le_ipk.fit_transform(data_uji['kategori_ipk'].astype(str))
        data_uji['kategori_lama_studi'] = le_studi.fit_transform(data_uji['kategori_lama_studi'].astype(str))
        data_uji['kerja tim'] = le_kerjatim.fit_transform(data_uji['kerja tim'].astype(str))
        data_uji['beasiswa'] = le_beasiswa.fit_transform(data_uji['beasiswa'].astype(str))
        data_uji['ketereratan'] = le_ketereratan.fit_transform(data_uji['ketereratan'].astype(str))

        # 🔹 3. Pisahkan fitur dan target
        X = data_uji[['kategori_ipk', 'kategori_lama_studi', 'kerja tim', 'beasiswa']]
        y = data_uji['ketereratan']

        # 🔹 4. Normalisasi fitur
        X_scaled = X.values

        # 🔹 5. Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # 🔹 6. Hyperparameter Tuning dengan GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10, None],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }

        grid_search = GridSearchCV(
            XGBClassifier(random_state=42),
            param_grid, cv=5, n_jobs=-1, scoring='accuracy'
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # 🔹 7. Prediksi
        y_pred = best_model.predict(X_test)

        # 🔹 8. Evaluasi Model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=le_ketereratan.classes_, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Precision, Recall, F1
        if binary_class:
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
        else:
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

        # ROC AUC Score
        y_proba = best_model.predict_proba(X_test)

        if binary_class:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

        # 🔹 9. Decode Hasil Prediksi
        y_pred_labels = le_ketereratan.inverse_transform(y_pred)
        y_test_labels = le_ketereratan.inverse_transform(y_test)

        # 🔹 10. Feature Importance
        importance = best_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Fitur': X.columns,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        # Plot grafik importance
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Fitur'], feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.title('Feature Importance XGBoost')
        plt.gca().invert_yaxis()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

        # 🔹 11. Visualisasi ROC Curve (MULTIKELAS)
        plt.figure(figsize=(8, 6))

        if binary_class:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        else:
            y_test_binarized = label_binarize(y_test, classes=np.unique(y))
            n_classes = len(le_ketereratan.classes_)
            colors = ['blue', 'red', 'green', 'orange', 'purple']

            for i, color in zip(range(n_classes), colors):
                fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, lw=2, label=f'Kelas {le_ketereratan.classes_[i]} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Kurva ROC-AUC untuk XGBoost')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig("roc_curve.png")
        plt.close()

        # 🔹 12. Return semua hasil
        results = {
            'accuracy': accuracy,
            'classification_report': report_df,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,

            'feature_importance': feature_importance_df,
            'feature_importance_image': 'feature_importance.png',
            'roc_auc': roc_auc,
            'roc_image': 'roc_curve.png',
            'y_test_labels': y_test_labels,
            'y_pred_labels': y_pred_labels,
            'label_encoder': le_ketereratan,
        }

        # Menyimpan model dan objek yang digunakan untuk prediksi di kemudian hari
        safe_save_model(best_model, 'models/Model_XGBoost.pkl')  # Menyimpan model XGBoost
        safe_save_model(le_ipk, 'le_ipk.pkl')
        safe_save_model(le_studi, 'le_lamastudi.pkl')
        safe_save_model(le_kerjatim, 'le_kerjatim.pkl')
        safe_save_model(le_beasiswa, 'le_beasiswa.pkl')
        safe_save_model(le_ketereratan, 'le_ketereratan.pkl')

        print("Model XGBoost telah disimpan.")

    return results
