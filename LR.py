# Import Library
import streamlit as st
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, auc)

def safe_save_model(model, filename):
    if os.path.exists(filename):
        os.remove(filename)
    joblib.dump(model, filename)

def run_logistic_regression(df_final, binary_class=True):
    """Train Logistic Regression, Evaluasi, dan Visualisasi dengan data kategorikal."""

    with st.spinner('Harap tunggu, sedang memproses prediksi...'):
        time.sleep(1)

        data_uji = df_final[['kategori_ipk', 'kategori_lama_studi', 'kerja tim', 'beasiswa', 'ketereratan']].copy()
        
        if not all(col in data_uji.columns for col in ["kategori_ipk", "kategori_lama_studi", "kerja tim", "beasiswa", "ketereratan"]):
            raise ValueError("Ada kolom yang kurang pada dataset.")

        # Label Encoding
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

        X = data_uji[['kategori_ipk', 'kategori_lama_studi', 'kerja tim', 'beasiswa']]
        y = data_uji['ketereratan']

        X_scaled = X.values  # Untuk logistic regression, scaling bisa ditambahkan jika ingin

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # 🔹 Model Logistic Regression
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=le_ketereratan.classes_, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        conf_matrix = confusion_matrix(y_test, y_pred)

        if binary_class:
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
        else:
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

        if binary_class:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

        y_pred_labels = le_ketereratan.inverse_transform(y_pred)
        y_test_labels = le_ketereratan.inverse_transform(y_test)

        # Feature Importance (koefisien)
        feature_importance_df = pd.DataFrame({
            'Fitur': X.columns,
            'Importance': np.abs(model.coef_[0])
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Fitur'], feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.title('Feature Importance Logistic Regression')
        plt.gca().invert_yaxis()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

        # ROC Curve
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
        plt.title('Kurva ROC-AUC untuk Logistic Regression')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig("roc_curve.png")
        plt.close()

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

        # Simpan model dan encoder
        safe_save_model(model, 'models/Model_LogisticRegression.pkl')
        safe_save_model(le_ipk, 'le_ipk.pkl')
        safe_save_model(le_studi, 'le_lamastudi.pkl')
        safe_save_model(le_kerjatim, 'le_kerjatim.pkl')
        safe_save_model(le_beasiswa, 'le_beasiswa.pkl')
        safe_save_model(le_ketereratan, 'le_ketereratan.pkl')

        print("Model Logistic Regression telah disimpan.")

    return results

