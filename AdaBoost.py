import streamlit as st
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, auc)

def safe_save_model(model, filename):
    if os.path.exists(filename):
        os.remove(filename)
    joblib.dump(model, filename)



def run_adaboost(df_final, binary_class=True):
    """Train AdaBoostClassifier, evaluasi, dan simpan model."""

    with st.spinner('Harap tunggu, sedang memproses prediksi...'):
        time.sleep(1)

        # 🔹 1. Siapkan dataset
        data_uji = df_final[['kategori_ipk', 'kategori_lama_studi', 'kerja tim', 'beasiswa', 'ketereratan']].copy()

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

        X = data_uji[['kategori_ipk', 'kategori_lama_studi', 'kerja tim', 'beasiswa']]
        y = data_uji['ketereratan']

        # 🔹 3. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 🔹 5. Model AdaBoost
        model = AdaBoostClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 🔹 6. Prediksi
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # 🔹 7. Evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        report_df = pd.DataFrame(
            classification_report(y_test, y_pred, target_names=le_ketereratan.classes_, output_dict=True)
        ).transpose()
        conf_matrix = confusion_matrix(y_test, y_pred)

        if binary_class:
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

        # 🔹 8. ROC Curve
        plt.figure(figsize=(8, 6))
        if binary_class:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='blue')
        else:
            y_test_bin = label_binarize(y_test, classes=np.unique(y))
            for i in range(y_test_bin.shape[1]):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                plt.plot(fpr, tpr, label=f"{le_ketereratan.classes_[i]} (AUC = {auc(fpr, tpr):.2f})")

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - AdaBoost')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig("roc_curve.png")
        plt.close()

        # 🔹 9. Return
        results = {
            'accuracy': accuracy,
            'classification_report': report_df,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'roc_auc': roc_auc,
            'roc_image': 'roc_curve.png',
            'y_test_labels': le_ketereratan.inverse_transform(y_test),
            'y_pred_labels': le_ketereratan.inverse_transform(y_pred),
            'label_encoder': le_ketereratan,
        }

        # 🔹 10. Save model dan encoder
        safe_save_model(model, 'models/Model_AdaBoost.pkl')
        safe_save_model(le_ipk, 'le_ipk.pkl')
        safe_save_model(le_studi, 'le_lamastudi.pkl')
        safe_save_model(le_kerjatim, 'le_kerjatim.pkl')
        safe_save_model(le_beasiswa, 'le_beasiswa.pkl')
        safe_save_model(le_ketereratan, 'le_ketereratan.pkl')

    return results
