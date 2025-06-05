import streamlit as st
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, auc)
def safe_save_model(model, filename):
    if os.path.exists(filename):
        os.remove(filename)
    joblib.dump(model, filename)
def run_catboost(df_final, binary_class=True):
    """Train CatBoost Classifier dan Evaluasi dengan SMOTE."""

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

        # 🔹 4. Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 🔹 5. Model CatBoost
        cat_features = list(range(X.shape[1]))  # semua kolom adalah kategorikal
        model = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.1, 
                                   loss_function='MultiClass' if not binary_class else 'Logloss',
                                   verbose=0)
        model.fit(X_train, y_train, cat_features=cat_features)

        # 🔹 6. Prediksi
        y_pred = model.predict(X_test).ravel()

        # 🔹 7. Evaluasi
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

        y_proba = model.predict_proba(X_test)

        if binary_class:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

        # 🔹 8. ROC Curve
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
                roc_auc_i = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, lw=2, label=f'Kelas {le_ketereratan.classes_[i]} (AUC = {roc_auc_i:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Kurva ROC-AUC untuk CatBoost')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig("roc_curve.png")
        plt.close()

        # 🔹 9. Feature Importance
        feature_importance_df = pd.DataFrame({
            'Fitur': X.columns,
            'Importance': model.get_feature_importance()
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Fitur'], feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.title('Feature Importance CatBoost')
        plt.gca().invert_yaxis()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

        # 🔹 10. Decode label
        y_pred_labels = le_ketereratan.inverse_transform(y_pred)
        y_test_labels = le_ketereratan.inverse_transform(y_test)

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
        safe_save_model(model, 'models/Model_CatBoost.pkl')
        safe_save_model(le_ipk, 'le_ipk.pkl')
        safe_save_model(le_studi, 'le_lamastudi.pkl')
        safe_save_model(le_kerjatim, 'le_kerjatim.pkl')
        safe_save_model(le_beasiswa, 'le_beasiswa.pkl')
        safe_save_model(le_ketereratan, 'le_ketereratan.pkl')

        print("Model CatBoost telah disimpan.")

    return results
