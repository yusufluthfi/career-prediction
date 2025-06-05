# Import Library
import streamlit as st
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, auc)

def safe_save_model(model, filename):
    if os.path.exists(filename):
        os.remove(filename)
    joblib.dump(model, filename)

# Fungsi utama
def run_naive_bayes(df_final, binary_class=True, n_splits=10):
    """Latih Naive Bayes di semua fold, pilih yang akurasinya paling tinggi"""

    with st.spinner('Mempelajari Data Anda...'):
        time.sleep(1)

        data_uji = df_final[['kategori_ipk', 'kategori_lama_studi', 'kerja tim', 'beasiswa', 'ketereratan']].copy()

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

        crossvalidation = StratifiedKFold(n_splits, shuffle=True, random_state=42)
        folds = list(crossvalidation.split(X, y))

        best_accuracy = 0
        best_fold_data = {}

        for i, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = CategoricalNB()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)

            if acc > best_accuracy:
                best_accuracy = acc

                # Simpan semua informasi fold terbaik
                best_fold_data = {
                    'model': model,
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'y_proba': model.predict_proba(X_test),
                    'fold': i + 1,
                    'X_train': X_train,
                    'y_train': y_train,
                }

        # Evaluasi akhir pada fold terbaik
        y_test = best_fold_data['y_test']
        y_pred = best_fold_data['y_pred']
        y_proba = best_fold_data['y_proba']
        model = best_fold_data['model']

        precision = precision_score(y_test, y_pred, average='binary' if binary_class else 'weighted')
        recall = recall_score(y_test, y_pred, average='binary' if binary_class else 'weighted')
        f1 = f1_score(y_test, y_pred, average='binary' if binary_class else 'weighted')

        if binary_class:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

        report_df = pd.DataFrame(classification_report(y_test, y_pred, target_names=le_ketereratan.classes_, output_dict=True)).transpose()
        conf_matrix = confusion_matrix(y_test, y_pred)
        y_test_labels = le_ketereratan.inverse_transform(y_test)
        y_pred_labels = le_ketereratan.inverse_transform(y_pred)

        # ROC Curve
        plt.figure(figsize=(8, 6))
        if binary_class:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        else:
            y_test_binarized = label_binarize(y_test, classes=np.unique(y))
            for i in range(len(le_ketereratan.classes_)):
                fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
                auc_score = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Kelas {le_ketereratan.classes_[i]} (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC Curve Fold Terbaik (Fold ke-{best_fold_data["fold"]})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig("roc_curve.png")
        plt.close()

        results = {
            'accuracy': best_accuracy,
            'classification_report': report_df,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'roc_auc': roc_auc,
            'roc_image': 'roc_curve.png',
            'y_test_labels': y_test_labels,
            'y_pred_labels': y_pred_labels,
            'label_encoder': le_ketereratan,
            'fold_terbaik': best_fold_data["fold"],
            'jumlah_data_training': len(best_fold_data['X_train']),
            'jumlah_data_testing': len(best_fold_data['X_test']),
        }

        # Simpan model terbaik dan encodernya
        safe_save_model(model, 'models/Model_NaiveBayes.pkl')
        safe_save_model(le_ipk, 'models/le_ipk.pkl')
        safe_save_model(le_studi, 'models/le_lamastudi.pkl')
        safe_save_model(le_kerjatim, 'models/le_kerjatim.pkl')
        safe_save_model(le_beasiswa, 'models/le_beasiswa.pkl')
        safe_save_model(le_ketereratan, 'models/le_ketereratan.pkl')

        print(f"Model terbaik ditemukan di fold ke-{best_fold_data['fold']} dengan akurasi {best_accuracy:.4f}")
        print("Model Categorical Naive Bayes telah disimpan.")

    return results