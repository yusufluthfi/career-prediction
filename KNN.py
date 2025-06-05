import streamlit as st
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, auc)

def safe_save_model(model, filename):
    if os.path.exists(filename):
        os.remove(filename)
    joblib.dump(model, filename)

def run_knn(df_final, binary_class=True):
    """Train KNN dengan GridSearchCV, Scaling, dan Evaluasi."""

    with st.spinner('Harap tunggu, sedang memproses KNN...'):
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

        # 🔹 3. Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )


        # 🔹 4. Pipeline KNN + Scaling
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier())
        ])

        # 🔹 5. GridSearch untuk cari k terbaik
        param_grid = {
            'knn__n_neighbors': list(range(1, 21)),
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['euclidean', 'manhattan', 'chebyshev']
        }

        grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)

        # 🔹 6. Prediksi & Evaluasi
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report_df = pd.DataFrame(classification_report(
            y_test, y_pred, target_names=le_ketereratan.classes_, output_dict=True
        )).transpose()
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

        # 🔹 7. ROC Curve Plot
        plt.figure(figsize=(8, 6))
        if binary_class:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        else:
            y_test_bin = label_binarize(y_test, classes=np.unique(y))
            for i in range(y_proba.shape[1]):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                plt.plot(fpr, tpr, lw=2, label=f'Kelas {i} (AUC = {auc(fpr, tpr):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Kurva ROC-AUC KNN')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig("roc_curve_knn.png")
        plt.close()

        # 🔹 8. Save Model & Encoder
        safe_save_model(best_model, 'models/Model_KNN.pkl')
        safe_save_model(le_ipk, 'le_ipk.pkl')
        safe_save_model(le_studi, 'le_lamastudi.pkl')
        safe_save_model(le_kerjatim, 'le_kerjatim.pkl')
        safe_save_model(le_beasiswa, 'le_beasiswa.pkl')
        safe_save_model(le_ketereratan, 'le_ketereratan.pkl')

        return {
            'accuracy': accuracy,
            'classification_report': report_df,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'roc_auc': roc_auc,
            'roc_image': "roc_curve_knn.png",
            'y_test_labels': le_ketereratan.inverse_transform(y_test),
            'y_pred_labels': le_ketereratan.inverse_transform(y_pred),
        }
