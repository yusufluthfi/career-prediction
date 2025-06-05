import numpy as np
import pandas as pd
import time
import os
import joblib
import streamlit as st
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
import matplotlib.pyplot as plt

def safe_save_model(model, filename):
    if os.path.exists(filename):
        os.remove(filename)
    joblib.dump(model, filename)

def run_voting_classifier(df_final, binary_class=True):

    with st.spinner("Memproses Voting Classifier..."):
        time.sleep(1)

        # Label encoding
        data_uji = df_final[['kategori_ipk', 'kategori_lama_studi', 'kerja tim', 'beasiswa', 'ketereratan']].copy()
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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Model ensemble
        knn = KNeighborsClassifier(n_neighbors=5)
        catboost = CatBoostClassifier(verbose=0, random_state=42)

        voting_clf = VotingClassifier(estimators=[
            ('knn', knn),
            ('catboost', catboost)
        ], voting='soft')

        voting_clf.fit(X_train, y_train)
        y_pred = voting_clf.predict(X_test)
        y_proba = voting_clf.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report_df = pd.DataFrame(classification_report(y_test, y_pred, target_names=le_ketereratan.classes_, output_dict=True)).transpose()
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

        # ROC Curve
        plt.figure(figsize=(8, 6))
        if binary_class:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='blue')
        else:
            y_test_binarized = label_binarize(y_test, classes=np.unique(y))
            for i in range(len(le_ketereratan.classes_)):
                fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
                plt.plot(fpr, tpr, lw=2, label=f'Kelas {le_ketereratan.classes_[i]} (AUC = {auc(fpr, tpr):.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Kurva ROC Voting Classifier")
        plt.legend(loc="lower right")
        plt.grid()
        plt.tight_layout()
        plt.savefig("roc_voting.png")
        plt.close()

        # Simpan model dan encoder
        os.makedirs("models", exist_ok=True)
        safe_save_model(voting_clf, 'models/Model_VotingClassifier.pkl')
        safe_save_model(le_ipk, 'models/le_ipk.pkl')
        safe_save_model(le_studi, 'models/le_lamastudi.pkl')
        safe_save_model(le_kerjatim, 'models/le_kerjatim.pkl')
        safe_save_model(le_beasiswa, 'models/le_beasiswa.pkl')
        safe_save_model(le_ketereratan, 'models/le_ketereratan.pkl')

        results = {
            'accuracy': accuracy,
            'classification_report': report_df,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'roc_auc': roc_auc,
            'roc_image': 'roc_voting.png',
            'y_test_labels': le_ketereratan.inverse_transform(y_test),
            'y_pred_labels': le_ketereratan.inverse_transform(y_pred),
            'label_encoder': le_ketereratan,
        }

        return results
