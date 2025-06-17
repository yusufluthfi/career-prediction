# decision_tree_model.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import joblib
from collections import Counter
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             precision_score, recall_score, f1_score, roc_auc_score, roc_curve,cohen_kappa_score, auc)
from sklearn.preprocessing import LabelEncoder
# from imblearn.combine import SMOTETomek

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

def run_decision_tree(df_final, n_splits=10):
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    with st.spinner('Harap tunggu, sedang memproses prediksi...'):
        progress_bar = st.empty()
        status_text = st.empty()
        progress_bar.progress(2)
        time.sleep(1)
        status_text.text("🔄 Menyiapkan data dan melakukan Label Encoding...")
        progress_bar.progress(5)

        # Preprocessing
        data_uji = df_final[['kategori_ipk', 'kategori_lama_studi', 'beasiswa', 'kerja tim', 'ketereratan']].copy()
        le_ipk, le_studi, le_beasiswa = LabelEncoder(), LabelEncoder(), LabelEncoder()
        le_kerjatim, le_ketereratan = LabelEncoder(), LabelEncoder()
        progress_bar.progress(10)
        data_uji['kategori_ipk'] = le_ipk.fit_transform(data_uji['kategori_ipk'])
        data_uji['kategori_lama_studi'] = le_studi.fit_transform(data_uji['kategori_lama_studi'])
        data_uji['beasiswa'] = le_beasiswa.fit_transform(data_uji['beasiswa'])
        data_uji['kerja tim'] = le_kerjatim.fit_transform(data_uji['kerja tim'])
        data_uji['ketereratan'] = le_ketereratan.fit_transform(data_uji['ketereratan'])

        progress_bar.progress(15)

        X = data_uji[['kategori_ipk', 'kategori_lama_studi', 'beasiswa', 'kerja tim']].values
        y = data_uji['ketereratan'].values

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        progress_bar.progress(20)

        param_grid = {
            'criterion': ['entropy'],
            'max_depth': [3, 5, 7, 10, 15, 20, 25, None],
            'min_samples_split': [2, 5, 10, 15, 20, 25, 30],
            'min_samples_leaf': [1, 2, 4, 6, 8, 10],
        }

        fold_errors = []
        max_depth_acc = {}
        min_samples_split_acc = {}
        min_samples_leaf_acc = {}
        all_y_test = []
        all_y_pred = []
        all_y_proba = []

        status_text.text("🧠 Melatih model Decision Tree dengan GridSearchCV...")
        total_folds = skf.get_n_splits()
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            status_text.text(f"🔁 Fold {i}/{total_folds} sedang diproses...")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            grid_search = GridSearchCV(
                DecisionTreeClassifier(random_state=42),
                param_grid,
                cv=5,
                scoring='accuracy'
            )
            grid_search.fit(X_train, y_train)

            results = grid_search.cv_results_
            for j in range(len(results['params'])):
                params = results['params'][j]
                mean_acc = results['mean_test_score'][j]
                max_d = params['max_depth']
                min_split = params['min_samples_split']
                min_leaf = params['min_samples_leaf']

                max_depth_acc[max_d] = max_depth_acc.get(max_d, []) + [mean_acc]
                min_samples_split_acc[min_split] = min_samples_split_acc.get(min_split, []) + [mean_acc]
                min_samples_leaf_acc[min_leaf] = min_samples_leaf_acc.get(min_leaf, []) + [mean_acc]

            model = grid_search.best_estimator_

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            error = 1 - accuracy
            fold_errors.append(error)

            best_params = grid_search.best_params_
            max_depth_val = best_params['max_depth']
            min_samples_split_val = best_params['min_samples_split']
            min_samples_leaf_val = best_params['min_samples_leaf']

            max_depth_acc[max_depth_val] = max_depth_acc.get(max_depth_val, []) + [accuracy]
            min_samples_split_acc[min_samples_split_val] = min_samples_split_acc.get(min_samples_split_val, []) + [accuracy]
            min_samples_leaf_acc[min_samples_leaf_val] = min_samples_leaf_acc.get(min_samples_leaf_val, []) + [accuracy]

            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_proba.extend(y_proba)

            progress = 20 + int((40 * i) / total_folds)
            progress_bar.progress(progress)

        # Evaluasi dan visualisasi

        avg_error = np.mean(fold_errors)
        all_y_test = np.array(all_y_test)
        all_y_pred = np.array(all_y_pred)
        all_y_proba = np.array(all_y_proba)
        y_pred_labels = le_ketereratan.inverse_transform(all_y_pred)
        y_test_labels = le_ketereratan.inverse_transform(all_y_test)

        status_text.text("🧾 Menyimpan hasil prediksi dan menghitung metrik evaluasi...")
        hasil_prediksi_df = pd.DataFrame({
            'Actual': y_test_labels,
            'Predicted': y_pred_labels
        })
        hasil_prediksi_df.to_excel('data_pred/data_prediksi_DT.xlsx', index=False)
        progress_bar.progress(65)
        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        report_df = pd.DataFrame(classification_report(y_test_labels, y_pred_labels, output_dict=True)).transpose()
        conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)

        if conf_matrix.shape == (2, 2):
            tn, fp, fn, tp = conf_matrix.ravel()
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
            npv = tn / (tn + fn) if (tn + fn) != 0 else 0
        else:
            specificity = 0
            npv = 0

        recall = recall_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0)
        precision = precision_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0)
        kappa = cohen_kappa_score(y_test_labels, y_pred_labels)
        f1 = f1_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0)

        progress_bar.progress(75)
        status_text.text("📉 Membuat visualisasi evaluasi model...")

        roc_auc = roc_auc_score(all_y_test, all_y_proba[:, 1])
        fpr, tpr, _ = roc_curve(all_y_test, all_y_proba[:, 1])
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Kurva ROC-AUC Decision Tree')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("roc_curve.png")
        plt.close()

        progress_bar.progress(80)

        final_model = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy'
        )
        final_model.fit(X, y)
        model = final_model.best_estimator_

        importance = model.feature_importances_
        feature_names = ['kategori_ipk', 'kategori_lama_studi', 'beasiswa', 'kerja tim']
        feature_importance_df = pd.DataFrame({
            'Fitur': feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Fitur'], feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.tight_layout()
        feature_importance_image = 'plots/feature_importance.png'
        plt.savefig(feature_importance_image)
        plt.close()

        progress_bar.progress(85)

        plt.figure(figsize=(16, 8))
        plot_tree(model, feature_names=feature_names, class_names=le_ketereratan.classes_, filled=True)
        plt.tight_layout()
        tree_image_path = 'plots/decision_tree.png'
        plt.savefig(tree_image_path)
        plt.close()

        avg_max_depth_acc = {k: np.mean(v) for k, v in max_depth_acc.items()}
        plt.figure()
        plt.plot(list(avg_max_depth_acc.keys()), list(avg_max_depth_acc.values()), marker='o', color='green')
        plt.title("Accuracy vs max_depth (Decision Tree)")
        plt.xlabel("max_depth")
        plt.ylabel("Accuracy")
        plt.grid(True)
        max_depth_plot_path = "plots/accuracy_vs_max_depth.png"
        plt.tight_layout()
        plt.savefig(max_depth_plot_path)
        plt.close()

        progress_bar.progress(90)

        avg_min_samples_split_acc = {k: np.mean(v) for k, v in min_samples_split_acc.items()}
        plt.figure()
        plt.plot(list(avg_min_samples_split_acc.keys()), list(avg_min_samples_split_acc.values()), marker='o', color='purple')
        plt.title("Accuracy vs min_samples_split (Decision Tree)")
        plt.xlabel("min_samples_split")
        plt.ylabel("Accuracy")
        plt.grid(True)
        min_split_plot_path = "plots/accuracy_vs_min_samples_split.png"
        plt.tight_layout()
        plt.savefig(min_split_plot_path)
        plt.close()

        avg_min_samples_leaf_acc = {k: np.mean(v) for k, v in min_samples_leaf_acc.items()}
        plt.figure()
        plt.plot(list(avg_min_samples_leaf_acc.keys()), list(avg_min_samples_leaf_acc.values()), marker='o', color='orange')
        plt.title("Accuracy vs min_samples_leaf (Decision Tree)")
        plt.xlabel("min_samples_leaf")
        plt.ylabel("Accuracy")
        plt.grid(True)
        min_leaf_plot_path = "plots/accuracy_vs_min_samples_leaf.png"
        plt.tight_layout()
        plt.savefig(min_leaf_plot_path)
        plt.close()

        progress_bar.progress(95)
        status_text.text("💾 Menyimpan model dan encoder...")

        results = {
            'accuracy': accuracy,
            'classification_report': report_df,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'npv': npv,
            'kappa': kappa,
            'confusion_matrix': conf_matrix,
            'roc_auc': roc_auc,
            'roc_image': 'roc_curve.png',
            'feature_importance': feature_importance_df,
            'feature_importance_image': feature_importance_image,
            'tree_image': tree_image_path,
            'max_depth_plot': max_depth_plot_path,
            'min_samples_split_plot': min_split_plot_path,
            'min_samples_leaf_plot': min_leaf_plot_path,
            'y_pred_labels': y_pred_labels,
            'y_actual_labels': y_test_labels,
            'average_error': avg_error,
            'average_accuracy': 1 - avg_error,
            'jumlah_data': len(X)
        }

        safe_save_model(model, 'models/Model_Decision_Tree.pkl')
        safe_save_model(le_ipk, 'models/le_ipk.pkl')
        safe_save_model(le_studi, 'models/le_lamastudi.pkl')
        safe_save_model(le_kerjatim, 'models/le_kerjatim.pkl')
        safe_save_model(le_beasiswa, 'models/le_beasiswa.pkl')
        safe_save_model(le_ketereratan, 'models/le_ketereratan.pkl')

        progress_bar.progress(100)
        status_text.text("✅ Proses selesai!")
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
        print("Model Decision Tree (mean fold evaluation) telah disimpan.")

    return results



