from sklearn.calibration import label_binarize
import streamlit as st 
import pandas as pd
import numpy as np
import os
import joblib
import time
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve, auc, cohen_kappa_score
from sklearn.inspection import permutation_importance

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

def calculate_gamma_value(gamma, X_train):
    if gamma == 'scale':
        return 1 / (X_train.shape[1] * X_train.var())
    elif gamma == 'auto':
        return 1 / X_train.shape[1]
    else:
        return gamma  # for numeric gamma values

def run_svm(df_final, binary_class=True, n_splits=10):
    import shap

    with st.spinner('Harap tunggu, sedang memproses prediksi...'):
        time.sleep(1)

        data = df_final[['kategori_ipk', 'kategori_lama_studi', 'kerja tim', 'beasiswa', 'ketereratan']].copy()

        le_ipk = LabelEncoder()
        le_studi = LabelEncoder()
        le_kerjatim = LabelEncoder()
        le_beasiswa = LabelEncoder()
        le_ketereratan = LabelEncoder()

        data['kategori_ipk'] = le_ipk.fit_transform(data['kategori_ipk'].astype(str))
        data['kategori_lama_studi'] = le_studi.fit_transform(data['kategori_lama_studi'].astype(str))
        data['kerja tim'] = le_kerjatim.fit_transform(data['kerja tim'].astype(str))
        data['beasiswa'] = le_beasiswa.fit_transform(data['beasiswa'].astype(str))
        data['ketereratan'] = le_ketereratan.fit_transform(data['ketereratan'].astype(str))

        X = data[['kategori_ipk', 'kategori_lama_studi', 'kerja tim', 'beasiswa']].values
        y = data['ketereratan'].values
        #  # Hitung korelasi fitur terhadap target ketereratan
        # encoded_data = data.copy()
        # correlation_matrix = encoded_data.corr(method='pearson')
        # correlation_with_target = correlation_matrix['ketereratan'].drop('ketereratan')

        # # Interpretasi arah korelasi
        # correlation_result = []
        # for feature, corr_value in correlation_with_target.items():
        #     arah = 'positif' if corr_value > 0 else 'negatif'
        #     correlation_result.append({
        #         'fitur': feature,
        #         'korelasi': corr_value,
        #         'arah': arah
        #     })
        # correlation_df = pd.DataFrame(correlation_result).sort_values(by='korelasi', key=abs, ascending=False)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        crossvalidation = StratifiedKFold(n_splits, shuffle=True, random_state=42)

        all_results = []
        y_tests = []
        y_preds = []
        y_probas = []
        all_fold_errors = []
        all_conf_matrices = []
        feature_importances = []

        for i, (train_index, test_index) in enumerate(crossvalidation.split(X_scaled, y)):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]

            param_grid = {
                'kernel': ['linear', 'poly', 'rbf'],
                'C': [0.01, 0.1, 1, 10.0],
                'gamma': ['scale'],
            }

            grid_search = GridSearchCV(SVC(probability=True, random_state=42),
                                       param_grid, cv=3, scoring='accuracy', n_jobs=-1, return_train_score=True)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Fold {i+1} - Best gamma: {grid_search.best_params_['gamma']}")

            for j in range(len(grid_search.cv_results_['params'])):
                result = grid_search.cv_results_['params'][j].copy()
                result['mean_test_accuracy'] = grid_search.cv_results_['mean_test_score'][j]
                result['fold'] = i + 1

                # Hitung nilai gamma aktual
                gamma_val = calculate_gamma_value(result['gamma'], X_train)
                result['gamma'] = f"{result['gamma']} ({gamma_val:.4f})"

                all_results.append(result)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            y_preds.extend(y_pred)
            y_tests.extend(y_test)
            y_probas.extend(y_proba)

            acc = accuracy_score(y_test, y_pred)
            all_fold_errors.append(1 - acc)
            all_conf_matrices.append(confusion_matrix(y_test, y_pred))

            # Feature importance
            if model.kernel == 'linear':
                imp = np.abs(model.coef_).mean(axis=0)
            else:
                result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
                imp = result.importances_mean
            feature_importances.append(imp)

        # Agregasi hasil
        avg_error = np.mean(all_fold_errors)
        avg_accuracy = 1 - avg_error

        y_tests = np.array(y_tests)
        y_preds = np.array(y_preds)
        y_probas = np.array(y_probas)

        # Buat DataFrame hasil prediksi
        hasil_prediksi_df = pd.DataFrame({
            'Actual': y_tests,
            'Predicted': y_preds
        })

        # Simpan ke Excel
        hasil_prediksi_df.to_excel('data_pred/data_prediksi_SVM.xlsx', index=False)

        precision = precision_score(y_tests, y_preds, average='binary' if binary_class else 'weighted')
        recall = recall_score(y_tests, y_preds, average='binary' if binary_class else 'weighted')
        f1 = f1_score(y_tests, y_preds, average='binary' if binary_class else 'weighted')
        report_df = pd.DataFrame(classification_report(y_tests, y_preds, target_names=le_ketereratan.classes_, output_dict=True)).transpose()
        conf_matrix = sum(all_conf_matrices)
        kappa = cohen_kappa_score(y_tests, y_preds)

        if binary_class:
            tn, fp, fn, tp = conf_matrix.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            roc_auc = roc_auc_score(y_tests, y_probas[:, 1])
        else:
            specificity = None
            npv = None
            roc_auc = roc_auc_score(y_tests, y_probas, multi_class='ovr')

        # Feature Importance rata-rata
        mean_importance = np.mean(feature_importances, axis=0)
        feature_importance_df = pd.DataFrame({
            'Fitur': ['kategori_ipk', 'kategori_lama_studi', 'kerja tim', 'beasiswa'],
            'Importance': mean_importance
        }).sort_values(by='Importance', ascending=False)

        # Visualisasi Feature Importance
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Fitur'], feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.title('Feature Importance SVM (Avg of Folds)')
        plt.gca().invert_yaxis()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

        # ROC Curve
        plt.figure(figsize=(8, 6))
        if binary_class:
            fpr, tpr, _ = roc_curve(y_tests, y_probas[:, 1])
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        else:
            y_tests_binarized = label_binarize(y_tests, classes=np.unique(y))
            for i, color in zip(range(len(le_ketereratan.classes_)), ['blue', 'red', 'green', 'orange', 'purple']):
                fpr, tpr, _ = roc_curve(y_tests_binarized[:, i], y_probas[:, i])
                auc_score = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, lw=2, label=f'Kelas {le_ketereratan.classes_[i]} (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Kurva ROC-AUC SVM')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig("roc_curve.png")
        plt.close()

        results_df = pd.DataFrame(all_results)
        kernel_best_detail_df = results_df.loc[results_df.groupby('kernel')['mean_test_accuracy'].idxmax()]
        kernel_best_detail_df = kernel_best_detail_df[['kernel', 'C', 'gamma', 'mean_test_accuracy']]
        kernel_best_detail_df.rename(columns={'mean_test_accuracy': 'best_accuracy'}, inplace=True)

        y_pred_labels = le_ketereratan.inverse_transform(y_preds)
        y_test_labels = le_ketereratan.inverse_transform(y_tests)

        # Simpan model dan encoder dari fold terakhir
        safe_save_model(model, 'models/Model_SVM.pkl')
        safe_save_model(le_ipk, 'models/le_ipk.pkl')
        safe_save_model(le_studi, 'models/le_lamastudi.pkl')
        safe_save_model(le_kerjatim, 'models/le_kerjatim.pkl')
        safe_save_model(le_beasiswa, 'models/le_beasiswa.pkl')
        safe_save_model(le_ketereratan, 'models/le_ketereratan.pkl')
        safe_save_model(scaler, 'models/scaler.pkl')

        print("Model SVM telah disimpan.")

        return {
            'accuracy': avg_accuracy,
            'classification_report': report_df,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'specificity': specificity,
            'npv': npv,
            'kappa': kappa,
            'feature_importance': feature_importance_df,
            'feature_importance_image': 'feature_importance.png',
            'roc_auc': roc_auc,
            'roc_image': 'roc_curve.png',
            # 'feature_target_correlation': correlation_df,
            'y_test_labels': y_test_labels,
            'y_pred_labels': y_pred_labels,
            'label_encoder': le_ketereratan,
            'average_error': avg_error,
            'average_accuracy': avg_accuracy,
            'jumlah_data_training': len(X_train),
            'jumlah_data_testing': len(X_test),
            'best_params': grid_search.best_params_,
            'grid_search_results': results_df,
            'best_accuracy_per_kernel': kernel_best_detail_df
        }
