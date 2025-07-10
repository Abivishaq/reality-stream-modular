import os
import time
import pickle
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    classification_report
)

from imblearn.over_sampling import SMOTE

from utils.utils import check_directory


class ModelRunner:
    def __init__(self, param, industries_dict, available_model_classes, model_name):
        """
        Initialize the ModelRunner.

        Args:
            param: The parameter object (usually a DictToObject).
            industries_dict (dict): Dictionary mapping industry codes to names.
            available_model_classes (dict): Dictionary mapping model names to their classes.
        """
        self.param = param
        self.industries_dict = industries_dict
        self.available_model_classes = available_model_classes
        self.model_name = model_name

    def display_header(self, feature_path, target_path, model_name):
        """
        Display the header for the model report.

        Args:
            featurePath (str): The path to the features.
            targetPath (str): The path to the targets.
            model (str): The name of the model.
        """
        print(f"\033[1mModel: {model_name}\033[0m")
        print(f"Feature path: {feature_path}")
        print(f"Target path: {target_path}")
        print(f"startyear: {self.param.features.startyear}, endyear: {self.param.features.endyear}, naics: {self.param.features.naics}, state: {self.param.features.state}")

    def train_model(self, model, X_train, y_train, X_test, y_test, over_sample):
        """
        Train the model and evaluate its performance.

        Args:
            model: The machine learning model to train.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training targets.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing targets.
            over_sample (bool): Flag to indicate if oversampling should be applied.

        Returns:
            tuple: Contains model, predictions, accuracy number, G-mean, and classification report dictionary.
        """
        if over_sample:
            sm = SMOTE(random_state=2)
            X_train, y_train = sm.fit_resample(X_train, y_train.ravel())
            print("Oversampling done for training data.")

        start = time.time() # Tarun
        model.fit(X_train, y_train)
        print("Model fitted successfully.")

        # Calculate predictions and metrics
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)
        end = time.time() # Tarun
        duration = end - start # Tarun


        # ROC-AUC score
        roc_auc = round(roc_auc_score(y_test, y_pred_prob[:, 1]), 2)
        print(f"\033[1mROC-AUC Score\033[0m: {roc_auc * 100} %")

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1], pos_label=1)
        gmeans = np.sqrt(tpr * (1 - fpr))
        ix = np.argmax(gmeans)

        print('\033[1mBest Threshold\033[0m: %.3f \n\033[1mG-Mean\033[0m: %.3f' % (thresholds[ix], gmeans[ix]))
        best_threshold_num = round(thresholds[ix], 3)
        gmeans_num = round(gmeans[ix], 3)

        # Update predictions based on the best threshold
        y_pred = (y_pred > thresholds[ix])

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_num = f"{accuracy * 100:.1f}"

        print("\033[1mModel Accuracy\033[0m: ", round(accuracy, 2) * 100, "%")
        print("\033[1m\nClassification Report:\033[0m")

        # Generate classification report
        cfc_report = classification_report(y_test, y_pred)
        cfc_report_dict = classification_report(y_test, y_pred, output_dict=True)
        print(cfc_report)

        return model, y_pred, accuracy_num, gmeans_num, roc_auc, best_threshold_num, cfc_report_dict, duration # Added duration as return Tarun

            

    def train(self, featurePath, targetPath, model_name, target_column, dataset_name, X_train, y_train, X_test, y_test, report_gen, all_model_list, valid_report_list, over_sample=False, model_saving=True,save_pickle=False, random_state=42):
        """
        Train the specified model and save it along with the reports.

        Args:
            featurePath (str): The path to the features.
            targetPath (str): The path to the targets.
            model_name (str): The name of the model to train.
            target_column (str): The target column name.
            dataset_name (str): The name of the dataset.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training targets.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing targets.
            report_gen (bool): Flag to indicate if a report should be generated.
            all_model_list (list): List of all available models.
            valid_report_list (list): List of models that support report generation.
            over_sample (bool): Flag to indicate if oversampling should be applied.
            model_saving (bool): Flag to indicate if the model should be saved.
            random_state (int): Random state for reproducibility.

        Returns:
            tuple: Contains paths and evaluation metrics.
        """
        assert model_name in all_model_list, f"Invalid model name: {model_name}. Must be one of {all_model_list}."

        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        # model_mapping = {
        # "LogisticRegression": LogisticRegression(max_iter=10000),  # from cuml.linear_model
        # "SVM": SVC(probability=True),  # from cuml.svm
        # "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=random_state),  # CPU model
        # "RandomForest": RandomForestClassifier(n_estimators=1000, criterion="gini", random_state=random_state),  # from cuml.ensemble
        # "XGBoost": xgb.XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', random_state=random_state, enable_categorical=True)  # GPU-enabled XGB
        # }


        # model = model_mapping.get(model_name)
        # Tarun changes and commented above model mapping code.
        model_class = self.available_model_classes.get(model_name)

        if not model_class:
            raise ValueError(f"Model class for {model_name} not found in available_model_classes.")

        # Customize default parameters
        if model_name == "LogisticRegression":
            model = model_class(max_iter=10000)
        elif model_name == "SVM":
            model = model_class(probability=True)
        elif model_name == "MLP":
            model = model_class(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=random_state)
        elif model_name == "RandomForest":
            model = model_class(n_estimators=1000, criterion="gini", random_state=random_state)
        elif model_name == "XGBoost":
            model = model_class(tree_method='gpu_hist', predictor='gpu_predictor', random_state=random_state, enable_categorical=True)
        else:
            model = model_class()

        model_fullname = model_name.replace("RandomForest", "Random Forest").replace("XGBoost", "XGBoost")

        self.displayModelHeader(featurePath, targetPath, model_fullname)

        if model_name == "XGBoost":
            model, y_pred, accuracy_num, gmeans_num, roc_auc, best_threshold_num, cfc_report_dict, runtime_seconds = train_model(model, X_train, y_train, X_test, y_test, over_sample)
        else:
            model, y_pred, accuracy_num, gmeans_num, roc_auc, best_threshold_num, cfc_report_dict, runtime_seconds = train_model(model, X_train_imputed, y_train, X_test_imputed, y_test, over_sample)

        save_dir = f"../output/{dataset_name}/saved"
        check_directory(save_dir)

        if model_saving and save_pickle:  # Tarun: Added save-pickle flag
            self.save_model(model, imputer if model_name != "XGBoost" else None, target_column, dataset_name, model_name, save_dir)

        if report_gen:
            if model_name in valid_report_list:
                if model_name == "RandomForest":
                    importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': model.feature_importances_})
                elif model_name == "XGBoost":
                    importance_df = pd.DataFrame(list(model.get_booster().get_score().items()), columns=["Feature", "Importance"])
                report = importance_df.sort_values(by='Importance', ascending=False)
                report["Feature_Name"] = report["Feature"].apply(self.report_modify)
                report = report.reindex(columns=["Feature", "Feature_Name", "Importance"])
                report.to_csv(os.path.join(save_dir, f"{target_column}-{dataset_name}-report-{model_name}.csv"), index=False)
            else:
                print("No valid report for the current model")

        return featurePath, targetPath, model, y_pred, report, model_fullname, cfc_report_dict, accuracy_num, gmeans_num, roc_auc, best_threshold_num


    def save_model(self, model, imputer, target_column, dataset_name, model_name, save_dir):
        """
        Save the trained model and imputer to disk.

        Args:
            model: The trained model to save.
            imputer: The imputer used for missing values, if applicable.
            target_column (str): The target column name.
            dataset_name (str): The name of the dataset.
            model_name (str): The name of the model.
            save_dir (str): The directory where the model will be saved.
        """
        data = {
            "model": model,
            "imputer": imputer
        }
        with open(os.path.join(save_dir, f"{target_column}-{dataset_name}-trained-{model_name}.pkl"), 'wb') as file:
            pickle.dump(data, file)

    def report_modify(self,value):
        """
        Modify feature names for better readability in reports.

        Args:
            value (str): The original feature name.

        Returns:
            str: The modified feature name.
        """
        splitted = value.split("-")
        if splitted[0] in ["Emp", "Est", "Pay"]:
            try:
                modified = splitted[0] + "-" + self.industries_dict[splitted[1]] + "-" + splitted[2]
            except KeyError:
                modified = value  # Keep original if not found
            return modified
        else:
            return value