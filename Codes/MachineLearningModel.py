import pandas as pd
import os
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import DataWranggling as DataWrang
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score,f1_score, roc_curve
from sklearn.metrics import confusion_matrix as cm


class MachineLearningModel:

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.df = self.read_data()
        self.traning_eval = {'model': [], 'auc': [], 'accuracy': [], 'recal': [],
                          'precision': [], 'score': [], 'CM': [], 'fpr': [], 'tpr':[]}
        self.test_eval = {'model': [], 'auc': [], 'accuracy': [], 'recal': [],
                          'precision': [], 'score': [], 'CM': [], 'fpr': [], 'tpr':[]}

    def read_data(self):
        try:
            df = pd.read_csv(os.path.join(self.folder_path, "outputdataWrangelling.csv"))
        except FileNotFoundError:
            print("outputdataWrangelling.csv is not found! please run data wrangling first")
        columns = df.columns.tolist()
        columns.remove('readmitted')
        columns.append('readmitted')
        df = df[df['gender']!="Unknown"]
        return df[columns]

    def resample_data_due_to_unbalance(self):
        not_readmitted = self.df[self.df.readmitted == 0]
        readmitted = self.df[self.df.readmitted == 1]
        length_readmitted = len(readmitted.index)
        random_selections = [0.1, 0.2, 0.3, 0.4, 0.5]

        resampled = []
        for run in range(0, 3):
            for random in random_selections:
                random_not_readmitted = not_readmitted.sample(n=round(random * length_readmitted))
                random_readmitted = readmitted.sample(n=round(random * length_readmitted))
                resampled.append(random_not_readmitted)
                resampled.append(random_readmitted)

        resampled = pd.concat(resampled)
        return resampled.sample(frac=1)

    def separate_featues_labels(self):
        df = self.resample_data_due_to_unbalance()
        columns = self.df.columns.tolist()
        columns.remove('readmitted')
        x, y = df[columns].values, df['readmitted'].values
        return df[columns].values, df['readmitted'].values

    def split_trian_test_data(self):
        x, y = self.separate_featues_labels()
        # Split data 70%-30% into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0, stratify=y)
        return X_train, X_test, y_train, y_test

    def build_model(self, solver_model):
        categorical_features = ['ethnicity', 'gender', 'medical_specialty', 'diag_1_Cate', 'diag_2_Cate', 'diag_3_Cate',
                                'admission_type',
                                'discharge_disposition', 'admission_source']
        orderbased_features = ['insulin', 'medication_change', 'diabetesMed', 'age']

        X_train, X_test, y_train, y_test = self.split_trian_test_data()

        columns = self.df.columns.tolist()
        columns.remove('readmitted')

        categorical_features_index = [columns.index(category) for category in categorical_features]
        orderbased_features_index = [columns.index(order) for order in orderbased_features]

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='error'))])

        orderbased_transformer = Pipeline(steps=[
            ('ordinal', OrdinalEncoder(handle_unknown='error'))])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features_index),
                ('order', orderbased_transformer, orderbased_features_index)])

        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', eval(solver_model))])

        model = pipeline.fit(X_train, y_train)
        return model

    @staticmethod
    def calculate_evaluate_matrix(y_actual, y_pred, thresh):

        auc = roc_auc_score(y_actual, y_pred)
        accuracy = accuracy_score(y_actual, (y_pred > thresh))
        recall = recall_score(y_actual, (y_pred > thresh))
        precision = precision_score(y_actual, (y_pred > thresh))
        fscore = f1_score(y_actual, (y_pred > thresh))

        return auc, accuracy, recall, precision, fscore

    @staticmethod
    def print_evaluation(auc, accuracy, recall, precision, fscore):
        print('AUC:...............................................{:.3f}'.format(auc))
        print('accuracy:..........................................{:.3f}'.format(accuracy))
        print('recall:............................................{:.3f}'.format(recall))
        print('precision:.........................................{:.3f}'.format(precision))
        print('fscore:............................................{:.3f}'.format(fscore))

    def run_evalaute_model(self, solver_model,model_abbre , thresh, printing=True):
        X_train, X_test, y_train, y_test = self.split_trian_test_data()
        model = self.build_model(solver_model)
        y_train_preds = model.predict_proba(X_train)[:, 1]
        y_test_preds = model.predict_proba(X_test)[:, 1]
        train_auc, train_accuracy, train_recall, \
        train_precision, train_fscore = MachineLearningModel.calculate_evaluate_matrix(y_train, y_train_preds, thresh)
        test_auc, test_accuracy, test_recall, \
        test_precision, test_fscore = MachineLearningModel.calculate_evaluate_matrix(y_test, y_test_preds, thresh)

        # prediction
        predictions = model.predict(X_train)
        train_score = round(accuracy_score(y_train, predictions), 3)
        cm_train = cm(y_train, predictions)

        predictions = model.predict(X_test)
        test_score = round(accuracy_score(y_test, predictions), 3)
        cm_test = cm(y_test, predictions)
        fpr, tpr, _ = roc_curve(y_test, predictions)

        self.traning_eval['model'].append(model_abbre)
        self.traning_eval['auc'].append(train_auc)
        self.traning_eval['accuracy'].append(train_accuracy)
        self.traning_eval['recal'].append(train_recall)
        self.traning_eval['precision'].append(train_precision)
        self.traning_eval['score'].append(train_score)
        self.traning_eval['CM'].append(cm_train)
        self.traning_eval['fpr'].append(None)
        self.traning_eval['tpr'].append(None)

        self.test_eval['model'].append(model_abbre)
        self.test_eval['auc'].append(test_auc)
        self.test_eval['accuracy'].append(test_accuracy)
        self.test_eval['recal'].append(test_recall)
        self.test_eval['precision'].append(test_precision)
        self.test_eval['score'].append(test_score)
        self.test_eval['CM'].append(cm_test)
        self.test_eval['fpr'].append(fpr)
        self.test_eval['tpr'].append(tpr)

        if print:
            print('Model: Train.........{}'.format(solver_model))
            MachineLearningModel.print_evaluation(train_auc, train_accuracy, train_recall, train_precision, train_fscore)
            print('Model: Test.........{}'.format(solver_model))
            MachineLearningModel.print_evaluation(test_auc, test_accuracy, test_recall, test_precision, test_fscore)

        return model

    def run_machine_learn_models(self, models_list_as_string, model_abbreviation):
        """

        :param folder_path:
        :param modelsListStringFormat:
        :return:
        """
        return [self.run_evalaute_model(model_as_string, model_abbreviation, 0.5)
                for model_as_string, model_abbreviation in zip(models_list_as_string,model_abbreviation)]


class MachineLearningVisulisation(MachineLearningModel):

    def __init__(self, folder_path, traning_eval, test_eval):
        self.traning_eval = traning_eval
        self.test_eval = test_eval
        self.folder_path = folder_path

    def plot_recal_precision(self):
        train_eval = pd.DataFrame(self.traning_eval)
        test_eval = pd.DataFrame(self.test_eval)

        train_eval['status'] = 'Train'
        test_eval['status'] = 'Test'
        full_eval = pd.concat([train_eval, test_eval])

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)

        sns.barplot(x='model', y='auc', hue='status', data=full_eval, ax=axes[0])
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, horizontalalignment='right')
        axes[0].set_title("AUC")

        sns.barplot(x='model', y='recal', hue='status', data=full_eval, ax=axes[1])
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, horizontalalignment='right')
        axes[1].set_title("recal")

        sns.barplot(x='model', y='precision', hue='status', data=full_eval,  ax=axes[2])
        axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, horizontalalignment='right')
        axes[1].set_title("precision")
        plt.savefig(os.path.join(self.folder_path, 'Eval.png'))
        plt.show()

    def plot_roc_curve(self):
        train_eval = pd.DataFrame(self.traning_eval)
        test_eval = pd.DataFrame(self.test_eval)

        train_eval['status'] = 'Train'
        test_eval['status'] = 'Test'

        fig = plt.figure(figsize=(10, 6))
        for i in test_eval.index:
            plt.plot(test_eval.loc[i]['fpr'],
                     test_eval.loc[i]['tpr'],
                     label="{}, AUC={:.3f}".format(test_eval.loc[i]['model'], test_eval.loc[i]['auc']))

        plt.plot([0, 1], [0, 1], color='black', linestyle='--')

        plt.xticks(np.arange(0.0, 1.1, step=0.1))
        plt.xlabel("False Positive Rate", fontsize=14)

        plt.yticks(np.arange(0.0, 1.1, step=0.1))
        plt.ylabel("True Positive Rate", fontsize=14)

        plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
        plt.legend(prop={'size': 10}, loc='lower right')
        plt.savefig(os.path.join(self.folder_path, 'ROC_Curve.png'))
        plt.show()










