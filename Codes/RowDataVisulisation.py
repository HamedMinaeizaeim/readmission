import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from Codes.ReAdmissionDataWrangling import ReAdmissionDataWrangling


class RowDataVisulasion:
    def __init__(self, folder_path, dF):
        self.folder_path = folder_path
        self.dF = dF
        self.prepare_data_for_ploting()

    def prepare_data_for_ploting(self):
        self.dF = self.dF.reset_index()
        self.dF['age'] = self.dF['age'].replace(['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)',
                                       '[70-80)', '[80-90)', '[90-100)'], ['(0-10)', '(10-20)', '(20-30)', '(30-40)',
                                                                           '(40-50)', '(50-60)', '(60-70)','(70-80)',
                                                                           '(80-90)', '(90-100)'])

    def plot_readmission_rate(self, var_holder):
        for mesuare in var_holder:
            agg_value = var_holder[mesuare]
            columns = agg_value.columns.tolist()

            plt.figure()
            rcParams.update({'figure.autolayout': True})
            plt.bar(agg_value[columns[0]], agg_value['Rate'])
            value_mesaure_name = mesuare[:-4]
            plt.title(value_mesaure_name + ' rate per 1,000 population')
            plt.xlabel(value_mesaure_name)
            plt.xticks(rotation=90)
            plt.ylabel(value_mesaure_name + ' rate per 1,000 population')
            plt.ylim(0, 1000)
            plt.grid(True)
            file_path = os.path.join(self.folder_path, value_mesaure_name + ' rate.png')
            plt.savefig(file_path, bbox_inches='tight')

    def plot_categories_distribution(self, catg_list):
        for category in catg_list:
            plt.figure(figsize=(10, 7))
            ax = sns.countplot(y=category, data=self.dF, order=self.dF[category].value_counts().index, palette="Set2");
            ax.set_alpha(0.8)
            ax.set_title("Bar plot : {}".format(category), fontsize=24)
            ax.set_xlabel("Number of patients", fontsize=14);
            ax.set_ylabel("");
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_xticks(range(0, 30000, 5000))
            ax.set_facecolor('xkcd:off white')
            ax.grid(alpha=0.2)

            # Add percentages to individual bars
            totals = []
            for i in ax.patches:
                totals.append(i.get_width())
            total = sum(totals)

            for i in ax.patches:
                ax.text(i.get_width() + .3, i.get_y() + .38, \
                        str(round((i.get_width() / total) * 100, 2)) + '%', fontsize=16,
                        color='black')
            file_path = os.path.join(self.folder_path, category + ' Distri.png')
            print(file_path)
            plt.savefig(file_path, bbox_inches='tight')

    def plot_numerical_distribution(self, numerical):
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 22))
        index = 1
        for num in numerical:
            plt.subplot(3, 3, index)
            sns.distplot(self.dF[num], kde=False)
            index = index + 1

        file_path = os.path.join(self.folder_path, ' numerical_Dis.png')
        print(file_path)
        plt.savefig(file_path, bbox_inches='tight')
        fig.show()

    def plot_numeric_columns_relationship(self):
        df_numeric = self.dF[['time_in_hospital', 'num_procedures', 'num_medications',
                        'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses', 'readmitted']]
        g = sns.PairGrid(df_numeric, hue="readmitted")
        g.map_diag(sns.histplot)
        g.map_offdiag(sns.scatterplot)
        g.add_legend()
        file_path = os.path.join(self.folder_path, 'Test.png')
        print(file_path)
        g.savefig(file_path)

    def plot_box_plot(self):
        numerric = ['time_in_hospital', 'num_lab_procedures', 'num_medications', 'number_inpatient']
        prameter = ["ethnicity", 'age', 'medical_specialty', 'insulin']

        for par in prameter:
            file_path = os.path.join(self.folder_path, par + '.png')
            _, ax = plt.subplots(2, 2, figsize=(20, 10))
            for index, num in enumerate(numerric):
                sns.boxplot(x=par, y=num, hue="readmitted", data=self.dF, ax=ax[index // 2][index % 2])
                ax[index // 2][index % 2].set_xticklabels(ax[index // 2][index % 2].get_xticklabels(), rotation=45,
                                                          horizontalalignment='right')
            print(file_path)
            plt.savefig(file_path)

    def plot_all_graphs(self, reAdmissionrate):
        self.plot_readmission_rate(reAdmissionrate)
        catList, Numerical = ReAdmissionDataWrangling.get_categories_numerics_columns()

        self.plot_categories_distribution(catList)
        self.plot_box_plot()
        self.plot_numerical_distribution(Numerical)
        self.plot_numeric_columns_relationship()












