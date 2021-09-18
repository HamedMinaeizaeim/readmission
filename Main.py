from ReAdmission_DataWrangling import ReAdmissionDataWrangling
from Row_Data_Visulisation import RowDataVisulasion
from MachineLearningModel import MachineLearningModel, MachineLearningVisulisation


def run_data_wrangling_and_visualisation(inpout_folder, ouput_graphs_folder, is_run_data_wrangling=True,
                                         is_run_data_visulsation= True):
    """

    :param inpout_folder: the location file where readmission_data.xlsx is saved
    :param ouput_graphs_folder: the location for publish putput
    :param is_run_data_wrangling: logic value to run data wrangling
    :param isRundataVisulsation: logic value to run data visulisation
    :return:
    """
    if is_run_data_wrangling and is_run_data_visulsation:
        data_wrangling = ReAdmissionDataWrangling(inpout_folder)
        readmission_rate = data_wrangling.run_all_data_wrangling_function(inpout_folder)
        df = data_wrangling.dF
        data_visulsation = RowDataVisulasion(ouput_graphs_folder, df)
        data_visulsation.plot_all_graphs(readmission_rate)

    elif is_run_data_wrangling and is_run_data_visulsation :
        data_wrangling = ReAdmissionDataWrangling(inpout_folder)
        readmission_rate = data_wrangling.run_all_data_wrangling_function(inpout_folder)
        df = data_wrangling.dF
        data_visulsation = RowDataVisulasion(ouput_graphs_folder, df)
        data_visulsation.plot_all_graphs(readmission_rate)

    elif is_run_data_wrangling and is_run_data_visulsation:
        data_wrangling = ReAdmissionDataWrangling(inpout_folder)
        readmission_rate = data_wrangling.run_all_data_wrangling_function(inpout_folder)


def run_machine_learning_model(input_folder_Path, output_graphs_folder_path):
    solver_models = ['RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100, max_depth=3)',
                     'GradientBoostingClassifier(random_state=42)', 'xgb.XGBClassifier()',
                     'AdaBoostClassifier(n_estimators=100, random_state=0)']
    solver_abbre = ['RandomForest', 'GradientBoosting', 'XGBOOST', 'AdaBoost']

    data_models = MachineLearningModel(input_folder_Path)
    models = data_models.run_machine_learn_models(solver_models, solver_abbre)
    calssification_visulise = MachineLearningVisulisation(output_graphs_folder_path, data_models.traning_eval,
                                                          data_models.test_eval )
    calssification_visulise.plot_recal_precision()
    calssification_visulise.plot_roc_curve()


def main(input_folder_path, output_graphs_folder_path, is_run_data_wrangling=True,
                                         is_run_data_visulsation= True):
    run_data_wrangling_and_visualisation(input_folder_path, output_graphs_folder_path, False, False)
    run_machine_learning_model()


if __name__ == '__main__':
    input_folder_path = r'C:\Users\HamedM.MANAIA.000\Documents\readmission'
    output_graphs_folder_path = r'C:\Users\HamedM.MANAIA.000\Documents\readmission\New folder'

    main(input_folder_path, output_graphs_folder_path, False, False)





