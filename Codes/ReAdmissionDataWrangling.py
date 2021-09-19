import pandas as pd
import numpy as np
import os


class ReAdmissionDataWrangling:

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.dF = self.read_files()

    def read_files(self):
        """

        :param filepath:
        :return:
        """
        file_path = os.path.join(self.folder_path, 'readmission_data.xlsx')
        return pd.read_excel(file_path, sheet_name='readmission_data_example', na_values=["NA", " ", "?", 'None'])

    def initial_assessment_data(self):
        """"
         :param file_path:
        :return:
        """

        dF = self.dF.replace('?', np.nan)
        uniques_per_columns = dF.nunique()
        df_info = dF.info()
        df_descr = dF.describe().T
        data_info = self.describle_data()
        print(dF.nunique())
        print(dF.shape)
        print(dF.info())
        print(dF.describe().T)
        print(data_info)
        self.dF = dF
        return data_info, uniques_per_columns, df_info, df_descr

    def write_initial_assessment_to_files(self):
        data_info, uniques_per_columns, df_info, df_descr = self.initial_assessment_data()
        output_excel_file_path = os.path.join(self.folder_path, 'InitialdataAssessment.xlsx')
        writer = pd.ExcelWriter(output_excel_file_path, engine='xlsxwriter')
        data_info.to_excel(writer, sheet_name='first dataset')
        df_descr.to_excel(writer, sheet_name='Second dataset')
        uniques_per_columns.to_excel(writer, sheet_name='Third dataset')
        writer.save()

    def describle_data(self):
        variable_name = []
        total_value = []
        total_missing_value = []
        missing_value_rate = []
        unique_value_list = []
        total_unique_value = []
        data_type = []

        for col in self.dF.columns:
            variable_name.append(col)
            data_type.append(self.dF[col].dtype)
            total_value.append(self.dF[col].shape[0])
            total_missing_value.append(self.dF[col].isnull().sum())
            missing_value_rate.append(round(self.dF[col].isnull().sum() / self.dF[col].shape[0], 4))
            unique_value_list.append(self.dF[col].unique())
            total_unique_value.append(len(self.dF[col].unique()))

        missing_data = pd.DataFrame({"Variable": variable_name,
                                     "#_Total_Value": total_value,
                                     "#_Total_Missing_Value": total_missing_value,
                                     "%_Missing_Value_Rate": missing_value_rate,
                                     "Data_Type": data_type, "Unique_Value": unique_value_list,
                                     "Total_Unique_Value": total_unique_value
                                     })

        missing_data = missing_data.set_index("Variable")
        return missing_data.sort_values("#_Total_Missing_Value", ascending=False)

    def modify_target(self):
        """

        :return:
        """

        self.dF['readmitted'] = self.dF['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

    @staticmethod
    def map_diagnostic_classification_code(diag_code):
        """

        :param diag_code:
        :return:
        """
        if diag_code.replace('.', '', 1).isdigit() and diag_code.count('.') < 2:
            diag_code = float(diag_code)
            if 1 >= diag_code <= 139:
                return "Infectious And Parasitic Diseases"
            elif 140 <= diag_code <= 239:
                return "Neoplasms"
            elif 240 <= diag_code <= 279:
                return "Nutritional, Metabolic Diseases, Immunity Disorders"
            elif 280 <= diag_code <= 289:
                return " Blood And Blood-Forming Organs"
            elif 290 <= diag_code <= 319:
                return "Mental Disorders"
            elif 320 <= diag_code <= 389:
                return "Nervous System And Sense Organs"
            elif 390 <= diag_code <= 459:
                return " Circulatory System"
            elif 460 <= diag_code <= 519:
                return " Respiratory System"
            elif 520 <= diag_code <= 579:
                return " Digestive System"
            elif 580 <= diag_code <= 629:
                return " Genitourinary System"
            elif 630 <= diag_code <= 679:
                return "f Pregnancy, Childbirth, And The Puerperium"
            elif 680 <= diag_code <= 709:
                return " Skin And Subcutaneous Tissue"
            elif 710 <= diag_code <= 739:
                return " Musculoskeletal System And Connective Tissue"
            elif 740 <= diag_code <= 759:
                return "Congenital Anomalies"
            elif 760 <= diag_code <= 779:
                return "Originating In The Perinatal Period"
            elif 780 <= diag_code <= 799:
                return "Symptoms, Signs, And Ill-Defined Conditions"
            else:
                return "Injury And Poisoning"
        elif diag_code == "unknown":
            return "unknown"
        else:  # if the code does not begin with a number
            return "EV_code"

    def map_diagnosis(self):
        """

        :return:
        """
        self.dF['diag_1'] = self.dF['diag_1'].astype(str)
        self.dF['diag_2'] = self.dF['diag_2'].astype(str)
        self.dF['diag_3'] = self.dF['diag_3'].astype(str)
        self.dF['diag_1_Cate'] = self.dF['diag_1'].apply(ReAdmissionDataWrangling.map_diagnostic_classification_code)
        self.dF['diag_2_Cate'] = self.dF['diag_2'].apply(ReAdmissionDataWrangling.map_diagnostic_classification_code)
        self.dF['diag_3_Cate'] = self.dF['diag_3'].apply(ReAdmissionDataWrangling.map_diagnostic_classification_code)

    @staticmethod
    def create_mapping_dicharge_admission_dictionaries(
            file_path=r'C:\Users\HamedM.MANAIA.000\Documents\readmission\Mapping.csv'):
        """

        :return:
        """

        mapping_data = pd.read_csv(file_path, header=None)
        admission_type_id = mapping_data[1:10]
        discharge_disposition_id = mapping_data[11:42]
        admission_source_id = mapping_data[43:68]

        admission_type_id_dict = dict(zip(admission_type_id[0], admission_type_id[1]))
        discharge_disposition_id_dict = dict(zip(discharge_disposition_id[0], discharge_disposition_id[1]))
        admission_source_id_dict = dict(zip(admission_source_id[0], admission_source_id[1]))
        return admission_type_id_dict, discharge_disposition_id_dict, admission_source_id_dict

    def add_discription_to_admission_discahrage(self):
        """

        :return:
        """
        admission_type_id_dict, discharge_disposition_id_dict, admission_source_id_dict = \
            ReAdmissionDataWrangling.create_mapping_dicharge_admission_dictionaries()
        id_cols = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
        self.dF[id_cols] = self.dF[id_cols].astype(str)

        self.dF["admission_type"] = self.dF["admission_type_id"].map(admission_type_id_dict)
        self.dF["discharge_disposition"] = self.dF["discharge_disposition_id"].map(discharge_disposition_id_dict)
        self.dF["admission_source"] = self.dF["admission_source_id"].map(admission_source_id_dict)

    def drop_expired_patients(self):
        """

        :return:
        """
        print("The number of expired patients is {}".format(len(self.dF[(self.dF["discharge_disposition_id"] == "11") |
                                                                        (self.dF["discharge_disposition_id"] == "19") |
                                                                        (self.dF["discharge_disposition_id"] == "20") |
                                                                        (self.dF["discharge_disposition_id"] == "21")])))
        self.dF = self.dF[~((self.dF["discharge_disposition_id"] == "11") |
                            (self.dF["discharge_disposition_id"] == "19") |
                            (self.dF["discharge_disposition_id"] == "20") |
                            (self.dF["discharge_disposition_id"] == "21"))
                         ]

    @staticmethod
    def create_category_medical_specialty(medical_special_code):

        if ((medical_special_code == "unknown") |
                (medical_special_code == "InternalMedicine") |
                (medical_special_code == "Family/GeneralPractice") |
                (medical_special_code == "Emergency/Trauma") |
                (medical_special_code == "Cardiology")
        ):
            return medical_special_code

        elif ((medical_special_code == "Surgery-General") |
              (medical_special_code == "Surgery-Colon&Rectal") |
              (medical_special_code == "Surgery-Cardiovascular") |
              (medical_special_code == "Surgery-Colon&Rectal") |
              (medical_special_code == 'Surgeon') |
              (medical_special_code == 'Surgery-Vascular') |
              (medical_special_code == 'Surgery-Maxillofacial') |
              (medical_special_code == 'Surgery-Pediatric') |
              (medical_special_code == 'Surgery-Thoracic') |
              (medical_special_code == 'Surgery-PlasticwithinHeadandNeck') |
              (medical_special_code == 'Surgery-Plastic') |
              (medical_special_code == "Surgery-General")):
            return 'Surgery'

        else:
            return "Other"

    def apply_category_medical_speciality(self):
        self.dF['medical_specialty'] = self.dF['medical_specialty'].replace(np.nan, 'unknown')
        self.dF['medical_specialty'] = \
            self.dF['medical_specialty'].apply(ReAdmissionDataWrangling.create_category_medical_specialty)

    @staticmethod
    def create_category_for_admission_sourse(admission_source):
        if ((admission_source == "unknown") |
                (admission_source == "Emergency Room") |
                (admission_source == "Physician Referral") |
                (admission_source == "Clinic Referral")):
            return admission_source

        elif ((admission_source == "Transfer from a hospital") |
              (admission_source == "Transfer from another health care facility") |
              (admission_source == "Transfer from a Skilled Nursing Facility (SNF)") |
              (admission_source == 'Transfer from another health care facility') |
              (admission_source == 'Transfer from critial access hospital') |
              (admission_source == 'Transfer from hospital inpt/same fac reslt in...') |
              (admission_source == 'Transfer from Ambulatory Surgery Center')):
            return 'Transfer'

        else:
            return "Other"

    def apply_category_for_admission_source(self):
        self.dF['admission_source'] = self.dF['admission_source'].replace(np.nan, 'unknown')
        self.dF['admission_source'] = \
            self.dF['admission_source'].replace(['Not Mapped', 'Not Available'], ['unknown', 'unknown'])
        self.dF['admission_source'] = \
            self.dF['admission_source'].apply(ReAdmissionDataWrangling.create_category_for_admission_sourse)

    @staticmethod
    def create_dis_charge_dispose_mapping(discha):
        if ((discha == "unknown") | (discha == "Discharged to home") |
                (discha == "Discharged/transferred to SNF") |
                (discha == "Discharged/transferred to home with home healt...") |
                (discha == "Discharged/transferred to another short term h...") |
                (discha == "Discharged/transferred to another rehab fac in...") |
                (discha == 'Discharged/transferred to another type of inpa...') |
                (discha == "Left AMA") |
                (discha == "Discharged/transferred to another rehab fac in...") |
                (discha == 'Discharged/transferred to a long term care hos...') |
                (discha == 'Hospice / home')
        ):
            return discha
        else:
            return "Other"

    def apply_discharge_disposal(self):
        self.dF['discharge_disposition'] = self.dF['discharge_disposition'].replace(np.nan, 'unknown')
        self.dF['discharge_disposition'] = self.dF['discharge_disposition'].replace(['Not Mapped', 'Not Available'],
                                                                          ['unknown', 'unknown'])
        self.dF['discharge_disposition'] = \
            self.dF['discharge_disposition'].apply(ReAdmissionDataWrangling.create_dis_charge_dispose_mapping)

    @staticmethod
    def create_admission_type_mapping(addmission):
        if ((addmission == "unknown") or (addmission == "Emergency") or (addmission == "Elective")
                or (addmission == "Urgent")
        ):
            return addmission
        else:
            return "Other"

    def apply_admission_type(self):
        self.dF['admission_type'] = self.dF['admission_type'].replace(np.nan, 'unknown')
        self.dF['admission_type'] = \
            self.dF['admission_type'].replace(['Not Mapped', 'Not Available'], ['unknown', 'unknown'])

        self.dF['admission_type'] = \
            self.dF['admission_type'].apply(ReAdmissionDataWrangling.create_admission_type_mapping)

    def drop_columns(self, column_lists):
        self.dF = self.dF.drop(labels=column_lists, axis=1)

    def calculate_readdmission_rate(self):
        var_holder = {}
        for column in self.dF.columns.tolist():
            agg_columns = self.dF.groupby([column]).agg({"readmitted": ["count", "sum"]})

            # agg_columns = agg_columns.reset_index()
            agg_columns['Rate'] = (agg_columns['readmitted']['sum'] / agg_columns['readmitted']['count']) * 1000
            agg_columns['No_notAddmitted'] = (agg_columns['readmitted']['count'] - agg_columns['readmitted']['sum'])
            level_one = agg_columns.columns.get_level_values(0).astype(str)
            level_two = agg_columns.columns.get_level_values(1).astype(str)
            agg_columns.columns = level_one + level_two
            agg_columns = agg_columns.reset_index()

            var_holder[str(column) + 'rate'] = agg_columns
        return var_holder

    def save_readmission_rate(self, folder_path, file_name):
        var_holder = self.calculate_readdmission_rate()
        output_excel_file_path = os.path.join(folder_path, file_name)
        writer = pd.ExcelWriter(output_excel_file_path, engine='xlsxwriter')
        for measuare in var_holder:
            agg_value = var_holder[measuare]
            agg_value.to_excel(writer, sheet_name=measuare)
        writer.close()

    @staticmethod
    def get_categories_numerics_columns():
        categories = ['ethnicity', 'gender', 'age', 'medical_specialty', 'insulin',
                      'medication_change', 'diabetesMed', 'diag_1_Cate',
                      'diag_2_Cate', 'diag_3_Cate', 'admission_type', 'discharge_disposition',
                      'admission_source', 'readmitted']

        numerical = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                     'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses', 'readmitted']
        return categories, numerical

    def replace_ehnicity_unkown(self):
        self.dF['ethnicity'] = self.dF['ethnicity'].replace(np.nan, 'unknown')

    def drop_na_save_data(self):
        self.dF = self.dF.dropna()
        self.dF.to_csv(os.path.join(self.folder_path, 'outputdataWrangelling.csv'), index=False)

    def run_all_data_wrangling_function(self, output_folder_path):

        self.read_files()
        self.modify_target()
        self.map_diagnosis()
        self.add_discription_to_admission_discahrage()
        self.apply_category_medical_speciality()
        self.apply_category_for_admission_source()
        self.apply_discharge_disposal()
        self.apply_admission_type()
        self.drop_expired_patients()
        self.write_initial_assessment_to_files()

        column_lists = ['diag_1', 'diag_2', 'diag_3', 'weight', 'admission_source_id', 'admission_type_id',
                       'discharge_disposition_id', 'encounter_id','patient_nbr', 'max_glu_serum', 'A1Cresult']
        self.drop_columns(column_lists)

        # calculate Readmission
        # initial assessment
        readmission_rate = self.calculate_readdmission_rate()
        self.save_readmission_rate(output_folder_path, 'ReadmissionRate.xlsx')

        # self.replace_ehnicity_unkown()
        self.drop_na_save_data()
        return readmission_rate





