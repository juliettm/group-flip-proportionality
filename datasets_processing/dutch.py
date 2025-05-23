# preprocesing
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import numpy as np

from datasets_processing._dataset import BaseDataset, get_the_middle
import os, sys

sys.path.append(os.getcwd())


class DutchDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'binary'
        self._name = 'dutch'
        self._att = att
        self.preprocess()

    def preprocess(self):

        df_base = pd.read_csv("/Users/juls/Documents/Repositories/classification_complexity/datasets/dutch/preprocessed_dutch.csv", sep=',', index_col=0)

        # this is always binary
        # EconomicStatus
        # 111: At work and student (1)
        # 120: Have job, not at work in reference period (2)
        # 112: At work and housework (3)

        status =  df_base['EconomicStatus']
        df_base.drop(labels=['EconomicStatus'], axis=1, inplace = True)
        df_base['status'] = status
        df_base.loc[df_base['status'] == 1, 'status'] = 0
        df_base.loc[df_base['status'] == 2, 'status'] = 1
        df_base.loc[df_base['status'] == 3, 'status'] = 1

        #Sex
        #1: Male
        #2: Female
        # df_base.loc[df_base['Sex'] == 1, 'Sex'] = 0
        df_base.loc[df_base['Sex'] == 2, 'Sex'] = 0



        df_base['binary_status'] = df_base['status']

        target_variable_ordinal = 'status'
        target_variable_binary = 'binary_status'

        self._ds = df_base

        self._explanatory_variables = ['Sex', 'Age', 'EducationLevel', 'HouseholdPosition', 'HouseholdSize',
                                        'Country', 'CurEcoActivity', 'MaritalStatus', 'Occupation']

        if self.outcome_type == 'binary':
            self._outcome_label = target_variable_binary
        else:
            self._outcome_label = target_variable_ordinal

        self._favorable_label_binary = [1]
        self._favorable_label_continuous = [1]

        # TODO change this to correspond with the paper values
        self._protected_att_name = [self._att]
        self._privileged_classes = [[1]]  # Male
        self._privileged_groups = [{self._att: 1}]
        self._unprivileged_groups = [{self._att: 0}]

        self._continuous_label_name = target_variable_ordinal
        self._binary_label_name = target_variable_binary

        self._cut_point = 1
        self._non_favorable_label_continuous = [0]

        self._fav_dict = self.assign_ranges_to_ordinal(fav=True)
        self._nonf_dict = self.assign_ranges_to_ordinal(fav=False, sort_desc=True)
        self._middle_fav = get_the_middle(self.favorable_label_continuous)
        self._middle_non_fav = get_the_middle(self.non_favorable_label_continuous)


# df = DutchDataset('Sex', outcome_type='binary')
# df._ds.to_csv('../data/dutch.csv', index = False)