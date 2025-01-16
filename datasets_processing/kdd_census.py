# preprocesing
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import os, sys

sys.path.append(os.getcwd())

from datasets_processing._dataset import BaseDataset, get_the_middle


class KddCensusDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'binary'
        self._name = 'kdd-census'
        self._att = att
        self.preprocess()

    def preprocess(self):

        df_base = pd.read_csv("datasets/kdd_census/preprocessed_kdd_census.csv", sep=',', index_col=0)

        df_base.loc[df_base['Sex'] == 2,'Sex'] = 0 # Male 1
        df_base.loc[df_base['Race'] == 2,'Race'] = 0 # White 1

        # this is always binary
        df_base['Label_binary'] = df_base['Label']

        target_variable_ordinal = 'Label'
        target_variable_binary = 'Label_binary'

        self._ds = df_base

        self._explanatory_variables = ['Sex', 'Race', 'Age', 'WageHour', 'CapitalGain', 'CapitalLoss',
                                        'Dividends', 'WorkWeeksYear', 'Industry', 'Occupation']

        if self.outcome_type == 'binary':
            self._outcome_label = target_variable_binary
        else:
            self._outcome_label = target_variable_ordinal  # [1-9]

        self._favorable_label_binary = [1]
        self._favorable_label_continuous = [1]

        self._protected_att_name = [self._att]
        self._privileged_classes = [[1]]  # caucasian
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
