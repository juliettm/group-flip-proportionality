import os
import sys
import warnings

import numpy as np
import pandas as pd

from datasets_processing.compas import CompasDataset
from datasets_processing.crime import CrimeDataset
from datasets_processing.drugs import DrugsDataset
from datasets_processing.insurance import InsuranceDataset
from datasets_processing.lsat import LsatDataset
from datasets_processing.obesity import ObesityDataset
from datasets_processing.older_adults import OlderAdultsDataset
from datasets_processing.parkinson import ParkinsonDataset
from datasets_processing.singles import SinglesDataset
from datasets_processing.student import StudentDataset
from datasets_processing.tic import TicDataset
from datasets_processing.wine import WineDataset
from methods.c45_classification import apply_c45_classifier
from datasets_processing.academic import AcademicDataset
from datasets_processing.adult import AdultDataset
from datasets_processing.arrhytmia import ArrhythmiaDataset
from datasets_processing.bank import BankDataset
from datasets_processing.catalunya import CatalunyaDataset
from datasets_processing.credit import CreditDataset
from datasets_processing.default import DefaultDataset
from datasets_processing.diabetes import DiabetesDataset
from datasets_processing.diabetes_w import DiabetesWDataset
from datasets_processing.dutch import DutchDataset
from datasets_processing.german import GermanDataset
from datasets_processing.heart import HeartDataset
from datasets_processing.hrs import HrsDataset
from datasets_processing.kdd_census import KddCensusDataset
from datasets_processing.nursery import NurseryDataset
from datasets_processing.oulad import OuladDataset
from datasets_processing.ricci import RicciDataset
from datasets_processing.synthetic_athlete import SyntheticAthleteDataset
from datasets_processing.synthetic_disease import SyntheticDiseaseDataset
from datasets_processing.toy import ToyDataset

warnings.filterwarnings("ignore")

to_insert = os.getcwd()
sys.path.append(to_insert)


def get_datasets(outcome):
    if outcome == 'binary':
        return [
                CompasDataset('race', outcome_type=outcome),
                CompasDataset('sex', outcome_type=outcome),
                WineDataset('color', outcome_type=outcome),
                SinglesDataset('sex', outcome_type=outcome),
                # TicDataset('religion', outcome_type=outcome),
                ObesityDataset('Gender', outcome_type=outcome),
                DrugsDataset('Gender', outcome_type=outcome),
                # Continuous
                InsuranceDataset('sex', outcome_type=outcome),
                ParkinsonDataset('sex', outcome_type=outcome),
                CrimeDataset('race', outcome_type=outcome),
                OlderAdultsDataset('sex', outcome_type=outcome),
                LsatDataset('race', outcome_type=outcome),
                LsatDataset('gender', outcome_type=outcome),
                StudentDataset('sex', outcome_type=outcome),
                # New
                AcademicDataset('ge', outcome_type=outcome),
                AdultDataset('Sex', outcome_type=outcome),
                AdultDataset('Race', outcome_type=outcome),
                AdultDataset('NativeCountry', outcome_type=outcome),
                ArrhythmiaDataset('sex', outcome_type=outcome),
                BankDataset('AgeGroup', outcome_type=outcome),
                CatalunyaDataset('foreigner', outcome_type=outcome),
                CatalunyaDataset('sex', outcome_type=outcome),
                CatalunyaDataset('NatG', outcome_type=outcome),
                CreditDataset('sex', outcome_type=outcome),
                DefaultDataset('SEX', outcome_type=outcome),
                DiabetesDataset('Sex', outcome_type=outcome),
                DiabetesDataset('Race', outcome_type=outcome),
                DiabetesWDataset('Age', outcome_type=outcome),
                DutchDataset('Sex', outcome_type=outcome),
                GermanDataset('Sex', outcome_type=outcome),
                HeartDataset('Sex', outcome_type=outcome),
                HrsDataset('gender', outcome_type=outcome),
                HrsDataset('race', outcome_type=outcome),
                KddCensusDataset('Sex', outcome_type=outcome),
                KddCensusDataset('Race', outcome_type=outcome),
                NurseryDataset('finance', outcome_type=outcome),
                OuladDataset('Sex', outcome_type=outcome),
                RicciDataset('Race', outcome_type=outcome),
                SyntheticAthleteDataset('Sex', outcome_type=outcome),
                SyntheticDiseaseDataset('Age', outcome_type=outcome),
                ToyDataset('sst', outcome_type=outcome)
                ]
    elif outcome == 'ordinal':
        return [CompasDataset('race', outcome_type=outcome),
                WineDataset('color', outcome_type=outcome),
                SinglesDataset('sex', outcome_type=outcome),
                TicDataset('religion', outcome_type=outcome),
                ObesityDataset('Gender', outcome_type=outcome),
                DrugsDataset('Gender', outcome_type=outcome)
                ]
    elif outcome == 'continuous':
        return [InsuranceDataset('sex', outcome_type=outcome),
                ParkinsonDataset('sex', outcome_type=outcome),
                CrimeDataset('race', outcome_type=outcome),
                OlderAdultsDataset('sex', outcome_type=outcome),
                LsatDataset('race', outcome_type=outcome),
                StudentDataset('sex', outcome_type=outcome)
                ]
    else:
        raise AssertionError('not a valid outcome: ', outcome)


def run_experiment(dataset, mitigation, outcome, rand_state):
    if dataset.name == 'older-adults':
        if outcome == 'binary':
            return apply_c45_classifier(dataset, splits=3, mitigation=mitigation, rand_state=rand_state)
        else:
            raise AssertionError('not a valid outcome: ', outcome)
    else:
        if outcome == 'binary':
            return apply_c45_classifier(dataset, splits=10, mitigation=mitigation, rand_state=rand_state)
        else:
            raise AssertionError('not a valid outcome: ', outcome)


outcomes = ['binary']  # , 'ordinal', 'continuous'
mitigation = [False]
stats_results = True
for outcome in outcomes:
    for mit in mitigation:
        datasets = [CompasDataset('race', outcome_type=outcome)]
        for dataset in datasets:
            print(dataset, outcome, mit)
            appended_results = []
            rand_state = 1
            print('Random state: ', rand_state)
            results = run_experiment(dataset, mit, outcome, rand_state)
            results.replace([np.inf, -np.inf], np.nan, inplace=True)
            results['seed'] = rand_state
            appended_results.append(results)
            # Concatenate all results and print them
            appended_data = pd.concat(appended_results)
            if stats_results:
                # Para imprimir los resultados por quartiles
                stats = appended_data.describe()
                stats.to_csv('results/{name}_{att}_{outcome}_{mitigation}_quartiles.csv'.format(name=dataset.name,
                                                                                                 outcome=outcome,
                                                                                                 mitigation=mit,
                                                                                                att=dataset._att))


            appended_data.to_csv('results/{name}_{att}_{outcome}_{mitigation}.csv'.format(name=dataset.name, outcome=outcome,
                                                                                mitigation=mit, rand=rand_state,
                                                                                                att=dataset._att), index=False)
