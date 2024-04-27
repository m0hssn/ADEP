import sqlite3
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn import preprocessing

class DDIDataset(Dataset):
    def __init__(self, df, extraction):
        self.df = df
        self.extraction = extraction.copy()
        self.extraction.rename(columns={'drugA':'drugB', 'drugB':'drugA'}, inplace=True)
        self.extraction = pd.concat([self.extraction, extraction], axis=0, ignore_index=True)

    def __len__(self):
        return len(self.extraction)

    def __getitem__(self, idx):
        drugA = torch.tensor(self.df[self.df['name'] == self.extraction.loc[idx]['drugA']].drop(columns=['name']).values.astype('float32'))
        drugB = torch.tensor(self.df[self.df['name'] == self.extraction.loc[idx]['drugB']].drop(columns=['name']).values.astype('float32'))
        return torch.cat([(drugA), (drugB)]).flatten(), self.extraction.loc[idx]['side']


def get_dataset(ds_name):
    def feature_extractor(df, f_list):
        for feature in f_list:
            unique = set('|'.join(df[feature].values.tolist()).split('|'))

            for side in unique:
                df[side] = 0

            for index, row in df.iterrows():
                for side in row[feature].split('|'):
                    df.at[index, side] = 1
        df.drop(columns=f_list, inplace=True)

    extraction = None
    df_drug = None

    if ds_name == 'DS1':
        df_drug = pd.read_pickle('../Data/df.pkl')
        conn = sqlite3.connect('../Data//event.db')
        extraction = pd.read_sql('select * from extraction;', conn)
        extraction.drop(columns=['index'], inplace=True)

        feature_extractor(df_drug, ['side', 'target', 'enzyme', 'pathway', 'smile'])

    elif ds_name == 'DS2':
        extraction = pd.read_csv("../Data/drug_interaction.csv")
        df_drug = pd.read_csv("../Data//drug_information_1258.csv")
        extraction.drop(columns=['index', "Unnamed: 0"], inplace=True)

        feature_extractor(df_drug, ['target', 'enzyme', 'smile'])
    elif ds_name == 'DS3':
        pass

    extraction['side'] = extraction['mechanism'] + ' ' + extraction['action']
    extraction.drop(columns=['mechanism', 'action'], inplace=True)
    df_drug.drop(columns=['id', 'index'], inplace=True)

    le = preprocessing.LabelEncoder()
    extraction['side'] = le.fit_transform(extraction['side'])

    return df_drug, extraction, le
