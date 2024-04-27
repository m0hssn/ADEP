from Utils import DataLoading
from Utils.trainer import  ADEPTrainer

if __name__ == '__main__':
    ds_name = 'DS1'
    df_drug, extraction, le = DataLoading.get_dataset(ds_name)
    input_size = 0

    if ds_name == 'DS1':
        input_size = 25658
    elif ds_name == 'DS2':
        input_size = 7738

    trainer = ADEPTrainer(df_drug=df_drug, extraction=extraction, Input_size=input_size
                          , label_size=len(extraction['side'].unique()))
    trainer.train()
