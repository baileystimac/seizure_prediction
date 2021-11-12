import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from glob import glob
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import loggers as pl_loggers
from models import MeiselClassifier
from evaluation import evaluate
import warnings

warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
interictal_data = torch.load(r"C:\Users\Bailey\OneDrive\Documents\Thesis\data\interictal.pt")
preictal_data = torch.load(r"C:\Users\Bailey\OneDrive\Documents\Thesis\data\preictal.pt")
patient_ids = list(interictal_data.keys())

for test_patient in patient_ids:
    ##----------------------------------------OVERSAMPLING-----------------------------------------------------------------------------##
    train_patients = [patient for patient in patient_ids if patient != test_patient]
    train_X = []
    train_y = []
    for patient in train_patients:
        patient_interictal_X = interictal_data[patient]
        patient_preictal_X = preictal_data[patient]
        # Oversampling preictal data
        sample_indexes = np.random.randint(low=0, high=len(patient_preictal_X), size=len(patient_interictal_X))
        patient_preictal_X = patient_preictal_X[sample_indexes]

        patient_interictal_y = torch.zeros(patient_interictal_X.shape[0])
        patient_preictal_y = torch.ones(patient_preictal_X.shape[0])
        

        patient_train_X = torch.cat([patient_interictal_X, patient_preictal_X])
        patient_train_y = torch.cat([patient_interictal_y, patient_preictal_y])

        train_X.append(patient_train_X)
        train_y.append(patient_train_y)

    train_X = torch.cat(train_X)
    train_y = torch.cat(train_y)

    ##----------------------------------------UNDERSAMPLING-----------------------------------------------------------------------------##
    # interictal_train_X = torch.cat([patient_data for patient_id, patient_data in interictal_data.items() if patient_id!=test_patient])
    # preictal_train_X = torch.cat([patient_data for patient_id, patient_data in preictal_data.items() if patient_id!=test_patient])
    # # Undersample interictal data
    # interictal_train_X = interictal_train_X[::int(len(interictal_train_X)/len(preictal_train_X))]
    # train_X = torch.cat([interictal_train_X, preictal_train_X])

    # interictal_train_y = torch.zeros(interictal_train_X.shape[0])
    # preictal_train_y = torch.ones(preictal_train_X.shape[0])
    # train_y = torch.cat([interictal_train_y, preictal_train_y])
    ##----------------------------------------------------------------------------------------------------------------------------------##

    train_dataset = TensorDataset(train_X, train_y)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    try:
        model_path = glob(f"C:/Users/Bailey/OneDrive/Documents/Thesis/experiments/oversampling/meisel/{test_patient}/version_0/checkpoints/*.ckpt")[0]
    except IndexError:
        model = MeiselClassifier()
        tb_logger = pl_loggers.TensorBoardLogger("C:/Users/Bailey/OneDrive/Documents/Thesis/experiments/oversampling", name=f"meisel/{test_patient}")
        trainer = pl.Trainer(gpus=1, max_epochs=200, logger=tb_logger)
        trainer.fit(model, train_dataloader)