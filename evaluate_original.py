import json
import pandas as pd
import numpy as np

def evaluate_on_patient(patient_id, prediction_function, threshold=0.6, integration_window=600000, occurance_period=3600000):
    index_period = occurance_period/30000
    labels = pd.read_csv(f"data\data\{patient_id}\{patient_id}_labels.csv")
    with open("labels.json") as f:
        label_notes = json.load(f)

    start_time = labels["startTime"][0]
    seizure_start_times = labels["labels.startTime"] - start_time
    seizure_indexes = np.floor(seizure_start_times/30000)
    seizure_indexes = seizure_indexes.loc[np.array([label_notes[note] for note in labels["labels.note"]])==1]

    n_segments = int((labels["duration"][0]-labels["duration"][0]%30000)/30000)

    predictions = pd.DataFrame(data={"prediction":np.nan, "time":np.nan, "alarm": False}, index=pd.RangeIndex(start=0, stop=n_segments, step=1))
    for segment_id in range(n_segments):
        try:
            segment = pd.read_parquet(f"data/segments/test/{patient_id}/{patient_id}_test_segment_{segment_id}.parquet")
        except FileNotFoundError:
            continue
        predictions.loc[segment_id, "time"] = segment.index[-1] + 250
        predictions.loc[segment_id, "prediction"] = prediction_function(segment)
    predictions = predictions.loc[:predictions["prediction"].last_valid_index()]

    predictions["Moving Average"] = predictions["prediction"].rolling(int(integration_window/30000)).mean()
    predictions["alarm"] = False

    shift = 0
    while not predictions[(predictions["Moving Average"]>threshold)&~predictions["alarm"]].empty:
        shift += 1
        try:
            start_index = predictions.loc[(predictions["Moving Average"]>threshold)&~predictions["alarm"]].index[shift]
        except IndexError:
            if len(predictions.loc[(predictions["Moving Average"]>threshold)&~predictions["alarm"]]) == shift - 1:
                break
            else:
                raise Exception
        end_index = start_index + index_period - 1
        predictions.loc[start_index:end_index, "alarm"] = True

    seizure_indexes = seizure_indexes[seizure_indexes < predictions.index[-1]]

    sensitivity = predictions.loc[seizure_indexes, "alarm"].sum()/len(seizure_indexes)
    time_in_warning = predictions["alarm"].sum()/len(predictions)
    improvement_over_chance = sensitivity - time_in_warning

    return {"S": sensitivity, "TiW": time_in_warning, "IoC": improvement_over_chance}