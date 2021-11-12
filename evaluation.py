import torch
import pandas as pd
import numpy as np

from scipy.stats import binom

def predict(model, segments):
    predictions = pd.Series(index=(3630000+30000*np.arange(len(segments))))
    with torch.no_grad():
        predictions[:] = model(segments).numpy()>0.5
    return predictions


def get_alarm_triggers(predictions, integration_windows, thresholds):
    alarm_triggers = pd.DataFrame(index=predictions.index)
    for integration_window in integration_windows:
        window_average = predictions.rolling(int(integration_window/30000)).mean()
        for threshold in thresholds:
            parameter_triggers = window_average > threshold
            alarm_triggers[f"{integration_window}_{threshold}"] = parameter_triggers
    return alarm_triggers


def p_value(n_correct, n_seizures, chance_sensitivity):
    if n_correct/n_seizures > chance_sensitivity:
        k_f = np.floor(2*n_seizures*chance_sensitivity-n_correct)
        return 1-binom.cdf(n_correct-1, n_seizures, chance_sensitivity)+binom.cdf(k_f, n_seizures, chance_sensitivity)
    elif  n_correct/n_seizures < chance_sensitivity:
        k_c = np.ceil(2*n_seizures*chance_sensitivity-n_correct)
        return 1-binom.cdf(k_c-1, n_seizures, chance_sensitivity)+binom.cdf(n_correct, n_seizures, chance_sensitivity)
    else:
        return 100


def get_metrics(seizure_start_times, triggers, timer_duration, detection_interval, num_seizures, recording_duration):
    metrics = pd.DataFrame()
    for col in triggers.columns:
        integration_window, threshold = col.split("_")
        num_seizures_detected = 0
        total_time_in_warning = 0
        
        current_alarm_end = 0
        for trigger_time in triggers[col].index[triggers[col]]:
            if trigger_time > current_alarm_end:
                current_alarm_start = trigger_time
                current_alarm_end = current_alarm_start + timer_duration
                num_seizures_detected += ((seizure_start_times>current_alarm_start)&(seizure_start_times<current_alarm_end)).sum()
                total_time_in_warning += timer_duration

        # try:
        #     alarm_start = triggers[col].index[triggers[col]][0]
        # except IndexError:
        #     continue
        # alarm_end = alarm_start + timer_duration
        # for trigger_time in triggers[col].index[triggers[col]]:
        #     if trigger_time > alarm_end:
        #         num_seizures_detected += ((seizure_start_times>alarm_start)&(seizure_start_times<alarm_end)).sum()
        #         total_time_in_warning += alarm_end-alarm_start
        #         alarm_start = trigger_time
        #     alarm_end = trigger_time + timer_duration
        
        sensitivity = num_seizures_detected/num_seizures
        time_in_warning = total_time_in_warning/recording_duration

        rate_parameter = (-1/timer_duration)*np.log(1-time_in_warning)
        chance_sensitivity = 1 - np.exp((-rate_parameter*timer_duration)+(1 - np.exp(-rate_parameter*detection_interval)))
        improvement_over_chance = sensitivity - chance_sensitivity
        p = p_value(num_seizures_detected, num_seizures, chance_sensitivity)
        m = pd.DataFrame(index=[col], data={"Integration Window":integration_window, "threshold":threshold, "n":num_seizures_detected, "N":num_seizures, "S":sensitivity, "Sc":chance_sensitivity, "TiW":time_in_warning, "IoC":improvement_over_chance, "p":p})
        metrics = pd.concat([metrics, m])
    return metrics


def evaluate(model, patient, integration_windows, thresholds, timer_duration, detection_interval):
    seizures = pd.read_csv("data\seizures.csv", index_col=0)
    seizure_start_times = seizures.loc[seizures["patient_id"]==patient, "start_time"].unique()

    segments = torch.load(f"data/test/{patient}_test_segments.pt")
    predictions = predict(model, segments)

    num_seizures = len(seizure_start_times)
    recording_duration = predictions.index[-1] - predictions.index[0]
    triggers = get_alarm_triggers(predictions, integration_windows, thresholds)
    metrics = get_metrics(seizure_start_times, triggers, timer_duration, detection_interval, num_seizures, recording_duration)

    return metrics