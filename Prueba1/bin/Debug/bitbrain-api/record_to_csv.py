# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 12:00:00 2022

@author: vgini
"""

# %% ENVIRONEMENT
# We load the required libraries.
# _____________________________________________________________________________

from contextlib import ExitStack
import csv
import sys
import time

import numpy as np
import pandas as pd
from bbt import Signal, Device, SensorType, ImpedanceLevel
from circe.classification.methods import ClassContext



# %% SETTING PARAMETERS
# We define the parameters for this script execution.
# _____________________________________________________________________________

model_path = 'models/'


# %% CUSTOM SDK
# Custom BitBrain SDK for EEG data processing in streaming.
# _____________________________________________________________________________

def try_to(condition, action, tries, socket,message=None):
    t = 0
    while (not condition() and t < tries):
        t += 1
        if message:
            print("{} ({}/{})".format(message, t, tries))
            str = "{} ({}/{})".format(message, t, tries)
            time.sleep(1)
            socket.emit("msg",str)
        action()
    return condition()


def config_signals(device):
    signals = device.get_signals()
    for s in signals:
        s.set_mode(1)


def csv_filename(signal_number, signal_type):
    return f"signal_{signal_number}({signal_type}).csv"


def record_one(device):
    sequence, battery, flags, signals = device.read()
    ts = time.time_ns()
    data_streaming = pd.DataFrame(columns=['ts', 'EEG-ch1', 'EEG-ch2', 'EEG-ch3',
                                           'EEG-ch4', 'EEG-ch5', 'EEG-ch6', 'EEG-ch7',
                                           'EEG-ch8'])
    n_rows = device.read_data_size()
    signals = signals[:-25]
    signals = np.array(signals).reshape(8, len(signals)//8).tolist()
    for i in range(0, len(signals)):
        data_streaming = data_streaming.append({'ts': ts,
                                                'EEG-ch1': signals[0][i],
                                                'EEG-ch2': signals[1][i],
                                                'EEG-ch3': signals[2][i],
                                                'EEG-ch4': signals[3][i],
                                                'EEG-ch5': signals[4][i],
                                                'EEG-ch6': signals[5][i],
                                                'EEG-ch7': signals[6][i],
                                                'EEG-ch8': signals[7][i]}, ignore_index=True)


    return data_streaming


def record_data(device, length):
    # create the csv files
    with ExitStack() as stack:
        active_signals = [s for s in device.get_signals() if s.mode() != 0]
        # open csv files
        files = [stack.enter_context(open(csv_filename(i, s.type()), 'w', newline='')) for i, s in
                 enumerate(active_signals)]

        # write headers
        writers = [csv.writer(f) for f in files]
        for (s, w) in zip(active_signals, writers):
            common_header = ["timestamp", "sequence", "battery", "flags"]
            channels_header = [f"channel_{i}" for i in range(s.channels())]
            w.writerow(common_header + channels_header)

        # record data
        device.start()
        f = int(device.get_frequency())
        fs =  f * 8

        data = pd.DataFrame(columns=['ts', 'EEG-ch1', 'EEG-ch2', 'EEG-ch3',
                                           'EEG-ch4', 'EEG-ch5', 'EEG-ch6', 'EEG-ch7',
                                           'EEG-ch8'])
        for i in range(length * f):
            data0=record_one(device)
            data = data.append(data0)

        eeg = ClassContext(data=data,
                           target='Target',
                           eeg_process=True,
                           metadata={'sfreq': fs,
                                     'ch_names': ['EEG-ch1', 'EEG-ch2',
                                                  'EEG-ch3', 'EEG-ch4',
                                                  'EEG-ch5', 'EEG-ch6',
                                                  'EEG-ch7', 'EEG-ch8'],
                                     'ch_types': 'eeg'},
                           chunk=2,
                           freq_filter={'l_freq': 1,
                                        'h_freq': 45,
                                        'h_trans_bandwidth': 5,
                                        'fir_design': 'firwin2',
                                        'phase': 'zero-double'},
                           scaling=1e-6)
        device.stop()
    print("Stopped: ", not device.is_running())
    return eeg


def call_model(eeg, feats_folder, model_folder):
    result = eeg.normalize() \
                .select_k_feats(load_folder=feats_folder) \
                .load_apply(folder=model_folder)
    return result

# __init__.py

if __name__ == "__main__":

    if (len(sys.argv) <= 1):
        print("Usage: " + sys.argv[0] + " <device name> [time (s) default = 10]")
        exit(0)

    name = sys.argv[1]

    length = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    with Device.create_bluetooth_device(name) as device:
        if not try_to(device.is_connected, device.connect, 10, "Connecting to {}".format(name)):
            print("unable to connect")
            exit(1)
        print("Connected")

        print(f"Recording {length} seconds of data into csv files from device {name}")

        config_signals(device)
        eeg = record_data(device, length)
        result = call_model(eeg=eeg, feats_folder='models/feats01/',
                            model_folder='models/model01')
        print('Inference result: {}'.format(result[0]))

        if not try_to(lambda: not device.is_connected(), device.disconnect, 10):
            print("unable to disconnect")
            exit(1)
        print("Connected")
