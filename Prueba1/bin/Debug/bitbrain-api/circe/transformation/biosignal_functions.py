
# %% ENVIRONEMENT
# We load the required libraries.
# _____________________________________________________________________________

import pandas as pd
import numpy as np

import mne

from circe.transformation.functions import GroupBy
from circe.utilities.typing import strn, floatn, listn, dictn, fi, rawt



# %% VERBOSITY
# We reduce the verbosity level of MNE methods.
# _____________________________________________________________________________

mne.set_log_level(verbose='ERROR')



# %% SIGNAL-PROCESSING FUNCTIONS
# We define several signal-processing functions.
# _____________________________________________________________________________

def PreprocessEEG(data,
                  metadata: dict,
                  scaling: float = 10e6,
                  annotations: dictn = None):
    data0 = data.loc[:, metadata['ch_names']] \
                .transpose() \
                .mul(scaling)
    info = mne.create_info(**metadata)
    raw = mne.io.RawArray(data=data0, info=info)
    if type(annotations) == dict:
        annotations = mne.Annotations(**annotations)
        raw.set_annotations(annotations)
    return raw

def ProcessEEG(raw: rawt,
               chs: listn = None,
               ch_bads: listn = None,
               chunk: fi = 2,
               amp_filter: floatn = 150e-6,
               freq_filter: strn = 'firwin2',
               target_values: listn = None,
               target_dict: dictn = None,
               plot_raw: bool = False,
               plot_psd: bool = False,
               plot_epochs: bool = False):
    if raw.preload == False:
        raw.load_data()
    if type(chs) == list:
        chs0 = chs
        raw.pick_channels(ch_names=chs0)
    elif chs is None:
        chs0 = raw.ch_names
    if type(ch_bads) == list:
        raw.info['bads'] = ch_bads
    if plot_raw == True:
        raw.plot(scalings={'eeg': amp_filter})
    if type(freq_filter) == dict:
        print('Applying band-pass frequency filter...')
        raw.filter(**freq_filter)
    if plot_psd == True:
        raw.plot_psd(tmax=np.inf, fmax=128, average=True)
    raw_freq = int(raw.info['sfreq'])
    if len(raw.annotations) > 0:
        print('Generating events from annotations...')
        (events, event_dict) \
        = mne.events_from_annotations(raw=raw, chunk_duration=chunk)
        print('Generating epochs from events...')
        epochs = mne.Epochs(raw=raw, events=events, event_id=event_dict,
                            reject_by_annotation=True, event_repeated='merge',
                            preload=True, reject={'eeg': amp_filter})
        consts = ['Target']
    else:
        print('Generating {}s sequential events...'.format(chunk))
        events = None
        epochs = mne.make_fixed_length_epochs(raw=raw, duration=chunk,
                                              preload=True)
        consts = []
    epochs_n = epochs.get_data().shape[0]
    epochs_t = epochs.get_data().shape[2]
    epochs_n0 = epochs_n * epochs_t
    print('{} epochs generated.'.format(epochs_n0))
    if epochs_n0 == 0:
        data0 = pd.DataFrame(columns=['Epoch'])
    else:
        if plot_epochs == True:
            epochs.plot(events=events, scalings={'eeg': amp_filter})
        data0 = epochs.to_data_frame(scalings={'eeg': amp_filter}) \
                      .rename(columns={'epoch': 'Epoch',
                                       'condition': 'Target'})
        if len(raw.annotations) > 0 and type(target_values) == list:
            data0 = data0.loc[data0['Target'].isin(values=target_values), :] \
                         .copy()
            epochs_n1 = len(data0)
            print('{}/{} epochs containing given target values.' \
                  .format(epochs_n1, epochs_n0))
        if len(raw.annotations) > 0 and type(target_dict) == dict:
            data0['Target'] = data0['Target'].replace(to_replace=target_dict)
        data0 = GroupBy(data=data0,
                        columns=['Epoch'],
                        feats=chs0,
                        funcs=['min', 'max', 'avg', 'std', 'median', 'kurt',
                               'fft', 'dwt_db2', 'spect'],
                        consts=consts,
                        fft_n=180,
                        dwt_mode='symmetric',
                        spect_freq=raw_freq)
    return data0
