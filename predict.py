import numpy as np
import pandas as pd
from obspy import read
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from scipy import signal
from matplotlib import cm
from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import highpass
from obspy.signal.trigger import classic_sta_lta, plot_trigger, trigger_onset

"""Parameters"""
noise_level = 15
sta_len = 40
lta_len = 1000

submit = {"filename":[], "time_rel(sec)":[]}
directory = ['./data/lunar/test/data/S12_GradeB', './data/lunar/test/data/S15_GradeA', './data/lunar/test/data/S15_GradeB', './data/lunar/test/data/S16_GradeA', './data/lunar/test/data/S16_GradeB', './data/mars/test/data']
for cat_directory in directory:
    for file_name in os.listdir(cat_directory):
        if file_name.endswith(".mseed"):
            file_path = os.path.join(cat_directory, file_name)
            st = read(file_path)
            minfreq = 0.5
            maxfreq = 1.0
            tr = st.traces[0].copy()
            tr_times = tr.times()
            tr_data = tr.data
            st_filt = st.copy()
            st_filt = st_filt.filter('bandpass',freqmin=minfreq,freqmax=maxfreq)
            tr_filt = st_filt.traces[0].copy()
            tr_times_filt = tr_filt.times()
            tr_data_filt = tr_filt.data
            stream = st_filt.sort()
            data = []
            for trace in stream:
                data.append(trace.data)
            data = np.array(data).T
            data_id = stream[0].get_id()[:-1]
            timestamp = stream[0].stats.starttime.datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")

            """Add noise"""
            #noisy_data = add_noise(data,min_freq=0.4, max_freq=0.6, noise_level=1e-6)
            noisy_data = data + np.random.randn(*data.shape)*np.max(data)/noise_level
            
            """STA/LTA"""
            df = tr_filt.stats.sampling_rate
            cft = classic_sta_lta(noisy_data[:,0], int(sta_len * df), int(lta_len * df))
            
            """Selection rule"""
            thr_on = 4
            thr_off = 0.5
            triggers = []
            while True:
                on_off = np.array(trigger_onset(cft, thr_on, thr_off, max_len_delete=2))
                #on_off.sort(key=map)
                if len(on_off) == 1:
                    if (noisy_data[on_off[0][0]:on_off[0][0]+50].max() > noisy_data[on_off[0][0]-500:on_off[0][0]+500].mean() + 2.5*noisy_data[on_off[0][0]-500:on_off[0][0]+500].std()):
                        thr_on -= 0.5
                        continue
                    else:
                        triggers.append(on_off[0])
                        break
                elif len(on_off) == 0:
                    thr_on -= 0.5
                    continue
                else:
                    for i in range(len(on_off)):
                        if (noisy_data[on_off[i][0]:on_off[i][0]+50].max() > noisy_data[on_off[i][0]-500:on_off[i][0]+500].mean() + 2.5*tr_data_filt[on_off[i][0]-500:on_off[i][0]+500].std()):
                            continue
                        else:
                            if len(triggers) < 2:
                                triggers.append(on_off[i])
                    if len(triggers) > 0:
                        break
                    else:
                        thr_on -= 0.5
                        continue

            """Save figure"""
            fig = plt.figure(figsize=(10, 5))
            ax = plt.subplot(1, 1, 1)
            for i in triggers:
                ax.axvline(x = tr_times_filt[i[0]], color='red', label='Trig. On')
            ax.plot(tr_times,tr_data_filt)
            ax.set_xlim([min(tr_times),max(tr_times)])
            ax.legend()
            plt.savefig(f"./figs/{file_name[:-6]}.png")
            plt.clf()

            for i in triggers:
                submit["filename"].append(file_name[:-6])
                submit["time_rel(sec)"].append(tr_times_filt[i[0]])

df = pd.DataFrame.from_dict(submit)
df.to_csv("prediction.csv",index=False)