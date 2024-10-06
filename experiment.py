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
noise_level = 10
sta_len = 40
lta_len = 800

# lunar data
mars_directory = ''
lunar_file = './data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'
mars_file = './data/mars/training/catalogs/Mars_InSight_training_catalog_final.csv'

loss = []
#for noise_level in [10,20,30]:
#    for sta_len in [40,60,80]:
#        for lta_len in [600,800,1000]:
for file in [lunar_file, mars_file]:
    cat = pd.read_csv(file)
    for i in range(len(cat)):
        """Read data"""
        row = cat.iloc[i]
        arrival_time = datetime.strptime(row['time_abs(%Y-%m-%dT%H:%M:%S.%f)'],'%Y-%m-%dT%H:%M:%S.%f')
        arrival_time_rel = row['time_rel(sec)']
        test_filename = row.filename
        if file == lunar_file:
            data_directory = './data/lunar/training/data/S12_GradeA/'
            mseed_file = f'{data_directory}{test_filename}.mseed'
        else:
            data_directory = './data/mars/training/data/'
            mseed_file = f'{data_directory}{test_filename[:-4]}.mseed'
    
        """Filter"""
        minfreq = 0.5
        maxfreq = 1.0
        st = read(mseed_file)
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
        
        starttime = tr.stats.starttime.datetime
        arrival = (arrival_time - starttime).total_seconds()

        """STA/LTA"""
        df = tr_filt.stats.sampling_rate
        cft = classic_sta_lta(noisy_data[:,0], int(sta_len * df), int(lta_len * df))
        
        """Selection rule"""
        thr_on = 4
        thr_off = 0.5
        triggers = []
        while True:
            on_off = np.array(trigger_onset(cft, thr_on, thr_off))
            if len(on_off) == 1:
                if (noisy_data[on_off[0][0]:on_off[0][0]+50].max() > noisy_data[on_off[0][0]-500:on_off[0][0]+500].mean() + 3*noisy_data[on_off[0][0]-500:on_off[0][0]+500].std()):
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
                    if (noisy_data[on_off[i][0]:on_off[i][0]+50].max() > noisy_data[on_off[i][0]-500:on_off[i][0]+500].mean() + 3*tr_data_filt[on_off[i][0]-500:on_off[i][0]+500].std()):
                        continue
                    else:
                        triggers.append(on_off[i])
                break

        """Save figure"""
        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot(1, 1, 1)
        for i in triggers:
            ax.axvline(x = tr_times_filt[i[0]], color='red', label='Trig. On')
        ax.plot(tr_times,tr_data_filt)
        ax.axvline(x = arrival_time_rel, color = 'purple', label = "ground truth")
        ax.set_xlim([min(tr_times),max(tr_times)])
        ax.legend()
        plt.savefig(f"./val/{row.filename}.png")
        plt.clf()

        loss.append(abs(tr_times_filt[triggers[0][0]] - arrival_time_rel))
         
print("noise_level:", noise_level)
print("sta_len:", sta_len)
print("lta_len:", lta_len)
print("MAE:", (sum(loss)/len(loss)))