import os
import scipy.io
from scipy import stats
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, detrend, lfilter, savgol_filter
import neurokit2 as nk
import pywt
from sklearn.calibration import LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.discriminant_analysis import StandardScaler
from sklearn.mixture import GaussianMixture
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class ECGExtractor:
    def __init__(self, heart_signal, sampling_rate=500, denoise_first = True, wavelet = 'db5', peaks_method = 'dwt', missing_values=False):
        self.sampling_rate = sampling_rate
        self.original_signal = heart_signal  # assign original unprocessed signal
        self.heart_signal = self.preprocess_signal(heart_signal)
        _, self.rpeaks = nk.ecg_peaks(self.heart_signal, sampling_rate=self.sampling_rate)
        self.rpeaks = np.asarray(self.rpeaks['ECG_R_Peaks'], dtype=float)
        _, waves_peak = nk.ecg_delineate(self.heart_signal, self.rpeaks, sampling_rate=self.sampling_rate, method="dwt") # assess other methods
        self.ppeaks = np.asarray(waves_peak['ECG_P_Peaks'], dtype=float)
        self.qpeaks = np.asarray(waves_peak['ECG_Q_Peaks'], dtype=float)
        self.speaks = np.asarray(waves_peak['ECG_S_Peaks'], dtype=float)
        self.tpeaks = np.asarray(waves_peak['ECG_T_Peaks'], dtype=float)
        self.p_onsets = np.asarray(waves_peak['ECG_P_Onsets'], dtype=float)
        self.p_offsets = np.asarray(waves_peak['ECG_P_Offsets'], dtype=float)
        self.missing_values = missing_values

        
        
        if not self.missing_values:
            for i in range(len(self.rpeaks)):
                # Correct missing Q peaks
                if np.isnan(self.qpeaks[i]):
                    qp = int(self.rpeaks[i])
                    while qp > 0 and self.heart_signal[qp] >= self.heart_signal[qp-1]:
                        qp -= 1
                    self.qpeaks[i] = qp

                # Correct missing S peaks
                if np.isnan(self.speaks[i]):
                    sp = int(self.rpeaks[i])
                    while sp < len(self.heart_signal) - 1 and self.heart_signal[sp] >= self.heart_signal[sp+1]:
                        sp += 1
                    self.speaks[i] = sp
                
                # Correct missing P peaks
                if np.isnan(self.ppeaks[i]):
                    # look backwards from Q peak if Q is available; otherwise, from the R peak
                    search_start_index = int(self.qpeaks[i]) if not np.isnan(self.qpeaks[i]) else int(self.rpeaks[i])
                    pp = search_start_index
                    while pp > max(0, search_start_index - 200):  # Search window 
                        pp = int(pp)
                        if self.heart_signal[pp] > self.heart_signal[pp-1]:
                            break
                        pp -= 1
                    self.ppeaks[i] = pp if pp != search_start_index else -1  # assign P peak if found
                
                # Correct missing T peaks
                if np.isnan(self.tpeaks[i]):
                    # search start (after S peak) and end (before the next P peak) indices
                    search_start_index = int(self.speaks[i]) if not np.isnan(self.speaks[i]) else int(self.rpeaks[i])
                    # use the next P peak if available; otherwise, extend the search window
                    search_end_index = int(self.ppeaks[i + 1]) if i + 1 < len(self.ppeaks) and not np.isnan(self.ppeaks[i + 1]) else search_start_index + 150
                    # ensuring search_end_index does not go beyond the signal length
                    search_end_index = min(search_end_index, len(self.heart_signal) - 1)
                    
                    if search_start_index < search_end_index:
                        highest_peak_index = np.argmax(self.heart_signal[search_start_index:search_end_index]) + search_start_index
                        self.tpeaks[i] = highest_peak_index
                    else:
                        self.tpeaks[i] = -1

                # TODO: need to address the wrong asignment of T at t=10s
                        
    def calculate_intervals_and_segments(self):
        # calculating intervals and segments based on the peaks
        self.pr_intervals = []
        self.qt_intervals = []
        self.st_segments = []
        self.rr_intervals = []
        self.pp_intervals = []
        self.qrs_durations = []
        self.p_wave_durations = []
        self.pr_segments = []

        for i in range(len(self.rpeaks)):
            # RR interval
            if i > 0:
                self.rr_intervals.append((self.rpeaks[i] - self.rpeaks[i-1]) / self.sampling_rate)

            # PP interval (considering P peaks)
            if i < len(self.ppeaks) - 1 and not np.isnan(self.ppeaks[i]) and not np.isnan(self.ppeaks[i+1]):
                self.pp_intervals.append((self.ppeaks[i+1] - self.ppeaks[i]) / self.sampling_rate)

            # QT interval (from the beginning of Q to the end of T)
            if not np.isnan(self.qpeaks[i]) and not np.isnan(self.tpeaks[i]):
                self.qt_intervals.append((self.tpeaks[i] - self.qpeaks[i]) / self.sampling_rate)

            # ST segment (from the end of S to the beginning of T)
            if not np.isnan(self.speaks[i]) and not np.isnan(self.tpeaks[i]):
                self.st_segments.append((self.tpeaks[i] - self.speaks[i]) / self.sampling_rate)

            # QRS duration
            if not np.isnan(self.qpeaks[i]) and not np.isnan(self.speaks[i]):
                self.qrs_durations.append((self.speaks[i] - self.qpeaks[i]) / self.sampling_rate)

            # P-wave duration
            if i < len(self.p_onsets) and not np.isnan(self.p_onsets[i]) and i < len(self.p_offsets) and not np.isnan(self.p_offsets[i]):
                self.p_wave_durations.append((self.p_offsets[i] - self.p_onsets[i]) / self.sampling_rate)

            # PR segment (from end of P to beginning of Q)
            if i < len(self.p_offsets) and not np.isnan(self.p_offsets[i]) and not np.isnan(self.qpeaks[i]):
                self.pr_segments.append((self.qpeaks[i] - self.p_offsets[i]) / self.sampling_rate)

    def bandpass_filter(self, data, lowcut, highcut, order=4):
        nyq = 0.5 * self.sampling_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    def notch_filter(self, data, freq, Q=30):
        nyq = 0.5 * self.sampling_rate
        freq = freq / nyq
        b, a = iirnotch(freq, Q)
        y = filtfilt(b, a, data)
        return y

    def preprocess_signal(self, heart_signal):
        heart_signal = detrend(heart_signal) # detrend to remove baseline wander
        heart_signal = self.bandpass_filter(heart_signal, lowcut=0.5, highcut=45, order=4) # apply bandpass filter
        heart_signal = self.notch_filter(heart_signal, freq=50) # apply notch filter to remove power line noise
        heart_signal = self.wavelet_denoising(heart_signal) # apply wavelet denoising
        try:
            heart_signal = savgol_filter(heart_signal, window_length=31, polyorder=2) # apply Savitzky-Golay filter (note: window_length must be odd number)
        except:
            print(heart_signal)
        return heart_signal

    def wavelet_denoising(self, data, wavelet='db5', level=1):
        coeff = pywt.wavedec(data, wavelet, mode="per", level=level)
        sigma = (1/0.6745) * np.median(np.abs(coeff[-level] - np.median(coeff[-level]))) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        coeff[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeff[1:])
        return pywt.waverec(coeff, wavelet, mode="per")

    def calculate_beat(self):
        if len(self.rpeaks) < 2:
            return None
        dx = np.diff(self.rpeaks) / self.sampling_rate * 1000  # time between peaks in ms
        return 60 / np.mean(dx) * 1000  # bpm

    def variance_beat(self):
      return np.var(self.heart_beats) # beat rythm
    

    def point_extractor(self, start, end):
        points = []
        matrix_points = []
        for i, rpeak in enumerate(self.rpeaks):
            # Strip first and last peaks
            if i == 0 or i == len(self.rpeaks)-1:
                continue
            if start <= rpeak < end:
                P_peak = float(self.ppeaks[i]) if i < len(self.ppeaks) and self.ppeaks[i] is not None else np.nan
                Q_peak = float(self.qpeaks[i]) if i < len(self.qpeaks) and self.qpeaks[i] is not None else np.nan
                S_peak = float(self.speaks[i]) if i < len(self.speaks) and self.speaks[i] is not None else np.nan
                T_peak = float(self.tpeaks[i]) if i < len(self.tpeaks) and self.tpeaks[i] is not None else np.nan

                # nested = [( P_peak / ecg_extractor.sampling_rate) if not np.isnan(P_peak) else None,( Q_peak / ecg_extractor.sampling_rate) if not np.isnan(Q_peak) else None, ( rpeak / ecg_extractor.sampling_rate) if not np.isnan(rpeak) else None, ( S_peak / ecg_extractor.sampling_rate) if not np.isnan(S_peak) else None,( T_peak / ecg_extractor.sampling_rate) if not np.isnan(T_peak) else None,ecg_extractor.heart_signal[int(P_peak)] / 1000 if not np.isnan(P_peak) else None,ecg_extractor.heart_signal[int(Q_peak)] / 1000 if not np.isnan(Q_peak) else None, ecg_extractor.heart_signal[int(rpeak)] / 1000 if not np.isnan(rpeak) else None, ecg_extractor.heart_signal[int(S_peak)] / 1000 if not np.isnan(S_peak) else None,ecg_extractor.heart_signal[int(T_peak)] / 1000 if not np.isnan(T_peak) else None]
                nested = [P_peak if not np.isnan(P_peak) else None,Q_peak if not np.isnan(Q_peak) else None,rpeak,S_peak if not np.isnan(S_peak) else None,T_peak if not np.isnan(T_peak) else None]
                point = {
                    'peak_id': i,
                    'P': P_peak if not np.isnan(P_peak) else None,
                    'Q': Q_peak if not np.isnan(Q_peak) else None,
                    'R': rpeak,
                    'S': S_peak if not np.isnan(S_peak) else None,
                    'T': T_peak if not np.isnan(T_peak) else None,
                }
                points.append(point)
                matrix_points.append(nested)
        return points, np.array(matrix_points)


    def print_intervals_and_segments(self):
        print("Calculated Intervals and Segments:")

        # RR intervals
        if self.rr_intervals:
            print("\nRR Intervals (s):")
            for i, interval in enumerate(self.rr_intervals):
                print(f"  RR Interval {i+1}: {interval:.3f} s")

        # PP intervals
        if self.pp_intervals:
            print("\nPP Intervals (s):")
            for i, interval in enumerate(self.pp_intervals):
                print(f"  PP Interval {i+1}: {interval:.3f} s")

        # PR intervals
        if self.pr_intervals:
            print("\nPR Intervals (s):")
            for i, interval in enumerate(self.pr_intervals):
                print(f"  PR Interval {i+1}: {interval:.3f} s")

        # QT intervals
        if self.qt_intervals:
            print("\nQT Intervals (s):")
            for i, interval in enumerate(self.qt_intervals):
                print(f"  QT Interval {i+1}: {interval:.3f} s")

        # ST segments
        if self.st_segments:
            print("\nST Segments (s):")
            for i, segment in enumerate(self.st_segments):
                print(f"  ST Segment {i+1}: {segment:.3f} s")

        # QRS durations
        if self.qrs_durations:
            print("\nQRS Durations (s):")
            for i, duration in enumerate(self.qrs_durations):
                print(f"  QRS Duration {i+1}: {duration:.3f} s")

        # P-wave durations
        if self.p_wave_durations:
            print("\nP-wave Durations (s):")
            for i, duration in enumerate(self.p_wave_durations):
                print(f"  P-wave Duration {i+1}: {duration:.3f} s")

        # PR segments
        if self.pr_segments:
            print("\nPR Segments (s):")
            for i, segment in enumerate(self.pr_segments):
                print(f"  PR Segment {i+1}: {segment:.3f} s")

hidden = []
def dx_to_string(value):
    if value == 426177001:
        return "Sinus Bradycardia"
    if value == 55827005:
        return "Left Ventricular Hypertrophy"
    if value == 164889003:
        return "Atrial Fibrillation"
    if value == 59118001:
        return "Right Bundle Branch Block"
    if value == 164934002:
        return "T wave abnormal"
    if value == 164890007:
        return "Atrial Flutter"
    if value == 429622005:
        return "ST Depression"
    if value == 428750005:
        return "Nonspecific ST-T"
    if value == 426783006:
        return "Sinus rhythm"
    if value == 427084000:
        return "sinus tachycardia"
    if value == 270492004:
        return "First degree atrioventricular block"
    if value == 251173003:
        return "Atrial bigeminy"
    if value == 59931005:
        return "Inverted T wave"
    if value == 164909002:
        return "left bundle branch block"
    if value == 164912004:
        return "P wave abnormal"
    if value == 164917005:
        return "Q wave abnormal"
    if value == 284470004:
        return "Premature atrial contraction"
    if value == 233917008:
        return "Atrioventricular block"
    if value == 17338001:
        return "Ventricular premature contraction"
    if value == 698252002:
        return "Nonspecific intraventricular conduction disorder"
    if value == 251146004:
        return "Low QRS voltages"
    if value == 39732003:
        return "Left axis deviation"
    if value == 47665007:
        return "Right axis deviation"
    if value == 251199005:
        return "Counterclockwise cardiac rotation"
    if value == 164865005:
        return "myocardial infarction"
    if value == 164931005:
        return "ST segment"
    if value == 251180001:
        return "ventricular ectopics"
    if value == 111975006:
        return "Electrocardiogram abnormal"
    if value == 164937009:
        return "U wave abnormal"
    if value == 428417006:
        return "Early repolarization"
    if value == 164947007:
        return "Prolonged PR interval"
    if value == 251198002:
        return "Clockwise cardiac rotation"
    if value == 164873001:
        return "left ventricle hypertrophy"
    if value == 446358003:
        return "right atrial hypertrophy"
    if value == 67751000119106:
        return "right atrial high voltage"
    if value not in hidden:
        hidden.append(value)
        print(hidden)

def extract_dx(file_path):
    dx_values = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#Dx:'):
                # Extract and split the values after '#Dx:'
                dx_values = line.strip().split(': ')[1].split(',')
                break

    # print(dx_values)
    dx_values = [dx_to_string(int(value)) for value in dx_values]
    
    # Print the extracted Dx values
    # print(dx_values)
    return(dx_values)

## Detect additional outliers
def identify_outliers(d,t,l, threshold=3):
    # print(d)
    if len(d) != 0:
        l.append(len(d))
    z_scores = np.abs(stats.zscore(d))
    outliers = d[z_scores > threshold]
    if len(outliers) > 0:
        # print("Outlier found " + t,outliers)
        return True, l
    return False, l

def additional_outliers_exist(ecg_extractor):
    additional_outlier_exists = False
    
    lengths = []
    additional_outlier_exists, lengths = identify_outliers(np.array(ecg_extractor.rr_intervals),"RR",lengths)
    
    if additional_outlier_exists:
        return True
    additional_outlier_exists, lengths = identify_outliers(np.array(ecg_extractor.pp_intervals),"PP",lengths)
    if additional_outlier_exists:
        return True
    additional_outlier_exists, lengths = identify_outliers(np.array(ecg_extractor.pr_intervals),"PR",lengths)
    if additional_outlier_exists:
        return True
    additional_outlier_exists, lengths = identify_outliers(np.array(ecg_extractor.qt_intervals),"QT",lengths)
    if additional_outlier_exists:
        return True
    additional_outlier_exists, lengths = identify_outliers(np.array(ecg_extractor.st_segments),"ST",lengths)
    if additional_outlier_exists:
        return True
    additional_outlier_exists, lengths = identify_outliers(np.array(ecg_extractor.qrs_durations),"QRS",lengths)
    if additional_outlier_exists:
        return True
    additional_outlier_exists, lengths = identify_outliers(np.array(ecg_extractor.p_wave_durations),"P WAVE",lengths)
    if additional_outlier_exists:
        return True
    additional_outlier_exists, lengths = identify_outliers(np.array(ecg_extractor.pr_segments),"PR",lengths)
    if additional_outlier_exists:
        return True
    # print(lengths)
    match = np.all(lengths == 0)
    return match

def check_file(filename, show_plot=True):
    import matplotlib.pyplot as plt
    # Testing
    # Testing
    file_name = filename  # select file

    # Determine if healthy patient..
    is_healthy = False
    dx_values = extract_dx(file_name[:-3] + "hea")
    if len(dx_values) == 1:
        if dx_values[0] == "426177001":
            is_healthy = True

    data = scipy.io.loadmat(file_name)
    ecg_data = data['val'][2, :]  # select lead
    # print(ecg_data)
    # try:
    # print("hello")
    ecg_extractor = ECGExtractor(ecg_data)
    # print
    # except:
    #     print("EXCEPTION")
    #     print(ecg_data)
    #     return
    ecg_extractor.calculate_intervals_and_segments() # extract intervals and segments
    # ecg_extractor.print_intervals_and_segments() # print intervals and segments
    # print("test")
    start, end = 0, 5000 # specify start and end points 
    points, np_points = ecg_extractor.point_extractor(start, end) # extract points

    time_vector = np.linspace(start / ecg_extractor.sampling_rate, end / ecg_extractor.sampling_rate, end - start) # convert indices to time (s)

    heart_rate = ecg_extractor.calculate_beat() # normal range: 60-100 bpm
    if heart_rate is not None: # check classification of rhythm 
        if heart_rate > 100:
            rhythm_label = "Sinus Tachycardia"
        elif heart_rate < 60:
            rhythm_label = "Sinus Bradycardia"
        else:
            rhythm_label = "Normal Sinus Rhythm"

        heart_rate_text = f"Heart Rate: {heart_rate:.2f} bpm ({rhythm_label})"
    else:
        heart_rate_text = "Heart Rate: N/A"

    if show_plot:
        # Plotting
        plt.figure(figsize=(20, 10))

        # Plot 1: Original vs. Processed Signal
        plt.subplot(2, 1, 1)
        plt.plot(time_vector, ecg_extractor.original_signal[start:end] / 1000, label="Original ECG Signal", alpha=0.5)  # Convert μV to mV
        plt.plot(time_vector, ecg_extractor.heart_signal[start:end] / 1000, label="Processed ECG Signal", alpha=0.75)  # Convert μV to mV
        plt.title("Original vs. Processed ECG Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (mV)")
        plt.legend()

        plt.text(0.98, 0.10, heart_rate_text, horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # print("test")
    ## Detect point outliers 
    try:
        test = np.diff(np_points, axis = 1).T
        # print(test)
        outlier_exists = False
        for i in range(0,len(test)):
            mean = np.mean(test[i])
            std = np.std(test[i])

            threshold = 3
            outliers = []
            for x in test[i]:
                if std == 0:
                    continue
                z_score = (x - mean) / std
                
                # print("z score", z_score)
                if abs(z_score) > threshold:
                    outliers.append(x)
            # print("Mean: ",mean)
            # print("\nStandard deviation: ",std)
            # print("\nOutliers  : ", outliers)
            if len(outliers) > 0:
                outlier_exists = True
    except:
        outliers_exist = True

    # Detect additional outliers
    # addtional_outliers_exist = additional_outliers_exist(ecg_extractor)
    # if additional_outliers_exist:
    #     if is_healthy:
    #         # if we can't label this correctly and the patient has a normal beat, then we throw this out
    #         # But there how are we not just training a model to detect poorly extracted features?
    #         return
    
    if show_plot:
        # Plot 2: Processed Signal with Extracted Features
        plt.subplot(2, 1, 2)
        plt.plot(time_vector, ecg_extractor.heart_signal[start:end] / 1000, label="Processed ECG Signal")  # Convert μV to mV
        for point in points:
            # only plot if the index is not None and not np.nan, ensuring it's a valid integer
            if point['P'] is not None and not np.isnan(point['P']):
                plt.plot(point['P'] / ecg_extractor.sampling_rate, ecg_extractor.heart_signal[int(point['P'])] / 1000, 'go', label='P-peak' if point['peak_id'] == 0 else "")
            if point['Q'] is not None and not np.isnan(point['Q']):
                plt.plot(point['Q'] / ecg_extractor.sampling_rate, ecg_extractor.heart_signal[int(point['Q'])] / 1000, 'bo', label='Q-peak' if point['peak_id'] == 0 else "")
            if point['R'] is not None:  # assuming R-peaks are always present and valid
                plt.plot(point['R'] / ecg_extractor.sampling_rate, ecg_extractor.heart_signal[int(point['R'])] / 1000, 'ro', label='R-peak' if point['peak_id'] == 0 else "")
            if point['S'] is not None and not np.isnan(point['S']):
                plt.plot(point['S'] / ecg_extractor.sampling_rate, ecg_extractor.heart_signal[int(point['S'])] / 1000, 'mo', label='S-peak' if point['peak_id'] == 0 else "")
            if point['T'] is not None and not np.isnan(point['T']):
                plt.plot(point['T'] / ecg_extractor.sampling_rate, ecg_extractor.heart_signal[int(point['T'])] / 1000, 'yo', label='T-peak' if point['peak_id'] == 0 else "")

        " Uncomment to visualize extracted intervals and segments "
        for i, point in enumerate(points):
            # Color PR Interval
            if point['P'] is not None and not np.isnan(point['P']) and point['Q'] is not None and not np.isnan(point['Q']):
                plt.plot(time_vector[int(point['P']):int(point['Q'])], 
                         ecg_extractor.heart_signal[int(point['P']):int(point['Q'])] / 1000, 
                         'b', linewidth=2, label='PR Interval' if i == 0 else "")

            # Color QT Interval
            if point['Q'] is not None and not np.isnan(point['Q']) and point['T'] is not None and not np.isnan(point['T']):
                plt.plot(time_vector[int(point['Q']):int(point['T'])], 
                         ecg_extractor.heart_signal[int(point['Q']):int(point['T'])] / 1000, 
                         'g', linewidth=2, label='QT Interval' if i == 0 else "")

            # Color ST Segment
            if point['S'] is not None and not np.isnan(point['S']) and point['T'] is not None and not np.isnan(point['T']):
                plt.plot(time_vector[int(point['S']):int(point['T'])], 
                         ecg_extractor.heart_signal[int(point['S']):int(point['T'])] / 1000, 
                         'r', linewidth=2, label='ST Segment' if i == 0 else "")

            # Color QRS Complex 
            if point['Q'] is not None and not np.isnan(point['Q']) and point['S'] is not None and not np.isnan(point['S']):
                plt.plot(time_vector[int(point['Q']):int(point['S'])],
                         ecg_extractor.heart_signal[int(point['Q']):int(point['S'])] / 1000,
                         'c', linewidth=2, label='QRS Duration' if i == 0 else "")

            # Color P-wave 
            if i < len(ecg_extractor.p_onsets) and not np.isnan(ecg_extractor.p_onsets[i]) and i < len(ecg_extractor.p_offsets) and not np.isnan(ecg_extractor.p_offsets[i]):
                plt.plot(time_vector[int(ecg_extractor.p_onsets[i]):int(ecg_extractor.p_offsets[i])],
                         ecg_extractor.heart_signal[int(ecg_extractor.p_onsets[i]):int(ecg_extractor.p_offsets[i])] / 1000,
                         'm', linewidth=2, label='P-wave Duration' if i == 0 else "")    

            # Color PR Segment 
            if i < len(ecg_extractor.p_offsets) and not np.isnan(ecg_extractor.p_offsets[i]) and point['Q'] is not None and not np.isnan(point['Q']):
                plt.plot(time_vector[int(ecg_extractor.p_offsets[i]):int(point['Q'])],
                         ecg_extractor.heart_signal[int(ecg_extractor.p_offsets[i]):int(point['Q'])] / 1000,
                         'y', linewidth=2, label='PR Segment' if i == 0 else "")

        if outlier_exists:
            plt.title("Processed ECG Signal with Extracted Features" + filename + (" (OUTLIERS EXIST)" if outlier_exists else ""), color="red")
        else:
            plt.title("Processed ECG Signal with Extracted Features" + filename + (" (OUTLIERS EXIST)" if outlier_exists else ""))
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (mV)")

        # To avoid repeating labels in the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        plt.show()

    
    # print(np_points)
    # only take the first 5 beats, since some samples only have that many
    np_points = np_points.flatten()
    # Normalize
    # print(np_points)
    np_points = (np_points-np.min(np_points))/(np.max(np_points)-np.min(np_points))
    # print(np_points)
    # print(ecg_extractor.rr_intervals)
    # print(ecg_extractor.pp_intervals)
    # print(ecg_extractor.pr_intervals)
    # print(ecg_extractor.qt_intervals)
    # print(ecg_extractor.st_segments)
    # print(ecg_extractor.qrs_durations)
    # print(ecg_extractor.p_wave_durations)
    # print(ecg_extractor.pr_segments)
    # features = np.concatenate((np_points,ecg_extractor.rr_intervals[:5],ecg_extractor.pp_intervals[:5],ecg_extractor.pr_intervals[:5],ecg_extractor.qt_intervals[:5],ecg_extractor.st_segments[:5],ecg_extractor.qrs_durations[:5],ecg_extractor.p_wave_durations[:5],ecg_extractor.pr_segments[:5]))
    # print(features)


    


    ## K-means clustering


    # from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import normalize
    # from sklearn.preprocessing import MinMaxScaler
    # from sklearn.decomposition import PCA
    # from sklearn.cluster import KMeans
    # from sklearn.mixture import GaussianMixture
    # import matplotlib.pyplot as plt

    
    # print(np.std(test.T, axis = 1))
    # print(time_vector)
    # print(np_points)
    # scaler = MinMaxScaler()
    # features_scaled = scaler.fit_transform(test)
    # print(features_scaled)

    # features_normalized = (test-np.min(test))/(np.max(test)-np.min(test)) 
    # features_normalized_by_row = normalize(np_points, axis=1, norm='l1')
    # print(features_normalized)
    # print(features_normalized_by_row)
    # averaged_features = np.mean(features_normalized_by_row, axis=0)
    # print(averaged_features)
    # if len(np_points) != 50:
        # return

    # just using first dx_label for starters.

    if "Sinus Bradycardia" in dx_values:
        return np_points,"Sinus Bradycardia", is_healthy
    if "Left Ventricular Hypertrophy" in dx_values:
        return np_points,"Left Ventricular Hypertrophy", is_healthy
    if "Atrial Fibrillation" in dx_values:
        return np_points,"Atrial Fibrillation", is_healthy
    else:
        return
    # kmeans = KMeans(n_clusters=1)  # Adjust the number of clusters
    # clusters = kmeans.fit_predict(features_normalized_by_row)

    # plt.scatter(features_normalized_by_row[:, 0], features_normalized_by_row[:, 1], c=clusters, cmap='viridis')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # plt.title('ECG Data Clustering')
    # plt.colorbar(label='Cluster ID')
    # plt.show()


    # if isinstance(kmeans, KMeans):  # Check if k-means was used
    #     centers = kmeans.cluster_centers_
    #     distances = np.linalg.norm(features_normalized_by_row - centers[clusters], axis=1)
    #     threshold = np.percentile(distances, 98)  # Adjust the percentile as needed
    #     anomalies = features_normalized_by_row[distances > threshold]
    #     print(anomalies)

    #     # Plot anomalies
    #     plt.scatter(features_normalized_by_row[:, 0], features_normalized_by_row[:, 1], c='blue')
    #     plt.scatter(anomalies[:, 0], anomalies[:, 1], c='red', label='Anomalies')
    #     plt.xlabel('Feature 1')
    #     plt.ylabel('Feature 2')
    #     plt.title('Anomaly Detection in ECG Data')
    #     plt.legend()
    #     plt.show()

exceptions = []
count = 40000
i = 0
feature_set = []
labels = []
heath_labels = []
count1 = 0
count2 = 0
count3 = 0
for subdir, dirs, files in os.walk("../../WFDBRecords/"):
    if i == count:
        break
    for file in files:
        file_path = os.path.join(subdir, file)
        if file_path[-3:] != "mat":
            continue
        try:
            feats, l, is_healthy = check_file(file_path, show_plot=False)
            if l == "Sinus Bradycardia":
                if count1 > 1500:
                    continue
                else:
                    count1 = count1 + 1
            if l == "Left Ventricular Hypertrophy":
                if count2 > 1500:
                    continue
                else:
                    count2 = count2 + 1
            if l == "Atrial Fibrillation":
                if count3 > 1500:
                    continue
                else:
                    count3 = count3 + 1
            feature_set.append(feats)
            labels.append(l)
            heath_labels.append(is_healthy)
        except:
            pass
            # print("BAD FILE",file_path)
        i = i + 1
        if i == count:
            break
# for i in range(1,30):
#     try:
#         if i > 9:
#             check_file('./WFDBRecords/01/010/JS000' + str(i) + '.mat')
#         else:
#             check_file('./WFDBRecords/01/010/JS0000' + str(i) + '.mat')
#     except Exception as e:
#         print(e)
#         exceptions.append(e)
#         continue
print("DONE")
# print(exceptions)

import matplotlib.pyplot as plt

# print(feature_set)

# Resize everything to be the same size
max_size = max(len(arr) for arr in feature_set)
feature_set = [np.resize(arr, max_size) for arr in feature_set]

print(labels)
label_encoder = LabelEncoder()
health_labels = label_encoder.fit_transform(labels)

# health_labels_int = np.array(heath_labels, dtype=int)

# PCA
print(len(health_labels))
X_train, X_test, y_train, y_test = train_test_split(feature_set, health_labels, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Random Forest test
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_pca, y_train)

# Check accuracy
y_pred = classifier.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the model with PCA-applied features: {accuracy*100:.2f}%')



import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()  # Use seaborn for a nicer-looking plot

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


# print(len(feature_set))
# features_df = pd.DataFrame(feature_set, columns=['P1','P2','P3','P4','P5'])

gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(feature_set)
pred_labels = gmm.predict(feature_set)
# features_df['cluster labels'] = pred_labels

feature_set = np.array(feature_set)
plt.figure(figsize=(12, 6))

# Plot GMM cluster labels
plt.subplot(1, 2, 1)
plt.scatter(feature_set[:, 0], feature_set[:, 1], c=pred_labels, cmap='viridis', alpha=0.5)
plt.title('GMM Cluster Labels')

# Plot Ground Truth Labels
plt.subplot(1, 2, 2)
plt.scatter(feature_set[:, 0], feature_set[:, 1], c=heath_labels, cmap='viridis', alpha=0.5)
plt.title('Ground Truth Labels')

plt.show()


from sklearn.metrics import adjusted_rand_score


print("RAND",adjusted_rand_score(heath_labels, pred_labels))

def calculate_accuracy(gmm_labels, true_labels):
    correct_labels = np.sum(gmm_labels == true_labels)
    total_labels = len(true_labels)
    accuracy = (correct_labels / total_labels) * 100
    return accuracy

# Calculate the accuracy
accuracy = calculate_accuracy(pred_labels, heath_labels)
print(f"Percentage of correctly clustered labels: {accuracy}%")


# sns.pairplot(features_df, hue='cluster labels', palette='bright')
# plt.show()
print(labels[0:30])
# print(np_labels)
# print(heath_labels)
# a = np.array([pred_labels,heath_labels])
# print(np.transpose(feature_set))
# Get probabilities of each point belonging to each cluster
probabilities = gmm.predict_proba(feature_set)
# print(probabilities)
# print(feature_set)

x = 2
top_x_labels = np.argsort(-probabilities, axis=1)[:, :x]
print(top_x_labels)

