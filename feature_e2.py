import scipy.io

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, detrend, lfilter
import neurokit2 as nk
import pywt

class ECGExtractor:
    def __init__(self, heart_signal, sampling_rate=500, missing_values=False):
        self.sampling_rate = sampling_rate
        self.original_signal = heart_signal  # Assign the original unprocessed signal
        self.heart_signal = self.preprocess_signal(heart_signal)
        _, self.rpeaks = nk.ecg_peaks(self.heart_signal, sampling_rate=self.sampling_rate)
        self.rpeaks = self.rpeaks['ECG_R_Peaks']
        _, waves_peak = nk.ecg_delineate(self.heart_signal, self.rpeaks, sampling_rate=self.sampling_rate, method="dwt")
        self.ppeaks = waves_peak['ECG_P_Peaks']
        self.qpeaks = waves_peak['ECG_Q_Peaks']
        self.speaks = waves_peak['ECG_S_Peaks']
        self.tpeaks = waves_peak['ECG_T_Peaks']
        self.missing_values = missing_values

        if not self.missing_values:
            for i in range(len(self.rpeaks)):
                # Correct missing Q peaks
                if np.isnan(self.qpeaks[i]):
                    qp = self.rpeaks[i]
                    while qp > 0 and self.heart_signal[qp] >= self.heart_signal[qp-1]:
                        qp -= 1
                    self.qpeaks[i] = qp

                # Correct missing S peaks
                if np.isnan(self.speaks[i]):
                    sp = self.rpeaks[i]
                    while sp < len(self.heart_signal) - 1 and self.heart_signal[sp] >= self.heart_signal[sp+1]:
                        sp += 1
                    self.speaks[i] = sp
                
                # Correct missing P peaks
                if np.isnan(self.ppeaks[i]):
                    # look backwards from Q peak if Q is available; otherwise, from the R peak
                    search_start_index = self.qpeaks[i] if not np.isnan(self.qpeaks[i]) else self.rpeaks[i]
                    pp = search_start_index
                    while pp > max(0, search_start_index - 200):  # Search window 
                        if self.heart_signal[pp] > self.heart_signal[pp-1]:
                            break
                        pp -= 1
                    self.ppeaks[i] = pp if pp != search_start_index else None  # assign P peak if found
                
                # Correct missing T peaks
                if np.isnan(self.tpeaks[i]):
                    # search start (after S peak) and end (before the next P peak) indices
                    search_start_index = self.speaks[i] if not np.isnan(self.speaks[i]) else self.rpeaks[i]
                    # use the next P peak if available; otherwise, extend the search window
                    search_end_index = self.ppeaks[i + 1] if i + 1 < len(self.ppeaks) and not np.isnan(self.ppeaks[i + 1]) else search_start_index + 150
                    # ensuring search_end_index does not go beyond the signal length
                    search_end_index = min(search_end_index, len(self.heart_signal) - 1)
                    
                    if search_start_index < search_end_index:
                        highest_peak_index = np.argmax(self.heart_signal[search_start_index:search_end_index]) + search_start_index
                        self.tpeaks[i] = highest_peak_index
                    else:
                        self.tpeaks[i] = None

        #         # TODO: need to address the wrong asignment of T at t=10s

    def bandpass_filter(self, data, lowcut, highcut, order=4):
        nyq = 0.5 * self.sampling_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    def bandpass_filter2(data, lowcut=5.0, highcut=15.0, fs=250.0, order=1):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y


    def notch_filter(self, data, freq, Q=30):
        nyq = 0.5 * self.sampling_rate
        freq = freq / nyq
        b, a = iirnotch(freq, Q)
        y = filtfilt(b, a, data)
        return y
    
    def differentiate(signal):
        diff = np.diff(signal)
        return np.append(diff, 0)  # Append 0 to make the array the same size as the input
    
    def square(signal):
        return np.square(signal)

    def preprocess_signal(self, heart_signal):
        heart_signal = detrend(heart_signal) # detrend to remove baseline wander
        heart_signal = self.bandpass_filter(heart_signal, lowcut=0.5, highcut=45, order=4) # apply bandpass filter
        heart_signal = self.notch_filter(heart_signal, freq=50) # apply notch filter to remove power line noise
        heart_signal = self.wavelet_denoising(heart_signal) # apply wavelet denoising
        return heart_signal
    
    def moving_window_integration(signal, window_size=30):
        window = np.ones(window_size) / float(window_size)
        return np.convolve(signal, window, 'same')

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

    def point_extractor(self, start, end):
        points = []
        matrix_points = []
        for i, rpeak in enumerate(self.rpeaks):
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


def check_file(filename):
    import matplotlib.pyplot as plt
    # Testing
    file_name = filename  # select file
    data = scipy.io.loadmat(file_name)
    ecg_data = data['val'][11, :]  # select lead

    ecg_extractor = ECGExtractor(ecg_data)

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


    ## EXTRA

    test = np.diff(np_points, axis = 1).T
    print(test)
    outlier_exists = False
    for i in range(0,len(test)):
        mean = np.mean(test[i])
        std = np.std(test[i])

        threshold = 3
        outliers = []
        for x in test[i]:
            z_score = (x - mean) / std
            if abs(z_score) > threshold:
                outliers.append(x)
        print("Mean: ",mean)
        print("\nStandard deviation: ",std)
        print("\nOutliers  : ", outliers)
        if len(outliers) > 0:
            outlier_exists = True



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



    print(points)
    print(np_points)

    ## K-means clustering


    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import normalize
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    import matplotlib.pyplot as plt

    
    # print(np.std(test.T, axis = 1))
    # print(time_vector)
    # print(np_points)
    # scaler = MinMaxScaler()
    # features_scaled = scaler.fit_transform(test)
    # print(features_scaled)

    # features_normalized = (test-np.min(test))/(np.max(test)-np.min(test)) 
    # features_normalized_by_row = normalize(np_points, axis=1, norm='l1')
    # print(features_normalized)
    # # print(features_normalized_by_row)

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

for i in range(1,104):
    try:
        if i > 9:
            check_file('./WFDBRecords/01/010/JS000' + str(i) + '.mat')
        else:
            check_file('./WFDBRecords/01/010/JS0000' + str(i) + '.mat')
    except:
        continue