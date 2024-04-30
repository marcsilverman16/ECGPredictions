import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from ecg_denoising import ECGdenoising
import neurokit2 as nk
import pywt


class ECGExtractor:
    def __init__(self, heart_signal, sampling_rate=1000, missing_values=False):
        self.sampling_rate = sampling_rate
        self.original_signal = heart_signal  # assign original unprocessed signal
        denoiser = ECGdenoising(self.original_signal)
        self.heart_signal = denoiser.heart_signal
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
                    self.ppeaks[i] = pp if pp != search_start_index else None  # assign P peak if found
                
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
                        self.tpeaks[i] = np.nan

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

    # def bandpass_filter(self, data, lowcut, highcut, order=4):
    #     nyq = 0.5 * self.sampling_rate
    #     low = lowcut / nyq
    #     high = highcut / nyq
    #     b, a = butter(order, [low, high], btype='band')
    #     y = filtfilt(b, a, data)
    #     return y

    # def notch_filter(self, data, freq, Q=30):
    #     nyq = 0.5 * self.sampling_rate
    #     freq = freq / nyq
    #     b, a = iirnotch(freq, Q)
    #     y = filtfilt(b, a, data)
    #     return y

    # def preprocess_signal(self, heart_signal):
    #     heart_signal = detrend(heart_signal) # detrend to remove baseline wander
    #     heart_signal = self.bandpass_filter(heart_signal, lowcut=0.5, highcut=45, order=4) # apply bandpass filter
    #     heart_signal = self.notch_filter(heart_signal, freq=50) # apply notch filter to remove power line noise
    #     heart_signal = self.wavelet_denoising(heart_signal) # apply wavelet denoising
    #     heart_signal = savgol_filter(heart_signal, window_length=31, polyorder=2) # apply Savitzky-Golay filter (note: window_length must be odd number)
    #     return heart_signal

    # def wavelet_denoising(self, data, wavelet='db5', level=1):
    #     coeff = pywt.wavedec(data, wavelet, mode="per", level=level)
    #     sigma = (1/0.6745) * np.median(np.abs(coeff[-level] - np.median(coeff[-level]))) / 0.6745
    #     threshold = sigma * np.sqrt(2 * np.log(len(data)))
    #     coeff[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeff[1:])
    #     return pywt.waverec(coeff, wavelet, mode="per")

    def calculate_beat(self):
        if len(self.rpeaks) < 2:
            return None
        dx = np.diff(self.rpeaks) / self.sampling_rate * 1000  # time between peaks in ms
        return 60 / np.mean(dx) * 1000  # bpm

    def point_extractor(self, start, end):
        points = []
        for i, rpeak in enumerate(self.rpeaks):
            if start <= rpeak < end:
                P_peak = float(self.ppeaks[i]) if i < len(self.ppeaks) and self.ppeaks[i] is not None else np.nan
                Q_peak = float(self.qpeaks[i]) if i < len(self.qpeaks) and self.qpeaks[i] is not None else np.nan
                S_peak = float(self.speaks[i]) if i < len(self.speaks) and self.speaks[i] is not None else np.nan
                T_peak = float(self.tpeaks[i]) if i < len(self.tpeaks) and self.tpeaks[i] is not None else np.nan

                point = {
                    'peak_id': i,
                    'P': P_peak if not np.isnan(P_peak) else None,
                    'Q': Q_peak if not np.isnan(Q_peak) else None,
                    'R': rpeak,
                    'S': S_peak if not np.isnan(S_peak) else None,
                    'T': T_peak if not np.isnan(T_peak) else None,
                }
                points.append(point)
        
        
        return points

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

# Testing
# file_name = '../raw_data/WFDBRecords/01/018/JS00909.mat'  # select file
# data = scipy.io.loadmat(file_name)
# print(data['val'].shape)
# ecg_data = data['val'][6, :]  # select lead

# ecg_extractor = ECGExtractor(ecg_data)
# ecg_extractor.calculate_intervals_and_segments() # extract intervals and segments
# ecg_extractor.print_intervals_and_segments() # print intervals and segments

# start, end = 0, 5000 # specify start and end points 
# points = ecg_extractor.point_extractor(start, end) # extract points

# time_vector = np.linspace(start / ecg_extractor.sampling_rate, end / ecg_extractor.sampling_rate, end - start) # convert indices to time (s)

# heart_rate = ecg_extractor.calculate_beat() # normal range: 60-100 bpm
# if heart_rate is not None: # check classification of rhythm 
#     if heart_rate > 100:
#         rhythm_label = "Sinus Tachycardia"
#     elif heart_rate < 60:
#         rhythm_label = "Sinus Bradycardia"
#     else:
#         rhythm_label = "Normal Sinus Rhythm"
    
#     heart_rate_text = f"Heart Rate: {heart_rate:.2f} bpm ({rhythm_label})"
# else:
#     heart_rate_text = "Heart Rate: N/A"

# # Plotting
# plt.figure(figsize=(20, 10))

# # Plot 1: Original vs. Processed Signal
# plt.subplot(2, 1, 1)
# plt.plot(time_vector, ecg_extractor.original_signal[start:end] / 1000, label="Original ECG Signal", alpha=0.5)  # Convert μV to mV
# plt.plot(time_vector, ecg_extractor.heart_signal[start:end] / 1000, label="Processed ECG Signal", alpha=0.75)  # Convert μV to mV
# plt.title("Original vs. Processed ECG Signal")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude (mV)")
# plt.legend()

# plt.text(0.98, 0.10, heart_rate_text, horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# # Plot 2: Processed Signal with Extracted Features
# plt.subplot(2, 1, 2)
# plt.plot(time_vector, ecg_extractor.heart_signal[start:end] / 1000, label="Processed ECG Signal")  # Convert μV to mV
# for point in points:
#     # only plot if the index is not None and not np.nan, ensuring it's a valid integer
#     if point['P'] is not None and not np.isnan(point['P']):
#         plt.plot(point['P'] / ecg_extractor.sampling_rate, ecg_extractor.heart_signal[int(point['P'])] / 1000, 'go', label='P-peak' if point['peak_id'] == 0 else "")
#     if point['Q'] is not None and not np.isnan(point['Q']):
#         plt.plot(point['Q'] / ecg_extractor.sampling_rate, ecg_extractor.heart_signal[int(point['Q'])] / 1000, 'bo', label='Q-peak' if point['peak_id'] == 0 else "")
#     if point['R'] is not None:  # assuming R-peaks are always present and valid
#         plt.plot(point['R'] / ecg_extractor.sampling_rate, ecg_extractor.heart_signal[int(point['R'])] / 1000, 'ro', label='R-peak' if point['peak_id'] == 0 else "")
#     if point['S'] is not None and not np.isnan(point['S']):
#         plt.plot(point['S'] / ecg_extractor.sampling_rate, ecg_extractor.heart_signal[int(point['S'])] / 1000, 'mo', label='S-peak' if point['peak_id'] == 0 else "")
#     if point['T'] is not None and not np.isnan(point['T']):
#         plt.plot(point['T'] / ecg_extractor.sampling_rate, ecg_extractor.heart_signal[int(point['T'])] / 1000, 'yo', label='T-peak' if point['peak_id'] == 0 else "")

# " Uncomment to visualize extracted intervals and segments "
# for i, point in enumerate(points):
#     # Color PR Interval
#     if point['P'] is not None and not np.isnan(point['P']) and point['Q'] is not None and not np.isnan(point['Q']):
#         plt.plot(time_vector[int(point['P']):int(point['Q'])], 
#                  ecg_extractor.heart_signal[int(point['P']):int(point['Q'])] / 1000, 
#                  'b', linewidth=2, label='PR Interval' if i == 0 else "")

#     # Color QT Interval
#     if point['Q'] is not None and not np.isnan(point['Q']) and point['T'] is not None and not np.isnan(point['T']):
#         plt.plot(time_vector[int(point['Q']):int(point['T'])], 
#                  ecg_extractor.heart_signal[int(point['Q']):int(point['T'])] / 1000, 
#                  'g', linewidth=2, label='QT Interval' if i == 0 else "")

#     # Color ST Segment
#     if point['S'] is not None and not np.isnan(point['S']) and point['T'] is not None and not np.isnan(point['T']):
#         plt.plot(time_vector[int(point['S']):int(point['T'])], 
#                  ecg_extractor.heart_signal[int(point['S']):int(point['T'])] / 1000, 
#                  'r', linewidth=2, label='ST Segment' if i == 0 else "")

#     # Color QRS Complex 
#     if point['Q'] is not None and not np.isnan(point['Q']) and point['S'] is not None and not np.isnan(point['S']):
#         plt.plot(time_vector[int(point['Q']):int(point['S'])],
#                  ecg_extractor.heart_signal[int(point['Q']):int(point['S'])] / 1000,
#                  'c', linewidth=2, label='QRS Duration' if i == 0 else "")
        
#     # Color P-wave 
#     if i < len(ecg_extractor.p_onsets) and not np.isnan(ecg_extractor.p_onsets[i]) and i < len(ecg_extractor.p_offsets) and not np.isnan(ecg_extractor.p_offsets[i]):
#         plt.plot(time_vector[int(ecg_extractor.p_onsets[i]):int(ecg_extractor.p_offsets[i])],
#                  ecg_extractor.heart_signal[int(ecg_extractor.p_onsets[i]):int(ecg_extractor.p_offsets[i])] / 1000,
#                  'm', linewidth=2, label='P-wave Duration' if i == 0 else "")    
    
#     # Color PR Segment 
#     if i < len(ecg_extractor.p_offsets) and not np.isnan(ecg_extractor.p_offsets[i]) and point['Q'] is not None and not np.isnan(point['Q']):
#         plt.plot(time_vector[int(ecg_extractor.p_offsets[i]):int(point['Q'])],
#                  ecg_extractor.heart_signal[int(ecg_extractor.p_offsets[i]):int(point['Q'])] / 1000,
#                  'y', linewidth=2, label='PR Segment' if i == 0 else "")


# plt.title("Processed ECG Signal with Extracted Features")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude (mV)")

# # To avoid repeating labels in the legend
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys())

# plt.tight_layout()
# plt.show()