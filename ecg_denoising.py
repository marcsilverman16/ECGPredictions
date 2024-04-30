from scipy.signal import butter, filtfilt, iirnotch, detrend, savgol_filter

# ECG Denoising and Transformation to 2D Variation Map
class ECGdenoising:
    def __init__(self, heart_signal, sampling_rate=500):
        self.sampling_rate = sampling_rate
        self.original_signal = heart_signal  # Original unprocessed signal
        self.heart_signal = self.preprocess_signal(heart_signal)  # Preprocessed signal

    def preprocess_signal(self, heart_signal):
        heart_signal = detrend(heart_signal)
        heart_signal = self.bandpass_filter(heart_signal, lowcut=0.5, highcut=45)
        heart_signal = self.notch_filter(heart_signal, freq=50)
        heart_signal = savgol_filter(heart_signal, window_length=31, polyorder=2)
        return heart_signal

    def bandpass_filter(self, data, lowcut, highcut, order=4):
        nyq = 0.5 * self.sampling_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def notch_filter(self, data, freq, Q=30):
        nyq = 0.5 * self.sampling_rate
        freq = freq / nyq
        b, a = iirnotch(freq, Q)
        return filtfilt(b, a, data)