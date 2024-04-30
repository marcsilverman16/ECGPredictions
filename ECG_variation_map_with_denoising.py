import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from ecg_denoising import ECGdenoising

'''
Info: The 2D maps generated from ECG signals represent variations in voltage over time, transformed into a spatial domain. 
Each pixel's intensity corresponds to the normalized voltage at a specific time point, i.e., the color intensity in 
the map indicates the relative magnitude of the ECG signal after normalization. Brighter areas signify higher voltage 
values, while darker areas indicate lower voltage values.
'''

class ECGMap():
    def __init__(self, ecg_path, map_width = 100):
        """
        Args:
            mat_files (list of str): List of paths to .mat files.
            labels (numpy.ndarray): Corresponding labels.
        """
        self.ecg_data = loadmat(ecg_path)['val']
        self.map_width = map_width


    def normalize_data(cls, data):
        # Handle NaN and Inf by replacing them, for example with the mean of non-NaN, finite data
        finite_data = data[np.isfinite(data)]
        if finite_data.size == 0:
            # Handle case where no finite values exist
            return np.zeros_like(data)
        
        mean_val = np.mean(finite_data)
        data = np.where(np.isfinite(data), data, mean_val)  # Replace NaN and Inf with mean

        min_val = np.min(data)
        max_val = np.max(data)
        range_val = max_val - min_val

        if range_val == 0:
            return data
        else:
            return (data - min_val) / range_val

    # Conversion of 1D ECG Signal to 2D Map
    def convert_to_2D_map(cls, ecg_lead, map_width):
        ecg_lead = cls.normalize_data(ecg_lead)
        map_height = len(ecg_lead) // map_width
        return ecg_lead[:map_height * map_width].reshape((map_height, map_width))

    def feature_map(self):
        ecg_3d_map = []
        if isinstance(self.ecg_data, np.ndarray) and self.ecg_data.ndim == 2 and self.ecg_data.shape[0] == 12:
            preprocessed_maps = []

            for i, lead in enumerate(self.ecg_data):
                extractor = ECGdenoising(lead)
                preprocessed_lead = extractor.heart_signal
                preprocessed_maps.append(self.convert_to_2D_map(preprocessed_lead, self.map_width))
            
            # 3D array of stacked 2D maps
            ecg_3d_map = np.stack(preprocessed_maps, axis=0)

        return ecg_3d_map


# Main Execution Block
# if __name__ == "__main__":
#     file_path = 'JS00020.mat'  # adjust this
#     ecg_data = load_ecg_data(file_path)

#     if isinstance(ecg_data, np.ndarray) and ecg_data.ndim == 2 and ecg_data.shape[0] == 12:
#         preprocessed_maps = []
#         original_maps = []

#         for i, lead in enumerate(ecg_data):
#             extractor = ECGdenoising(lead)
#             preprocessed_lead = extractor.heart_signal
#             preprocessed_maps.append(convert_to_2D_map(preprocessed_lead))
        
#         # 3D array of stacked 2D maps
#         ecg_3d_map = np.stack(preprocessed_maps, axis=0)
        
#         # Plot Processed 2D maps
#         plt.figure(figsize=(20, 15))
#         for i, preprocessed_map in enumerate(preprocessed_maps):
#             plt.subplot(3, 4, i+1)  
#             plt.imshow(preprocessed_map, cmap='hot', interpolation='nearest')
#             plt.title(f'Lead {i+1} Preprocessed Map')
#             plt.colorbar()
            
#         plt.tight_layout()
#         plt.show()
        
#         print(f"Shape of the 3D ECG map: {ecg_3d_map.shape}")

#         # # Unprocessed and Processed Signal Comparison 
#         # plt.figure(figsize=(20, 15))
        
#         # for i, lead in enumerate(ecg_data):
#         #     extractor = ECGdenoising(lead)
#         #     preprocessed_lead = extractor.heart_signal

#         #     plt.subplot(6, 4, i+1)
#         #     plt.plot(lead, label='Original', alpha=0.5)
#         #     plt.plot(preprocessed_lead, label='Preprocessed', alpha=0.75)
#         #     plt.title(f'Lead {i+1} Signal')
#         #     plt.legend()

#         #     original_maps.append(convert_to_2D_map(lead))
#         #     preprocessed_maps.append(convert_to_2D_map(preprocessed_lead))
        
#         # plt.tight_layout()
#         # plt.show()

#         # # Unprocessed and Processed 2D Map Comparison 
#         # plt.figure(figsize=(20, 15))
#         # for i, (original_map, preprocessed_map) in enumerate(zip(original_maps, preprocessed_maps)):
#         #     plt.subplot(6, 8, 2*i+1)
#         #     plt.imshow(original_map, cmap='hot', interpolation='nearest')
#         #     plt.title(f'Lead {i+1} Original Map')
#         #     plt.colorbar()

#         #     plt.subplot(6, 8, 2*i+2)
#         #     plt.imshow(preprocessed_map, cmap='hot', interpolation='nearest')
#         #     plt.title(f'Lead {i+1} Preprocessed Map')
#         #     plt.colorbar()

#         # plt.tight_layout()
#         # plt.show()

#     else:
#         print("The loaded data doesn't match the expected format.")
