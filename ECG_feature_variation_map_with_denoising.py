import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from feature_extraction_with_denoising import ECGExtractor

'''
Info: The 2D maps generated from ECG signals represent PQRST complex in time/voltage over time, transformed into a spatial domain. 
Each pixel's intensity corresponds to the normalized voltage at a specific time point, i.e., the color intensity in 
the map indicates the relative magnitude of the ECG signal after normalization. Brighter areas signify higher voltage 
values, while darker areas indicate lower voltage values.
'''

class FeatureMap():
    def __init__(self, ecg_path, num_peaks = 5, num_wavelets = 12, nan_time = 0, nan_lead = 0):
        """
        Args:
            mat_files (list of str): List of paths to .mat files.
            labels (numpy.ndarray): Corresponding labels.
        """
        self.pth = ecg_path
        self.ecg_data = loadmat(self.pth)['val']
        self.num_peaks = num_peaks
        self.num_wavelets = num_wavelets
        self.nan_time = nan_time
        self.nan_lead = nan_lead


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


    def periodic_correct_shape(cls, arr, max_shape, offset=0):
        """
        Extends an array to a specified max_shape by cyclically appending elements from the array,
        with an offset at the start of each new cycle.

        Args:
            arr (np.ndarray or list): The input array to be extended.
            max_shape (int): The desired length of the output array.
            offset (int): The offset with which to start each new cycle of the array elements.

        Returns:
            np.ndarray: The modified array with length equal to max_shape.
        """
        l = len(arr)
        if len(arr) >= max_shape:
            return arr[:max_shape]
        
        while len(arr) < max_shape:

            # Calculate how much more to fill
            remaining_length = min(max_shape-len(arr), l)

            arr = np.concatenate([arr, arr[:remaining_length]+offset])
            
            # Increase the offset for the next cycle
            offset += offset  # Change this if a different offset progression is needed

        return arr

    def feature_map(self):
        ecg_3d_map = []
        if isinstance(self.ecg_data, np.ndarray) and self.ecg_data.ndim == 2 and self.ecg_data.shape[0] == 12:
            preprocessed_maps = []
            data = []

            try:
                lead = self.ecg_data[11]
                extractor = ECGExtractor(lead)
            
            except Exception as e:
                try:
                    print("This data is not extractable via lead 11:", self.pth)
                    lead = self.ecg_data[1]
                    extractor = ECGExtractor(lead)
                except Exception as e:
                    print("This data is not extractable:", self.pth)
                    return []
                
            lead_point_data = np.empty((len(extractor.rpeaks), 5))
            lead_amp_data = np.empty((len(extractor.rpeaks), 5))

            for x in range(len(extractor.rpeaks)):
                # Correct missing P peaks
                if np.isnan(extractor.ppeaks[x]):
                    lead_point_data[x][0] = self.nan_time
                    lead_point_data[x][0] = self.nan_lead
                else:
                    lead_point_data[x][0] = extractor.ppeaks[x]
                    lead_point_data[x][0] = lead[int(extractor.ppeaks[x])]

                # Correct missing Q peaks
                if np.isnan(extractor.qpeaks[x]):
                    lead_point_data[x][1] = self.nan_time
                    lead_point_data[x][1] = self.nan_lead
                else:
                    lead_point_data[x][1] = extractor.qpeaks[x]
                    lead_point_data[x][1] = lead[int(extractor.qpeaks[x])]
                        
                # Correct missing Q peaks
                if np.isnan(extractor.rpeaks[x]):
                    lead_point_data[x][1] = self.nan_time
                    lead_point_data[x][1] = self.nan_lead
                else:
                    lead_point_data[x][1] = extractor.rpeaks[x]
                    lead_point_data[x][1] = lead[int(extractor.rpeaks[x])]

                lead_point_data[x][1] = extractor.rpeaks[x]
                lead_point_data[x][1] = lead[int(extractor.rpeaks[x])]

                # Correct missing S peaks
                if np.isnan(extractor.speaks[x]):
                    lead_point_data[x][3] = self.nan_time
                    lead_point_data[x][3] = self.nan_lead
                else:
                    lead_point_data[x][3] = extractor.speaks[x]
                    lead_point_data[x][3] = lead[int(extractor.speaks[x])]
                        
                # Correct missing T peaks 
                if np.isnan(extractor.tpeaks[x]):
                    lead_point_data[x][4] = self.nan_time
                    lead_point_data[x][4] = self.nan_lead
                else:
                    lead_point_data[x][4] = extractor.tpeaks[x]
                    lead_point_data[x][4] = lead[int(extractor.tpeaks[x])]
                    

            lead_point_data[np.isnan(lead_point_data)] = self.nan_time
            lead_amp_data[np.isnan(lead_amp_data)] = self.nan_lead
            data.append([lead_point_data.flatten(), lead_amp_data.flatten()])
                    

            for i in range(len(data)):
                for j in range(len(data[i])):
                    if(j == 0):
                        data[i][j] = self.periodic_correct_shape(data[i][0], max_shape=self.num_peaks*self.num_wavelets, offset=self.ecg_data.shape[1])
                    else:
                        data[i][j] = self.periodic_correct_shape(data[i][0], max_shape=self.num_peaks*self.num_wavelets)

                    preprocessed_maps.append(self.convert_to_2D_map(data[i][j], self.num_peaks))

            
            # 3D array of stacked 2D maps
            ecg_3d_map = np.stack(preprocessed_maps, axis=0)

        return ecg_3d_map

# Main Execution Block
# if __name__ == "__main__":
#     file_path = 'testing/JS00020.mat'  # adjust this
#     ecg_data = load_ecg_data(file_path)

#     if isinstance(ecg_data, np.ndarray) and ecg_data.ndim == 2 and ecg_data.shape[0] == 12:
#         preprocessed_maps = []
#         original_maps = []
#         data = []

#         for i, lead in enumerate(ecg_data):
#             extractor = ECGExtractor(lead)
#             lead_point_data = np.array([[extractor.ppeaks[i], extractor.qpeaks[i], extractor.rpeaks[i], extractor.speaks[i], extractor.tpeaks[i]] for i in range(len(extractor.rpeaks))]).flatten()
#             lead_amp_data = np.array([[lead[int(extractor.ppeaks[i])], lead[int(extractor.qpeaks[i])], lead[int(extractor.rpeaks[i])], lead[int(extractor.speaks[i])], lead[int(extractor.tpeaks[i])]] for i in range(len(extractor.rpeaks))]).flatten()
#             max_shape = max(max_shape, len(lead_point_data))
#             data.append([lead_point_data, lead_amp_data])
        

#         print("Max shape:", max_shape)
#         for i in range(len(data)):
#             for j in range(len(data[i])):
#                 if(j == 0):
#                     data[i][j] = periodic_correct_shape(data[i][0], max_shape=max_shape, offset=ecg_data.shape[1])
#                 else:
#                     data[i][j] = periodic_correct_shape(data[i][0], max_shape=max_shape)

#                 preprocessed_maps.append(convert_to_2D_map(data[i][j]))

#         for e in preprocessed_maps:
#             print(e.shape)
        
#         # 3D array of stacked 2D maps
#         ecg_3d_map = np.stack(preprocessed_maps, axis=0)
        
#         # Plot Processed 2D maps
#         plt.figure(figsize=(20, 15))
#         for i, preprocessed_map in enumerate(preprocessed_maps[:12]):
#             plt.subplot(3, 4, i+1)  
#             plt.imshow(preprocessed_map, cmap='hot', interpolation='nearest')
#             plt.title(f'Lead {i+1} Preprocessed Map')
#             plt.colorbar()
            
#         plt.tight_layout()
#         plt.show()
        
#         print(f"Shape of the 3D ECG map: {ecg_3d_map.shape}")

#     else:
#         print("The loaded data doesn't match the expected format.")
