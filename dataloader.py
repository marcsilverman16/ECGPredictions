import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
from ECG_variation_map_with_denoising import ECGMap
from ECG_feature_variation_map_with_denoising import FeatureMap
from scipy.io import loadmat



class FilterData():
    def __init__(self, file_path):
        """
        Args:
            mat_files (list of str): List of paths to .mat files.
            labels (numpy.ndarray): Corresponding labels.
        """
        self.df = pq.read_table(file_path).to_pandas().to_numpy()

    def top_labels(self, threshold_map):
        d_count = {}
        for col in range(len(self.df[0])):
            d_count[col] = {}

        for row in self.df:
            for col in range(len(self.df[0])):
                if(row[col] in d_count[col]):
                    d_count[col][row[col]] += 1
                else:
                    d_count[col][row[col]] = 1
            
        
        # Order by descending order and apply threshold map
        for col, counts in d_count.items():
            # Sort the dictionary by counts in descending order and keep only top n items
            top_n = threshold_map.get(col, len(counts))  # Get threshold or default to all
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            d_count[col] = sorted_counts[:top_n]

        return d_count
    

    def select_data(self, filter_dict):
     # Initialize mask as True for all rows
        mask = np.ones(len(self.df), dtype=bool)
        
        # Apply filtering for each specified column
        for col, filtered_values in filter_dict.items():
            # Update mask to include current column's conditions
            mask &= np.isin(self.df[:, col], filtered_values)
        
        # Apply final mask to dataframe to get filtered data
        filtered_df = self.df[mask]
        return filtered_df

    def balance_dataset(cls, data, disease_col_index):
        # Get unique diseases and their counts
        unique, counts = np.unique(data[:, disease_col_index], return_counts=True)
        disease_counts = dict(zip(unique, counts))
        
        # Find the minimum count of the diseases to balance the dataset
        min_count = min(disease_counts.values())
        
        # Sample from each disease to make the distribution nearly equal
        balanced_data = np.array([], dtype=object).reshape(0, data.shape[1])
        for disease in unique:
            disease_data = data[data[:, disease_col_index] == disease]
            indices = np.random.choice(disease_data.shape[0], min_count, replace=False)
            sampled_data = disease_data[indices]
            balanced_data = np.vstack((balanced_data, sampled_data))
        
        unique, counts = np.unique(balanced_data[:, disease_col_index], return_counts=True)
        disease_counts = dict(zip(unique, counts))

        return balanced_data, disease_counts

    

# Sample DataFrame
# data = {
#     'patient_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'disease': ['Flu', 'Cold', 'COVID', 'Flu', 'Cancer', 'COVID', 'Cold', 'Cancer', 'Flu', 'Cold']
# }
# df = pd.DataFrame(data)

# def filter_top_k_diseases(df, k):
#     # Calculate the frequency of each disease
#     disease_counts = df['disease'].value_counts()
    
#     # Identify the top k diseases
#     top_k_diseases = disease_counts.nlargest(k).index
    
#     # Filter the DataFrame to keep only the rows with the top k diseases
#     filtered_df = df[df['disease'].isin(top_k_diseases)]
    
#     # Find the minimum count of the top k diseases to balance the dataset
#     min_count = disease_counts.loc[top_k_diseases].min()

#     # Sample from each disease to make the distribution nearly equal
#     # We use `min_count` to equalize the representation in the DataFrame
#     balanced_df = pd.DataFrame()
#     for disease in top_k_diseases:
#         sampled_df = filtered_df[filtered_df['disease'] == disease].sample(min_count, replace=False)
#         balanced_df = pd.concat([balanced_df, sampled_df])

#     return balanced_df

# # Filter the DataFrame to retain rows with only the top 3 diseases, equally split
# result_df = filter_top_k_diseases(df, 3)
# print(result_df)



class MATDataset(Dataset):
    def __init__(self, maps, labels):
        """
        This is a custom Pytorch Dataset class.
        See: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        As required for a custom Pytorch Dataset class,
        it implements three functions: __init__, __len__, and __getitem__.
        Args:
            mat_files (list of str): List of data maps
            labels (numpy.ndarray): Corresponding labels.
        """
        self.labels = labels
        self.maps = maps


    def __len__(self):
        # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        # "The __len__ function returns the number of samples in our dataset."
        return len(self.maps)

    def __getitem__(self, idx):
        # Load .mat file

        ecg_3d_map = self.maps[idx]

        # Convert data to torch tensor
        data_tensor = torch.tensor(ecg_3d_map, dtype=torch.float32)
        
        # Get corresponding label and convert to tensor
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)  # Use torch.long if it's a classification label
        
        return data_tensor, label_tensor
    


    
