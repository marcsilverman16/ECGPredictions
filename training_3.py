import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataloader import FilterData, MATDataset
from cnn_gen_arch import ECGCNN
import os
import argparse
from lstm import LSTM
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from ECG_variation_map_with_denoising import ECGMap
from ECG_feature_variation_map_with_denoising import FeatureMap
from scipy.io import loadmat

def format_index(x):
    # Check if x has at most 5 digits and is non-negative
    if not (0 <= x < 100000):
        raise ValueError("X must be between 1 and 99999")
    
    # Format the number with leading zeros to ensure it has exactly 5 digits
    return "JS{:05d}".format(x)

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Example Script with Argument Parsing")
    parser.add_argument("-data_path", type=str, help="The ECG file directory", default="/Users/theodoroszournatzis/Documents/GitHub/ml_project/src/raw_data/WFDBRecords")
    parser.add_argument("-label_path", type=str, help="The label file directory", default="all_header_file_data.parquet.snappy")
    parser.add_argument("-data_threshold", type=int, help="The number of patient records", default=1000)
    parser.add_argument("-label_threshold", type=int, help="The number of label records", default=10)
    parser.add_argument("-feature_include", type=bool, default=False)
    parser.add_argument("-lstm", type=bool, default=False)
    args = parser.parse_args()

    # Prepare the dataset
    col = 3
    selector = FilterData(file_path=args.label_path)
    common_labels = selector.top_labels({col: args.label_threshold})[col]
    print("Common labels:", common_labels)

    index_files = [format_index(x) for x in range(args.data_threshold)]
    index_files.sort()
    filter_data = selector.select_data({0: index_files, 3: common_labels})
    patient_map = np.sort(np.unique(filter_data[:, 0]))
    print("Num patients:", len(patient_map))

    pred_labels = np.zeros((len(patient_map), args.label_threshold))
    for row in filter_data:
        pid = np.where(patient_map == row[0])[0][0]
        did = common_labels.index(row[col])
        pred_labels[pid][did] = 1

    def find_file(filename, root_directory):
        """
        Search for a file within a given directory and its subdirectories.
        
        Args:
            filename (str): The name of the file to search for.
            root_directory (str): The root directory to start the search from.
            
        Returns:
            str: The full path to the file if found, otherwise None.
        """
        for dirpath, dirnames, filenames in os.walk(root_directory):
            if filename in filenames:
                # Construct the full path to the file
                return os.path.join(dirpath, filename)
            
        return None  # Return None if the file is not found

    mat_files = [find_file(f"{patient_map[i]}.mat", args.data_path) for i in range(len(patient_map))]

    # Loading data before hand to avoid reload during training
    maps = []
    
    # Determine if LSTM or CNN is used
    if(args.lstm):  # LSTM
        batch_size = 64  # Example batch size
        input_size = 12  # Number of features per time step
        sequence_length = 5000  # Number of time steps per sample
        num_layers = 2  # Example number of LSTM layers

        model = LSTM(input_size=input_size, hidden_size=100, num_layers=num_layers, output_size=args.label_threshold)
        print("Performing LSTM")

        for i in range(len(mat_files)):
            maps.append(loadmat(mat_files[i])['val'])

    else:
        if(args.feature_include):
            model = ECGCNN(num_channels=2, output_labels=args.label_threshold, features=args.feature_include)
            print("Performing ECGCNN (with features)")

            for i in range(len(mat_files)):
                ecg_3d_map = FeatureMap(ecg_path=mat_files[i]).feature_map()
                
                if(len(ecg_3d_map) == 0):
                    ecg_3d_map = maps[0]

                maps.append(ecg_3d_map)
                    
        else:
            print("Performing ECGCNN (without features)")
            model = ECGCNN(num_channels=12, output_labels=args.label_threshold, features=args.feature_include)

            for i in range(len(mat_files)):
                ecg_3d_map = ECGMap(ecg_path=mat_files[i]).feature_map()
                maps.append(ecg_3d_map)
    
    # print(len(maps), len(mat_files))
    assert(len(maps) == len(mat_files))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # Setup K-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(mat_files)):
        train_files = [maps[i] for i in train_idx]
        valid_files = [maps[i] for i in valid_idx]
        train_labels = pred_labels[train_idx]
        valid_labels = pred_labels[valid_idx]

        train_dataset = MATDataset(train_files, train_labels)
        valid_dataset = MATDataset(valid_files, valid_labels)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

        # Loss function and optimizer
        model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0003)

        train_losses, valid_losses = [], []
        train_accuracies, valid_accuracies = [], []

        # Training and validation loop
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            running_loss, running_accuracy = 0.0, 0.0
            total, correct = 0, 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.float().to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                predicted = torch.sigmoid(outputs).data > 0.5
                correct += (predicted == labels.byte()).sum().item()
                total += labels.numel()

            train_loss = running_loss / len(train_loader)
            train_accuracy = correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            
            model.eval()
            valid_loss, valid_accuracy = 0.0, 0.0
            total_v, correct_v = 0, 0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.float().to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item()
                    predicted = torch.sigmoid(outputs).data > 0.5
                    correct_v += (predicted == labels.byte()).sum().item()
                    total_v += labels.numel()

            valid_loss /= len(valid_loader)
            valid_accuracy = correct_v / total_v
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)
            
            print(f"Fold {fold+1}, Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {valid_loss:.4f}, Val Acc: {valid_accuracy:.4f}")
        
        fold_results.append((train_losses, valid_losses, train_accuracies, valid_accuracies))

    print('Finished Training and Validation across all folds')

    # Average results from all folds and plot
    avg_train_losses = np.mean([fr[0] for fr in fold_results], axis=0)
    avg_valid_losses = np.mean([fr[1] for fr in fold_results], axis=0)
    avg_train_accuracies = np.mean(np.array([fr[2] for fr in fold_results]), axis=0)
    avg_valid_accuracies = np.mean([fr[3] for fr in fold_results], axis=0)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(avg_train_losses, label='Average Train Loss')
    plt.plot(avg_valid_losses, label='Average Validation Loss')
    plt.xticks(np.arange(0, len(avg_train_losses), 1))
    plt.title('Average Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(avg_train_accuracies, label='Average Train Accuracy')
    plt.plot(avg_valid_accuracies, label='Average Validation Accuracy')
    plt.xticks(np.arange(0, len(avg_train_losses), 1))
    plt.title('Average Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

           