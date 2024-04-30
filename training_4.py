import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataloader import FilterData, MATDataset
from cnn_gen_arch import ECGCNN, CustomCNN
import os
import argparse
from lstm import LSTM
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from ECG_variation_map_with_denoising import ECGMap
from ECG_feature_variation_map_with_denoising import FeatureMap
from scipy.io import loadmat
from training import format_index, train_epoch, validate_epoch


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Example Script with Argument Parsing")
    parser.add_argument("-data_path", type=str, help="The ECG file directory", default="../raw_data/WFDBRecords")
    parser.add_argument("-label_path", type=str, help="The label file directory", default="../output_labels/all_header_file_data.parquet.snappy")
    parser.add_argument("-data_threshold", type=int, help="The number of patient records", default=1000)
    parser.add_argument("-label_threshold", type=int, help="The number of label records", default=10)
    parser.add_argument("-feature_include", type=bool, default=False)
    parser.add_argument("-lstm", type=bool, default=False)
    args = parser.parse_args()

    # Prepare the dataset
    col = 3
    selector = FilterData(file_path=args.label_path)
    common_labels = selector.top_labels({col: args.label_threshold})[col]
    common_labels = [common_labels[i][0] for i in range(len(common_labels))]

    index_files = [format_index(x) for x in range(args.data_threshold)]
    index_files.sort()
    filter_data = selector.select_data({0: index_files, col: common_labels})
    filter_data, dis_count = selector.balance_dataset(filter_data, col)
    patient_map = np.sort(np.unique(filter_data[:, 0]))
    common_labels = [k for k in dis_count]

    print("Number of patients:", len(patient_map))

    if(args.label_threshold != len(common_labels)):
        print(f"There are only {len(dis_count)} diseases in the limited dataset")
        args.label_threshold = len(common_labels)
    
    print("Disease count:", dis_count)

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
        num_layers = 2  # Example number of LSTM layers

        if(args.feature_include):
            print("Performing LSTM (with features)")

            for i in range(len(mat_files)):
                ecg_3d_map = FeatureMap(ecg_path=mat_files[i]).feature_map()

                
                if(len(ecg_3d_map) == 0):
                    ecg_3d_map = maps[0]

                maps.append(np.array([ecg_3d_map[j].flatten() for j in range(len(ecg_3d_map))]))

        else:
            print("Performing LSTM (without features)")

            for i in range(len(mat_files)):
                maps.append(loadmat(mat_files[i])['val'])
        
        input_size, sequence_length = maps[0].shape
        model = LSTM(input_size=input_size, hidden_size=100, num_layers=num_layers, output_size=args.label_threshold)

    else:
        if(args.feature_include):
            # model = ECGCNN(num_channels=2, output_labels=args.label_threshold, features=args.feature_include)
            print("Performing ECGCNN (with features)")

            for i in range(len(mat_files)):
                ecg_3d_map = FeatureMap(ecg_path=mat_files[i]).feature_map()
                
                if(len(ecg_3d_map) == 0):
                    ecg_3d_map = maps[0]
                
                maps.append(ecg_3d_map)

            num_channels, map_height, map_width = maps[0].shape
            # print(num_channels, map_height, map_width)
            model = CustomCNN(map_width, map_height, args.label_threshold, num_channels)
                    
        else:
            print("Performing ECGCNN (without features)")
            model = ECGCNN(num_channels=12, output_labels=args.label_threshold, features=args.feature_include)

            for i in range(len(mat_files)):
                ecg_3d_map = ECGMap(ecg_path=mat_files[i]).feature_map()
                maps.append(ecg_3d_map)
    
    assert(len(maps) == len(mat_files))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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

        # Model initialization
        if(args.lstm):
            model = LSTM(input_size=input_size, hidden_size=100, num_layers=num_layers, output_size=args.label_threshold)
        else:
            if(args.feature_include):
                num_channels, map_height, map_width = maps[0].shape
                model = CustomCNN(map_width, map_height, args.label_threshold, num_channels)
            else:
                model = ECGCNN(num_channels=12, output_labels=args.label_threshold, features=args.feature_include)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Loss function and optimizer
        model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        train_losses, valid_losses = [], []
        train_accuracies, valid_accuracies, train_ones_accuracies, valid_ones_accuracies = [], [], [], []

        # Training and validation loop
        num_epochs = 1000
        for epoch in range(num_epochs):
            train_loss, train_accuracy, train_ones_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            train_ones_accuracies.append(train_ones_accuracy)

            valid_loss, valid_accuracy, valid_ones_accuracy = validate_epoch(model, valid_loader, criterion, device)

            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)
            valid_ones_accuracies.append(valid_ones_accuracy)

            print(f"Fold {fold+1}, Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train One Acc: {train_ones_accuracy:.4f}, Val Loss: {valid_loss:.4f}, Val Acc: {valid_accuracy:.4f}, Val One Acc: {valid_ones_accuracy:.4f}")
        
        fold_results.append((train_losses, valid_losses, train_accuracies, train_ones_accuracies, valid_accuracies, valid_ones_accuracies))

    print('Finished Training and Validation across all folds')

    # Average results from all folds and plot
    avg_train_losses = np.mean([fr[0] for fr in fold_results], axis=0)
    avg_valid_losses = np.mean([fr[1] for fr in fold_results], axis=0)
    avg_train_ones_accuracies = np.mean(np.array([fr[3] for fr in fold_results]), axis=0)
    avg_valid_ones_accuracies = np.mean([fr[5] for fr in fold_results], axis=0)

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
    plt.plot(avg_train_ones_accuracies, label='Average Train Accuracy')
    plt.plot(avg_valid_ones_accuracies, label='Average Validation Accuracy')
    plt.xticks(np.arange(0, len(avg_train_losses), 1))
    plt.title('Average Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()