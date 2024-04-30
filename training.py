import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import FilterData, MATDataset
from cnn_gen_arch import ECGCNN, CustomCNN
import os
import argparse
from lstm import LSTM
from ECG_variation_map_with_denoising import ECGMap
from ECG_feature_variation_map_with_denoising import FeatureMap
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


def format_index(x):
    # Check if x has at most 5 digits and is non-negative
    if not (0 <= x < 100000):
        raise ValueError("X must be between 1 and 99999")
    
    # Format the number with leading zeros to ensure it has exactly 5 digits
    return "JS{:05d}".format(x)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_predictions = 0
    correct_predictions = 0
    correct_ones = 0
    total_ones = 0
    all_labels = []
    all_predictions = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        predicted = torch.sigmoid(outputs).data > 0.5
        all_labels.append(labels.byte().cpu().numpy())
        all_predictions.append(predicted.cpu().numpy())

        correct_predictions += (predicted == labels.byte()).sum().item()
        correct_ones += ((predicted == 1) & (labels.byte() == 1)).sum().item()
        total_ones += (labels.byte() == 1).sum().item()
        total_predictions += labels.numel()

    # Calculate F1 Score
    all_labels = np.vstack(all_labels)
    all_predictions = np.vstack(all_predictions)
    train_f1_score = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_predictions / total_predictions 
    train_accuracy_ones = correct_ones / total_ones if total_ones > 0 else 0

    return train_loss, train_accuracy, train_accuracy_ones, train_f1_score

def validate_epoch(model, valid_loader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    total_predictions = 0
    correct_predictions = 0
    correct_ones = 0
    total_ones = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            predicted = torch.sigmoid(outputs).data > threshold
            all_labels.append(labels.byte().cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())

            correct_predictions += (predicted == labels.byte()).sum().item()
            correct_ones += ((predicted == 1) & (labels.byte() == 1)).sum().item()
            total_ones += (labels.byte() == 1).sum().item()
            total_predictions += labels.numel()

    # Calculate F1 Score
    all_labels = np.vstack(all_labels)
    all_predictions = np.vstack(all_predictions)
    valid_f1_score = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

    valid_loss = running_loss / len(valid_loader)
    valid_accuracy = correct_predictions / total_predictions
    valid_accuracy_ones = correct_ones / total_ones if total_ones > 0 else 0

    return valid_loss, valid_accuracy, valid_accuracy_ones, valid_f1_score


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Example Script with Argument Parsing")

    # Add arguments
    parser.add_argument("-data_path", type=str, help="The ECG file directory", default="/Users/theodoroszournatzis/Documents/GitHub/ml_project/src/raw_data/WFDBRecords")
    parser.add_argument("-label_path", type=str, help="The label file directory", default="all_header_file_data.parquet.snappy")
    parser.add_argument("-data_threshold", type=int, help="The number of patient records", default=1000) # JS0000 -> JS1000
    parser.add_argument("-label_threshold", type=int, help="The number of label records", default=10)
    parser.add_argument("-feature_include", type=bool, default=False)
    parser.add_argument("-lstm", type=bool, default=False)


    # Parse the arguments
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

    # Prepare prediction labels
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


    # Example paths and labels
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
            print("Performing ECGCNN (with features)")

            for i in range(len(mat_files)):
                ecg_3d_map = FeatureMap(ecg_path=mat_files[i]).feature_map()
                
                if(len(ecg_3d_map) == 0):
                    ecg_3d_map = maps[0]

                maps.append(ecg_3d_map)
            
            num_channels, map_height, map_width = maps[0].shape
            model = CustomCNN(map_width, map_height, args.label_threshold, num_channels)
                    
        else:
            print("Performing ECGCNN (without features)")

            for i in range(len(mat_files)):
                ecg_3d_map = ECGMap(ecg_path=mat_files[i]).feature_map()
                maps.append(ecg_3d_map)
            
            num_channels, _, _ = maps[0].shape
            model = ECGCNN(args.label_threshold, num_channels, args.feature_include)
    
    assert(len(maps) == len(mat_files))

    # Assuming we're using a GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # Define the loss function and optimizer for multi-label classification
    criterion = nn.BCEWithLogitsLoss()  # Suitable for multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Example sizes
    total_size = len(maps)  # Assuming maps and pred_labels have the same length

    # Shuffle indices
    indices = np.arange(total_size)
    np.random.shuffle(indices)

    # Calculate split index
    split_idx = int(total_size * 0.8)  # 80% for training

    # Split indices into training and validation sets
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    actual_labelop = [pred_labels[i] for i in test_indices]

    # Create datasets using the shuffled indices
    train_dataset = MATDataset([maps[i] for i in train_indices], [pred_labels[i] for i in train_indices])
    valid_dataset = MATDataset([maps[i] for i in test_indices], [pred_labels[i] for i in test_indices])

    # # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    # Training loop
    num_epochs = 400
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []
    train_ones_accuracies, valid_ones_accuracies = [], []
    train_f1_scores, valid_f1_scores = [], []

    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_ones_accuracy, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_accuracy, valid_ones_accuracy, valid_f1 = validate_epoch(model, valid_loader, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_ones_accuracies.append(train_ones_accuracy)
        train_f1_scores.append(train_f1)

        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
        valid_ones_accuracies.append(valid_ones_accuracy)
        valid_f1_scores.append(valid_f1)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train One Acc: {train_ones_accuracy:.4f}, Train F1: {train_f1:.4f}, Val Loss: {valid_loss:.4f}, Val One Acc: {valid_ones_accuracy:.4f}, Val F1: {valid_f1:.4f}")

    print('Finished Training')

    # Plotting the results...
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Plot for losses
    axs[0].plot(train_losses, label='Training Loss')
    axs[0].plot(valid_losses, label='Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend(loc='upper right')
    axs[0].set_title('Training vs Validation Loss')

    # Plot for accuracies
    axs[1].plot(train_ones_accuracies, label='Training Accuracy')
    axs[1].plot(valid_ones_accuracies, label='Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend(loc='upper left')
    axs[1].set_title('Training vs Validation Accuracy')

    # Plot for F1 scores
    axs[2].plot(train_f1_scores, label='Training F1 Score')
    axs[2].plot(valid_f1_scores, label='Validation F1 Score')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('F1 Score')
    axs[2].legend(loc='upper left')
    axs[2].set_title('Training vs Validation F1 Score')

    plt.tight_layout()
    plt.show()

