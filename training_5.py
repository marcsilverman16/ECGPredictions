import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import FilterData, MATDataset
from cnn_gen_arch import ECGCNN, CustomCNN, CNNAwareLSTM, ECGCNNLSTM
import os
import argparse
from lstm import LSTM
from ECG_variation_map_with_denoising import ECGMap
from ECG_feature_variation_map_with_denoising import FeatureMap
from scipy.io import loadmat


def format_index(x):
    # Check if x has at most 5 digits and is non-negative
    if not (0 <= x < 100000):
        raise ValueError("X must be between 1 and 99999")
    
    # Format the number with leading zeros to ensure it has exactly 5 digits
    return "JS{:05d}".format(x)

def train_epoch(model, train_loader, criterion, optimizer, device, threshold = 0.5):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    total_predictions = 0
    correct_predictions = 0
    correct_ones = 0  # to count correct predictions that are 1
    total_ones = 0  # to count actual ones in labels
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # print(inputs.shape)
        optimizer.zero_grad()  # Clear the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backpropagate the errors
        optimizer.step()  # Update the weights
        
        running_loss += loss.item()

        # Calculate accuracy
        predicted = torch.sigmoid(outputs).data > threshold  # Apply threshold to obtain binary results
        correct_predictions += (predicted == labels.byte()).sum().item()
        # Calculating accuracy for ones
        correct_ones += ((predicted == 1) & (labels.byte() == 1)).sum().item()
        total_ones += (labels.byte() == 1).sum().item()
        total_predictions += labels.numel()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_predictions / total_predictions 
    train_accuracy_ones = correct_ones / total_ones if total_ones > 0 else 0  # Avoid division by zero

    return train_loss, train_accuracy, train_accuracy_ones

def validate_epoch(model, valid_loader, criterion, device, threshold = 0.5):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    total_predictions = 0
    correct_predictions = 0
    correct_ones = 0  # to count correct predictions that are 1
    total_ones = 0  # to count actual ones in labels

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            running_loss += loss.item()

            # Calculate accuracy
            predicted = torch.sigmoid(outputs).data > threshold  # Apply threshold to obtain binary results
            correct_predictions += (predicted == labels.byte()).sum().item()
            total_predictions += labels.numel()
             # Calculating accuracy for ones
            correct_ones += ((predicted == 1) & (labels.byte() == 1)).sum().item()
            total_ones += (labels.byte() == 1).sum().item()

    valid_loss = running_loss / len(valid_loader)
    valid_accuracy = correct_predictions / total_predictions  # Correct over total elements
    valid_accuracy_ones = correct_ones / total_ones if total_ones > 0 else 0  # Avoid division by zero

    return valid_loss, valid_accuracy, valid_accuracy_ones



if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Example Script with Argument Parsing")

    # Add arguments
    parser.add_argument("-data_path", type=str, help="The ECG file directory", default="../raw_data/WFDBRecords")
    parser.add_argument("-label_path", type=str, help="The label file directory", default="../output_labels/all_header_file_data.parquet.snappy")
    parser.add_argument("-data_threshold", type=int, help="The number of patient records", default=1000) # JS0000->JS1000
    parser.add_argument("-label_threshold", type=int, help="The number of label records", default=10)
    parser.add_argument("-feature_include", type=bool, default=False)
    parser.add_argument("-lstm", type=bool, default=False)


    # Parse the arguments
    args = parser.parse_args()

    # Prepare the dataset
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
            
            _, num_channels, map_height, map_width = maps[0].shape
            # model = CustomCNN(map_width, map_height, args.label_threshold, num_channels)
            model = CNNAwareLSTM(map_width, map_height, args.label_threshold, num_channels, 128, 2)
                    
        else:
            print("Performing ECGCNN (without features)")

            for i in range(len(mat_files)):
                ecg_3d_map = ECGMap(ecg_path=mat_files[i]).feature_map()
                maps.append(ecg_3d_map)
            
            num_channels, _, _ = maps[0].shape
            print(maps[0].shape)
            model = ECGCNN(args.label_threshold, num_channels, args.feature_include)
            print(model)
            # model = ECGCNNLSTM(num_channels, args.label_threshold)
    
    assert(len(maps) == len(mat_files))

    # Assuming we're using a GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # # Define the loss function and optimizer for multi-label classification
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
    num_epochs = 500  # Adjust according to your needs

    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies, train_ones_accuracies, valid_ones_accuracies = [], [], [], []

    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_ones_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_ones_accuracies.append(train_ones_accuracy)

        valid_loss, valid_accuracy, valid_ones_accuracy = validate_epoch(model, valid_loader, criterion, device)

        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
        valid_ones_accuracies.append(valid_ones_accuracy)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train One Acc: {train_ones_accuracy:.4f}, Val Loss: {valid_loss:.4f}, Val Acc: {valid_accuracy:.4f}, Val One Acc: {valid_ones_accuracy:.4f}")


    print('Finished Training')

    
    # model.eval()  # Set the model to evaluation mode
    # predictions = []
    # actual_labels = []

    # threshold = 0.5  # Define a threshold to classify predictions as 0 or 1

    # with torch.no_grad():
    #     for inputs, labels in valid_loader:
    #         inputs, labels = inputs.to(device), labels.float().to(device)
    #         outputs = model(inputs)

    #         # Apply a threshold to convert to binary
    #         predicted_labels = (outputs > threshold).int()  # Convert probabilities to 0 or 1 based on the threshold

    #         # Store the predicted labels
    #         predictions.extend(predicted_labels.cpu().numpy())
    #         actual_labels.extend(labels.cpu().numpy())


    # predictions = np.array(predictions)
    # total_correct = 0
    # total_correct_ones = 0
    # total_size = 0
    # total_size_ones = 0
    # for i in range(len(predictions)):
    #     total_size += len(actual_labels[i])
    #     print(predictions[i], actual_labels[i])
    #     for j in range(len(predictions[i])):
    #         if(actual_labels[i][j] == 1):
    #                 total_size_ones += 1
    #         if(predictions[i][j] == actual_labels[i][j]):
    #             total_correct += 1
    #             if(predictions[i][j] == 1):
    #                 total_correct_ones += 1
    
    # print(f"Acc: {total_correct/total_size} Acc_ones: {total_correct_ones/total_size_ones}")

