''' This module takes in as input a user provided pretrained network architecture (e.g. 'vgg16'), accompanying training/test/validation
data of flower images and hyperparameters (e.g. 'learning_rate'), and trains the provided network architecture on the provided data,
then saves the resulting trained model as a checkpoint.
User provided inputs, including the directory for saving the trained model are passed via the commandline using the argparse module '''


# Import Libraries

import argparse
import time

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# =============================================================================
# Define Utility Functions
# =============================================================================

# Function arg_parser parses user provided keyword parameters from the command line
def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser(description="Parameters used for training neural network")

    # Add network architecture argument to parser
    parser.add_argument('--arch',
                        type=str,
                        default='vgg16',
                        help='Select VGG network architecture from torchvision.models as str. Default=vgg16')

    parser.add_argument('--data_dir',
                        type=str,
                        #default='flowers', # required parameter (e.g. 'flowers')
                        help='Identify source directory for training/validation/testing datasets')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./',
                        help='directory for saving checkpoint of trained model.  Default=current directory')
    parser.add_argument('--checkpt_file',
                        type=str,
                        default ="checkpoint.pth",
                        help='name of checkpoint file for saving trained model, Default="checkpoint.pth"')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.0007,
                        help='Learning rate for Adam optimizer. Default = 0.0007')
    parser.add_argument('--epochs',
                        type=int,
                        default=3,
                        help='Number of epochs for training neural network model. Default=3')
    parser.add_argument('--hidden_units',
                        type=int,
                        default = 500,
                        help='Number of hidden units in fully connected classifier. Default=500')
    parser.add_argument('--device',
                        action="store",
                        default='cpu',
                        help='select device cpu or cuda (GPU)for model training, testing, validation.  Default=cpu')

    # Parse args
    args = parser.parse_args()
    return args


# Function load_data loads and transforms data for training, testing and validation of model
def load_data(data_dir):
    # Differentiate the train, validation, and test directories (user provided data directory must have these subfolders)
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Perform transformations for train data
    # Transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    # Validation & testing tranformations use same transformations
    test_transforms = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    # Load the datasets with ImageFolder module
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = test_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

    return train_datasets, valid_datasets, test_datasets

# Function data_loader takes the train/valid/test datasets and converts them into iterable dataloaders
def data_loader(train_datasets, valid_datasets, test_datasets):
# Using the image datasets and the transforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
    valid_dataloader = torch.utils.data.DataLoader(valid_datasets,batch_size = 32, shuffle = False)
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size = 20, shuffle = False)

    return train_dataloader, valid_dataloader, test_dataloader


''' Function train_network trains and validates model w/ training and validation data via the following steps: \
1. Loads the pre-trained network \
2. Defines a new, untrained feed-forward network as a classifier, using ReLU activations and dropout \
3. trains the classifier layers using backpropagation using the pre-trained network to get the features \
4. Tracks the loss and accuracy on the validation set to help determine the best hyperparameters \
5. Prints out training loss, validation loss, and validation accuracy as the network trains '''

def train_network(arch, train_dataloader, valid_dataloader, learning_rate, hidden_units, epochs, device):
    # Load user provided pretrained network architecture
    model = eval("models.{}(pretrained=True)".format(arch))
    model.name = arch
    # Freeze parameters of pretrained network so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Find Input Layers (works for VGG models)
    input_features = model.classifier[0].in_features

    # Replace Classifier layer of model such that we have appropriate number of possible outputs for flower classifications (102 categories)
    model.classifier = nn.Sequential(nn.Linear(input_features, hidden_units),
                                    nn.ReLU(),
                                    nn.Dropout(0.4),
                                    nn.Linear(hidden_units, 102),
                                    nn.LogSoftmax(dim=1))

    # Use Negative Log Likelihood Loss given we're using Softmax as final activation function
    criterion = nn.NLLLoss()

    # Use Adam Optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Use GPU if user requests and it's available
    device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(device)
    # Set parameters related to periodically printing out training/validation loss and validation accuracy during training
    train_steps = 0
    print_every = 50
    begin = time.time()

    # Training iteration
    for epoch in range(epochs):
        start = time.time()
        running_loss = 0
        for images, labels in train_dataloader:
            train_steps += 1
            # Move model parameters, and test set input and label tensors to the GPU if available
            images, labels = images.to(device), labels.to(device)
            # zero out gradient to avoid cumulative totals on iterations
            optimizer.zero_grad()
            # forwards pass
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            # backwards pass and update running loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if train_steps % print_every == 0:
                valid_step = 0
                valid_loss = 0
                accuracy = 0
                model.eval()
                # turn off gradient for validation
                with torch.no_grad():
                    for images, labels in valid_dataloader:
                        valid_step += 1
                        images, labels = images.to(device), labels.to(device)
                        # forwards pass
                        log_ps = model.forward(images)

                        # backwards pass and update validation loss
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss

                        # calculate the accuracy starting w/ probabilities
                        ps = torch.exp(log_ps)
                        # take the top classification
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                        print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Step {train_steps}.."
                        f" Validation Step {valid_step}.."
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Validation loss: {valid_loss/len(valid_dataloader):.3f}.. "
                        f"Validation accuracy: {accuracy/len(valid_dataloader):.3f}")
                        print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds.."
                        f"Total Time: {(time.time() - begin)/60/3:.3f} minutes")

                        # reset running_loss for next validation iteration and turn training back on
                        running_loss = 0
                        model.train()

    return model

# Function save_checkpoint saves trained model as a checkpoint
def save_checkpoint(model, save_dir, checkpt_file, train_datasets):
    # Save mapping of classes to indices used in training the model
    model.class_to_idx = train_datasets.class_to_idx

    checkpoint = {'architecture': model.name,
                'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}

    # checkpoint filename
    filename = checkpt_file
    torch.save(checkpoint, save_dir + filename)

# Function main calls and executes all of the above supporting/utility functions
def main():
    # Get arguments for helper Functions
    args = arg_parser()

    # Load and transform the datasets for training, validation, testing
    train_datasets, valid_datasets, test_datasets = load_data(args.data_dir)

    # Convert datasets into iterable dataloaders for training, validation, testing
    train_dataloader, valid_dataloader, test_dataloader = data_loader(train_datasets, valid_datasets, test_datasets)

    # Train model using dataloaders and provided hyperparameters
    trained_model = train_network(args.arch, train_dataloader, valid_dataloader, args.learning_rate, args.hidden_units, args.epochs, args.device)

    # Print confirmation for completion of training and validation
    print("\nTraining and validation is now complete!!")

    # Save trained model as a checkpoint
    save_checkpoint(trained_model, args.save_dir, args.checkpt_file, train_datasets)

# =============================================================================
# Run Program
# =============================================================================
if __name__ == '__main__': main()
