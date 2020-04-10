''' This module takes in the path to an image of a flower along with a checkpoint for a saved neural
network model and predicts the type of flower as its output along with the prediction probability
User provided inputs, including the image path and model are passed via the commandline using the
argparse module '''

# Import Libraries
import argparse
import json

import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
import numpy as np

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
    parser.add_argument('--checkpt_file',
                        type=str,
                        #default='checkpoint.pth', #required parameter (e.g. 'checkpoint.pth')
                        help='file path for saved pretrained network checkpoint')

    parser.add_argument('--image_path',
                        type=str,
                        #default='flowers/test/1/image_06760.jpg', # required parameter
                        help='provide image file path for flower class prediction')
    parser.add_argument('--cat_to_name',
                        type=str,
                        default='cat_to_name.json',
                        help='provide a mapping of categories to real names. Default=cat_to_name.json')
    parser.add_argument('--top_k',
                        type=int,
                        default=5,
                        help='select number of top K most likely predicted classes to be returned. Default=5')
    parser.add_argument('--device',
                        action="store",
                        default='cpu',
                        help='select device cpu or cuda (GPU)for model training, testing, validation. Default=cpu')

    # Parse args
    args = parser.parse_args()
    return args


# Function load_checkpoint loads the model for predicting flower type
def load_checkpoint(filepath, device):
    # Use GPU if user requests and it's available
    device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")

    # Load the saved checkpoint file
    checkpoint = torch.load(filepath)

    # Identify model architecture
    arch = checkpoint['architecture']
    # Download pretrained model
    model = eval("models.{}(pretrained=True)".format(arch))
    model.to(device)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Load checkpoint details
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

# Function process_image preprocesses the image so it can be used as an input for the model prediction
from PIL import Image
import numpy as np

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Import Image module from PIL (PILLOW) aka Python Imaging Library
    im = Image.open(image_path)
    im.thumbnail((256,256))
    # get image size
    width, height = im.size
    # center and crop to 224 x 224
    left = (width-224)/2
    top = (height-224)/2
    right = left + 224
    bottom = top + 224
    im = im.crop((left, top, right, bottom))
    # convert values from int b/w 0-255 to float b/w 0-1 using Numpy array
    np_im = np.array(im)/255
    mean = [0.485, 0.456, 0.406]
    std_dev = [0.229, 0.224, 0.225]
    np_im = (np_im - mean)/std_dev
    # transpose the dimensions for PyTorch consumption
    np_im = np_im.transpose((2, 0,  1))
    image = torch.from_numpy(np_im)

    return image


# Function predict_class predicts the class of flower from an image file
def predict_class(image_path, model, top_k, device, cat_to_name):

    # Use GPU for model evaluation if user requests and it's available
    device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(device)

    # Pre-process image
    image = process_image(image_path)

    # Convert image from numpy array to torch tensor and add dimensions to comply w/ input of model
    image = torch.from_numpy(np.expand_dims(image,
                                                  axis=0)).type(torch.FloatTensor)
    # Move image to GPU if it's available
    image = image.to(device)

    # Turn off gradient as we're not further training the model as part of our predictions
    with torch.no_grad():
        # set the model to evaluation mode
        model.eval()
        # calculate probabilities
        probs = torch.exp(model.forward(image))
        # find the top k results
        top_p, top_classes = probs.topk(top_k, dim =1)

        # load the class_to_idx mapping to convert from indices to class labels
        idx2class = model.class_to_idx
        # Inverting index-class dictionary
        idx2class = {x: y for y, x in idx2class.items()}
        probs = top_p.tolist()[0]
        # Inverting index-class dictionary
        classes = [idx2class[i] for i in top_classes.tolist()[0]]

        # Convert classes to flower names
        with open(cat_to_name, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[cat] for cat in classes]

        return probs, classes

# Function main calls and executes all of the above supporting/utility functions
# Function end result is to print the predicted flower class(es) and probabilities for the image

def main():
# Get arguments for helper Functions
    args = arg_parser()

    # First load the model and define image path
    model = load_checkpoint(args.checkpt_file, args.device)

    # Pre-process image for prediction
    im = process_image(args.image_path)

    # predict top_k flower class(es) for image
    probs, classes = predict_class(args.image_path, model, args.top_k, args.device, args.cat_to_name)

    # print top_k predicted flowers along with probabilities
    print("Top {} predicted flower(s) and probabilities for image are as follows:".format(args.top_k))

    for flower, prob in zip(classes, probs):
        print(f"{flower}: {prob:.3f}")

# =============================================================================
# Run Program
# =============================================================================
if __name__ == '__main__': main()
