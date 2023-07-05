import os
import sys
import numpy as np
import joblib as jb
import matplotlib.pyplot as plt
from glob import glob

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets

from sklearn.metrics import f1_score, recall_score, precision_score
from collections import defaultdict
from PIL import Image, ImageFile

# the following import is required for training to be robust to truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_image_classifier(
    n_epochs, 
    dataloaders, 
    model, 
    optimizer, 
    criterion, 
    device, 
    save_path
):     
    '''
    Trains and saves a PyTorch model
    inputs:
        n_epochs: int, number of epochs to train 
        dataloaders: dictionary containing at least two keys ('train', 'valid') and the respective DataLoaders as values
        model: pytorch modek class 
        optimizer: torch.optim, step optimizer  
        criterion: defined loss function 
        device: torch.device('cpu'), or torch.device('cuda')
        save_path: string, path to save the trained model
    
    '''
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    train_batches = int(np.ceil(len(dataloaders['train'])))
    model = model.to(device)
    
    epoch_loss = []
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        total = 0
        train_loss = 0.0
        valid_loss = 0.0
        correct_acc = 0.0
        correct_valid = 0.0
        
        ###################
        # train the model #
        ###################
        
        model.train()
        for batch_idx, (data, target) in enumerate(dataloaders['train']):
            
            # move to GPU
            data, target = data.to(device), target.to(device)
            
            # reset optimizer every iteration
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # predict
            _, pred = torch.max(output, 1)
           
            equals = pred == target
            correct_acc += torch.sum(equals.type(torch.cuda.FloatTensor)).item()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
            total += target.size(0)
            
            sys.stdout.write('\r')
            sys.stdout.write(f"Epoch: {epoch}\tBatch {batch_idx} out of {train_batches}\tClassified correctly: {round(correct_acc / total, 5)}")
            sys.stdout.flush()
            
        ##############################    
        # validate model performance #
        ##############################
        
        with torch.no_grad():
            model.eval()
            for batch_idx, (data, target) in enumerate(dataloaders['valid']):
                
                # move to GPU
                data, target = data.to(device), target.to(device)
                
                ## update the average validation loss
                output = model(data)
                loss = criterion(output, target)
                
                _, pred = torch.max(output, 1)
                equals = pred == target
                correct_valid += torch.sum(equals.type(torch.cuda.FloatTensor)).item()
                valid_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
                
        epoch_loss.append((train_loss, valid_loss))         
        # print training/validation statistics 
        print('\nEpoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \t Training accuracy: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss,
            correct_acc / total
            ))
        
        ## save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model, save_path)
            print(f'\nValidation loss improved from {valid_loss_min} to {valid_loss}')
            valid_loss_min = valid_loss
        else: 
            print("Validation loss hasn't improved. Model not saved\n")
    
    # make sure to free up gpu memory
    data, target = data.to(torch.device('cpu')), target.to(torch.device('cpu'))
    model = model.to(torch.device('cpu'))
    return epoch_loss


def test_image_classifier(dataloader, model, criterion, device):
    '''
    tests a PyTorch model
    inputs:
        dataloader: DataLoader object
        model: pytorch model class  
        criterion: defined loss function 
        device: torch.device('cpu'), or torch.device('cuda')   
    '''
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.
    model = model.to(device)
    precision_arr = []
    recall_arr = []
    f1_arr = []
    
    model.eval()
    for batch_idx, (data, target) in enumerate(dataloader):
        
        # move to gpu
        data, target = data.to(device), target.to(device)
        
        
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        
        # calculate the loss_arr
        loss = criterion(output, target)
        
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
        
        precision_arr.append(precision_tensor(output, target))
        recall_arr.append(recall_tensor(output, target))
        f1_arr.append(f1_tensor(output, target))
        
    recall = round(sum(recall_arr) / len(recall_arr), 3) * 100
    precision = round(sum(precision_arr) / len(precision_arr),  3) * 100
    f1 = round(sum(f1_arr) / len(f1_arr),  3) * 100
    accuracy = round(100.0 * correct / total, 3)
    
    # make sure to free up gpu memory
    data, target = data.to(torch.device("cpu")), target.to(torch.device("cpu"))
    model = model.to(torch.device("cpu"))
    return {
        "Test Results": {
            "Loss": test_loss.item(),
            "Accuracy": f"{accuracy}%",
            "Rcall": f"{recall}%",
            "Precision": f"{precision}%",
            "F1": f"{f1}%",
            "Correct": int(correct),
            "Total": int(total)  
        }
    }


def precision_tensor(outputs, targets):
    out = outputs.cpu()
    tar = targets.cpu()
    _, preds = torch.max(out, dim=1)
    return precision_score(tar, preds, average='weighted')


def recall_tensor(outputs, targets):
    out = outputs.cpu()
    tar = targets.cpu()
    _, preds = torch.max(out, dim=1)
    return recall_score(tar, preds, average='weighted')


def f1_tensor(outputs, targets):
    out = outputs.cpu()
    tar = targets.cpu()
    _, preds = torch.max(out, dim=1)
    return f1_score(tar, preds, average='weighted')