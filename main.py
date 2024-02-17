import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import TensorDataset
import numpy as np
import os
import pdb
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

def get_data_from_abide():
  downloads = 'abide/downloads/Outputs/dparsf/filt_global/rois_aal/'
  pheno_file = 'abide/Phenotypic_V1_0b_preprocessed1.csv'

  pheno_file = open(pheno_file, 'r')
  pheno_list = pheno_file.readlines()
  
  labels_dict = {}
  for i in pheno_list[1:]:
    file_name = i.split(',')[6]
    diagnosis = i.split(',')[7]

    labels_dict[file_name] = float(diagnosis) # Save labels alongisde their filenames

  data = []
  labels = []

  for filename in os.listdir(downloads):
    if filename.endswith('.1D'):  # Check if the file is a .1D file
      filepath = os.path.join(downloads, filename)
      dataset = np.loadtxt(filepath)  # Load the file
      data.append(dataset)  # Append the dataset to the list

      file_id = '_'.join(filename.split('_')[:-2]) # Get file ID from filename
      labels.append(labels_dict[file_id])

  return data, labels

def get_feature_vecs(data):
  roi_size = data[0].shape[1]
  feature_vec_size = int(roi_size * (roi_size - 1) / 2)
  feature_vecs = np.zeros([len(data), feature_vec_size])

  vectorized_fisher_transfrom = np.vectorize(fishers_z_transform)
  
  for i in range(len(data)):
    corr_coefs = np.corrcoef(data[i], rowvar=False)
    corr_coefs = np.nan_to_num(corr_coefs)

    transformed_corr_coefs = vectorized_fisher_transfrom(corr_coefs)

    lower_triangular_indices = np.tril_indices(transformed_corr_coefs.shape[0], -1)
    feature_vector = transformed_corr_coefs[lower_triangular_indices]

    feature_vecs[i] = feature_vector

  return feature_vecs

def get_top_features_from_SVM_RFE(X, Y, N):
  svm = SVC(kernel="linear")

  rfe = RFE(estimator=svm, n_features_to_select=N, step=20, verbose=1)

  rfe.fit(X, Y)

  top_features = rfe.transform(X)

  return top_features

def fishers_z_transform(x):
  # Handling the case where correlation coefficient is 1 or -1
  if x == 1:
    return np.inf
  elif x == -1:
    return -np.inf
  else:
    return 0.5 * np.log((1 + x) / (1 - x))
  

class SparseAutoencoder(nn.Module):
  def __init__(self, input_size, encoded_output_size):
    super(SparseAutoencoder, self).__init__()

    self.encoder = nn.Linear(input_size, encoded_output_size)
    self.decoder = nn.Linear(encoded_output_size, input_size)
  
  def forward(self, x):
    encoded = torch.relu(self.encoder(x))
    decoded = self.decoder(encoded)
    return encoded, decoded
  
class SoftmaxClassifier(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):  # Add hidden_size
    super(SoftmaxClassifier, self).__init__()
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()  # Introduce a nonlinear activation function
    self.linear2 = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    out = self.linear1(x)
    out = self.relu(out)  # Apply the activation
    out = self.linear2(out)
    return out
  
class CustomDataset(Dataset):
  def __init__(self, data, labels):
    'Initialization'
    self.labels = labels
    self.data = data

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]


def encode_data(dataloader, sae1, sae2, device):
  encoded_data = []
  labels = []

  for batch in dataloader:
    data, label = batch
    data = data.float().to(device)

    with torch.no_grad():
      encoded_features, _ = sae1(data)
      encoded_features, _ = sae2(encoded_features)
      encoded_data.append(encoded_features)
      labels.append(label)

  return encoded_data, labels
  


if __name__ == "__main__":
  print("Hi this is main")
  
  use_cuda = torch.cuda.is_available()
  print("Torch Cuda is Available =",use_cuda)

  device = torch.device("cuda:0" if use_cuda else "cpu")
  torch.backends.cudnn.benchmark = True

  seed = int(np.random.rand() * (2**32 - 1))
  seed = 723708028

  # 2071878563 ccs aal 84%
  # 723708028 dparsf aal 84% with 1 softmax layer
  # 723708028 dparsf aal 85.13% with 2 softmax layer

  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  if use_cuda:
      torch.cuda.manual_seed_all(seed)

  data, labels = get_data_from_abide()
  labels = np.array(labels)
  
  #Convert labels from 1, 2 to 0, 1 for PyTorch compatibility
  labels = labels - 1


  #feature_vecs = get_feature_vecs(data)

  #top_features = get_top_features_from_SVM_RFE(feature_vecs, labels, 1000)
  #np.savetxt("top_features_dparsf_aal_116_step20.csv", top_features, delimiter=",")
  
  top_features = np.loadtxt('top_features_dparsf_aal_116_step20.csv', delimiter=',')
  
  train_idx, test_idx = train_test_split(list(range(len(top_features))), test_size=0.2)

  dataset = {}
  label = {}

  dataset['train'] = Subset(top_features, train_idx)
  label['train'] = Subset(labels, train_idx)

  dataset['test'] = Subset(top_features, test_idx)
  label['test'] = Subset(labels, test_idx)

  train_set = CustomDataset(dataset['train'], label['train'])
  test_set = CustomDataset(dataset['test'], label['test'])

  params = {
    'batch_size': 128,
    'shuffle': True,
    'num_workers': 0
  }

  test_params = {
    'batch_size': 128,
    'num_workers': 0
  }

  train_dataloader = DataLoader(train_set, **params)
  test_dataloader = DataLoader(test_set, **test_params)

  SAE1 = SparseAutoencoder(1000, 500).to(device)
  SAE2 = SparseAutoencoder(500, 200).to(device)
  classifier = SoftmaxClassifier(200, 100, 2).to(device) 

  train_model = True
  if (train_model):
    SAE1_epochs = 200
    optimizer_sae1 = optim.Adam( SAE1.parameters(), lr=0.001, weight_decay=1e-4 )
    
    SAE2_epochs = 200
    optimizer_sae2 = optim.Adam( SAE2.parameters(), lr=0.001, weight_decay=1e-4 )

    classifier_epochs = 150
    optimizer_classifier = optim.Adam( classifier.parameters(), lr=0.001, weight_decay=1e-4 )

    sae_criterion = nn.MSELoss()
    classifier_criterion = nn.CrossEntropyLoss()

    
    loss_sae1 =[]
    #Train SAE 1
    for epoch in range(SAE1_epochs):
      for batch in train_dataloader:
        data, labels = batch
        data = data.float().to(device) 

        optimizer_sae1.zero_grad()

        encoded_features, decoded_featues = SAE1(data)

        loss = sae_criterion(decoded_featues, data)
        loss.backward()
        optimizer_sae1.step()
      loss_sae1.append(loss.item())
      print(f"SAE 1: Epoch {epoch}, loss {loss.item()}")
    
    print("======================================\nTrained SAE 1\n======================================")
    
    encoded_features_from_sae1 = []
    labels_from_sae1 = []

    for batch in train_dataloader:
      data, labels = batch
      data = data.float().to(device) 

      with torch.no_grad():
        encoded_features, _ = SAE1(data)
        encoded_features_from_sae1.append(encoded_features)
        labels_from_sae1.append(labels)

    encoded_dataset_tensor = torch.cat(encoded_features_from_sae1, dim=0)
    labels_tensor = torch.cat(labels_from_sae1, dim=0)

    encoded_dataset = TensorDataset(encoded_dataset_tensor, labels_tensor) 

    encoded_dataset_loader = DataLoader(encoded_dataset, **params)

    loss_sae2 = []
    # Train SAE 2
    for epoch in range(SAE2_epochs):
      for batch in encoded_dataset_loader:
        data, labels = batch
        data = data.float().to(device) 

        optimizer_sae2.zero_grad()

        encoded_features, decoded_featues = SAE2(data)

        loss = sae_criterion(decoded_featues, data)
        loss.backward()
        optimizer_sae2.step()
      loss_sae2.append(loss.item())
      print(f"SAE 2: Epoch {epoch}, loss {loss.item()}")

    print("======================================\nTrained SAE 2\n======================================")
    
    encoded_features_from_sae2 = []
    labels_from_sae2 = []
    for batch in encoded_dataset_loader:
      data, labels = batch
      data = data.float().to(device) 

      with torch.no_grad():
        encoded_features, _ = SAE2(data)
        encoded_features_from_sae2.append(encoded_features)
        labels_from_sae2.append(labels)

    encoded_dataset_tensor = torch.cat(encoded_features_from_sae2, dim=0)
    labels_tensor = torch.cat(labels_from_sae2, dim=0)

    encoded_dataset = TensorDataset(encoded_dataset_tensor, labels_tensor) 

    encoded_dataset_loader = DataLoader(encoded_dataset, **params)

    loss_classifier = []
    # Train classifier
    for epoch in range(classifier_epochs):
      for batch in encoded_dataset_loader:
        data, labels = batch
        data = data.float().to(device) 
        labels = labels.long().to(device) 

        optimizer_classifier.zero_grad()

        classifier_output = classifier(data)

        loss = classifier_criterion(classifier_output, labels)
        loss.backward()
        optimizer_classifier.step()
      loss_classifier.append(loss.item())
      print(f"Classifier: Epoch {epoch} loss: {loss.item()}")

    print("======================================\nTrained classifier\n======================================")

    dig, axs = plt.subplots(1, 3, figsize=(15,5))
    
    axs[0].plot(range(SAE1_epochs), loss_sae1)
    axs[0].set_title('SAE1 Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')

    # Plot for SAE2
    axs[1].plot(range(SAE2_epochs), loss_sae2)
    axs[1].set_title('SAE2 Loss')
    axs[1].set_xlabel('Epoch')
    # axs[1].set_ylabel('Loss')  # Optional, as it shares the y-axis with the first plot

    # Plot for Classifier
    axs[2].plot(range(classifier_epochs), loss_classifier)
    axs[2].set_title('Classifier Loss')
    axs[2].set_xlabel('Epoch')
    # axs[2].set_ylabel('Loss')  # Optional, as it shares the y-axis with the first plot

    plt.tight_layout()  # Adjust the padding between and around subplots
    plt.show()

    torch.save(SAE1.state_dict(), 'SAE1.pth')
    torch.save(SAE2.state_dict(), 'SAE2.pth')
    torch.save(classifier.state_dict(), 'classifier.pth')
  else:
    SAE1.load_state_dict(torch.load('SAE1.pth'))
    SAE2.load_state_dict(torch.load('SAE2.pth'))
    classifier.load_state_dict(torch.load('classifier.pth'))

  print("Infer data from trained SAE")
  encoded_test_data, test_labels = encode_data(test_dataloader, SAE1, SAE2, device)

  classifier.eval()

  total = 0
  correct = 0


  for i in range(len(encoded_test_data)):
    data = encoded_test_data[i].float().to(device) 
    labels = test_labels[i].long().to(device) 

    outputs = classifier(data)

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

  accuracy = 100 * correct / total
  print(f'Accuracy of the model on the test dataset: {accuracy:.2f}%')

  pdb.set_trace()