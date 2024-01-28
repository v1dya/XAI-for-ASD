import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import numpy as np
import os
import pdb
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

def get_data_from_abide():
  downloads = 'abide/downloads/Outputs/ccs/filt_global/rois_aal/'
  pheno_file = 'abide/Phenotypic_V1_0b_preprocessed1.csv'

  pheno_file = open(pheno_file, 'r')
  pheno_list = pheno_file.readlines()
  
  labels_dict = {}
  for i in pheno_list[1:]:
    file_name = i.split(',')[6]
    diagnosis = i.split(',')[7]

    labels_dict[file_name] = int(diagnosis) # Save labels alongisde their filenames

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

  pdb.set_trace()
  rfe.fit(X, Y)
  pdb.set_trace()
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
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return encoded, decoded
  
class CustomDataset(Dataset):
  def __init__(self, data, labels):
    'Initialization'
    self.labels = labels
    self.data = data

  def __len__(self):
    return len(data)
  
  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]

  


if __name__ == "__main__":
  print("Hi this is main")
  
  use_cuda = torch.cuda.is_available()
  print("Torch Cuda is Available =",use_cuda)

  device = torch.device("cuda:0" if use_cuda else "cpu")
  torch.backends.cudnn.benchmark = True

  data, labels = get_data_from_abide()
  labels = np.array(labels)

  # feature_vecs = get_feature_vecs(data)

  # top_features = get_top_features_from_SVM_RFE(feature_vecs, labels, 1000)
  
  top_features = np.loadtxt('top_features_116_step20.csv', delimiter=',')  
  
  batch_size = 64

  pdb.set_trace()
  

  SAE1 = SparseAutoencoder(1000, 200)
  SAE1_epochs = 20

  SAE2 = SparseAutoencoder(200, 100)
  SAE2_epochs = 20

  train_idx, test_idx = train_test_split(list(range(len(top_features))), test_size=0.2)

  dataset = {}
  label = {}

  dataset['train'] = Subset(top_features, train_idx)
  label['train'] = Subset(labels, train_idx)

  dataset['test'] = Subset(top_features, test_idx)
  label['test'] = Subset(labels, test_idx)

  train_set = CustomDataset(dataset['train'].dataset, label['train'].dataset)
  test_set = CustomDataset(dataset['test'].dataset, label['test'].dataset)
  pdb.set_trace()
  params = {
    'batch_size': 64,
    'shuffle': True,
    'num_workers': 1
  }

  train_dataloader = DataLoader(train_set, **params)
  test_dataloader = DataLoader(test_set, **params)

  criterion = nn.BCELoss()  # Binary Cross Entropy Loss
  

  # Loop over epochs
  for epoch in range(SAE1_epochs):
      # Training
      for batch in train_dataloader:
          # Transfer to GPU
          pdb.set_trace()
          X, y = batch

          # Model computations
          print(len(X))
          print("label",len(y))


