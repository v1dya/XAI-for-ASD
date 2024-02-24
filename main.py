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
from sklearn.model_selection import train_test_split, StratifiedKFold

def get_data_from_abide():
  downloads = 'abide/downloads/Outputs/ccs/filt_global/rois_aal/'
  pheno_file = 'data/Phenotypic_V1_0b_preprocessed1.csv'

  pheno_file = open(pheno_file, 'r')
  pheno_list = pheno_file.readlines()

  labels_dict = {}
  for i in pheno_list[1:]:
    file_name = i.split(',')[6]
    diagnosis = i.split(',')[7]

    labels_dict[file_name] = float(diagnosis) # Save labels alongisde their filenames

  data = []
  labels = []

  for filename in sorted(os.listdir(downloads)):
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

def get_top_features_from_SVM_RFE(X, Y, N, step):
  svm = SVC(kernel="linear")

  rfe = RFE(estimator=svm, n_features_to_select=N, step=step, verbose=1)

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
  def __init__(self, input_size, encoded_output_size, rho=0.2, beta=2, criterion=nn.MSELoss()):
    """
    rho: desired sparsity parameter
    beta: weight of the KL divergence term
    """
    super(SparseAutoencoder, self).__init__()

    self.encoder = nn.Linear(input_size, encoded_output_size)
    self.decoder = nn.Linear(encoded_output_size, input_size)
    self.rho = rho
    self.beta = beta
    self.criterion = criterion
  
  def kl_divergence(self, rho, rho_hat):
    """Calculates KL divergence for regularization."""
    return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)) 
  
  def forward(self, x):
    encoded = torch.relu(self.encoder(x))

    # Compute average activation of hidden neurons
    rho_hat = torch.mean(encoded, dim=0) 

    kl_loss = self.kl_divergence(self.rho, rho_hat).sum()

    decoded = self.decoder(encoded)

    # Total loss: Reconstruction loss + KL divergence
    mse_loss = self.criterion(decoded, x)
    loss = mse_loss + self.beta * kl_loss 

    return encoded, decoded, loss
  
class SoftmaxClassifier(nn.Module):
  def __init__(self, input_size, num_classes):
    super(SoftmaxClassifier, self).__init__()
    self.linear = nn.Linear(input_size, num_classes)

  def forward(self, x):
    out = self.linear(x)
    return out
  
class StackedSparseAutoencoder(nn.Module):
  def __init__(self, SAE1, SAE2, classifier):
      super(StackedSparseAutoencoder, self).__init__()
      self.sae1 = SAE1  # Assuming you have your pre-trained SAE1
      self.sae2 = SAE2  # Assuming you have your pre-trained SAE2
      self.classifier = classifier 

  def forward(self, x):
      x = self.sae1.encoder(x)  # Pass through the encoder of SAE1
      x = self.sae2.encoder(x)  # Pass through the encoder of SAE2
      x = self.classifier(x)
      return x
  
class CustomDataset(Dataset):
  def __init__(self, data, labels):
    'Initialization'
    self.labels = labels
    self.data = data

  def __len__(self):
    return len(self.data.indices)
  
  def __getitem__(self, idx):
    data_idx = self.data.indices[idx]  # Get index into the original dataset
    labels_idx = self.labels.indices[idx]  # Get index into the original dataset
    return self.data.dataset[data_idx], self.labels.dataset[labels_idx] 


def encode_data(dataloader, sae1, sae2, device):
  encoded_data = []
  labels = []

  for batch in dataloader:
    data, label = batch
    data = data.float().to(device)

    with torch.no_grad():
      encoded_features, _, __ = sae1(data)
      encoded_features, _, __ = sae2(encoded_features)
      encoded_data.append(encoded_features)
      labels.append(label)

  return encoded_data, labels

def get_encoded_data(model, dataloader, dataloader_params, device):
  """Encodes data from a dataloader using a given model.
  
  Args:
      model: The PyTorch model used for encoding.
      dataloader: The PyTorch dataloader containing the data.
      device: The device (e.g., 'cuda:0' or 'cpu') where the model and data should be sent.

  Returns:
      A tuple: (encoded_dataset, encoded_dataset_loader), where
          * encoded_dataset is a TensorDataset containing the encoded features and labels.
          * encoded_dataset_loader is a DataLoader for the encoded_dataset.
  """  

  encoded_features_from_model = []
  labels_from_model = []

  for batch in dataloader:
    data, labels = batch
    data = data.float().to(device) 

    with torch.no_grad():
      encoded_features, _, __ = model(data)  # Assuming your model outputs encoded features, ...
      encoded_features_from_model.append(encoded_features)
      labels_from_model.append(labels)

  encoded_dataset_tensor = torch.cat(encoded_features_from_model, dim=0)
  labels_tensor = torch.cat(labels_from_model, dim=0)

  encoded_dataset = TensorDataset(encoded_dataset_tensor, labels_tensor)
  encoded_dataset_loader = DataLoader(encoded_dataset, **dataloader_params)

  return encoded_dataset, encoded_dataset_loader 
  


if __name__ == "__main__":
  use_cuda = torch.cuda.is_available()
  print("Torch Cuda is Available =",use_cuda)

  device = torch.device("cuda:0" if use_cuda else "cpu")

  # seed = int(np.random.rand() * (2**32 - 1))
  seed = 2071878563

  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  if use_cuda:
      torch.cuda.manual_seed_all(seed)

  data, labels = get_data_from_abide()
  labels_from_abide = np.array(labels)
  
  #Convert labels from 1, 2 to 0, 1 for PyTorch compatibility
  labels_from_abide = labels_from_abide - 1


  #feature_vecs = get_feature_vecs(data)

  #top_features = get_top_features_from_SVM_RFE(feature_vecs, labels, 1000, 20)
  #np.savetxt("sorted_top_features_116_step20.csv", top_features, delimiter=",")
  
  top_features = np.loadtxt('sorted_top_features_116_step20.csv', delimiter=',')
  
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)  # Example with 5 folds

  avg_TP, avg_FP, avg_FN, avg_TN = [], [], [], []

  fold = 0
  for train_idx, test_idx in skf.split(top_features, labels_from_abide):
    fold+=1
    print(f'======================================\nSplit {fold}\n======================================')
    dataset = {}
    label = {}

    # Split the training set into training and validation
    train_subidx, val_subidx = train_test_split(train_idx, test_size=0.1, random_state=seed)  # Adjust test_size as needed

    dataset['train'] = Subset(top_features, train_subidx)
    label['train'] = Subset(labels_from_abide, train_subidx)

    dataset['val'] = Subset(top_features, val_subidx)
    label['val'] = Subset(labels_from_abide, val_subidx)

    dataset['test'] = Subset(top_features, test_idx)
    label['test'] = Subset(labels_from_abide, test_idx)

    print("Total: ", len(top_features))  # Original dataset size
    print("Train: ", len(dataset['train'].indices)) 
    print("Test: ", len(dataset['test'].indices)) 
    print("Validation: ", len(dataset['val'].indices)) 

    train_set = CustomDataset(dataset['train'], label['train'])
    test_set = CustomDataset(dataset['test'], label['test'])
    val_set = CustomDataset(dataset['val'],label['val'])

    params = {
      'batch_size': 128,
      'shuffle': True,
      'num_workers': 0
    }

    val_params = {
      'batch_size': 128,
      'shuffle': False,
      'num_workers': 0
    }

    test_params = {
      'batch_size': 128,
      'shuffle': False,
      'num_workers': 0
    }
    
    train_dataloader = DataLoader(train_set, **params)
    test_dataloader = DataLoader(test_set, **test_params)
    val_dataloader = DataLoader(val_set, **val_params)

    SAE1 = SparseAutoencoder(1000, 500).to(device)
    SAE2 = SparseAutoencoder(500, 100).to(device)
    classifier = SoftmaxClassifier(100, 2).to(device)
    model = StackedSparseAutoencoder(SAE1, SAE2, classifier).to(device)

    verbose = True
    train_model = False
    save_model = False
    if (train_model):
      SAE1_epochs = 50
      optimizer_sae1 = optim.Adam( SAE1.parameters(), lr=0.001, weight_decay=1e-4 )
      
      SAE2_epochs = 50
      optimizer_sae2 = optim.Adam( SAE2.parameters(), lr=0.001, weight_decay=1e-4 )

      classifier_epochs = 100
      optimizer_classifier = optim.Adam( classifier.parameters(), lr=0.001, weight_decay=1e-4 )

      sae_criterion = nn.MSELoss()
      classifier_criterion = nn.CrossEntropyLoss()

      fine_tuning_epochs = 2000

      loss_sae1 = []
      val_sae1 = []

      for epoch in range(SAE1_epochs):
        for batch in train_dataloader:
          data, labels = batch
          data = data.float().to(device) 

          optimizer_sae1.zero_grad()

          encoded_features, decoded_featues, loss = SAE1(data)

          loss.backward()
          optimizer_sae1.step()
        loss_sae1.append(loss.item())

        val_loss = 0.0
        with torch.no_grad():
          for batch in val_dataloader:
            data, labels = batch
            data = data.float().to(device)

            encoded_features, decoded_featues, loss = SAE1(data)
            val_loss += loss.item()

        val_loss /= len(val_dataloader)
        val_sae1.append(val_loss)

        if verbose:
          print(f"SAE 1: Epoch {epoch}, loss {loss.item()}, validation loss {val_loss}")
      
      print("======================================\nTrained SAE 1\n======================================")

      encoded_dataset, encoded_dataset_loader = get_encoded_data(SAE1, train_dataloader, params, device)

      val_encoded_dataset, val_encoded_dataset_loader = get_encoded_data(SAE1, val_dataloader, val_params, device)

      loss_sae2 = []
      val_sae2 = []

      for epoch in range(SAE2_epochs):
        for batch in encoded_dataset_loader:
          data, labels = batch
          data = data.float().to(device) 

          optimizer_sae2.zero_grad()

          encoded_features, decoded_featues, loss = SAE2(data)

          loss.backward()
          optimizer_sae2.step()
        loss_sae2.append(loss.item())

        val_loss = 0.0
        with torch.no_grad():
          for batch in val_encoded_dataset_loader:
            data, labels = batch
            data = data.float().to(device)

            encoded_features, decoded_featues, loss = SAE2(data)
            val_loss += loss.item()

        val_loss /= len(val_encoded_dataset_loader)
        val_sae2.append(val_loss)

        if verbose:
          print(f"SAE 2: Epoch {epoch}, loss {loss.item()}, validation loss {val_loss}")

      print("======================================\nTrained SAE 2\n======================================")

      encoded_dataset, encoded_dataset_loader = get_encoded_data(SAE2, encoded_dataset_loader, params, device)

      val_encoded_dataset, val_encoded_dataset_loader = get_encoded_data(SAE2, val_encoded_dataset_loader, val_params, device)

      loss_classifier = []
      val_classifier = []

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

        val_loss = 0.0
        with torch.no_grad():
          for batch in val_encoded_dataset_loader:
            data, labels = batch
            data = data.float().to(device)
            labels = labels.long().to(device) 

            classifier_output = classifier(data)
            loss = classifier_criterion(classifier_output, labels)

            val_loss += loss.item()

        val_loss /= len(val_encoded_dataset_loader)  # Average validation loss
        val_classifier.append(val_loss)

        if verbose:
          print(f"Classifier: Epoch {epoch}, loss {loss.item()}, validation loss {val_loss}")

      print("======================================\nTrained classifier\n======================================")

      optimizer = optim.Adam(model.parameters(), lr=0.0001)

      loss_model = []
      accuracy_model = []
      val_model = []
      val_accuracy_model = []
      for epoch in range(fine_tuning_epochs):
        total = 0
        correct = 0
        for batch in train_dataloader:
          data, labels = batch
          data = data.float().to(device)
          labels = labels.long().to(device) 

          optimizer.zero_grad()
          outputs = model(data)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

          loss = classifier_criterion(outputs, labels) 
          loss.backward()
          optimizer.step()
        loss_model.append(loss.item())
        accuracy_model.append(100 * correct / total)

        val_loss = 0.0
        val_total = 0
        val_correct = 0
        with torch.no_grad():
          for batch in val_dataloader:
            data, labels = batch
            data = data.float().to(device)
            labels = labels.long().to(device) 

            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)

            loss = classifier_criterion(outputs, labels) 

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            val_loss += loss.item()

        val_accuracy_model.append(100 * val_correct / val_total)
        val_loss /= len(val_dataloader)
        val_model.append(val_loss)

        if verbose:
          print(f"Model: Epoch {epoch}, loss {loss.item()}, validation loss {val_loss}")

      print("======================================\nFine tuned model\n======================================")

      dig, axs = plt.subplots(1, 5, figsize=(15,5))
      
      # Plot for SAE1
      axs[0].plot(range(SAE1_epochs), loss_sae1, label='Training Loss')
      axs[0].plot(range(SAE1_epochs), val_sae1, label='Validation Loss')
      axs[0].set_title('SAE1 Loss')
      axs[0].set_xlabel('Epoch')
      axs[0].set_ylabel('Loss')
      axs[0].legend()

      # Plot for SAE2
      axs[1].plot(range(SAE2_epochs), loss_sae2, label='Training Loss')
      axs[1].plot(range(SAE2_epochs), val_sae2, label='Validation Loss')
      axs[1].set_title('SAE2 Loss')
      axs[1].set_xlabel('Epoch')
      axs[1].legend()

      # Plot for classifier
      axs[2].plot(range(classifier_epochs), loss_classifier, label='Training Loss')
      axs[2].plot(range(classifier_epochs), val_classifier, label='Validation Loss')
      axs[2].set_title('Classifier Loss')
      axs[2].set_xlabel('Epoch')
      axs[2].legend()

      # Plot for Model
      axs[3].plot(range(fine_tuning_epochs), loss_model, label='Training Loss')
      axs[3].plot(range(fine_tuning_epochs), val_model, label='Validation Loss')
      axs[3].set_title('Model Loss')
      axs[3].set_xlabel('Epoch')
      axs[3].legend()

      # Plot for Accuracy over fine tuning
      axs[4].plot(range(fine_tuning_epochs), accuracy_model, label='Training Accuracy')
      axs[4].plot(range(fine_tuning_epochs), val_accuracy_model, label='Validation Accuracy')
      axs[4].set_title('Accuracy')
      axs[4].set_xlabel('Epoch')
      axs[4].legend()

      plt.tight_layout()
      
      if verbose:
        plt.show()

      # pdb.set_trace()

      if save_model:
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

    true_labels = np.array([])
    predicted_labels = np.array([])

    for i in range(len(encoded_test_data)):
      data = encoded_test_data[i].float().to(device) 
      labels = test_labels[i].long().to(device) 

      outputs = classifier(data)

      _, predicted = torch.max(outputs.data, 1)

      true_labels = np.concatenate((true_labels,labels.cpu().numpy()),axis=0)
      predicted_labels = np.concatenate((predicted_labels,predicted.cpu().numpy()),axis=0)

    TP,FP,TN,FN = 0,0,0,0

    for true_label, predicted_label in zip(true_labels, predicted_labels):
      if true_label == predicted_label == 0:
          TP += 1  # True Positive
      elif true_label == predicted_label == 1:
          TN += 1  # True Negative
      elif true_label == 1 and predicted_label == 0:
          FP += 1  # False Positive
      elif true_label == 0 and predicted_label == 1:
          FN += 1  # False Negative
    
    avg_TP.append(TP)
    avg_FP.append(FP)
    avg_FN.append(FN)
    avg_TN.append(TN)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN) 
    recall = TN / (TN + FP) 
    precision = TP / (TP + FP)
    f1 = (2 * precision * sensitivity) / (precision + sensitivity)
    cm = np.array([[TP,FP],[FN,TN]])

    print(f'Accuracy: {(accuracy * 100):.2f}%')
    print(f'Recall: {recall:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'F1_Score: {f1:.2f}')
    print(f'Confusion Matrix:\n{cm}')
  
  print("======================================\nCompleted splits\n======================================")
  TP = sum(avg_TP)/len(avg_TP)
  FP = sum(avg_FP)/len(avg_FP)
  FN = sum(avg_FN)/len(avg_FN)
  TN = sum(avg_TN)/len(avg_TN)

  accuracy = (TP + TN) / (TP + TN + FP + FN)
  sensitivity = TP / (TP + FN) 
  recall = TN / (TN + FP) 
  precision = TP / (TP + FP)
  f1 = (2 * precision * sensitivity) / (precision + sensitivity)
  cm = np.array([[TP,FP],[FN,TN]])

  print(f'Accuracy: {(accuracy * 100):.2f}%')
  print(f'Recall: {recall:.2f}')
  print(f'Precision: {precision:.2f}')
  print(f'F1_Score: {f1:.2f}')
  print(f'Confusion Matrix:\n{cm}')
