import random
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import colormaps, colorbar
import collections
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd
import os
import pdb
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, StratifiedKFold
from captum.attr import IntegratedGradients, DeepLiftShap, DeepLift, GradientShap, ShapleyValueSampling, ShapleyValues, FeatureAblation, GuidedBackprop, Occlusion
from nilearn import datasets, plotting
import networkx as nx
from lime import lime_tabular

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def get_data_from_abide(pipeline):
  downloads = f'abide/downloads/Outputs/{pipeline}/filt_global/rois_aal/'
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
  feature_vecs = []
  feature_indices = []

  vectorized_fisher_transfrom = np.vectorize(fishers_z_transform)
  
  for i in range(len(data)):
    corr_coefs = np.corrcoef(data[i], rowvar=False)
    corr_coefs = np.nan_to_num(corr_coefs)
    f = []
    idx = []

    transformed_corr_coefs = vectorized_fisher_transfrom(corr_coefs)

    lower_triangular_indices = np.tril_indices(transformed_corr_coefs.shape[0], -1)

    for row_idx, col_idx in zip(*lower_triangular_indices):  # Unpack indices
      coefficient = transformed_corr_coefs[row_idx, col_idx]
      f.append(coefficient)
      idx.append([row_idx, col_idx])

    feature_vecs.append(f)
    feature_indices.append(idx)

  feature_vecs = np.array(feature_vecs)
  feature_indices = np.array(feature_indices)

  return feature_vecs, feature_indices

def get_top_features_from_SVM_RFE(X, Y, indices, N, step):
  svm = SVC(kernel="linear")
  rfe = RFE(estimator=svm, n_features_to_select=N, step=step, verbose=1)

  rfe.fit(X, Y)

  top_features = rfe.transform(X)
  top_indices = np.where(rfe.support_)[0]

  top_ROIs = []

  for i in top_indices:
    roi = indices[0][i]
    top_ROIs.append(roi)

  top_ROIs = np.array(top_ROIs)

  return top_features, top_ROIs 

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

def find_top_rois_using_LIME(N, model, test_dataloader, train_dataloader, rois):
  features_list = []
  labels_list = []

  for features, labels in train_dataloader:
      # Move data to CPU if it's on a GPU
      features = features.cpu().numpy()
      labels = labels.cpu().numpy()
      
      # Append to lists
      features_list.append(features)
      labels_list.append(labels)

  # Concatenate all features and labels
  X_train = np.concatenate(features_list, axis=0)
  Y_train = np.concatenate(labels_list, axis=0)

  # Assuming 'dataloader' is your DataLoader instance
  features_list = []
  labels_list = []

  for features, labels in test_dataloader:
      # Move data to CPU if it's on a GPU
      features = features.cpu().numpy()
      labels = labels.cpu().numpy()
      
      # Append to lists
      features_list.append(features)
      labels_list.append(labels)

  # Concatenate all features and labels
  X_test = np.concatenate(features_list, axis=0)
  Y_test = np.concatenate(labels_list, axis=0)

  # Initialize LIME Explainer for tabular data
  explainer = lime_tabular.LimeTabularExplainer(
      training_data=X_train,  # Use your training data here
      feature_names=list(range(X_train.shape[1])),  # Feature names or indices
      class_names=['Class 0', 'Class 1'],  # Output classes
      mode='classification'
  )

  # Select an instance to explain
  instance_index = 0  # Example index, choose appropriately
  instance = X_test[instance_index]

  # Generate explanation for the selected instance
  explanation = explainer.explain_instance(
      data_row=instance, 
      predict_fn=model_predict_lime,  # Use the prediction function defined above
      num_features=1000,  # Number of top features you want to show
      top_labels=1  # Number of top labels for multi-class classification
  )

  feature_weights = explanation.as_list(label=explanation.top_labels[0])

  sorted_features = sorted(feature_weights, key=lambda x: abs(x[1]), reverse=True)

  sorted_feature_indices = [find_index_from_string(feature[0]) for feature in sorted_features]

  sorted_feature_weights = [abs(feature[1]) for feature in sorted_features]

  return rois[sorted_feature_indices[:N]], sorted_feature_weights[:N]

def find_top_rois_using_SHAP(N, model, test_dataloader, train_dataloader, rois):
  # Select a background dataset from train_dataloader
  background_data = []
  for batch in train_dataloader:
      data, labels = batch
      data = data.float().to(device) 
      labels = labels.long().to(device)

      background_data.append(data)
      if len(background_data) >= 100:  # Collect 100 samples, adjust as needed
          break
  background_data = torch.cat(background_data)[:100]  # Adjust size as needed

  # Select test instances from test_dataloader
  test_instances = []
  for batch in test_dataloader:
      data, labels = batch
      data = data.float().to(device) 
      labels = labels.long().to(device) 

      test_instances.append(data)
      if len(test_instances) >= 5:  # Let's say we want to explain 5 test instances
          break
  test_instances = torch.cat(test_instances)[:5]  # Adjust size as needed

  background_data = background_data.to(device)
  test_instances = test_instances.to(device)

  # Initialize SHAP DeepExplainer
  explainer = shap.DeepExplainer(model, background_data)

  # Compute SHAP values for test_instances
  shap_values = explainer.shap_values(test_instances)

  mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)

  feature_importance = np.mean(mean_abs_shap_values, axis=1)

  # Step 4: Find indices of top 100 features
  top_indices = np.argsort(feature_importance)[-(N):][::-1]

  return rois[top_indices], feature_importance[top_indices]
  
def find_top_rois_using_integrated_gradients(N, model, test_dataloader, rois):
  for batch in test_dataloader:
    data, labels = batch
    data = data.float().to(device) 
    labels = labels.long().to(device) 
    break

  model.eval()
  data.requires_grad = True  # Enable gradient computation on the input

  # Baseline (here, a tensor of zeros)
  baseline = torch.zeros_like(data).to(device)

  # Initialize Integrated Gradients with model
  integrated_gradients = IntegratedGradients(model)

  # Compute attributions for the autism positive class
  # Assuming the first output (index 0) corresponds to autism positive
  # Convergence delta is approximate error
  attributions_ig, delta = integrated_gradients.attribute(data, baselines=baseline, target=0, return_convergence_delta=True)

  # Calculate the mean of the attributions across all input samples to get an average importance
  # Postive attribution means positive contribution of that feature
  # Negative attribution means negative contribution of that feature
  # 0 attribution means 0 contribution of that feature
  attributions_mean = attributions_ig.mean(dim=0).cpu().detach().numpy()

  # To get top features we need to take the features in descending order of attribution value to get most contributing features
  abs_attribution = np.abs(attributions_mean)

  top_indices = np.argsort(abs_attribution)[-(N):][::-1]

  return rois[top_indices], abs_attribution[top_indices]

def find_top_rois_using_DeepLift(N, model, test_dataloader, rois):
  for batch in test_dataloader:
    data, labels = batch
    data = data.float().to(device) 
    labels = labels.long().to(device) 
    break

  model.eval()
  data.requires_grad = True  # Enable gradient computation on the input

  deep_lift = DeepLift(model)
  attributions_dl = deep_lift.attribute(data, target=0)

  attributions_mean = attributions_dl.mean(dim=0).cpu().detach().numpy()

  abs_attribution = np.abs(attributions_mean)

  top_indices = np.argsort(abs_attribution)[-(N):][::-1]

  return rois[top_indices], abs_attribution[top_indices]

def find_top_rois_using_DeepLiftShap(N, model, test_dataloader, rois):
  for batch in test_dataloader:
    data, labels = batch
    data = data.float().to(device) 
    labels = labels.long().to(device) 
    break

  model.eval()
  data.requires_grad = True  # Enable gradient computation on the input

  baseline = torch.zeros_like(data).to(device)

  deep_lift_shap = DeepLiftShap(model)
  attributions_dls = deep_lift_shap.attribute(data, baselines=baseline, target=0)

  attributions_mean = attributions_dls.mean(dim=0).cpu().detach().numpy()

  abs_attribution = np.abs(attributions_mean)

  top_indices = np.argsort(abs_attribution)[-(N):][::-1]

  return rois[top_indices], abs_attribution[top_indices]

def find_top_rois_using_GradientShap(N, model, test_dataloader, rois):
  for batch in test_dataloader:
    data, labels = batch
    data = data.float().to(device) 
    labels = labels.long().to(device) 
    break

  model.eval()
  data.requires_grad = True  # Enable gradient computation on the input

  baseline = torch.zeros_like(data).to(device)

  gradient_shap = GradientShap(model)

  attributions_gs = gradient_shap.attribute(data, baselines=baseline, target=0)

  attributions_mean = attributions_gs.mean(dim=0).cpu().detach().numpy()

  abs_attribution = np.abs(attributions_mean)

  top_indices = np.argsort(abs_attribution)[-(N):][::-1]

  return rois[top_indices], abs_attribution[top_indices]

def find_top_rois_using_GuidedBackprop(N, model, test_dataloader, rois):
  for batch in test_dataloader:
    data, labels = batch
    data = data.float().to(device) 
    labels = labels.long().to(device) 
    break

  model.eval()
  data.requires_grad = True  # Enable gradient computation on the input

  baseline = torch.zeros_like(data).to(device)

  guided_backprop = GuidedBackprop(model)

  attributions_gb = guided_backprop.attribute(data, target=0)

  attributions_mean = attributions_gb.mean(dim=0).cpu().detach().numpy()

  abs_attribution = np.abs(attributions_mean)

  top_indices = np.argsort(abs_attribution)[-(N):][::-1]

  return rois[top_indices], abs_attribution[top_indices]

def get_threshold_from_percentile(adjacency_matrix, percentile):
  all_weights = adjacency_matrix[np.nonzero(adjacency_matrix)]
  threshold = np.percentile(all_weights, percentile) 
  return threshold

def expand_relative_coords(coordinates, percent):
  # Calculate center
  center = np.mean(coordinates, axis=0)

  # Center the coordinates
  centered_coordinates = coordinates - center 

  # Scale the coordinates
  scaled_coordinates = centered_coordinates * percent

  # Shift back to original center
  spread_coordinates = scaled_coordinates + center 

  return spread_coordinates


def print_connections(rois, weights, method, pipeline, show_now=False, save=False):
  atlas = datasets.fetch_atlas_aal(version='SPM5')
  labels = atlas.labels  # List of AAL region labels
  weights = np.array(weights)
  rois = rois.astype(int)

  weights = ((weights - weights.min()) / (weights.max() - weights.min())) * 10

  edge_cmap = colormaps['viridis']  # Colormap choice 

  num_connections = len(rois)

  cmap = colormaps['viridis']

  fig = plt.figure(figsize=(15, 8))
  ax_connection_connectome = fig.add_axes([0.05, 0.55, 0.8, 0.40])
  ax_connection_colorbar = fig.add_axes([0.85, 0.55, 0.05, 0.40])

  ax_roi_connectome = fig.add_axes([0.05, 0.05, 0.8, 0.40])
  ax_roi_colorbar = fig.add_axes([0.85, 0.05, 0.05, 0.40])

  # Set the figure-wide title
  fig.suptitle(f'Top {num_connections} connections and ROI Importance using {method}', fontsize=16)


  G = nx.Graph()

  # Add nodes (brain regions)
  for label in labels:
    G.add_node(label)

  # Add edges with weights
  for i, roi_pair in enumerate(rois):
      roi1_index = int(roi_pair[0])
      roi2_index = int(roi_pair[1])
      roi1_name = labels[roi1_index]
      roi2_name = labels[roi2_index]
      weight = weights[i]
      G.add_edge(roi1_name, roi2_name, weight=weight) 

  node_color = 'grey'

  coordinates = expand_relative_coords(plotting.find_parcellation_cut_coords(atlas.maps), 1.08) 

  adjacency_matrix = nx.adjacency_matrix(G).todense()

  # Dynamic Thresholding
  edge_threshold = get_threshold_from_percentile(adjacency_matrix, 0)  # Show all 

  plotting.plot_connectome(adjacency_matrix, coordinates,
                          node_color=node_color,
                          edge_vmin=0,
                          edge_vmax=weights.max(),
                          edge_cmap=edge_cmap,
                          edge_threshold=edge_threshold,
                          axes=ax_connection_connectome)
  
  norm = Normalize(vmin=weights.min(), vmax=weights.max())

  cb = colorbar.ColorbarBase(ax_connection_colorbar, cmap=cmap,
                                  norm=norm,
                                  orientation='vertical')
  cb.set_label('Importance')

  # Count the occurrence of each ROI
  roi_counts = np.zeros(len(labels))
  all_rois = [int(roi) for pair in rois for roi in pair]  # Flatten list of ROI pairs
  roi_counts = collections.Counter(all_rois)  # Count occurrences of each ROI 
  top_rois, top_counts = zip(*roi_counts.most_common())

  adjacency_matrix = np.zeros((len(coordinates), len(coordinates)))  # No edges

  roi_importances = []

  for idx, label in enumerate(labels):
    if idx in top_rois:
      i = top_rois.index(idx)

      weight = 1

      possible_weights = []
      count_in_rois = 0

      for j in range(len(rois)):
        if float(idx) in rois[j]:
          count_in_rois += 1  
          possible_weights.append(weights[j])  

      if count_in_rois >= 1:
        weight = max(possible_weights)

      roi_importances.append((top_counts[i] + 1)*weight)      
    else:
      roi_importances.append(0.0001)
  
  roi_importances = np.array(roi_importances)

  # Normalize the importance scores for node sizes and colors
  normalized_sizes = 20 + (roi_importances - roi_importances.min()) / (roi_importances.max() - roi_importances.min()) * 180  # Scale between 20 and 200

  normalized_colors = cmap((roi_importances - roi_importances.min()) / (roi_importances.max() - roi_importances.min()))

  plotting.plot_connectome(adjacency_matrix, coordinates,
                         node_color=normalized_colors,
                         node_size=normalized_sizes,
                         display_mode='ortho',
                         colorbar=False,
                         axes=ax_roi_connectome)

  norm = Normalize(vmin=roi_importances.min(), vmax=roi_importances.max())

  cb = colorbar.ColorbarBase(ax_roi_colorbar, cmap=cmap,
                                  norm=norm,
                                  orientation='vertical')
  cb.set_label('Importance')

  if save:
    filename = f"{pipeline}_plot_{method}_{num_connections}_connections.png"
    plt.savefig(filename)

  if show_now:
    plt.show()

  labels = np.array(atlas.labels)
  top_connections = labels[rois[:10]]
  connections_with_weights = np.array([(connection[0], connection[1], np.round(weight, 2)) for connection, weight in zip(top_connections, weights)])

  # Convert the top connections to a DataFrame for nice formatting
  top_connections_df = pd.DataFrame(connections_with_weights, columns=['ROI 1', 'ROI 2', 'Importance'])

  return top_connections_df

def find_index_from_string(stri):
  stri = stri.split(' ')

  for i in stri:
    if i.isnumeric() and float(i) >= 1:
      return int(i)

def model_predict_lime(data):
  # Convert data to tensor, pass through model, and return softmax probabilities
  data_tensor = torch.tensor(data).float().to(device)
  model.eval()
  with torch.no_grad():
    outputs = model(data_tensor)
    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

  return probabilities


if __name__ == "__main__":
  print("Torch Cuda is Available =",use_cuda)
  # seed = int(np.random.rand() * (2**32 - 1))
  seed = 2071878563

  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  if use_cuda:
      torch.cuda.manual_seed_all(seed)

  pipeline = 'ccs'

  data, labels = get_data_from_abide(pipeline)
  labels_from_abide = np.array(labels)
  
  #Convert labels from 1, 2 to 0, 1 for PyTorch compatibility
  labels_from_abide = labels_from_abide - 1


  # feature_vecs, feature_vec_indices = get_feature_vecs(data)

  # top_features, top_rois = get_top_features_from_SVM_RFE(feature_vecs, labels, feature_vec_indices, 1000, 1)

  # np.savetxt(f'sorted_top_features_{pipeline}_116_step1.csv', top_features, delimiter=",")
  # np.savetxt(f'sorted_top_frois_{pipeline}_116_step1.csv', top_rois, delimiter=",")
  
  top_features = np.loadtxt(f'sorted_top_features_{pipeline}_116_step1.csv', delimiter=',')
  top_rois = np.loadtxt(f'sorted_top_rois_{pipeline}_116_step1.csv', delimiter=',')

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
      'batch_size': 64,
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

    verbose = False
    train_model = True
    save_model = True
    if (train_model):
      SAE1_epochs = 50
      optimizer_sae1 = optim.Adam( SAE1.parameters(), lr=0.001, weight_decay=1e-4 )
      
      SAE2_epochs = 50
      optimizer_sae2 = optim.Adam( SAE2.parameters(), lr=0.001, weight_decay=1e-4 )

      classifier_epochs = 100
      optimizer_classifier = optim.Adam( classifier.parameters(), lr=0.001, weight_decay=1e-4 )

      sae_criterion = nn.MSELoss()
      classifier_criterion = nn.CrossEntropyLoss()

      fine_tuning_epochs = 50

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

      if save_model:
        torch.save(model.state_dict(), f'model_{pipeline}_step1.pth')
    else:
      model.load_state_dict(torch.load(f'model_{pipeline}_step1.pth', map_location=torch.device(device)))

    print("======================================\nTesting Model\n======================================")

    model.eval()

    true_labels = np.array([])
    predicted_labels = np.array([])

    for batch in test_dataloader:
      data, labels = batch
      data = data.float().to(device)
      labels = labels.long().to(device)

      outputs = model(data)

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

  N_rois = 50

  rois_ig, weights_ig = find_top_rois_using_integrated_gradients(N_rois, model, test_dataloader, top_rois)

  rois_shap, weights_shap = find_top_rois_using_SHAP(N_rois, model, test_dataloader, train_dataloader, top_rois)

  rois_lime, weights_lime = find_top_rois_using_LIME(N_rois, model, test_dataloader, train_dataloader, top_rois)

  rois_deeplift, weights_deeplift = find_top_rois_using_DeepLift(N_rois, model, test_dataloader, top_rois)

  rois_deepliftshap, weights_deepliftshap = find_top_rois_using_DeepLiftShap(N_rois, model, test_dataloader, top_rois)

  rois_gradientshap, weights_gradientshap = find_top_rois_using_GradientShap(N_rois, model, test_dataloader, top_rois)

  rois_guidedbackprop, weights_guidedbackprop = find_top_rois_using_GuidedBackprop(N_rois, model, test_dataloader, top_rois)

  interpretation_results = [
    (rois_ig, weights_ig, "Integrated Gradients"),
    (rois_shap, weights_shap, "SHAP"),
    (rois_lime, weights_lime, "LIME"),
    (rois_deeplift, weights_deeplift, "DeepLift"),
    (rois_deepliftshap, weights_deepliftshap, "DeepLiftShap"),
    (rois_gradientshap, weights_gradientshap, "GradientShap"),
    (rois_guidedbackprop, weights_guidedbackprop, "GuidedBackprop"),
  ]

  for i in interpretation_results:
    print("=" * 65,f'\n{i[2]}\n' + ('=' * 65))
    print(print_connections(i[0], i[1], i[2], pipeline).to_string(index=False))

  pdb.set_trace()

  plt.show()
