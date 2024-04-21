import sys
sys.path.append('/Users/suryansv/git/COMP3000-remote')

import pytest
import pdb
import numpy as np
from sklearn.svm import SVC
from app.main import CustomDataset, find_top_rois_using_SHAP, find_top_rois_using_GuidedBackprop,find_top_rois_using_GradientShap, find_top_rois_using_integrated_gradients, find_top_rois_using_DeepLiftShap, find_top_rois_using_DeepLift, get_data_from_abide, fishers_z_transform, Autoencoder, StackedAutoencoder, SoftmaxClassifier
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import numpy as np

"""
Test Data Preparation Functions
"""

@pytest.fixture
def sample_data():
    return np.random.rand(20, 20, 50)  # 20 samples, 20 time instances, 50 ROIs

@pytest.fixture
def sample_labels():
    return np.random.randint(0, 2, size=20)  # 20 labels (0 or 1)

# Tests for get_data_from_abide:
def test_get_data_from_abide_valid_file(tmpdir):
    data, labels = get_data_from_abide('ccs')
    assert isinstance(data, list)
    assert isinstance(labels, list)

def test_get_data_from_abide_missing_file():
    with pytest.raises(FileNotFoundError):
        get_data_from_abide('non_existing_file.1D')

# Tests for fishers_z_transform
def test_fishers_z_transform_edge_cases():
    assert fishers_z_transform(1) == np.inf
    assert fishers_z_transform(-1) == -np.inf
    assert np.isclose(fishers_z_transform(0), 0)

"""
Test Modeling components
"""

test_data = torch.randn(20, 50)  # 10 samples, 50 features, dummy data
test_dataloader = DataLoader(test_data, batch_size=4)

def test_autoencoder_reconstruction():
    autoencoder = Autoencoder(50, 25)  # Input size 50, encoded size 25
    criterion = nn.MSELoss()

    for data in test_dataloader:  # Use sample data
        _, decoded, loss = autoencoder(data)
        assert loss.item() < 1.5  # Assert reconstruction loss within a threshold

def test_softmax_classifier_probabilities():
    classifier = SoftmaxClassifier(5, 2)  # Input size 5, 2 output classes
    input_data = torch.randn(2, 5)  # 2 samples, 5 features

    output = classifier(input_data).detach().numpy()

    assert output.shape == (2, 2)  # Ensure correct output shape

"""
Test Explanation Functions
"""

explain_data = torch.randn(20, 50)  # 20 samples, 50 features, dummy data
explain_labels = np.random.randint(0, 2, size=20)  # 20 labels (0 or 1)

explain_train_data = torch.randn(20, 50)  # 20 samples, 50 features, dummy data
explain_train_labels = np.random.randint(0, 2, size=20)  # 20 labels (0 or 1)

test_idx = [i for i in range(20)]
random.shuffle(test_idx)
train_idx = [i for i in range(20)]
random.shuffle(train_idx)

dataset, label = {}, {}
dataset['test'] = Subset(explain_data, test_idx)
label['test'] = Subset(explain_labels, test_idx)
test_set = CustomDataset(dataset['test'], label['test'])
explain_dataloader = DataLoader(test_set, batch_size=4)

dataset['train'] = Subset(explain_data, train_idx)
label['train'] = Subset(explain_labels, train_idx)
train_set = CustomDataset(dataset['train'], label['train'])
explain_train_dataloader = DataLoader(train_set, batch_size=4)

rois = np.random.rand(50, 2)
ae1 = Autoencoder(50, 25)
ae2 = Autoencoder(25, 10)
classifier = SoftmaxClassifier(10, 2)  # 2 output classes for classification
stacked_ae = StackedAutoencoder(ae1, ae2, classifier)

def test_find_top_rois_using_DeepLift():
    N = 5
    rois_dl, weights, indices = find_top_rois_using_DeepLift(N, stacked_ae, explain_dataloader, rois)
    assert rois_dl.shape == (N, 2)
    assert weights.shape == (N,)
    assert indices.shape == (N,)

def test_find_top_rois_using_DeepLiftShap():
    N = 5
    rois_dlshap, weights, indices = find_top_rois_using_DeepLiftShap(N, stacked_ae, explain_dataloader, rois)

    assert rois_dlshap.shape == (N, 2)
    assert weights.shape == (N,)
    assert indices.shape == (N,)

def test_find_top_rois_using_GradientShap():
    N = 5
    rois_gradientshap, weights, indices = find_top_rois_using_GradientShap(N, stacked_ae, explain_dataloader, rois)

    assert rois_gradientshap.shape == (N, 2)
    assert weights.shape == (N,)
    assert indices.shape == (N,)

def test_find_top_rois_using_GuidedBackprop():
    N = 5
    rois_GuidedBackprop, weights, indices = find_top_rois_using_GuidedBackprop(N, stacked_ae, explain_dataloader, rois)

    assert rois_GuidedBackprop.shape == (N, 2)
    assert weights.shape == (N,)
    assert indices.shape == (N,)

def test_find_top_rois_using_integrated_gradients():
    N = 5
    rois_integrated_gradients, weights, indices = find_top_rois_using_integrated_gradients(N, stacked_ae, explain_dataloader, rois)

    assert rois_integrated_gradients.shape == (N, 2)
    assert weights.shape == (N,)
    assert indices.shape == (N,)

def test_find_top_rois_using_SHAP():
    N = 5
    rois_SHAP, weights, indices = find_top_rois_using_SHAP(N, stacked_ae, explain_dataloader, explain_train_dataloader, rois)

    assert rois_SHAP.shape == (N, 2)
    assert weights.shape == (N,)
    assert indices.shape == (N,)