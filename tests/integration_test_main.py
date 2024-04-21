import sys
sys.path.append('/Users/suryansv/git/COMP3000-remote')

from app.main import Autoencoder, SoftmaxClassifier, StackedAutoencoder, get_feature_vecs, get_top_features_from_SVM_RFE
import torch
import numpy as np
from torch.utils.data import DataLoader
import pytest

test_data = torch.randn(20, 50)  # 10 samples, 50 features, dummy data
test_dataloader = DataLoader(test_data, batch_size=4)

@pytest.fixture
def sample_data():
    return np.random.rand(20, 20, 50)  # 20 samples, 20 time instances, 50 ROIs

@pytest.fixture
def sample_labels():
    return np.random.randint(0, 2, size=20)  # 20 labels (0 or 1)

def test_stacked_autoencoder_integration():
     ae1 = Autoencoder(50, 25)
     ae2 = Autoencoder(25, 10)
     classifier = SoftmaxClassifier(10, 2)  # 2 output classes for classification

     stacked_ae = StackedAutoencoder(ae1, ae2, classifier)

     for data in test_dataloader: 
        outputs = stacked_ae(data).detach().numpy()
        assert outputs.shape == (4, 2)  # Ensure correct output shape (batch, output_classes)

# Tests for get_top_features_from_SVM_RFE
def test_get_top_features_svm_rfe(sample_data, sample_labels):
    feature_vecs, feature_vec_indices = get_feature_vecs(sample_data)
    top_features, _ = get_top_features_from_SVM_RFE(feature_vecs, sample_labels, feature_vec_indices, N=10, step=1)
    assert top_features.shape == (20, 10)  # Expect 10 top features from 20 samples

# Tests for get_feature_vecs
def test_get_feature_vecs_dimensions(sample_data):
    feature_vecs, feature_indices = get_feature_vecs(sample_data)
    roi_size = sample_data.shape[2]
    expected_num_features = int(roi_size * (roi_size - 1) / 2)

    assert feature_vecs.shape == (sample_data.shape[0], expected_num_features)
    assert feature_indices.shape == (sample_data.shape[0], expected_num_features, 2)