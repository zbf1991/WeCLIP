"""Infrastructure validation tests."""

import pytest
import numpy as np
from pathlib import Path


@pytest.mark.unit
def test_pytest_working():
    """Test that pytest is working correctly."""
    assert True


@pytest.mark.unit
def test_numpy_import():
    """Test that numpy can be imported and used."""
    arr = np.array([1, 2, 3])
    assert len(arr) == 3
    assert arr.sum() == 6


@pytest.mark.unit
def test_temp_dir_fixture(temp_dir):
    """Test that temp_dir fixture works."""
    assert isinstance(temp_dir, Path)
    assert temp_dir.exists()
    
    # Create a test file
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists()
    assert test_file.read_text() == "test content"


@pytest.mark.unit
def test_sample_config_fixture(sample_config):
    """Test that sample_config fixture provides valid config."""
    assert isinstance(sample_config, dict)
    assert "model" in sample_config
    assert "dataset" in sample_config
    assert "training" in sample_config
    assert sample_config["model"]["num_classes"] == 21


@pytest.mark.unit
def test_sample_image_fixture(sample_image):
    """Test that sample_image fixture creates valid image."""
    assert isinstance(sample_image, np.ndarray)
    assert sample_image.shape == (224, 224, 3)
    assert sample_image.dtype == np.uint8
    assert 0 <= sample_image.min() <= sample_image.max() <= 255


@pytest.mark.unit
def test_sample_mask_fixture(sample_mask):
    """Test that sample_mask fixture creates valid mask."""
    assert isinstance(sample_mask, np.ndarray)
    assert sample_mask.shape == (224, 224)
    assert sample_mask.dtype == np.uint8
    assert 0 <= sample_mask.min() <= sample_mask.max() <= 20


@pytest.mark.unit
def test_mock_dataset_paths_fixture(mock_dataset_paths):
    """Test that mock dataset paths are created correctly."""
    paths = mock_dataset_paths
    assert "root" in paths
    assert "images" in paths
    assert "masks" in paths
    
    for path in paths.values():
        assert path.exists()
    
    # Check sample files exist
    images_dir = paths["images"]
    image_files = list(images_dir.glob("*.jpg"))
    assert len(image_files) >= 3


@pytest.mark.integration
def test_mock_clip_model_fixture(mock_clip_model):
    """Test that mock CLIP model works."""
    model = mock_clip_model
    
    # Test image encoding
    dummy_image = np.random.randn(1, 3, 224, 224).astype(np.float32)
    image_features = model.encode_image(dummy_image)
    assert image_features.shape == (1, 512)
    
    # Test text encoding  
    dummy_text = ["a photo of a dog"]
    text_features = model.encode_text(dummy_text)
    assert text_features.shape == (1, 512)


@pytest.mark.unit
def test_sample_class_labels_fixture(sample_class_labels):
    """Test that sample class labels are valid."""
    labels = sample_class_labels
    assert labels.shape == (5, 21)
    assert labels.dtype == np.float32
    assert np.all((labels == 0) | (labels == 1))  # Binary labels


@pytest.mark.unit  
def test_mock_logger_fixture(mock_logger):
    """Test that mock logger captures log messages."""
    logger = mock_logger
    
    logger.info("test info")
    logger.warning("test warning")
    logger.error("test error")
    
    assert len(logger.logs) == 3
    assert logger.logs[0] == ('INFO', 'test info')
    assert logger.logs[1] == ('WARNING', 'test warning')
    assert logger.logs[2] == ('ERROR', 'test error')


@pytest.mark.slow
def test_slow_operation():
    """Test marked as slow for filtering."""
    import time
    time.sleep(0.1)  # Simulate slow operation
    assert True


class TestInfrastructureClass:
    """Test class to verify class-based test discovery."""
    
    def test_class_based_test(self):
        """Test that class-based tests work."""
        assert True
    
    def test_with_fixture(self, sample_config):
        """Test that fixtures work in class-based tests."""
        assert "model" in sample_config