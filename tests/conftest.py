"""Shared pytest fixtures for testing infrastructure."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
import numpy as np
from PIL import Image


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    try:
        yield Path(temp_path)
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide a sample configuration for testing."""
    return {
        "model": {
            "backbone": "RN50",
            "num_classes": 21,
            "pretrained": True,
        },
        "dataset": {
            "name": "voc12",
            "root": "/path/to/dataset",
            "split": "train",
        },
        "training": {
            "batch_size": 8,
            "learning_rate": 0.001,
            "epochs": 100,
        },
        "evaluation": {
            "metrics": ["accuracy", "miou"],
        }
    }


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample RGB image for testing."""
    return np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask() -> np.ndarray:
    """Create a sample segmentation mask for testing."""
    return np.random.randint(0, 21, size=(224, 224), dtype=np.uint8)


@pytest.fixture
def sample_image_pil(sample_image) -> Image.Image:
    """Convert sample numpy image to PIL Image."""
    return Image.fromarray(sample_image)


@pytest.fixture
def mock_dataset_paths(temp_dir) -> Dict[str, Path]:
    """Create mock dataset directory structure."""
    paths = {
        "root": temp_dir / "dataset",
        "images": temp_dir / "dataset" / "JPEGImages",
        "masks": temp_dir / "dataset" / "SegmentationClass",
        "annotations": temp_dir / "dataset" / "Annotations",
    }
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    # Create sample files
    sample_files = ["2007_000032.jpg", "2007_000039.jpg", "2007_000042.jpg"]
    for filename in sample_files:
        (paths["images"] / filename).touch()
        (paths["masks"] / filename.replace(".jpg", ".png")).touch()
    
    return paths


@pytest.fixture
def mock_clip_model():
    """Mock CLIP model for testing."""
    class MockCLIPModel:
        def __init__(self):
            self.visual = MockVisualEncoder()
            self.text = MockTextEncoder()
            
        def encode_image(self, image):
            return np.random.randn(1, 512).astype(np.float32)
            
        def encode_text(self, text):
            return np.random.randn(1, 512).astype(np.float32)
    
    class MockVisualEncoder:
        def forward(self, x):
            return np.random.randn(x.shape[0], 512).astype(np.float32)
    
    class MockTextEncoder:
        def forward(self, x):
            return np.random.randn(x.shape[0], 512).astype(np.float32)
    
    return MockCLIPModel()


@pytest.fixture
def mock_config_file(temp_dir, sample_config) -> Path:
    """Create a temporary config file."""
    import yaml
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    return config_path


@pytest.fixture
def sample_class_labels() -> np.ndarray:
    """Provide sample class labels for multi-label classification."""
    # 21 classes (VOC) with some samples having multiple labels
    labels = np.zeros((5, 21), dtype=np.float32)
    labels[0, [0, 5]] = 1.0  # person, chair
    labels[1, [2]] = 1.0     # car
    labels[2, [1, 6, 12]] = 1.0  # bicycle, bus, dog
    labels[3, [15]] = 1.0    # person
    labels[4, [7, 9]] = 1.0  # cat, cow
    return labels


@pytest.fixture
def mock_transforms():
    """Mock image transforms for testing."""
    class MockTransform:
        def __call__(self, image):
            if isinstance(image, Image.Image):
                image = np.array(image)
            # Simple resize simulation
            return image.astype(np.float32) / 255.0
    
    return MockTransform()


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Clean up matplotlib figures after each test."""
    yield
    import matplotlib.pyplot as plt
    plt.close('all')


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    class MockLogger:
        def __init__(self):
            self.logs = []
        
        def info(self, msg):
            self.logs.append(('INFO', msg))
        
        def warning(self, msg):
            self.logs.append(('WARNING', msg))
        
        def error(self, msg):
            self.logs.append(('ERROR', msg))
        
        def debug(self, msg):
            self.logs.append(('DEBUG', msg))
    
    return MockLogger()


@pytest.fixture
def set_random_seeds():
    """Set random seeds for reproducible testing."""
    np.random.seed(42)
    import random
    random.seed(42)
    # If torch is available
    try:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    except ImportError:
        pass


# Pytest configuration hooks
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")