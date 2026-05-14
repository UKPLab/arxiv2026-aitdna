from .anonymization import main as run_dataset_anonymization
from .generate_synthetic_texts import main as run_synthetic_data_generation
__all__ = [
    "run_dataset_anonymization",
    "run_synthetic_data_generation"
]