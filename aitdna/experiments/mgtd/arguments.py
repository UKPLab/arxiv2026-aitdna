from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
        default=None
    )
    reference_model_name_or_path: Optional[str] = field(default=None)
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field( 
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    sampling_model_name: Optional[str] = field(default=None)
    scoring_model_name: Optional[str] = field(default=None)

    method: str = field(default=384)

    # Seq2Seq model specific args
    generation_max_len: int = field(default=128)
    generation_beam_size: int = field(default=4)
    generation_do_sample: bool = field(default=False)
    generation_length_penalty: float = field(default=1.0)
    generation_uid_regularization: float = field(default=0.0)
    generation_no_repeat_ngram_size: int = field(default=3)
    generation_temperature: float = field(default=0.6)
    num_return_sequences: int = field(default=1)
    num_sequences_to_keep: int = field(default=1)
    num_labels: int = field(default=None)

    # Tokenization
    max_input_length: int = field(default=1024)

    fine_tuned_model: Optional[str] = field(default=None)

@dataclass
class DataPredictionArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_path: str = field()

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_restrict_to: str = field(default=None)
    dataset_threshold: str = field(default=None)

    metric_output_dir: Optional[str] = field(default=None)

    detection_level: Optional[str] = field(default="document", metadata={"help": "Options: document, boundary, sentence, span"})

    commercial_evaluation_folder: Optional[str] = field(default=None, metadata={"help": "Folder to store  commercial evaluation data into"})

    eval_batch_size: Optional[int] = field(default=4)
