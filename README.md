<p  align="center">
  <img src='logo.png' width='200'>
</p>

# ’Your AI Text is not Mine’: Redefining and Evaluating AI-generated Text Detection under Realistic Assumptions
[![Arxiv](https://img.shields.io/badge/Arxiv-YYMM.NNNNN-red?style=flat-square&logo=arxiv&logoColor=white)](https://put-here-your-paper.com)
[![License](https://img.shields.io/github/license/UKPLab/ukp-project-template)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.11-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/UKPLab/ukp-project-template/actions/workflows/main.yml/badge.svg)](https://github.com/UKPLab/ukp-project-template/actions/workflows/main.yml)

This repository implements the code for our paper on the notions of AI-generated text and their evaluation.

> **Abstract:** Although it is generally agreed that AI-generated text poses a broad societal risk, there is no common understanding in the AI-generated text detection literature on what constitutes harmful use.  Rather, existing datasets and approaches often define their own criteria and make their own assumptions, sometimes implicitly, and often only loosely related to real-world needs and applications. To address this gap, we here systematically define various notions of AI-generated text and their characteristics. To study these, we collect AITDNA - a new benchmark of human-machine co-constructed texts that is annotated with detailed genesis information, such as the entire edit and AI-interaction history. We benchmark various machine-generated text detectors and find that they often only perform well for specific notions but not as broad detectors. We release code and data publicly.
> 
Contact person: [Marina Sakharova](mailto:marina.sakharova@stud.tu-darmstadt.de), [Nico Daheim](mailto:nico.daheim@tu-darmstadt.de)

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.


## Quickstart

1. Create conda env with `conda create -n aitdna_env python=3.11`, then activate it `conda activate aitdna_env`
2. Install flit `pip install flit`
3. Run `python -m flit install --symlink`
4. Download spacy tokenizer `python -m spacy download en_core_web_lg`
5. Run the initialization script `python -m aitdna install`
6. Get started (see below)

## Datasets
Our framework represents AITDNA and five other datasets (CoAuthor, DetectRL, Mixset, SenDetEx, Boundary Detection for TriBERT) in different notions of AI text and evaluates predictors on these notions. For the notion representation, we use a wrapper, AitdDataset. AitdDataset allows to represent all six datasets in different notions with custom hyperparameter values (for instance, you can specify AI token threshold for document-level labeling). 
When instantiating AitdDataset, you have to specify:
- The dataset you want to load (see `aitdna/notions/data_loading/Datasets.py` for all options).
- The notion you want to use (see `aitdna/notions/data_loading/Notion.py` for all options).
- (Optional) Hyperparameter values for your notion (see `aitdna/notions/data_loading/AitdDataset.py` for all options).
- (Optional, only for other datasets) Path to dataset root.

### Use AITDNA
The framework uses our AITDNA dataset from huggingface [TODO url] and load it in the AitdDataset wrapper.

```python
from torch.utils.data import DataLoader

from aitdna.notions.data_loading import AitdDataset, DatasetName, Notion

dataset = AitdDataset(dataset=DatasetName.AITDNA,
                  notion=Notion.DOCUMENT_LEVEL,
                  document_level_threshold=0.75)
loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
for batch in loader:
    for data_point in batch:
        for snippet in data_point:
          # do things
          ...
```
### Use other datasets
 To create the different notions for other datasets, we represent each text as a list of quill-delta edits. For instance, if the data format is (human original text, AI rewritten version), we first insert the original text as a human edit, and then incrementally add AI edits. We use [diff-match-patch](https://github.com/google/diff-match-patch), a package that balances semantic and syntactic differences in edit creation.

For the edit creation process, run 
```console
python -m aitdna create_dataset --dataset_type mixset --dst_root data/processed/mixset
```
`dst_root` will store the processed version of the dataset, including recreated edits and all notions with their default hyperparameter values.

To use the dataset with AitdDataset, do:
```python
from torch.utils.data import DataLoader

from aitdna.notions.data_loading import AitdDataset, DatasetName, Notion

dataset = AitdDataset(dataset=DatasetName.MIXSET,
                  notion=Notion.DOCUMENT_LEVEL,
                  root_dir="data/processed/mixset") # path where you saved your dataset
loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
for batch in loader:
    for data_point in batch:
        for snippet in data_point:
          # do things
          ...
```

## Evaluate Models
To evaluate predictors, run this command. 

```console
python -m txaitd run_predictors --path_to_config_json aitdna/experiments/config/search_config.json --cache_dir your-cache-dir
```
Adapt the search config file to your needs. Specify what predictor and dataset you want to evaluate, the path to the dataset, and hyperparameter values.


## Statistics generation
To generate all statistics, use
```console
python -m aitdna compute_dataset_stats -r data/processed/mixset -n MIXSET -d all_stats.json
```
The following statistics will be generated:
1. Statistics per sentence (computed for user, bot, and mixed):
- Avg syntax tree depth
- Avg syntax tree width
- Avg syntax tree #leaves
- Avg #tokens in a sentence
- Avg #characters in a sentence
- Total #sentences

2. Statistics per user (author: user and bot)
- Avg #tokens (of this author in a text)
- Total vocabulary size
- Avg #distinct lemmas (vocabulary size / total # tokens by this user)
- Counts of all POS
- Avg readability scores (computed only for snippets > 100 words): Flesch Reading Ease, Flesch Kincaid Grade Level, Gunning Fog, and Dale-Chall.


3. Statistics per span (for span-level notation; authors: user and bot)
- Avg #sentences per span (computed by sent_tokenize'ing each span)
- Avg #tokens per span
- Avg #boundaries sentence-level (how often does the authorship change in the sorted list of sentences)
- Avg #boundaries span-level


## Repository structure
```bash
aitdna/
├── analysis
│   ├── argument_evaluation.py # Evaluation of argument number for argumentative essays
│   ├── BakgroundInfoProcessor.py # Stats for user background information
│   ├── DependencyTree.py # Linguistic stats computer
│   ├── eval_polcies.py # Policy evaluation
│   ├── organize_stats_table.py
│   ├── RawDataAnalyser.py # Stats computer for raw AITDNA data
│   └── StatsComputer.py # Stats computer for all datasets
├── cli.py # CLI command definition
├── datasets
│   ├── aitdna_dataset
│   │   ├── huggingface_upload
│   │   │   └── push_to_hf.py # Upload AITDNA to HF
│   │   ├── postprocessing
│   │   │   ├── anonymization.py # AITDNA data anonymization
│   │   │   └── generate_synthetic_texts.py # (Experimental) Synthetic AITDNA version generation
│   │   ├── preprocessing
│   │   │   └── process_csv.py # Processing of raw CARE data for AITDNA creation
│   │   └── processing
│   │       ├── DataFormatter.py # Formatter for AITDNA creation
│   │       └── format_data.py # Main script for AITDNA creation
│   ├── boundary_detection.py # Creation of the BD dataset
│   ├── coauthor.py # Creation of the Coauthor dataset
│   ├── detectRL.py # Creation of the DetectRL dataset
│   ├── mixset.py # Creation of the Mixset dataset
│   └── senDetEx.py # Creation of the SenDetEx dataset
├── experiments
│   ├── config
│   │   ├── dataset_paths.json
│   │   └── search_config.json
│   ├── mgtd
│   │   ├── arguments.py # Prediction argument specification
│   │   ├── evaluate.py # Evaluation of commercial detectors
│   │   ├── majority_baseline.py # Majority baseline code
│   │   ├── methods
│   │   │   ├── base.py # Base detection method
│   │   │   ├── generation.py # Definition of all MGTD methods
│   │   │   └── preprocessing
│   │   │       ├── base.py # Base preprocessor for MGTD
│   │   │       └── generation.py # Data preprocessing for MGTD
│   │   ├── mgtd_datasets
│   │   │   └── DetectionDataset.py # Dataset wrapper for evaluation
│   │   ├── trainer
│   │   │   └── custom_trainer.py # Trainer class with prediction code
│   │   ├── train_predict.py # Main script for evaluation
│   │   └── utils.py
│   ├── predict.py
│   └── utils.py
├── install.py # Package installation script
├── __main__.py # Main entry point
├── notions
│   ├── AITDNotions.py # Creation of different notions
│   ├── ContentPolicy.py # Content policy for content-based notion
│   ├── CostFunction.py # Cost function for boundary-level notion
│   ├── data_loading
│   │   ├── AitdDataset.py # Representation of datasets in notions
│   │   ├── DatasetName.py # Available datasets
│   │   ├── Notion.py # Available notions
│   │   └── Population.py # Population for membership and authorship notions
│   ├── IntentPolicy.py # Intent policy for intent-based notion
│   └── NotionConverter.py # Converter for genesis notions
└── utils.py
```
## Cite

Please use the following citation:

```
TBA
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
