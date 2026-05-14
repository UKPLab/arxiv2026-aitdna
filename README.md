<p  align="center">
  <img src='logo.png' width='200'>
</p>

# Benchmarking Notions of MGTD
[![Arxiv](https://img.shields.io/badge/Arxiv-YYMM.NNNNN-red?style=flat-square&logo=arxiv&logoColor=white)](https://put-here-your-paper.com)
[![License](https://img.shields.io/github/license/UKPLab/ukp-project-template)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.11-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/UKPLab/ukp-project-template/actions/workflows/main.yml/badge.svg)](https://github.com/UKPLab/ukp-project-template/actions/workflows/main.yml)

<Introtext TBA>

> **Abstract:** Although it is generally agreed that AI-generated text poses a broad societal risk, there is no common understanding in the AI-generated text detection literature on what constitutes harmful use.  Rather, existing datasets and approaches often define their own criteria and make their own assumptions, sometimes implicitly, and often only loosely related to real-world needs and applications. To address this gap, we here systematically define various notions of AI-generated text and their characteristics. To study these, we collect AITDNA - a new benchmark of human-machine co-constructed texts that is annotated with detailed genesis information, such as the entire edit and AI-interaction history. We benchmark various machine-generated text detectors and find that they often only perform well for specific notions but not as broad detectors. We release code and data publicly.
> 
Contact person: [Marina Sakharova](mailto:marina.sakharova@stud.tu-darmstadt.de)

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.


## Quickstart

1. Create conda env with `conda create -p ./venv python=3.11`, then source it `conda activate ./venv`
2. Download spacy tokenizer `python -m spacy download en_core_web_lg`
3. Install flit `pip install flit`
5. Run `flit install`
6. Run the initialization script `python -m txaitd install`
7. Get started (see below)

## Data Processing

### Preparation
Make sure to prepare the following data:
- Logs from CARE. These should include: 
```
document_edit.csv
document.csv
human_ai_perception.csv
nlp_editor_request.csv
nlp_editor_response.csv
user.csv
```
- Consent form data in `.csv` format
- Surveys data (background and UX survey) in `.csv` format
- User-task-model assignment in `JSON` format

### CSV Data Processing

Data preprocessing. Processes CARE data between cutoff dates. Src_root should contain logs from CARE. One run processes one study session. Results in raw edits saved in JSON format, in form study/user/task/edits.json

```console
python txaitd/datasets/aitdna/preprocessing/process_csv.py --src_root data/raw_data/2026-02-20-snapshot --dst_root "data/datasets/original/2020-01-22" --consent_form_path data/raw_data/surveys_data/processed/consent.csv --earliest_cutoff_date "2026-01-18 09:00:00.00+00" --latest_cutoff_date "2026-02-20 00:00:00.00+00"
```

### Data Formatting
Data formatting. Formats edits themselves (better naming of operations, users, time starting from 0s), filters and flags bad data, computes notions and statistics. If --process_all flag passed, processes all data for all studies. Otherwise, processes only data for one user study. Format of root and dst is then: --src_root data/datasets/original/YOUR STUDY --dst_root data/datasets/formatted/YOUR STUDY, and do not pass the process_all flag.

```console
python -m txaitd format_dataset --src_root data/test/original --dst_root data/test/formatted --ns_segments 2 5 10 --survey_paths data/raw_data/surveys_data/processed/background.csv data/raw_data/surveys_data/processed/ux_survey.csv --user_task_assignment data/raw_data/user_task_assignment/user_task_assignment.json --process_all

```

### Data Loading
The code below shows a simple example of data loading. Each data point corresponds to one text in one notion, represented as a list of dicts.

You need to specify:
- Path to dataset root. For AITDNA, it's the folder with all study folders. For AITDNA-SYNTHETIC,
it's the one with all model folders. For the others, it's the one with all data points.
- What dataset you want to load (see `scripts/data_scripts/data_loading/Datasets.py` for all options).
- The notion that you want to use (see `scripts/data_scripts/data_loading/Notion.py` for all options)

```python
from torch.utils.data import DataLoader

from txaitd.notions.data_loading import AitdDataset, DatasetName, Notion

root_dir = "dataset-root-directory"
dataset = AitdDataset(dataset=DatasetName.COAUTHOR, root_dir=root_dir,
                  notion=Notion.DOCUMENT_LEVEL)
loader = DataLoader(dataset, batch_size=2, collate_fn=lambda data_point: data_point)
for batch in loader:
    for data_point in batch:
        for snippet in data_point:
          # do things
          ...
```

### Statistics generation
To generate all statistics, use
```console
python -m txaitd compute_dataset_stats -r data/aitdna_anonymized/formatted -n AITDNA -d all_stats.json
```
Following statistics will be generated:
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
- Avg readability scores (computed only for snippets > 100 words): flesch reading ease, flesch kincaid grade level, gunning fog, and dalle-chall.


3. Statistics per span (for span-level notation; authors: user and bot)
- Avg #sentences per span (computed by sent_tokenize'ing each span)
- Avg #tokens per span
- Avg #boundaries sentence-level (how often does the authorship change in the sorted list of sentences)
- Avg #boundaries span-level

## Cite

Please use the following citation:

```
TBA
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
