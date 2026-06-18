import os
from aitdna.analysis.StatsComputer import StatsComputer
from aitdna.notions.data_loading import DatasetName

def test_avg_n_tokens():
    stats = StatsComputer(dataset_name=DatasetName.AITDNA)
    tokens = stats.get_avg_n_tokens_per_span()
    print(tokens)

def test_avg_n_chars():
    stats = StatsComputer(dataset_name=DatasetName.AITDNA)
    sents = stats.get_avg_n_sentences_per_span()
    print(sents)

def test_avg_n_boud_charwise():
    stats = StatsComputer(dataset_name=DatasetName.AITDNA)
    n = stats.get_avg_n_boundaries_span_level()
    print(n)

def test_avg_n_boud_sentwise():
    stats = StatsComputer(dataset_name=DatasetName.AITDNA)
    n = stats.get_avg_n_boundaries_sentence_level()
    print(n)

def test_sentence_stats():
    stats = StatsComputer(dataset_name=DatasetName.AITDNA)
    sent = stats.get_avg_stats_per_sentence()
    print(sent)

def test_span_stats():
    stats = StatsComputer(dataset_name=DatasetName.AITDNA)
    sent = stats.get_avg_stats_per_span()
    print(sent)
