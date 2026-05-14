import os
import json
import pickle
from typing import Callable

import bm25s
import nltk

from . import AitdDataset

from .Notion import Notion


class Population:
    retriever: bm25s.BM25 = None
    corpus_tokens: list = None
    corpus: list = None
    tokenizer = None

    def __init__(self, dataset: AitdDataset | None = None, cache_dir: str | None = None,
                 filter_fun: Callable|None = None, user: str | None = None):
        
        self._cache_dir = cache_dir

        assert self._cache_dir is None or os.path.exists(self._cache_dir), f"the provided cache path {self._cache_dir} does not exist"

        self.caching = self._cache_dir is not None
        self.filter_fun = filter_fun
        self.dataset = dataset
        self.user = user
        self.detokenizer = nltk.tokenize.TreebankWordDetokenizer()

        self._load()


    def _load_cache(self):
        self.retriever = bm25s.BM25.load(os.path.join(self._cache_dir, "retriever"), load_corpus=True)
        
        with open(os.path.join(self._cache_dir, "tokenizer.pkl"), "rb") as f:
            self.tokenizer = pickle.load(f)
        
        with open(os.path.join(self._cache_dir, "corpus_tokens.pkl"), "rb") as f:
            self.corpus_tokens = pickle.load(f)

        with open(os.path.join(self._cache_dir, "corpus.json"), "r", encoding="utf-8") as f:
            self.corpus = json.load(f)


    def _cache(self):
        assert self.retriever is not None
        assert self._cache_dir is not None and os.path.exists(self._cache_dir)

        self.retriever.save(os.path.join(self._cache_dir, "retriever"), corpus=self.corpus)

        corpus_path = os.path.join(self._cache_dir, "corpus.json")
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(self.corpus, f)
        
        corpus_tokens_path = os.path.join(self._cache_dir, "corpus_tokens.pkl")
        with open(corpus_tokens_path, "wb") as f:
            pickle.dump(self.corpus_tokens, f)
        
        tokenizer_path = os.path.join(self._cache_dir, "tokenizer.pkl")
        with open(tokenizer_path, "wb") as f:
            pickle.dump(self.tokenizer, f)

    def _load(self):
        # load dataset if path is specified (overrules a present cache path)
        if self.dataset is not None:
            if self.filter_fun and not self.dataset.with_meta:
                raise ValueError("To filter by user, you need metadata in your dataset!")
            corpus = []
            for doc in self.dataset:
                if self.dataset.with_meta:
                    d, meta = doc
                else:
                    d = doc
                    meta = {}
                # skip any document containing machine text
                if self.filter_fun and not self.filter_fun(meta, self.user):
                    continue

                if any(token for token in d if token["author"] == "Bot"):
                    continue
                text_reconstructed = self.detokenizer.detokenize([token["text"] for token in d])
                corpus += [text_reconstructed]

            # index
            self.corpus = corpus
            self.tokenizer = bm25s.tokenization.Tokenizer(
                lower=True,  # lowercase the tokens
                stopwords=False,
                splitter=nltk.word_tokenize  # consistency with notion tokenization
            )

            self.corpus_tokens = self.tokenizer.tokenize(corpus, update_vocab=True)
            self.retriever = bm25s.BM25()
            self.retriever.index(self.corpus_tokens)

            # cache
            if self.caching:
                self._cache()
        else:  # otherwise load from cache
            self._load_cache()

    def _search_docs_for_ngram_overlap(self, n, query_token_ids):
        def find(entry):
            if isinstance(entry[0], dict):
                doc_id = entry[0]["id"]
            else:
                doc_id = entry[0]
            doc = self.corpus_tokens[doc_id]

            # search for n-gram
            return any(doc[i:i + n] == query_token_ids for i in range(len(doc) - n + 1))

        return find

    def __contains__(self, n_gram_info:tuple[str, int]) -> bool:
        """

        Args:
            n_gram_info (tuple[str, int]): tuple of (n gram, n gram length)

        Returns:
            bool: True if the population contains the n-gram
        """
        ngram, ngram_len = n_gram_info
        query_token_ids = self.tokenizer.tokenize([ngram.lower()])[0]
        if len(query_token_ids) != ngram_len:
            return False
        n = len(query_token_ids)

        matches = self.retriever.retrieve([query_token_ids], k=5)  # focus on top 5 documents
        return any(map(self._search_docs_for_ngram_overlap(n, query_token_ids),
                       filter(lambda x: x[1] > 0,
                              zip(matches[0][0], matches[1][0]))))
