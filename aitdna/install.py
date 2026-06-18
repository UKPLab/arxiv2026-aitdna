import nltk
import spacy


def install_prerequisites():
    # nltk
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download("words")

    # spacy
    spacy.cli.download("en_core_web_lg")