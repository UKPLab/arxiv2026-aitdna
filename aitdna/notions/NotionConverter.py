from .data_loading.AitdDataset import AitdDataset
from .data_loading.Notion import Notion
from torch.utils.data import DataLoader
import nltk

class Converter:
    def __init__(self):
        self.order = [
            Notion.DOCUMENT_LEVEL,
            Notion.SPAN_LEVEL,
            Notion.SENTENCE_LEVEL,
            Notion.TOKEN_LEVEL
        ]
        self.detokenizer = nltk.tokenize.TreebankWordDetokenizer()
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()


    def convert(self, dataset: AitdDataset, notion: Notion, **kwargs) -> AitdDataset:
        """Main converting function. Either splits or aggregates dataset's notion into the given one, depending on the notion order.

        Args:
            dataset (AitdDataset): the dataset to change the notion for
            notion (Notion): the new notion

        Returns:
            AitdDataset: dataset with changed notion
        """
        if dataset.notion == notion:
            return dataset
        if self.order.index(dataset.notion) < self.order.index(notion):
            return self.split_notion(dataset, notion)
        else:
            return self.aggregate_notion(dataset, notion, **kwargs)


    def split_notion(self, dataset: AitdDataset, notion: Notion) -> AitdDataset:
        """Main splitting function. Splits the datasets into a more fine-grained notion.

        Args:
            dataset (AitdDataset): the dataset to change the notion for
            notion (Notion): the new notion

        Returns:
            AitdDataset: dataset with changed notion
        """
        if dataset.notion == Notion.SPAN_LEVEL:
            dataset.data = self.split_span(dataset=dataset, notion=notion)
        else:
            dataset.data = self.split(dataset=dataset, notion=notion)
        dataset.notion = notion
        return dataset


    def aggregate_notion(self, dataset: AitdDataset, notion: Notion, **kwargs) -> AitdDataset:
        """Main aggregating function. Aggregates the datasets into a more coarse-grained notion.

        Args:
            dataset (AitdDataset): the dataset to change the notion for
            notion (Notion): the new notion

        Returns:
            AitdDataset: dataset with changed notion
        """
        if notion == Notion.DOCUMENT_LEVEL:
            dataset.data = self.aggregate_to_document_level(dataset, **kwargs)
        elif notion == Notion.SPAN_LEVEL and dataset.notion == Notion.TOKEN_LEVEL:
            dataset.data = self.aggregate_tokens_to_span_level(dataset)
        elif notion == Notion.SPAN_LEVEL and dataset.notion == Notion.SENTENCE_LEVEL:
            dataset.data = self.aggregate_sentences_to_span_level(dataset)
        elif notion == Notion.SENTENCE_LEVEL:
            dataset.data = self.aggregate_tokens_to_sentence_level(dataset, **kwargs)
        else:
            raise ValueError("Unknown notion!")
        dataset.notion = notion
        return dataset


    def split(self, dataset: AitdDataset, notion: Notion) -> list[list[dict]]:
        """Implements splitting of document-level and document-level notions into fine-grained ones.

        Args:
            dataset (AitdDataset): the dataset to change the notion for
            notion (Notion): the new notion

        Raises:
            ValueError: if used to split a span-level dataset.

        Returns:
            list[list[dict]]: data in the new notion
        """
        if dataset.notion == Notion.SPAN_LEVEL:
            raise ValueError("Use the split_span notion to change this dataset!")
        data = []
        loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
        for batch in loader:
            for text in batch:
                text_data = []
                for snippet in text:
                    if notion == Notion.TOKEN_LEVEL:
                        tokens = self.tokenizer.tokenize(snippet["text"])
                    elif notion == Notion.SENTENCE_LEVEL:
                        tokens = nltk.sent_tokenize(snippet["text"])
                    elif notion == Notion.SPAN_LEVEL:
                        tokens = [snippet["text"]]
                    else:
                        raise ValueError("Unknown notion!")
                    for token in tokens:
                        text_data.append({
                            "text": token,
                            "author": snippet["author"],
                            "queries": snippet["queries"],
                        })
                data.append(text_data)
        return data


    def split_span(self, dataset: AitdDataset, notion: Notion) -> list[list[dict]]:
        """Implements splitting from span-notion notion into fine-grained ones.

        Args:
            dataset (AitdDataset): the dataset to change the notion for
            notion (Notion): the new notion

        Returns:
            list[list[dict]]: data in the new notion
        """
        data = []
        loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
        for batch in loader:
            for text in batch:
                text_data = []
                text_content = ""
                authors_by_letter = []
                queries_by_letter = []
                for snippet in text:
                    text_content += snippet["text"]
                    authors_by_letter += [snippet["author"]] * len(snippet["text"])
                    queries_by_letter += [snippet["queries"]] * len(snippet["text"])

                if notion == Notion.TOKEN_LEVEL:
                    tokens = self.tokenizer.tokenize(text_content)
                elif notion == Notion.SENTENCE_LEVEL:
                    tokens = nltk.sent_tokenize(text_content)
                else:
                    raise ValueError("Unknown notion!")

                current_index = 0
                for token in tokens:
                    if token == "``":
                            token = '"'
                    token_index = text_content[current_index:].find(token)
                    authors = authors_by_letter[current_index + token_index: current_index + token_index + len(token)]
                    queries = queries_by_letter[current_index + token_index: current_index + token_index + len(token)]
                    current_index += token_index + len(token)
                    if len(set(authors)) > 1:
                        author = "Mixed"
                    else:
                        author = authors[0]

                    text_data.append({
                        "text": token,
                        "author": author,
                        "queries": list(set(query for lst in queries for query in lst if lst))
                    })
                data.append(text_data)
        return data


    def aggregate_to_document_level(self, dataset: AitdDataset, **kwargs):
        data = []
        document_level_threshold = kwargs.get("document_level_threshold", 0.5)
        loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
        for batch in loader:
            for text in batch:
                document = []
                authors = []
                queries = []
                for snippet in text:
                    document.append(snippet["text"])
                    authors.append(snippet["author"])
                    queries.extend(snippet["queries"])
                if dataset.notion == Notion.TOKEN_LEVEL:
                    document_text = self.detokenizer.detokenize(document)
                else:
                    document_text = "".join(document)

                n_bot_snippets = len([author for author in authors if author != "User"])
                if n_bot_snippets / len(text) >= document_level_threshold:
                    author = "Bot"
                else:
                    author = "User"
                data.append(
                    [
                        {
                            "text": document_text,
                            "author": author,
                            "queries": list(set(queries))
                        }
                    ]
                )
        return data
    

    def aggregate_tokens_to_span_level(self, dataset: AitdDataset):
        data = []
        loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
        for batch in loader:
            for text in batch:
                text_data = []
                current_author = text[0]["author"]
                current_text = []
                queries = []
                for i, token in enumerate(text):
                    if token["author"] == current_author and i != len(text) - 1:
                        current_text.append(token["text"])
                        queries.extend(token["queries"])
                    else:
                        if i == len(text) - 1:
                            current_text.append(token["text"])
                            queries.extend(token["queries"])
                        text_data.append({
                            "text": self.detokenizer.detokenize(current_text),
                            "author": current_author,
                            "queries": list(set(queries))
                        })
                        current_author = token["author"]
                        queries = token["queries"]
                        current_text = [token["text"]]
                data.append(text_data)
        return data
    
    def aggregate_sentences_to_span_level(self, dataset: AitdDataset):
        data = []
        loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
        for batch in loader:
            for text in batch:
                text_data = []
                current_author = text[0]["author"]
                current_text = ""
                queries = []
                for i, sent in enumerate(text):
                    if sent["author"] == current_author and i != len(text) - 1:
                        current_text += sent["text"] + " "
                        queries.extend(sent["queries"])
                    else:
                        if i == len(text) - 1:
                            current_text += sent["text"]
                            queries.extend(sent["queries"])
                        text_data.append({
                            "text": current_text,
                            "author": current_author,
                            "queries": list(set(queries))
                        })
                        current_author = sent["author"]
                        queries = sent["queries"]
                        current_text = sent["text"] + " "
                data.append(text_data)
        return data

    def aggregate_tokens_to_sentence_level(self, dataset: AitdDataset, **kwargs):
        data = []
        loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
        for batch in loader:
            for text in batch:
                text_data = []
                text_content = self.detokenizer.detokenize([token["text"] for token in text])
                sentences = nltk.sent_tokenize(text_content)
                tokens = [token["text"] if token["text"] != "''" else '"' for token in text]
                current_token_index = 0
                for sent in sentences:
                    token_in_sent_index = 0
                    index = 0
                    sent_authors = []
                    sent_queries = []
                    while index != -1 and current_token_index < len(tokens):
                        index = sent.find(tokens[current_token_index], token_in_sent_index)
                        if index != -1:
                            sent_authors.append(text[current_token_index]["author"])
                            sent_queries.extend(text[current_token_index]["queries"])   
                            token_in_sent_index = index + len(tokens[current_token_index])
                            current_token_index += 1

                    if len(set(sent_authors)) > 1:
                        author = "Mixed"
                    else:
                        author = sent_authors[0]

                    text_data.append({
                        "text": sent,
                        "author": author,
                        "queries": list(set(sent_queries))
                    })
                data.append(text_data)
        return data

