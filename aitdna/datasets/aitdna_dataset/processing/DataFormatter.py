import json

from typing import List, Dict
from compact_json import Formatter
from datetime import datetime

from ..preprocessing.process_csv import modify_full_query
from aitdna.notions import AITDNotions

class DataFormatter():
    def __init__(self, max_inline_complexity: int, max_inline_length: int, unrelevant_keys: set,  guidelines_path: str = "scripts/data_scripts/processing/guidelines.txt"):
        with open(guidelines_path, "r") as f:
            self.guidelines = f.read()

        self.formatter = Formatter()
        self.notions = AITDNotions()
        self.formatter.max_inline_complexity = max_inline_complexity
        self.formatter.max_inline_length = max_inline_length
        self.unrelevant_keys = unrelevant_keys

    def save(self, data, path):
        """
        Save formatted data to path as JSON
        """
        self.formatter.dump(data, output_file=path, newline_at_eof=True)

    def remove_unrelevant_keys(self, data: List[Dict]):
        """
        Filter data by removing unrelevant keys
        """
        filtered = []
        for element in data:
            filtered_element = {key: value for key, value in element.items() if key not in self.unrelevant_keys}
            if "requestId" not in filtered_element:
                filtered_element.pop("updatedAt", None)
            else:
                filtered_element["decidedAt"] = filtered_element.pop("updatedAt", None)
            filtered.append(filtered_element)
        return filtered
    
    def remove_feedback_skill(self, edits: list[dict]):
        filtered = []
        for edit in edits:
            if "nlpService" in edit and edit["nlpService"] == "text_feedback":
                continue
            filtered.append(edit)
        return filtered


    def _convert_time(self, time):
        return datetime.fromisoformat(time)

    def edit_time(self, edits: list[dict]):
        new_edits = []
        if not edits:
            return []

        earliest_time = self._convert_time(edits[0]["createdAt"])
        for edit in edits:
            created_at = (self._convert_time(edit["createdAt"]) - earliest_time).total_seconds()
            edit["createdAt"] = created_at
            if edit.get("decidedAt"):
                decided_at = (self._convert_time(edit["decidedAt"]) - earliest_time).total_seconds()
                edit["decidedAt"] = decided_at
            new_edits.append(edit)
        return new_edits

    def change_mapping(self, edits: list[dict]):
        for edit in edits:
            if "operationType" in edit:
                if edit["operationType"] == 0:
                    edit["operationType"] = "insert"
                elif edit["operationType"] == 1:
                    edit["operationType"] = "delete"
                elif edit["operationType"] == 2:
                    edit["operationType"] = "retain"
                edit["user"] = "Bot" if edit["userId"] == 2 else "User"
                edit.pop("userId", None)
        return edits

    @staticmethod
    def get_final_text(text_by_user: list[tuple[str, str, list[str]]]) -> str:
        """
        Returns the resulting text (end state of editor) as a string.
        """
        return "".join([letter["text"] for letter in text_by_user])

    def find_copied_text(self, edits: list[dict], index: int, text: str):
        """Reconstructs older version of texts and returns whether the pasted text comes from a version.
        """
        if text in self.guidelines:
            return True
        for i in reversed(range(index)):
            partial_edits = edits[:i]
            text_by_user = self.notions.get_final_text_by_user_span_level(partial_edits)
            final_text = DataFormatter.get_final_text(text_by_user)
            if text in final_text:
                return True
        return False

    def find_copied_text_in_answers(self, edits: list[dict], text: str):
        for edit in edits:
            if "requestId" in edit and text in edit["response"]:
                return True
        return False

    def sanity_check(self, edits: list[dict]):
        violations = ""
        text_by_user = self.notions.get_final_text_by_user_span_level(edits)
        final_text = DataFormatter.get_final_text(text_by_user)
        if len(final_text.split(" ")) < 100:
            violations += "Too small text\n"

        for i, edit in enumerate(edits):
            if "operationType" in edit and edit["operationType"] == "insert":
                if edit["user"] == "User" and edit["span"] > 60:
                    found_in_written_text = self.find_copied_text(edits, i, edit["text"])
                    if found_in_written_text:
                        continue
                    found_in_answer = self.find_copied_text_in_answers(edits, edit["text"])
                    if found_in_answer:
                        edit["user"] = "Bot"
                        continue
                    violations += f"Inserted too much text in edit {i}" + "\n"
            if "model" in edit and edit["model"] in ["mistral:7b-instruct", "llama3.3:latest"]:
                violations += f"Used an old model\n"

        if violations == "" and (len(edits) < 100 or edits[-1]["createdAt"] - edits[0]["createdAt"] < 300):
            print("Allowing this")
        return violations, edits

    def format_and_save(self, original_data_path, formatted_data_path) -> bool:
        """
        Format and save new log file
        """
        with open(original_data_path, "r") as f:
            data = json.load(f)
        filtered_data = self.remove_unrelevant_keys(data)
        filtered_data = self.remove_feedback_skill(data)
        filtered_data = self.change_mapping(filtered_data)
        filtered_data = modify_full_query(filtered_data)
        data_time_edited = self.edit_time(filtered_data)
        violations, edits = self.sanity_check(data_time_edited)
        self.save(edits, formatted_data_path)
        return violations
