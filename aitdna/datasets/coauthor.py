import json
import os
import argparse

from .aitdna_dataset.processing.format_data import get_and_save_notions,\
    get_and_save_final_text, create_folders_for_analysis

def process_coauthor_traces(traces):
    new_traces = []
    for trace in traces:
        if trace["eventName"] == "system-initialize":
            new_trace = {}
            new_trace["user"] = "User"
            new_trace["offset"] = 0
            new_trace["operationType"] = "insert"
            new_trace["text"] = trace["currentDoc"]
            new_trace["span"] = len(trace["currentDoc"])
            new_traces.append(new_trace)
        if trace["eventName"] in ("text-insert", "text-delete"):
            retain_index = 0
            ops = trace["textDelta"]["ops"]
            # if a user selects text, delets it and inserts something instead, the insertion is logged first
            # so, deletion index is wrong. swap them for the right order.
            if len(ops) == 3 and "retain" in ops[0].keys() and "insert" in ops[1].keys() and "delete" in ops[2].keys():
                ops = [ops[0], ops[2], ops[1]]
            for op in ops:
                new_trace = {}
                if "retain" in op:
                    retain_index += op["retain"]
                if "insert" in op:
                    new_trace["user"] = "User" if trace["eventSource"] == "user" else "Bot"
                    new_trace["operationType"] = "insert"
                    new_trace["offset"] = retain_index
                    new_trace["text"] = op["insert"]
                    new_trace["span"] = len(op["insert"])
                if "delete" in op:
                    new_trace["user"] = "User" if trace["eventSource"] == "user" else "Bot"
                    new_trace["operationType"] = "delete"
                    new_trace["offset"] = retain_index
                    new_trace["text"] = None
                    new_trace["span"] = op["delete"]
                if new_trace:
                    new_traces.append(new_trace)
    return new_traces


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", '--src_root', type=str, default="data/other_datasets/original/coauthor-v1.0")
    parser.add_argument("-d", '--dst_root', type=str, default="data/other_datasets/processed/coauthor-v1.0")
    args = parser.parse_args(argv)

    SRC_ROOT = args.src_root
    DST_ROOT = args.dst_root
    for file in os.listdir(SRC_ROOT):
        if not os.path.exists(DST_ROOT):
            os.mkdir(DST_ROOT)
        dst_folder = os.path.join(DST_ROOT, file.replace(".jsonl", ""))
        if not os.path.exists(dst_folder):
            os.mkdir(dst_folder)
        stats_path, notions_path, boundary_path = create_folders_for_analysis(dst_folder)

        src_file_path = os.path.join(SRC_ROOT, file)
        with open(src_file_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]


        edits = process_coauthor_traces(data)
        dst_file_path = os.path.join(dst_folder, "edits.json")
        with open(dst_file_path, "w", encoding="utf-8") as f:
            json.dump(edits, f)

        _, _, _, text_by_user, _ = get_and_save_notions(edits,
                                                                boundary_path,
                                                                notions_path)
    
        get_and_save_final_text(text_by_user, dst_folder)