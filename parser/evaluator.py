import grouping
import os
import torch
import csv
import sys
import argparse
import llama_parser
import transformers
import regex as re
import pandas as pd
import regex_manager
import accuracy
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
from tqdm import tqdm
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="Apache")
parser.add_argument(
    "--model",
    type=str,
    default="../models/Meta-Llama-3-8B-Instruct",
)
parser.add_argument("--sample", type=str, default="3")
parser.add_argument("--similarity", type=str, default="jaccard")
parser.add_argument("--do_self_reflection", type=str, default="True")
args = parser.parse_args()

datasets_full = args.project.split(",")
model_path = args.model
similarity = args.similarity
regex_sample = int(args.sample)
do_self_reflection = args.do_self_reflection

benchmark_settings = {
    "HDFS": {
        "log_file": "HDFS/HDFS_2k.log",
        "log_format": "<Date> <Time> <Pid> <Level> <Component>: <Content>",
        "regex": [r"blk_-?\d+", r"(\d+\.){3}\d+(:\d+)?"],
        "st": 0.5,
        "depth": 4,
    },
    "Hadoop": {
        "log_file": "Hadoop/Hadoop_2k.log",
        "log_format": "<Date> <Time> <Level> \[<Process>\] <Component>: <Content>",
        "regex": [r"(\d+\.){3}\d+"],
        "st": 0.5,
        "depth": 4,
    },
    "Spark": {
        "log_file": "Spark/Spark_2k.log",
        "log_format": "<Date> <Time> <Level> <Component>: <Content>",
        "regex": [r"(\d+\.){3}\d+", r"\b[KGTM]?B\b", r"([\w-]+\.){2,}[\w-]+"],
        "st": 0.5,
        "depth": 4,
    },
    "Zookeeper": {
        "log_file": "Zookeeper/Zookeeper_2k.log",
        "log_format": "<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>",
        "regex": [r"(/|)(\d+\.){3}\d+(:\d+)?"],
        "st": 0.5,
        "depth": 4,
    },
    "BGL": {
        "log_file": "BGL/BGL_2k.log",
        "log_format": "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
        "regex": [r"core\.\d+"],
        "st": 0.5,
        "depth": 4,
    },
    "HPC": {
        "log_file": "HPC/HPC_2k.log",
        "log_format": "<LogId> <Node> <Component> <State> <Time> <Flag> <Content>",
        "regex": [r"=\d+"],
        "st": 0.5,
        "depth": 4,
    },
    "Thunderbird": {
        "log_file": "Thunderbird/Thunderbird_2k.log",
        "log_format": "<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>",
        "regex": [r"(\d+\.){3}\d+"],
        "st": 0.5,
        "depth": 4,
        # "st": 0.2,
        # "depth": 3,
    },
    "Windows": {
        "log_file": "Windows/Windows_2k.log",
        "log_format": "<Date> <Time>, <Level>                  <Component>    <Content>",
        "regex": [r"0x.*?\s"],
        "st": 0.7,
        "depth": 5,
    },
    "Linux": {
        "log_file": "Linux/Linux_2k.log",
        "log_format": "<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>",
        "regex": [r"(\d+\.){3}\d+", r"\d{2}:\d{2}:\d{2}"],
        "st": 0.39,
        "depth": 6,
    },
    "Android": {
        "log_file": "Android/Android_2k.log",
        "log_format": "<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>",
        "regex": [
            r"(/[\w-]+)+",
            r"([\w-]+\.){2,}[\w-]+",
            r"\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b",
        ],
        "st": 0.2,
        "depth": 6,
    },
    "HealthApp": {
        "log_file": "HealthApp/HealthApp_2k.log",
        "log_format": "<Time>\|<Component>\|<Pid>\|<Content>",
        "regex": [],
        "st": 0.2,
        "depth": 4,
    },
    "Apache": {
        "log_file": "Apache/Apache_2k.log",
        "log_format": "\[<Time>\] \[<Level>\] <Content>",
        "regex": [r"(\d+\.){3}\d+"],
        "st": 0.5,
        "depth": 4,
    },
    "Proxifier": {
        "log_file": "Proxifier/Proxifier_2k.log",
        "log_format": "\[<Time>\] <Program> - <Content>",
        "regex": [
            r"<\d+\ssec",
            r"([\w-]+\.)+[\w-]+(:\d+)?",
            r"\d{2}:\d{2}(:\d{2})*",
            r"[KGTM]B",
        ],
        "st": 0.6,
        "depth": 3,
    },
    "OpenSSH": {
        "log_file": "OpenSSH/OpenSSH_2k.log",
        "log_format": "<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>",
        "regex": [r"(\d+\.){3}\d+", r"([\w-]+\.){2,}[\w-]+"],
        "st": 0.6,
        "depth": 5,
    },
    "OpenStack": {
        "log_file": "OpenStack/OpenStack_2k.log",
        "log_format": "<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>",
        "regex": [r"((\d+\.){3}\d+,?)+", r"/.+?\s", r"\d+"],
        "st": 0.5,
        "depth": 5,
    },
    "Mac": {
        "log_file": "Mac/Mac_2k.log",
        "log_format": "<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>",
        "regex": [r"([\w-]+\.){2,}[\w-]+"],
        "st": 0.7,
        "depth": 6,
    },
}


def log_file_to_logs(
    log_file, logformat, first_lines_percent=100, start_line_percent=0
):
    """Function to transform log file to dataframe, reads from a specific start line and reads up to a given percent of lines."""
    headers, regex = generate_logformat_regex(logformat)
    log_messages = []
    with open(log_file, "r") as fin:
        lines = fin.readlines()
        total_lines = len(lines)  
        start_line = int(
            total_lines * start_line_percent / 100
        )  
        lines_to_read = int(
            (total_lines - start_line) * (first_lines_percent / 100)
        )  
        for i, line in enumerate(
            lines[start_line : start_line + lines_to_read], start=start_line
        ):

            try:
                match = regex.search(line.strip())
                if match:
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
            except Exception as e:
                print("Skip line: ", line)

    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(
        0, "LineId", range(start_line + 1, start_line + len(log_messages) + 1)
    )  

    array_result = logdf.loc[:, ["Content"]].values
    list_result = [list(row) for row in array_result]
    return list_result


def read_column_from_csv(file_path, column_name="Content"):
    column_data = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if column_name in row:
                column_data.append(row[column_name])
            else:
                raise ValueError(
                    f"The column '{column_name}' does not exist in the CSV file."
                )
    return column_data


def generate_logformat_regex(logformat):
    """
    Function to generate regular expression to split log messages

    """
    headers = []
    splitters = re.split(r"(<[^<>]+>)", logformat)
    regex = ""
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(" +", r"\s+", splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip("<").strip(">")
            regex += "(?P<%s>.*?)" % header
            headers.append(header)
    regex = re.compile("^" + regex + "$")
    return headers, regex


def group_logs_using_parser(grouped_logs):
    df = pd.DataFrame(grouped_logs, columns=["Content", "EventId", "EventTemplate"])
    df = df[["Content", "EventId", "EventTemplate"]]
    grouped = df.groupby("EventId")
    groups_dict = {}
    for name, group in grouped:
        groups_dict[name] = group.to_dict("records")
    return groups_dict


def get_logs_from_group(group_list):
    logs_from_group = []
    for ele in group_list:
        logs_from_group.append(ele["Content"])
    return logs_from_group


def check_group_count(groups_dict, removed_items=[]):
    for eventID, logs in list(groups_dict.items()):
        if len(logs) < 5:
            removed_items.extend(
                [[log["Content"], log["EventId"], log["EventTemplate"]] for log in logs]
            )
            del groups_dict[eventID]
    return removed_items, groups_dict


def res_list_to_file(res_list, out_path, regex_sample):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    file_path = os.path.join(out_path, str(regex_sample) + ".csv")

    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)

        if not file_exists:
            writer.writerow(["Content", "EventId", "RegexTemplate"])

        writer.writerows(res_list)

    return file_path


def one_result_to_file(one_result, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with open(out_path + str(regex_sample) + ".csv", "a", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(one_result)
    return out_path + str(regex_sample) + ".csv"


def reorder_csv_in_place(csv_path, order_list):
    data = []
    with open(csv_path, mode="r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader) 
        data = list(reader)  

    rows_by_key = {row[0]: row for row in data if row}

    sorted_data = []
    for key in order_list:
        if key in rows_by_key:
            sorted_data.append(rows_by_key[key])

    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)  
        writer.writerows(sorted_data)  


def prepare_results(output_dir, parser_name, sample_size, list_to_insert, order_list):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_file = "summary_[parser={},sample_size={}].csv".format(
        str(parser_name), str(sample_size)
    )
    result_file_path = os.path.join(output_dir, result_file)

    if not os.path.exists(result_file_path) or os.stat(result_file_path).st_size == 0:
        with open(result_file_path, "w", newline="") as csv_file:
            fw = csv.writer(csv_file)
            fw.writerow(
                [
                    "Dataset",
                    "Total_time",
                    "LLaMA_parsing_time",
                    "Drain_parsing_time",
                    "Regex_parsing_time",
                    "GA",
                    "PA",
                    "Event_count",
                ]
            )

    with open(result_file_path, "a", newline="") as csv_file:
        fw = csv.writer(csv_file)
        fw.writerow(list_to_insert)
    reorder_csv_in_place(result_file_path, order_list)
    return result_file


def sort_dict_by_content_length(input_dict):
    def count_words_in_content(entry):
        return len(entry["Content"].split())

    sorted_items = sorted(
        input_dict.items(), key=lambda item: count_words_in_content(item[1][0])
    )

    sorted_dict = {key: value for key, value in sorted_items}
    return sorted_dict


def append_unique_to_csv(data_list, file_path):
    new_data = pd.DataFrame(data_list)
    file = Path(file_path)

    if "Count" in new_data.columns:
        new_data = new_data.drop(columns="Count")
    new_data = new_data.groupby(new_data.columns.tolist(), as_index=False).size()
    new_data = new_data.rename(columns={"size": "Count"})

    if file.is_file():
        existing_data = pd.read_csv(file_path, dtype={1: str})
    else:
        existing_data = pd.DataFrame(columns=new_data.columns)

    combined_data = pd.concat([existing_data, new_data], ignore_index=True)

    combined_data.to_csv(file_path, index=False, header=True)
    return file_path


order_list = [
    "HDFS",
    "Hadoop",
    "Spark",
    "Zookeeper",
    "BGL",
    "HPC",
    "Thunderbird",
    "Windows",
    "Linux",
    "Android",
    "HealthApp",
    "Apache",
    "Proxifier",
    "OpenSSH",
    "OpenStack",
    "Mac",
]

if __name__ == "__main__":
    path_prefix = "../result_offline_similar/"
    if "chatglm" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(
            "../models/chatglm3-6b", trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            "../models/chatglm3-6b", trust_remote_code=True, device="cuda"
        )
        model = model.eval()
        pipeline = (model, tokenizer)
    else:
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
    print(f"{model_path} Pipeline is ready.", flush=True)
    for system in datasets_full:
        print(f"Start Parsing {system}", flush=True)
        log_file = f"../full_dataset/{system}/{system}_full.log_structured.csv"
        out_path = f"{path_prefix}{system}/"
        if not os.path.exists(path_prefix):
            os.makedirs(path_prefix)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        setting = benchmark_settings[system]
        start_time = datetime.now()
        Tree_parser1 = grouping.LogParser(
            rex=setting["regex"], depth=setting["depth"], st=setting["st"]
        )
        # Tree parser with default settings
        # Tree_parser1 = grouping.LogParser(
        #     rex=setting["regex"]
        # )
        logs = read_column_from_csv(log_file)
        grouped_logs = Tree_parser1.parse(logs)
        groups_dict = group_logs_using_parser(grouped_logs)
        groups_dict = sort_dict_by_content_length(groups_dict)
        print("==================", flush=True)
        print(
            "initial set grouping finished, start parsing. ",
            len(groups_dict.keys()),
            " groups in total for ",
            len(logs),
            " logs",
            flush=True,
        )
        print("==================", flush=True)
        regex_manager1 = regex_manager.RegexTemplateManager()
        llama_parser1 = llama_parser.LogParser(
            pipeline=pipeline,
            model=model_path,
            regex_manager1=regex_manager1,
            regex_sample=regex_sample,
            similarity=similarity,
            do_self_reflection=do_self_reflection,
        )
        for eventid in tqdm(groups_dict.keys(), desc=f"Processing events {system}"):
            append_unique_to_csv(groups_dict[eventid], out_path + "group.csv")
            res_list = []
            logs_from_group = get_logs_from_group(groups_dict[eventid])
            res_list = llama_parser1.parse(groups_dict[eventid], logs_from_group)
            out_file = res_list_to_file(res_list, out_path, regex_sample=regex_sample)

        Tree_parser1.print_time()
        regex_manager1.print_time()
        regex_manager1.print_regex_templates()
        total_time = datetime.now() - start_time
        print(
            system + " Parsing done. [Time taken: {!s}]".format(total_time), flush=True
        )
        ground_truth_file = f"../full_dataset/{system}/{system}_full.log_structured.csv"
        file_path = f"{out_path}/{str(regex_sample)}.csv"
        sorted_file = f"{out_path}/{str(regex_sample)}_sorted.csv"
        GA, PA, event_count = accuracy.evaluate_result(
            file_path, ground_truth_file, sorted_file, save_sorted=True
        )
        print("==================", flush=True)
        print(
            system,
            total_time,
            llama_parser1.total_time - regex_manager1.total_time,
            Tree_parser1.total_time,
            regex_manager1.total_time,
            GA,
            PA,
            event_count,
            flush=True,
        )
        prepare_results(
            output_dir=path_prefix,
            parser_name="Drain",
            sample_size=regex_sample,
            list_to_insert=[
                system,
                total_time,
                llama_parser1.total_time - regex_manager1.total_time,
                Tree_parser1.total_time,
                regex_manager1.total_time,
                GA,
                PA,
                event_count,
            ],
            order_list=order_list,
        )
        print("==================", flush=True)
