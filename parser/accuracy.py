import re
import pandas as pd
import numpy as np


def sort_csv_by_content_order(file1_df, file2_df, to_file, save_sorted=False):
    file1_df_unique = file1_df.drop_duplicates(subset='Content', keep='first')
    merged_df = pd.merge(file2_df[['Content']], file1_df_unique, on='Content', how='left')
    if save_sorted:
        merged_df.to_csv(to_file, index=False)
    return merged_df

def get_accuracy(series_groundtruth, series_parsedlog, debug=False):
    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    accurate_events = 0  # determine how many lines are correctly parsed
    for parsed_eventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
        error_eventIds = (
            parsed_eventId,
            series_groundtruth_logId_valuecounts.index.tolist(),
        )
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if (
                    logIds.size
                    == series_groundtruth[series_groundtruth == groundtruth_eventId].size
            ):
                accurate_events += logIds.size
                error = False
        if error and debug:
            print(
                "(parsed_eventId, groundtruth_eventId) =",
                error_eventIds,
                "failed",
                logIds.size,
                "messages",
            )
    precision = 0
    recall = 0
    f_measure = 0
    accuracy = float(accurate_events) / series_groundtruth.size
    return precision, recall, f_measure, accuracy

def evaluate_result(predic_file, gt_file, sorted_file, save_sorted=False,sort=True):
    column_names = ["Content", "RegexTemplate", "EventId"]
    if sort:
        df_parsedlog = pd.read_csv(
            predic_file
            , usecols=column_names, dtype=str
        )
        df_gtlog = pd.read_csv(
            gt_file, usecols=["Content", "EventId", "EventTemplate"], dtype=str
        )
        df_parsedlog = sort_csv_by_content_order(df_parsedlog, df_gtlog, sorted_file, save_sorted)
        print("df_parsedlog sorted! ", flush=True)
    else:
        df_parsedlog = pd.read_csv(
            sorted_file
            , usecols=column_names, dtype=str
        )
        df_gtlog = pd.read_csv(
            gt_file, usecols=["Content", "EventId", "EventTemplate"], dtype=str
        )
        print("df_parsedlog sorted file loaded! ", flush=True)
    df_parsedlog["RegexTemplate_NoSpaces_NoVar_cleaned"] = df_parsedlog[
        "RegexTemplate"
    ].apply(clean_regex_content)
    print("df_parsedlog RegexTemplate ready to be checked", flush=True)
    df_gtlog["EventTemplate_NoSpaces_NoVar_cleaned"] = df_gtlog["EventTemplate"].apply(
        clean_content
    )
    print("df_gtlog EventTemplate ready to be checked", flush=True)
    correctly_parsed_messages = df_parsedlog['RegexTemplate_NoSpaces_NoVar_cleaned'].eq(df_gtlog['EventTemplate_NoSpaces_NoVar_cleaned']).values.sum()
    PA = float(correctly_parsed_messages) / len(df_parsedlog[['Content']])
    print(f"PA: {PA}", flush=True)
    (precision, recall, f_measure, GA) = get_accuracy(
        df_gtlog["EventId"],df_parsedlog["RegexTemplate_NoSpaces_NoVar_cleaned"]
    )
    print(f"GA: {GA}", flush=True)
    event_count = str(df_parsedlog["RegexTemplate_NoSpaces_NoVar_cleaned"].nunique())
    return (
        GA,
        PA,
        event_count,
    )
    
def clean_content(content):
    content = content.replace(",", "")
    content = content.replace(".", "")
    segments = content.split(" ")
    cleaned_segments = ["" if "<*>" in segment else segment for segment in segments]
    return "".join(cleaned_segments)

def clean_regex_content(content):
    if pd.isna(content):
        return ""
    content = content.replace("</s>", '')
    pattern = r"\((?:\?P<[^>]+>)?(?:\\.|[^()\\])*?\)"
    segments = smart_split(content)
    cleaned_segments = []
    for segment in segments:
        if not (re.search(pattern, segment) or re.search(r"^([+\.?()\[\]{}]+)$", segment) or re.search(r'a-z',
                                                                                                     segment) or re.search(
            r'a-f', segment) or re.search(
            r'\\w', segment) or re.search(r'\\d', segment)):
            cleaned_segments.append(segment)
    result = "".join(cleaned_segments)
    if result.startswith("^"):
        result = result[1:]
    if result.endswith("$"):
        result = result[:-1]
    result = result.replace("\\", "")
    result = result.replace(".", "")
    return result

def smart_split(input_string):
    initial_segments = input_string.split(" ")
    final_segments = []
    for segment in initial_segments:
        if "\\s+" in segment:
            sub_segments = re.split(r"\\s\+", segment)
            final_segments.extend(sub_segments)
        elif "\\s*" in segment:
            sub_segments = re.split(r"\\s\*", segment)
            final_segments.extend(sub_segments)
        elif "\\s" in segment:
            sub_segments = re.split(r"\\s", segment)
            final_segments.extend(sub_segments)
        else:
            final_segments.append(segment)
    return final_segments