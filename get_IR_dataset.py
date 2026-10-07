import json
import argparse
import warnings
import requests
import gzip
import shutil
import os

import raqe_jvm  # noqa: F401  (must precede the pyserini import)

from pyserini.search import get_qrels
from beir import util, LoggingHandler

warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------------------------------------------------------
# 1) Main function: get_IR_dataset
#    - Download dataset via BEIR if dataset != dl19/dl20
#    - Download dataset via LuceneSearcher if dataset == dl19/dl20
#    - Remove leftover .zip files (if any) after unzipping
#    - Convert queries to JSON for the relevant dataset
#    - Filter them with qrels (Pyserini) and save to test_subsample_processed.json
# -----------------------------------------------------------------------------
def get_IR_dataset(dataset):
    print('-' * 50)
    print(f'Processing dataset : {dataset}')

    queries_dict = {}
    json_output_path = f'./datasets/IR/{dataset}/{dataset}/test_subsample_processed.json'

    # -------------------------------------------------------------------------
    # 1.1) Download datasets from BEIR if not dl19/dl20 or cqadupstack subsets
    # -------------------------------------------------------------------------
    if dataset not in ['dl19', 'dl20'] and not dataset.startswith('cqadupstack-'):
        # Download & unzip
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        out_dir = f"./datasets/IR/{dataset}"

        zip_file_path = os.path.join(out_dir, f"{dataset}.zip")
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)
            print(f"Removed existing zip file: {zip_file_path}")
        
        if not os.path.exists(out_dir + f'/{dataset}/queries.jsonl'):
            util.download_and_unzip(url, out_dir)

        # Handle cqadupstack subsets.
        if dataset.startswith('cqadupstack-'):
            subset = dataset.split('-', 1)[1]
            out_dir = "./datasets/IR/cqadupstack"
            qrels_key = f"beir-v1.0.0-cqadupstack-{subset}-test"
            queries_file = os.path.join(out_dir, subset, "queries.jsonl")
            json_output_path = f"./datasets/IR/cqadupstack/{subset}/test_subsample_processed.json"
        else:
            qrels_key = f"beir-v1.0.0-{dataset}-test"
            queries_file = os.path.join(out_dir, dataset, "queries.jsonl")
            json_output_path = f"./datasets/IR/{dataset}/{dataset}/test_subsample_processed.json"

        # For BEIR datasets, Pyserini exposes qrels under beir-v1.0.0-<dataset>-test.
        qrels = get_qrels(qrels_key)
        if os.path.exists(queries_file):
            with open(queries_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    queries_dict[data["_id"]] = data["text"]
        else:
            print(f"Warning: Could not find {queries_file}, so queries_dict may remain empty.")

    # -------------------------------------------------------------------------
    # 1.2) Download datasets from LuceneSearcher (for dl19/dl20)
    # -------------------------------------------------------------------------
    else:
        # For dl19/dl20, we get qrels from {dl19, dl20}-passage
        qrels = get_qrels(f'{dataset}-passage')

        # Prepare queries.tsv path
        save_path = f'./datasets/IR/{dataset}/{dataset}/queries.tsv.gz'
        query_path = f'./datasets/IR/{dataset}/{dataset}/queries.tsv'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Download the queries .tsv.gz
        num = 2019 if dataset == 'dl19' else 2020
        url = f"https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test{num}-queries.tsv.gz"

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print(f"Downloaded: {save_path}")
        else:
            print("Failed to download the file")
            exit(1)

        # Extract .tsv.gz → queries.tsv
        with gzip.open(save_path, "rb") as f_in:
            with open(query_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Extracted: {query_path}")

        os.remove(save_path)
        print(f"Removed: {save_path}")

        # Read queries from TSV
        with open(query_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue  # Skip malformed lines
                query_id, query_text = parts[0], parts[1]
                queries_dict[query_id] = query_text

    # -------------------------------------------------------------------------
    # 1.3) Filter queries by qrels and write them to JSON
    # -------------------------------------------------------------------------
    # Convert qrels keys to strings (the doc IDs are forced to int for safety)
    qrels = {
        str(qid): {str(docid): int(rel) for docid, rel in rel_dict.items()}
        for qid, rel_dict in qrels.items()
    }

    # Filter only queries that appear in qrels
    filtered_queries = {qid: queries_dict[qid] for qid in qrels.keys() if qid in queries_dict}

    # Write out only filtered queries to test_subsample_processed.json
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    with open(json_output_path, 'w', encoding='utf-8') as outfile:
        for idx, (qid, query_text) in enumerate(filtered_queries.items()):
            json_obj = {
                "qid": qid,
                "query_text": query_text,
            }
            outfile.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

    # Status output
    print(f"Wrote {len(filtered_queries)} test queries with judgements to {json_output_path}")
    print('-' * 50)

    return filtered_queries



# -----------------------------------------------------------------------------
# 2) Main entry point
#    - Parse command-line arguments
#    - (Optionally) handle multiple datasets at once
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get BeIR datasets")
    parser.add_argument('-dataset', type=str, required=True)
    args = parser.parse_args()

    if args.dataset == 'all':
        # The eight datasets evaluated in the paper: five with official training
        # splits, then the three out-of-distribution sets.
        datasets = [
            'nfcorpus', 'fiqa', 'fever', 'hotpotqa', 'msmarco',
            'arguana', 'nq', 'scidocs',
        ]
        for dataset in datasets:
            get_IR_dataset(dataset)
    else:
        get_IR_dataset(args.dataset)
