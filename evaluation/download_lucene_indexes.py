#!/usr/bin/env python3
"""
Download Lucene indexes for sparse retrieval (BM25) using Pyserini.
This script downloads the required Lucene indexes for the specified datasets.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add the project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import raqe_jvm  # noqa: F401  (must precede the pyserini import)

try:
    from pyserini.search.lucene import LuceneSearcher
except ImportError as e:
    print(f"Error importing LuceneSearcher: {e}")
    print("Make sure pyserini is installed.")
    sys.exit(1)

def download_lucene_index(dataset: str):
    """Download Lucene index for a dataset by attempting to load it."""
    # Must stay in sync with evaluation/evaluate_raqe.py: RAQE evaluates with BM25
    # over the `.flat` BEIR indexes, not the learned-sparse `.splade-pp-ed` ones.
    if dataset in ("msmarco", "dl19", "dl20"):
        index_key = "msmarco-v1-passage"
    elif dataset.startswith("cqadupstack-"):
        subset = dataset.split("-", 1)[1]
        index_key = f"beir-v1.0.0-cqadupstack-{subset}.flat"
    else:
        index_key = f"beir-v1.0.0-{dataset}.flat"
    
    print(f"Downloading Lucene index for dataset: {dataset} (index: {index_key})")
    try:
        # This will trigger download if the index doesn't exist
        searcher = LuceneSearcher.from_prebuilt_index(index_key)
        print(f"Successfully downloaded/verified index for {dataset}")
    except Exception as e:
        print(f"Failed to download index for {dataset}: {e}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Download Lucene indexes for sparse retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets
  python download_lucene_indexes.py
  
  # Download specific datasets only
  python download_lucene_indexes.py --datasets fever hotpotqa
  
  # Download CQADupStack subsets
  python download_lucene_indexes.py --datasets cqadupstack-wordpress cqadupstack-programmers
"""
    )
    
    # All available datasets
    all_datasets = [
        "nfcorpus",
        "fiqa", 
        "fever",
        "hotpotqa",
        "msmarco",
        "scifact",
        "arguana",
        "nq",
        "scidocs",
        "trec-covid",
        "webis-touche2020",
        "dl19",
        "dl20",
        # CQADupStack subsets
        "cqadupstack-android",
        "cqadupstack-english", 
        "cqadupstack-gaming",
        "cqadupstack-gis",
        "cqadupstack-mathematica",
        "cqadupstack-physics",
        "cqadupstack-programmers",
        "cqadupstack-stats",
        "cqadupstack-tex",
        "cqadupstack-unix",
        "cqadupstack-webmasters",
        "cqadupstack-wordpress"
    ]
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=all_datasets,
        help="Specific datasets to download (default: the eight paper datasets)"
    )
    
    args = parser.parse_args()

    # The eight datasets evaluated in the paper; the remaining entries of
    # `all_datasets` stay selectable via --datasets.
    paper_datasets = ["nfcorpus", "fiqa", "fever", "hotpotqa", "msmarco", "arguana", "nq", "scidocs"]
    datasets = args.datasets if args.datasets else paper_datasets

    print("Starting Lucene index downloads...")
    print(f"Indexes will be saved to: ~/.cache/pyserini/indexes/")
    print(f"Datasets to download: {', '.join(datasets)}")
    
    success_count = 0
    for dataset in datasets:
        if download_lucene_index(dataset):
            success_count += 1
    
    print(f"\nDownload complete: {success_count}/{len(datasets)} indexes downloaded successfully")
    
    if success_count == len(datasets):
        print("All indexes are ready for sparse retrieval!")
    else:
        print("Some downloads failed. Check the output above.")

if __name__ == "__main__":
    main()