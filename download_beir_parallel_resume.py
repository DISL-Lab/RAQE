"""
Parallel BEIR downloader with resume support.
- Downloads BEIR zip files to ./datasets/IR/<dataset>/<dataset>.zip
- Unzips into ./datasets/IR/<dataset>/ and tries to find queries.jsonl
- Resumes partial downloads using HTTP Range header when supported
- Retries on transient failures

Usage:
  python download_beir_parallel_resume.py --datasets hotpotqa msmarco fever --workers 3

Notes:
- Assumes BEIR zip URL: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip
- If a download was interrupted, re-running the script will resume where it left off
- If the remote server doesn't support Range, the script will re-download the file
"""

import argparse
import os
import time
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ReadTimeout, ConnectionError as ReqConnectionError
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile
from tqdm import tqdm

CHUNK_SIZE = 1024 * 1024  # 1MB


def download_with_resume(url: str, out_path: str, max_retries: int = 8, chunk_size: int = CHUNK_SIZE, timeout: int = 120):
    """Download a file with resume support using HTTP Range header.

    Writes to out_path + '.part' while downloading and renames on success.
    Retries with exponential backoff on failure.
    """
    temp_path = out_path + ".part"
    attempt = 0

    # Prepare a session with retry/backoff for transient network issues
    session = requests.Session()
    retries = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    while attempt < max_retries:
        attempt += 1
        try:
            existing = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
        except OSError:
            existing = 0

        headers = {}
        if existing > 0:
            headers['Range'] = f'bytes={existing}-'
            print(f"Resuming download from byte {existing} for {url}")
        else:
            print(f"Starting download for {url}")

        try:
            # Use a (connect, read) timeout tuple to avoid hanging on slow reads
            with session.get(url, stream=True, headers=headers, timeout=(10, timeout)) as r:
                # If server ignored Range and sent full content (200) while we had a partial file,
                # remove the partial and restart to avoid corrupt concatenation.
                if existing > 0 and r.status_code == 200:
                    print(f"Server ignored Range header; restarting download from scratch for {url}")
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                    # restart attempt loop; this counts as a retry
                    time.sleep(2 ** min(attempt, 5))
                    continue

                if r.status_code not in (200, 206):
                    # Some servers may redirect or deny; raise to retry
                    raise RuntimeError(f"Unexpected status code: {r.status_code} for {url}")

                mode = 'ab' if r.status_code == 206 and existing > 0 else 'wb'
                total = r.headers.get('Content-Length')
                try:
                    total = int(total) + existing if total is not None else None
                except Exception:
                    total = None

                downloaded = existing
                start = time.time()
                # Use tqdm to show progress per-file. total may be None if server doesn't send Content-Length
                t = tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024, initial=existing, desc=os.path.basename(out_path))
                try:
                    with open(temp_path, mode) as f:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if not chunk:
                                continue
                            f.write(chunk)
                            downloaded += len(chunk)
                            try:
                                t.update(len(chunk))
                            except Exception:
                                # Guard against tqdm update failures
                                pass
                finally:
                    try:
                        t.close()
                    except Exception:
                        pass

                # if Content-Length provided, verify
                if total is not None and downloaded < total:
                    print(f"[download_with_resume] incomplete ({downloaded}/{total}), retry {attempt}/{max_retries}")
                    time.sleep(2 ** min(attempt, 5))
                    continue

            # Success: rename
            os.replace(temp_path, out_path)
            elapsed = time.time() - start
            speed = downloaded / elapsed if elapsed > 0 else 0
            print(f"Downloaded {out_path} ({downloaded} bytes) in {elapsed:.1f}s ({speed/1024:.1f} KB/s)")
            return out_path

        except (ReadTimeout, ReqConnectionError, requests.exceptions.RequestException) as e:
            print(f"Transient error downloading {url}: {e} (attempt {attempt}/{max_retries})")
            time.sleep(2 ** min(attempt, 5))
            continue

    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts")


def safe_unzip(zip_path: str, extract_to: str):
    os.makedirs(extract_to, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)
        print(f"Unzipped {zip_path} -> {extract_to}")
        return True
    except zipfile.BadZipFile:
        print(f"Bad zip file: {zip_path}")
        return False
    except Exception as e:
        print(f"Unzip failed: {e}")
        return False


def prepare_dataset(dataset: str, out_root: str = './datasets/IR', timeout: int = 120):
    """Download and prepare a single BEIR dataset (resume-capable).

    Returns a dict with status and paths.
    """
    info = {'dataset': dataset, 'ok': False, 'zip_path': None, 'queries_path': None, 'error': None}

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join(out_root, dataset)
    zip_path = os.path.join(out_dir, f"{dataset}.zip")
    extract_dir = out_dir

    try:
        os.makedirs(out_dir, exist_ok=True)

        # If queries already present, skip download
        possible_queries = [
            os.path.join(out_dir, dataset, 'queries.jsonl'),
            os.path.join(out_dir, 'queries.jsonl'),
            os.path.join(out_dir, dataset, 'queries.jsonl')
        ]
        for p in possible_queries:
            if os.path.exists(p):
                info['ok'] = True
                info['zip_path'] = zip_path
                info['queries_path'] = p
                print(f"Queries already exist for {dataset}: {p} -> skipping download")
                return info

        # Download with resume
        print('-' * 60)
        print(f"Downloading dataset {dataset} to {zip_path}")
        try:
            download_with_resume(url, zip_path, timeout=timeout)
        except ReadTimeout as e:
            info['error'] = f'read_timeout: {e}'
            print(f"Read timed out while downloading {dataset}: {e}")
            return info
        except ReqConnectionError as e:
            info['error'] = f'connection_error: {e}'
            print(f"Connection error while downloading {dataset}: {e}")
            return info
        except Exception as e:
            info['error'] = str(e)
            print(f"Error downloading {dataset}: {e}")
            return info
        info['zip_path'] = zip_path

        # Unzip
        success = safe_unzip(zip_path, extract_dir)
        if not success:
            info['error'] = 'unzip_failed'
            return info

        # Try common locations for queries.jsonl
        candidates = [
            os.path.join(out_dir, dataset, 'queries.jsonl'),
            os.path.join(out_dir, 'queries.jsonl'),
        ]
        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break

        if found:
            info['queries_path'] = found
            info['ok'] = True
            print(f"Found queries for {dataset}: {found}")
        else:
            print(f"Could not find queries.jsonl for {dataset} under {out_dir}. You may need to inspect the extracted files.")
            info['error'] = 'queries_not_found'

        return info

    except Exception as e:
        info['error'] = str(e)
        print(f"Error preparing {dataset}: {e}")
        return info


def download_datasets_parallel(datasets, workers: int = 3, timeout: int = 120):
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(prepare_dataset, ds, './datasets/IR', timeout): ds for ds in datasets}
        # Show overall progress across datasets
        with tqdm(total=len(futures), desc='datasets') as overall_pbar:
            for fut in as_completed(futures):
                ds = futures[fut]
                try:
                    info = fut.result()
                    results[ds] = info
                except Exception as e:
                    results[ds] = {'dataset': ds, 'ok': False, 'error': str(e)}
                overall_pbar.update(1)
    return results


def main():
    parser = argparse.ArgumentParser(description='Download BEIR datasets in parallel with resume support')
    parser.add_argument('--datasets', nargs='+', default=['hotpotqa', 'msmarco', 'fever'], help='List of BEIR dataset names')
    parser.add_argument('--workers', type=int, default=3, help='Number of parallel workers')
    parser.add_argument('--timeout', type=int, default=120, help='Read timeout in seconds for each download chunk')
    args = parser.parse_args()

    # Normalize dataset name: allow user to pass 'msmacro' as typo -> assume 'msmarco'
    normalized = []
    for d in args.datasets:
        if d.lower() in ('msmacro', 'msmarco'):
            normalized.append('msmarco')
        else:
            normalized.append(d.lower())

    print(f"Downloading datasets: {normalized} with {args.workers} workers (timeout={args.timeout}s)")

    results = download_datasets_parallel(normalized, workers=args.workers, timeout=args.timeout)

    print('\nSummary:')
    for ds, info in results.items():
        status = 'OK' if info.get('ok', False) else f"FAILED ({info.get('error')})"
        print(f"- {ds}: {status}")


if __name__ == '__main__':
    main()
