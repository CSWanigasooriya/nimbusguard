import argparse
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from prometheus_api_client import PrometheusConnect

# Get the directory where the script is located
SCRIPT_DIR = Path(__file__).resolve().parent

# Configure logging
LOG_FILE = SCRIPT_DIR / "prometheus_export.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)


def test_connection(prometheus_url, prom_conn=None):
    """Test connection to Prometheus."""
    logging.info(f"Testing connection to {prometheus_url}...")
    try:
        if not prom_conn:
            prom_conn = PrometheusConnect(url=prometheus_url, disable_ssl=True)
        if prom_conn.check_prometheus_connection():
            logging.info("Connection successful!")
            return True
        else:
            logging.error("Connection failed. Check URL and Prometheus status.")
            return False
    except Exception as e:
        logging.error(f"Connection test failed: {e}")
        return False


def list_metrics(prometheus_url):
    """List available metrics from Prometheus."""
    logging.info(f"Fetching metric list from {prometheus_url}...")
    try:
        prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)
        if not test_connection(prometheus_url, prom_conn=prom):
            return

        metrics = prom.all_metrics()
        print("Available metrics:")
        for metric in sorted(metrics):
            print(metric)
    except Exception as e:
        logging.error(f"Could not fetch metrics: {e}")


def process_metric(
    metric_name, prom, start_time, end_time, step, output_dir, output_format
):
    """Fetch a single metric and return it as a DataFrame."""
    try:
        logging.info(f"Fetching data for {metric_name}...")
        metric_data = prom.custom_query_range(
            query=metric_name,
            start_time=start_time,
            end_time=end_time,
            step=step,
        )
        if not metric_data:
            logging.warning(f"No data for metric {metric_name}")
            return None

        df_list = []
        for d in metric_data:
            metric_labels = d["metric"]
            df = pd.DataFrame(d["values"], columns=["timestamp", "value"])
            df["metric_name"] = metric_labels.pop("__name__", metric_name)
            for label, value in metric_labels.items():
                df[label] = value
            df_list.append(df)

        if not df_list:
            return None

        metric_df = pd.concat(df_list, ignore_index=True)
        metric_df["timestamp"] = pd.to_datetime(metric_df["timestamp"], unit="s")
        metric_df["value"] = pd.to_numeric(metric_df["value"], errors="coerce")
        metric_df = metric_df.dropna(subset=["value"])

        if output_format == "individual":
            safe_metric_name = re.sub(r'[^0-9a-zA-Z_]', '_', metric_name)
            output_path = Path(output_dir) / f"{safe_metric_name}.csv"
            logging.info(f"Saving to {output_path}...")
            metric_df.to_csv(output_path, index=False)
            return None  # Don't return df to save memory
        else:
            return metric_df
    except Exception as e:
        logging.error(f"Failed to export {metric_name}: {e}", exc_info=True)
        return None


def export_metrics(
    prometheus_url,
    days,
    step,
    output_format,
    output_dir,
    metrics_file,
    filter_regex,
    workers,
    timeout,
    retries,
    dry_run,
):
    """Export metrics from Prometheus."""
    logging.info("Starting metrics export...")
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = SCRIPT_DIR / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logging.info("--- DRY RUN ---")
        logging.info(f"URL: {prometheus_url}")
        logging.info(f"Days: {days}, Step: {step}")
        logging.info(f"Output Format: {output_format}, Output Dir: {output_path}")
        if metrics_file:
            logging.info(f"Metrics File: {metrics_file}")
        if filter_regex:
            logging.info(f"Filter: {filter_regex}")
        logging.info("--- END DRY RUN ---")
        return

    prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)
    if not test_connection(prometheus_url, prom_conn=prom):
        return

    if metrics_file:
        try:
            with open(metrics_file) as f:
                metric_names = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logging.error(f"Metrics file not found: {metrics_file}")
            return
    else:
        metric_names = prom.all_metrics()

    if filter_regex:
        try:
            regex = re.compile(filter_regex)
            metric_names = [m for m in metric_names if regex.search(m)]
        except re.error as e:
            logging.error(f"Invalid regex '{filter_regex}': {e}")
            return

    logging.info(f"Found {len(metric_names)} metrics to export.")
    if not metric_names:
        logging.warning("No metrics matched criteria. Exiting.")
        return

    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    all_metrics_dfs = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_metric,
                metric,
                prom,
                start_time,
                end_time,
                step,
                output_path,
                output_format,
            )
            for metric in metric_names
        ]

        for future in futures:
            result_df = future.result()
            if result_df is not None:
                all_metrics_dfs.append(result_df)

    if output_format == "unified" and all_metrics_dfs:
        logging.info(f"Concatenating {len(all_metrics_dfs)} DataFrames...")
        final_df = pd.concat(all_metrics_dfs, ignore_index=True)
        unified_output_path = output_path / "unified_metrics.parquet"
        logging.info(f"Saving unified dataset to {unified_output_path}...")
        final_df.to_parquet(unified_output_path, index=False)

    logging.info("Export complete.")


def main():
    """Main function to parse arguments and execute commands."""
    parser = argparse.ArgumentParser(
        description="Export metrics from a Prometheus instance."
    )
    parser.add_argument(
        "--test-connection", action="store_true", help="Test connection to Prometheus."
    )
    parser.add_argument(
        "--list-metrics", action="store_true", help="List available metrics."
    )
    parser.add_argument(
        "--url",
        default="http://localhost:9090",
        help="Prometheus server URL.",
    )
    parser.add_argument("--days", type=int, default=7, help="Number of days to export.")
    parser.add_argument("--step", default="1m", help="Query resolution step width.")
    parser.add_argument(
        "--output-format",
        choices=["unified", "individual"],
        default="individual",
        help="Output format for metrics.",
    )
    parser.add_argument(
        "--output-dir",
        default="prometheus_data",
        help="Directory to save exported data.",
    )
    parser.add_argument(
        "--metrics", dest="metrics_file", help="File with a list of metrics to export."
    )
    parser.add_argument(
        "--filter", dest="filter_regex", help="Regex to filter metric names."
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers."
    )
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout.")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be exported."
    )

    args = parser.parse_args()

    try:
        if args.test_connection:
            test_connection(args.url)
        elif args.list_metrics:
            list_metrics(args.url)
        else:
            export_metrics(
                prometheus_url=args.url,
                days=args.days,
                step=args.step,
                output_format=args.output_format,
                output_dir=args.output_dir,
                metrics_file=args.metrics_file,
                filter_regex=args.filter_regex,
                workers=args.workers,
                timeout=args.timeout,
                retries=args.retries,
                dry_run=args.dry_run,
            )
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 