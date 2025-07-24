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


def find_actual_data_range(prom, metrics_sample=None, max_lookback_days=90):
    """Find the actual earliest and latest data in Prometheus."""
    logging.info("Finding actual data range in Prometheus...")
    
    if metrics_sample is None:
        # Get all metrics and sample a larger set for better coverage
        all_metrics = prom.all_metrics()
        # Sample 20% of metrics or at least 50 metrics for comprehensive check
        sample_size = max(50, len(all_metrics) // 5)
        metrics_sample = all_metrics[:sample_size]
    
    earliest_timestamp = None
    latest_timestamp = None
    successful_checks = 0
    
    # Start from a reasonable lookback and work backwards
    end_time = datetime.now()
    
    for lookback_days in [1, 7, 30, 60, 90]:
        if lookback_days > max_lookback_days:
            break
            
        start_time = end_time - timedelta(days=lookback_days)
        logging.info(f"Checking for data in last {lookback_days} days...")
        
        found_data_in_period = False
        
        for metric in metrics_sample[:10]:  # Check first 10 metrics for each period
            try:
                result = prom.custom_query_range(
                    query=metric,
                    start_time=start_time,
                    end_time=end_time,
                    step="1h",  # Use 1h step for broad search
                )
                
                if result:
                    for series in result:
                        if series["values"]:
                            timestamps = [float(ts) for ts, _ in series["values"]]
                            if timestamps:
                                found_data_in_period = True
                                successful_checks += 1
                                
                                min_ts = min(timestamps)
                                max_ts = max(timestamps)
                                
                                if earliest_timestamp is None or min_ts < earliest_timestamp:
                                    earliest_timestamp = min_ts
                                if latest_timestamp is None or max_ts > latest_timestamp:
                                    latest_timestamp = max_ts
                                
                                break  # Found data for this metric, move to next
                            
            except Exception as e:
                logging.debug(f"Error checking {metric} for {lookback_days} days: {e}")
                continue
        
        if found_data_in_period:
            logging.info(f"Found data in {lookback_days}-day period")
        else:
            logging.info(f"No data found in {lookback_days}-day period")
            break  # No data this far back, stop looking further
    
    if earliest_timestamp and latest_timestamp:
        earliest_dt = datetime.fromtimestamp(earliest_timestamp)
        latest_dt = datetime.fromtimestamp(latest_timestamp)
        
        logging.info(f"Actual data range found:")
        logging.info(f"  Earliest: {earliest_dt}")
        logging.info(f"  Latest: {latest_dt}")
        logging.info(f"  Duration: {latest_dt - earliest_dt}")
        logging.info(f"  Successful metric checks: {successful_checks}")
        
        return earliest_dt, latest_dt
    else:
        logging.warning("No data found in any reasonable time range")
        return None, None


def calculate_batch_plan(start_time, end_time, requested_step):
    """Calculate how to split the export into batches that fit within Prometheus limits."""
    total_seconds = (end_time - start_time).total_seconds()
    step_seconds = parse_duration(requested_step)
    
    # Calculate how many data points this would generate
    estimated_points = total_seconds / step_seconds
    
    # Prometheus limit is 11,000 points, use 10,000 for safety
    MAX_POINTS = 10000
    
    logging.info(f"Total time range: {end_time - start_time}")
    logging.info(f"Estimated points: {estimated_points:.0f}")
    
    if estimated_points <= MAX_POINTS:
        # No batching needed
        return [{
            'start_time': start_time,
            'end_time': end_time,
            'step': requested_step,
            'estimated_points': estimated_points
        }]
    
    # Calculate batch duration in seconds
    batch_seconds = int((MAX_POINTS * step_seconds) * 0.9)  # 90% of limit for safety
    num_batches = int(total_seconds / batch_seconds) + 1
    
    batches = []
    current_start = start_time
    
    for i in range(num_batches):
        batch_end = min(current_start + timedelta(seconds=batch_seconds), end_time)
        batch_points = (batch_end - current_start).total_seconds() / step_seconds
        
        batches.append({
            'start_time': current_start,
            'end_time': batch_end,
            'step': requested_step,
            'estimated_points': batch_points
        })
        
        current_start = batch_end
        
        if current_start >= end_time:
            break
    
    logging.info(
        f"Splitting into {len(batches)} batches of ~{batch_points:.0f} points each."
    )
    
    return batches


def parse_duration(duration_str):
    """Parse Prometheus duration format to seconds."""
    import re
    
    # Match patterns like "15s", "1m", "2h", etc.
    match = re.match(r'^(\d+)([smhd])$', duration_str)
    if not match:
        raise ValueError(f"Invalid duration format: {duration_str}")
    
    value, unit = match.groups()
    value = int(value)
    
    if unit == 's':
        return value
    elif unit == 'm':
        return value * 60
    elif unit == 'h':
        return value * 3600
    elif unit == 'd':
        return value * 86400
    else:
        raise ValueError(f"Unknown time unit: {unit}")


def process_metric_batch(
    metric_name, prom, start_time, end_time, step, output_dir, output_format, batch_num=None
):
    """Fetch a single metric for a specific time batch and return it as a DataFrame."""
    try:
        batch_info = f" (batch {batch_num})" if batch_num is not None else ""
        logging.info(f"Fetching data for {metric_name}{batch_info}...")
        logging.debug(f"Time range: {start_time} to {end_time}")
        
        metric_data = prom.custom_query_range(
            query=metric_name,
            start_time=start_time,
            end_time=end_time,
            step=step,
        )
        if not metric_data:
            logging.warning(f"No data for metric {metric_name}{batch_info}")
            return None

        df_list = []
        total_points = 0
        for d in metric_data:
            metric_labels = d["metric"]
            df = pd.DataFrame(d["values"], columns=["timestamp", "value"])
            df["metric_name"] = metric_labels.pop("__name__", metric_name)
            for label, value in metric_labels.items():
                df[label] = value
            df_list.append(df)
            total_points += len(df)

        if not df_list:
            logging.warning(f"No data points found for {metric_name}{batch_info}")
            return None

        metric_df = pd.concat(df_list, ignore_index=True)
        metric_df["timestamp"] = pd.to_datetime(metric_df["timestamp"], unit="s")
        metric_df["value"] = pd.to_numeric(metric_df["value"], errors="coerce")
        metric_df = metric_df.dropna(subset=["value"])

        if len(metric_df) == 0:
            logging.warning(f"No valid data points for {metric_name}{batch_info}")
            return None

        # CRITICAL FIX: Sort by timestamp to ensure chronological order
        metric_df = metric_df.sort_values('timestamp').reset_index(drop=True)
        
        logging.info(f"Found {len(metric_df)} data points for {metric_name}{batch_info}")

        if output_format == "individual":
            safe_metric_name = re.sub(r'[^0-9a-zA-Z_]', '_', metric_name)
            if batch_num is not None:
                output_path = Path(output_dir) / f"{safe_metric_name}_batch_{batch_num:03d}.csv"
            else:
                output_path = Path(output_dir) / f"{safe_metric_name}.csv"
            logging.info(f"Saving {len(metric_df)} points to {output_path}...")
            metric_df.to_csv(output_path, index=False)
            return None  # Don't return df to save memory
        else:
            return metric_df
    except Exception as e:
        logging.error(f"Failed to export {metric_name}{batch_info}: {e}", exc_info=True)
        return None


def merge_batch_files(output_dir, metric_names):
    """Merge individual batch files for each metric into single files, sorted by timestamp."""
    logging.info("Merging batch files for each metric...")
    output_path = Path(output_dir)
    
    for metric_name in metric_names:
        safe_metric_name = re.sub(r'[^0-9a-zA-Z_]', '_', metric_name)
        
        # Find all batch files for this metric
        batch_files = list(output_path.glob(f"{safe_metric_name}_batch_*.csv"))
        
        if len(batch_files) <= 1:
            # No batching was done or only one batch, rename if needed
            if len(batch_files) == 1:
                final_path = output_path / f"{safe_metric_name}.csv"
                batch_files[0].rename(final_path)
                logging.info(f"Renamed single batch file to {final_path}")
            continue
        
        logging.info(f"Merging {len(batch_files)} batch files for {metric_name}...")
        
        # Read all batch files
        dfs = []
        for batch_file in sorted(batch_files):
            try:
                df = pd.read_csv(batch_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                dfs.append(df)
            except Exception as e:
                logging.error(f"Error reading {batch_file}: {e}")
                continue
        
        if not dfs:
            logging.warning(f"No valid batch files found for {metric_name}")
            continue
        
        # Concatenate and sort by timestamp
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
        
        # Save merged file
        final_path = output_path / f"{safe_metric_name}.csv"
        merged_df.to_csv(final_path, index=False)
        logging.info(f"Merged {len(merged_df)} points to {final_path}")
        
        # Clean up batch files
        for batch_file in batch_files:
            batch_file.unlink()
            logging.debug(f"Removed batch file {batch_file}")


def export_metrics(
    prometheus_url,
    days,
    minutes,
    seconds,
    step,
    output_format,
    output_dir,
    metrics_file,
    filter_regex,
    workers,
    timeout,
    retries,
    dry_run,
    everything_available,
):
    """Export metrics from Prometheus."""
    logging.info("Starting metrics export...")
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = SCRIPT_DIR / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

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

    # Determine time range
    if everything_available:
        logging.info("Finding all available data...")
        start_time, end_time = find_actual_data_range(prom)
        if start_time is None:
            logging.error("No data found. Exiting.")
            return
    else:
        end_time = datetime.now()
        if seconds is not None:
            start_time = end_time - timedelta(seconds=seconds)
            logging.info(f"Exporting last {seconds} seconds")
        elif minutes is not None:
            start_time = end_time - timedelta(minutes=minutes)
            logging.info(f"Exporting last {minutes} minutes")
        else:
            start_time = end_time - timedelta(days=days)
            logging.info(f"Exporting last {days} days")

    logging.info(f"Time range: {start_time} to {end_time}")
    logging.info(f"Using step size: {step}")

    if dry_run:
        logging.info("--- DRY RUN ---")
        logging.info(f"URL: {prometheus_url}")
        logging.info(f"Time range: {start_time} to {end_time}")
        logging.info(f"Step: {step}")
        logging.info(f"Output Format: {output_format}, Output Dir: {output_path}")
        logging.info(f"Metrics to export: {len(metric_names)}")
        if metrics_file:
            logging.info(f"Metrics File: {metrics_file}")
        if filter_regex:
            logging.info(f"Filter: {filter_regex}")
        
        # Calculate estimated data volume
        total_seconds = (end_time - start_time).total_seconds()
        step_seconds = parse_duration(step)
        estimated_points_per_metric = total_seconds / step_seconds
        estimated_total_points = estimated_points_per_metric * len(metric_names)
        logging.info(f"Estimated points per metric: {estimated_points_per_metric:.0f}")
        logging.info(f"Estimated total points: {estimated_total_points:.0f}")
        logging.info("--- END DRY RUN ---")
        return

    # Calculate batch plan
    batches = calculate_batch_plan(start_time, end_time, step)
    logging.info(f"Total batches to process: {len(batches)}")

    all_metrics_dfs = []
    has_multiple_batches = len(batches) > 1
    
    for batch_idx, batch in enumerate(batches):
        logging.info(f"Processing batch {batch_idx + 1}/{len(batches)}: "
                    f"{batch['start_time']} to {batch['end_time']} "
                    f"({batch['estimated_points']:.0f} points)")
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    process_metric_batch,
                    metric,
                    prom,
                    batch['start_time'],
                    batch['end_time'],
                    batch['step'],
                    output_path,
                    output_format,
                    batch_idx + 1 if has_multiple_batches else None,
                )
                for metric in metric_names
            ]

            for future in futures:
                result_df = future.result()
                if result_df is not None:
                    all_metrics_dfs.append(result_df)
        
        # Brief pause between batches to avoid overwhelming Prometheus
        if batch_idx < len(batches) - 1:
            logging.info("Pausing 2 seconds between batches...")
            time.sleep(2)

    # Handle post-processing based on output format
    if output_format == "individual" and has_multiple_batches:
        # Merge batch files for each metric and sort by timestamp
        merge_batch_files(output_path, metric_names)
    elif output_format == "unified" and all_metrics_dfs:
        logging.info(f"Concatenating {len(all_metrics_dfs)} DataFrames...")
        final_df = pd.concat(all_metrics_dfs, ignore_index=True)
        
        # CRITICAL FIX: Sort the unified dataset by timestamp to ensure chronological order
        logging.info("Sorting unified dataset by timestamp...")
        final_df = final_df.sort_values('timestamp').reset_index(drop=True)
        
        unified_output_path = output_path / "unified_metrics.parquet"
        logging.info(f"Saving unified dataset with {len(final_df)} points to {unified_output_path}...")
        final_df.to_parquet(unified_output_path, index=False)

    logging.info("Export complete.")


def main():
    """Main function to parse arguments and execute commands."""
    parser = argparse.ArgumentParser(
        description="Export metrics from a Prometheus instance."
    )
    
    # Connection options
    parser.add_argument('--test-connection', action='store_true',
                      help='Test connection to Prometheus and exit')
    parser.add_argument('--list-metrics', action='store_true',
                      help='List available metrics and exit')
    parser.add_argument('--url', default='http://localhost:9090',
                      help='Prometheus server URL')
    
    # Export options
    parser.add_argument('--days', type=int, default=None,
                      help='Number of days to export')
    parser.add_argument('--minutes', type=int, default=None,
                      help='Number of minutes to export (alternative to --days)')
    parser.add_argument('--seconds', type=int, default=None,
                      help='Number of seconds to export (alternative to --days/--minutes)')
    parser.add_argument('--everything', action='store_true',
                      help='Export all available data (overrides --days/--minutes/--seconds)')
    parser.add_argument('--step', default='1m',
                      help='Query resolution step width (e.g., 15s, 30s, 1m, 5m, 1h). Supports Prometheus duration format.')
    parser.add_argument('--output-format', choices=['unified', 'individual'],
                      default='individual',
                      help='Output format: unified parquet or individual CSVs')
    parser.add_argument('--output-dir', default='prometheus_data',
                      help='Output directory for exported data')
    parser.add_argument('--metrics', dest='metrics_file',
                      help='File containing list of metrics to export')
    parser.add_argument('--filter', dest='filter_regex',
                      help='Regex pattern to filter metrics')
    
    # Performance options
    parser.add_argument('--workers', type=int, default=4,
                      help='Number of parallel workers')
    parser.add_argument('--timeout', type=int, default=30,
                      help='Query timeout in seconds')
    parser.add_argument('--retries', type=int, default=3,
                      help='Number of retries for failed queries')
    
    # Other options
    parser.add_argument('--dry-run', action='store_true',
                      help='Show what would be done without doing it')
    
    args = parser.parse_args()
    
    # Validate time range arguments
    if not args.everything and sum(1 for x in [args.days, args.minutes, args.seconds] if x is not None) > 1:
        logging.error("Cannot specify more than one of --days, --minutes, or --seconds")
        sys.exit(1)
    
    if not args.everything and args.days is None and args.minutes is None and args.seconds is None:
        args.days = 7  # Default to 7 days
    
    try:
        # Handle work directory - set output relative to script location if not absolute
        if not Path(args.output_dir).is_absolute():
            args.output_dir = SCRIPT_DIR / args.output_dir
        
        # Handle commands
        if args.test_connection:
            success = test_connection(args.url)
            sys.exit(0 if success else 1)
        
        if args.list_metrics:
            list_metrics(args.url)
            sys.exit(0)
        
        export_metrics(
            prometheus_url=args.url,
            days=args.days,
            minutes=args.minutes,
            seconds=args.seconds,
            step=args.step,
            output_format=args.output_format,
            output_dir=args.output_dir,
            metrics_file=args.metrics_file,
            filter_regex=args.filter_regex,
            workers=args.workers,
            timeout=args.timeout,
            retries=args.retries,
            dry_run=args.dry_run,
            everything_available=args.everything,
        )
        sys.exit(0)
        
    except Exception as e:
        logging.error(f"Export failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()