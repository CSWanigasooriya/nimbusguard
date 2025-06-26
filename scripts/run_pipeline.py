#!/usr/bin/env python3
"""
Pipeline Orchestrator
Coordinates the execution of the infrastructure monitoring pipeline scripts in sequence.
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
import subprocess
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

class PipelineOrchestrator:
    """Orchestrates the execution of pipeline scripts in sequence."""
    
    def __init__(self, config: dict):
        self.config = config
        # Get the scripts directory path (where this file is located)
        self.scripts_dir = Path(__file__).parent.resolve()
        self.work_dir = self.scripts_dir
        
        # Define pipeline stages and their corresponding scripts
        self.pipeline_stages = [
            {
                'name': 'Data Export',
                'script': 'export_prometheus.py',
                'skip_flag': 'skip_extraction',
                'description': 'Export metrics from Prometheus'
            },
            {
                'name': 'Data Preparation',
                'script': 'prepare_dataset.py',
                'skip_flag': 'skip_preparation',
                'description': 'Prepare and consolidate metrics data'
            },
            {
                'name': 'Feature Engineering',
                'script': 'feature_engineering.py',
                'skip_flag': 'skip_feature_engineering',
                'description': 'Engineer features for ML'
            },
            {
                'name': 'Data Inspection',
                'script': 'inspect_dataset.py',
                'skip_flag': 'skip_inspection',
                'description': 'Analyze final dataset'
            }
        ]
    
    def _run_script(self, script_name: str, args: list = None) -> bool:
        """Run a pipeline script with arguments."""
        script_path = self.scripts_dir / script_name
        if not script_path.exists():
            logging.error(f"Script not found: {script_path}")
            return False
        
        cmd = [sys.executable, str(script_path)]
        
        # No need to pass work-dir anymore as scripts handle their own paths
        
        if args:
            cmd.extend(args)
        
        try:
            logging.info(f"Running command: {' '.join(map(str, cmd))}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Script failed with exit code {e.returncode}")
            print(e.stdout)
            print(e.stderr, file=sys.stderr)
            return False
        except Exception as e:
            logging.error(f"Failed to run script: {e}")
            return False
    
    def run_pipeline(self) -> bool:
        """Run the complete pipeline by executing scripts in sequence."""
        start_time = datetime.now()
        
        print("🏗️  Infrastructure Monitoring Pipeline")
        print("=" * 60)
        print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Show configuration
        print("📋 Configuration:")
        print(f"   Prometheus URL: {self.config['prometheus_url']}")
        print(f"   Days to extract: {self.config['days']}")
        print(f"   Time resolution: {self.config['step']}")
        print(f"   Quick mode: {self.config.get('quick_mode', False)}")
        print()
        
        # Create necessary directories
        prometheus_data_dir = self.scripts_dir / 'prometheus_data'
        processed_data_dir = self.scripts_dir / 'processed_data'
        prometheus_data_dir.mkdir(parents=True, exist_ok=True)
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean existing data if requested
        if not self.config.get('keep_existing', False):
            if prometheus_data_dir.exists() and not self.config.get('skip_extraction', False):
                logging.info("Cleaning existing Prometheus data...")
                shutil.rmtree(prometheus_data_dir)
                prometheus_data_dir.mkdir(parents=True, exist_ok=True)
            
            if processed_data_dir.exists() and not self.config.get('skip_preparation', False):
                logging.info("Cleaning existing processed data...")
                for file in processed_data_dir.glob('dataset*'):
                    file.unlink()
                for file in processed_data_dir.glob('engineered_features*'):
                    file.unlink()
        
        # Execute pipeline stages
        for i, stage in enumerate(self.pipeline_stages, 1):
            print(f"🔄 STAGE {i}: {stage['name']}")
            print(f"   {stage['description']}")
            print("-" * 60)
            
            # Skip if requested
            if stage.get('skip_flag') and self.config.get(stage['skip_flag'], False):
                print(f"⏭️  Skipping {stage['name']} (configured to skip)")
                print()
                continue
            
            # Check dependencies
            if stage['script'] == 'prepare_dataset.py':
                # Check if we have prometheus data
                if not (prometheus_data_dir / 'prometheus_data').exists() and len(list(prometheus_data_dir.glob('*.csv'))) == 0:
                    print(f"⚠️  No Prometheus data found for {stage['name']}")
                    if not self.config.get('skip_extraction', False):
                        print("   Make sure data export completed successfully")
                        return False
            
            elif stage['script'] == 'feature_engineering.py':
                # Check if we have dataset
                dataset_file = processed_data_dir / 'dataset.parquet'
                if not dataset_file.exists():
                    print(f"⚠️  No dataset found for {stage['name']} at {dataset_file}")
                    if not self.config.get('skip_preparation', False):
                        print("   Make sure data preparation completed successfully")
                        return False
            
            elif stage['script'] == 'inspect_dataset.py':
                # Check if we have engineered features
                features_file = processed_data_dir / 'engineered_features.parquet'
                if not features_file.exists():
                    print(f"⚠️  No engineered features found for {stage['name']} at {features_file}")
                    if not self.config.get('skip_feature_engineering', False):
                        print("   Make sure feature engineering completed successfully")
                        return False
            
            # Build script arguments
            args = []
            
            # Add stage-specific arguments
            if stage['script'] == 'export_prometheus.py':
                args.extend([
                    '--url', self.config['prometheus_url'],
                    '--days', str(self.config['days']),
                    '--step', self.config['step']
                ])
                if self.config.get('metric_filter'):
                    args.extend(['--filter', self.config['metric_filter']])
            
            elif stage['script'] == 'prepare_dataset.py':
                if self.config.get('batch_size'):
                    args.extend(['--batch-size', str(self.config['batch_size'])])
                if self.config.get('output_format'):
                    args.extend(['--format', self.config['output_format']])
                if self.config.get('safe_mode'):
                    args.append('--safe-mode')
            
            elif stage['script'] == 'feature_engineering.py':
                if self.config.get('quick_mode'):
                    args.extend([
                        '--skip-dimensionality-reduction',
                        '--skip-correlation'
                    ])
                # Add feature engineering flags
                for flag in ['health_scores', 'utilization', 'performance', 'anomaly', 
                           'correlation', 'time_series', 'dimensionality_reduction']:
                    if self.config.get(f'skip_{flag}', False):
                        args.append(f'--skip-{flag}')
                
                # Add output format if specified
                if self.config.get('output_format'):
                    args.extend(['--output-format', self.config['output_format']])
            
            elif stage['script'] == 'inspect_dataset.py':
                # Use the engineered features as input
                processed_dir = self.scripts_dir / 'processed_data'
                args.append(str(processed_dir / 'engineered_features.parquet'))
            
            # Run the script
            if not self._run_script(stage['script'], args):
                print(f"❌ {stage['name']} failed")
                return False
            
            print(f"✅ {stage['name']} completed")
            print()
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 45)
        print(f"Duration: {duration}")
        print()
        
        print("📁 Output files:")
        print(f"   • dataset.parquet - Consolidated dataset")
        print(f"   • engineered_features.parquet - ML-ready dataset")
        print(f"   • engineered_features_metadata.txt - Feature descriptions")
        print(f"   • final_analysis.txt - Dataset analysis")
        print()
        
        print("🎯 Your data is ready for:")
        print("   • Machine Learning models")
        print("   • Anomaly detection")
        print("   • Performance monitoring")
        print("   • Capacity planning")
        print()
        
        return True


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Infrastructure Monitoring Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                         # Full pipeline with defaults
  python run_pipeline.py --quick                 # Quick mode (1 day, skip advanced)
  python run_pipeline.py --skip-extraction       # Use existing data
  python run_pipeline.py --work-dir ./scripts    # Specify working directory
  python run_pipeline.py --days 7 --cleanup     # 7 days with cleanup
        """
    )
    
    # Prometheus settings
    parser.add_argument('--url', default='http://localhost:9090',
                      help='Prometheus server URL')
    parser.add_argument('--days', type=int, default=7,
                      help='Number of days to extract')
    parser.add_argument('--step', default='1m',
                      help='Query resolution step width')
    
    # Pipeline control
    parser.add_argument('--skip-extraction', action='store_true',
                      help='Skip extraction, use existing data')
    parser.add_argument('--skip-preparation', action='store_true',
                      help='Skip data preparation, use existing dataset')
    parser.add_argument('--skip-feature-engineering', action='store_true',
                      help='Skip feature engineering')
    parser.add_argument('--skip-inspection', action='store_true',
                      help='Skip data inspection')
    parser.add_argument('--keep-existing', action='store_true',
                      help='Keep existing data during extraction')
    parser.add_argument('--quick', action='store_true',
                      help='Quick mode: 1 day, skip advanced features')
    parser.add_argument('--cleanup', action='store_true',
                      help='Clean up intermediate files')
    parser.add_argument('--batch-size', type=int, default=50,
                      help='Batch size for data processing')
    parser.add_argument('--safe-mode', action='store_true',
                      help='Enable safe mode with extra validation')
    
    # Feature engineering control
    parser.add_argument('--skip-health-scores', action='store_true',
                      help='Skip health score features')
    parser.add_argument('--skip-utilization', action='store_true',
                      help='Skip utilization features')
    parser.add_argument('--skip-performance', action='store_true',
                      help='Skip performance features')
    parser.add_argument('--skip-anomaly', action='store_true',
                      help='Skip anomaly detection features')
    parser.add_argument('--skip-time-series', action='store_true',
                      help='Skip time series features')
    parser.add_argument('--skip-correlation', action='store_true',
                      help='Skip correlation features')
    parser.add_argument('--skip-dimensionality-reduction', action='store_true',
                      help='Skip dimensionality reduction features')
    parser.add_argument('--output-format', choices=['parquet', 'csv'],
                      help='Output format for engineered features')
    
    # Filters
    parser.add_argument('--metric-filter',
                      help='Regex filter for metric names')
    
    args = parser.parse_args()
    
    # Handle quick mode
    if args.quick:
        args.days = 1
        args.skip_dimensionality_reduction = True
        args.skip_correlation = True
        args.cleanup = True
    
    # Build configuration
    config = {
        'prometheus_url': args.url,
        'days': args.days,
        'step': args.step,
        'skip_extraction': args.skip_extraction,
        'skip_preparation': args.skip_preparation,
        'skip_feature_engineering': args.skip_feature_engineering,
        'skip_inspection': args.skip_inspection,
        'keep_existing': args.keep_existing,
        'quick_mode': args.quick,
        'cleanup': args.cleanup,
        'batch_size': args.batch_size,
        'safe_mode': args.safe_mode,
        'skip_health_scores': args.skip_health_scores,
        'skip_utilization': args.skip_utilization,
        'skip_performance': args.skip_performance,
        'skip_anomaly': args.skip_anomaly,
        'skip_time_series': args.skip_time_series,
        'skip_correlation': args.skip_correlation,
        'skip_dimensionality_reduction': args.skip_dimensionality_reduction,
        'output_format': args.output_format,
        'metric_filter': args.metric_filter
    }
    
    # Create and run pipeline
    orchestrator = PipelineOrchestrator(config)
    
    try:
        success = orchestrator.run_pipeline()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
