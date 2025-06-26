#!/usr/bin/env python3
"""
Kubeflow Pipeline Definition for NimbusGuard
Orchestrates the data processing, model training, and serving pipeline for predictive scaling.
"""

import kfp
from kfp import dsl
from kfp.components import create_component_from_func, InputPath, OutputPath
import os
from typing import NamedTuple
from collections import namedtuple

# Define output types
MetricsOutput = namedtuple('MetricsOutput', ['metrics_path'])
DatasetOutput = namedtuple('DatasetOutput', ['dataset_path'])
FeaturesOutput = namedtuple('FeaturesOutput', ['features_path'])
ModelOutput = namedtuple('ModelOutput', ['model_path', 'metrics_path'])
ServiceOutput = namedtuple('ServiceOutput', ['service_url'])

# Define component for data export
@create_component_from_func(
    packages_to_install=['prometheus-api-client>=0.5.3', 'pandas>=1.5.0']
)
def export_prometheus_data(
    prometheus_url: str,
    days: int = 7,
    step: str = "1m",
    output_dir: str = "/mnt/data/prometheus_data"
) -> MetricsOutput:
    """Export metrics from Prometheus."""
    from pathlib import Path
    import logging
    from prometheus_api_client import PrometheusConnect
    import pandas as pd
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Connect to Prometheus
    prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)
    if not prom.check_prometheus_connection():
        raise RuntimeError("Could not connect to Prometheus")
    
    # Get all metrics
    metric_names = prom.all_metrics()
    logging.info(f"Found {len(metric_names)} metrics")
    
    # Export each metric
    for metric in metric_names:
        try:
            metric_data = prom.custom_query_range(
                query=metric,
                start_time=pd.Timestamp.now() - pd.Timedelta(days=days),
                end_time=pd.Timestamp.now(),
                step=step
            )
            
            if metric_data:
                df_list = []
                for d in metric_data:
                    metric_labels = d['metric']
                    df = pd.DataFrame(d['values'], columns=['timestamp', 'value'])
                    df['metric_name'] = metric_labels.pop('__name__', metric)
                    for label, value in metric_labels.items():
                        df[label] = value
                    df_list.append(df)
                
                if df_list:
                    metric_df = pd.concat(df_list, ignore_index=True)
                    safe_metric_name = metric.replace(':', '_').replace('/', '_')
                    output_file = output_path / f"{safe_metric_name}.csv"
                    metric_df.to_csv(output_file, index=False)
        except Exception as e:
            logging.error(f"Error exporting {metric}: {e}")
            continue
    
    return MetricsOutput(str(output_path))

# Define component for data preparation
@create_component_from_func(
    packages_to_install=['pandas>=1.5.0', 'pyarrow>=10.0.0']
)
def prepare_dataset(
    input_dir: str,
    output_path: str = "/mnt/data/dataset.parquet",
    batch_size: int = 50
) -> DatasetOutput:
    """Prepare and consolidate the dataset."""
    from pathlib import Path
    import pandas as pd
    import logging
    from prepare_dataset import CSVConsolidator
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize consolidator
    consolidator = CSVConsolidator(input_dir, output_path)
    
    # Run consolidation
    df = consolidator.consolidate(
        batch_size=batch_size,
        format='parquet',
        wide_format=True
    )
    
    return DatasetOutput(output_path)

# Define component for feature engineering
@create_component_from_func(
    packages_to_install=['pandas>=1.5.0', 'numpy>=1.20.0', 'scikit-learn>=1.0.0']
)
def engineer_features(
    input_path: str,
    output_path: str = "/mnt/data/engineered_features.parquet"
) -> FeaturesOutput:
    """Engineer features for the model."""
    from pathlib import Path
    import logging
    from feature_engineering import InfrastructureFeatureEngineer
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize feature engineer
    engineer = InfrastructureFeatureEngineer(input_path, output_path)
    
    # Run feature engineering
    df = engineer.engineer_features(
        include_health_scores=True,
        include_utilization=True,
        include_performance=True,
        include_anomaly=True,
        include_correlation=True,
        include_time_series=True,
        include_dimensionality_reduction=True
    )
    
    # Save features
    engineer.save_engineered_features(df, format='parquet')
    
    return FeaturesOutput(output_path)

# Define component for DQN model training
@create_component_from_func(
    packages_to_install=['torch>=2.1.0', 'pandas>=1.5.0', 'numpy>=1.20.0']
)
def train_dqn_model(
    features_path: str,
    model_path: str = "/mnt/models/dqn_model",
    epochs: int = 100,
    batch_size: int = 32
) -> ModelOutput:
    """Train the DQN model for predictive scaling."""
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from pathlib import Path
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Load features
    df = pd.read_parquet(features_path)
    
    # TODO: Implement DQN model training
    # This is a placeholder for the actual DQN implementation
    # You'll need to:
    # 1. Define the state space from your features
    # 2. Define the action space (scaling decisions)
    # 3. Implement the DQN architecture
    # 4. Train the model with your reward function
    
    # Save training metrics
    metrics_path = Path(model_path).parent / "training_metrics.csv"
    # TODO: Save actual training metrics
    
    return ModelOutput(model_path, str(metrics_path))

# Define component for model serving
@create_component_from_func(
    packages_to_install=['kubernetes>=28.1.0', 'kserve>=0.11.0']
)
def deploy_model(
    model_path: str,
    model_name: str = "nimbusguard",
    namespace: str = "kubeflow",
    replicas: int = 1
) -> ServiceOutput:
    """Deploy the model using KServe."""
    from kubernetes import client, config
    from kserve import KServeClient
    from kserve import constants
    from kserve import V1beta1InferenceService
    from kserve import V1beta1InferenceServiceSpec
    from kserve import V1beta1PredictorSpec
    from kserve import V1beta1TorchServeSpec
    import logging
    from pathlib import Path
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize KServe client
    config.load_incluster_config()
    kserve_client = KServeClient()
    
    # Create InferenceService spec
    isvc = V1beta1InferenceService(
        api_version=constants.KSERVE_V1BETA1,
        kind=constants.KSERVE_KIND,
        metadata=client.V1ObjectMeta(
            name=model_name,
            namespace=namespace
        ),
        spec=V1beta1InferenceServiceSpec(
            predictor=V1beta1PredictorSpec(
                min_replicas=1,
                max_replicas=replicas,
                pytorch=V1beta1TorchServeSpec(
                    storage_uri=f"pvc://{Path(model_path).parent.name}/{Path(model_path).name}",
                    resources=client.V1ResourceRequirements(
                        requests={'cpu': '100m', 'memory': '1Gi'},
                        limits={'cpu': '1', 'memory': '2Gi'}
                    )
                )
            )
        )
    )
    
    # Deploy the model
    kserve_client.create(isvc)
    
    # Wait for the service to be ready
    kserve_client.wait_isvc_ready(
        model_name, 
        namespace=namespace,
        timeout_seconds=300
    )
    
    # Get the service URL
    service_url = kserve_client.get_service_url(model_name, namespace=namespace)
    logging.info(f"Model deployed successfully at: {service_url}")
    
    return ServiceOutput(service_url)

# Define the full pipeline
@dsl.pipeline(
    name='NimbusGuard Training and Serving Pipeline',
    description='End-to-end pipeline for training and serving the NimbusGuard predictive scaling model'
)
def nimbusguard_pipeline(
    prometheus_url: str = "http://localhost:9090",
    days: int = 7,
    step: str = "1m",
    epochs: int = 100,
    batch_size: int = 32,
    model_name: str = "nimbusguard",
    serving_replicas: int = 1
):
    """Define the full training and serving pipeline."""
    
    # Create a volume for data persistence
    vop = dsl.VolumeOp(
        name="create-pvc",
        resource_name="nimbusguard-data-pvc",
        size="10Gi",
        modes=dsl.VOLUME_MODE_RWO
    )
    
    # Export data
    export_task = export_prometheus_data(
        prometheus_url=prometheus_url,
        days=days,
        step=step
    ).add_pvolumes({"/mnt/data": vop.volume})
    
    # Prepare dataset
    prepare_task = prepare_dataset(
        input_dir=export_task.outputs['metrics_path']
    ).add_pvolumes({"/mnt/data": vop.volume}).after(export_task)
    
    # Engineer features
    engineer_task = engineer_features(
        input_path=prepare_task.outputs['dataset_path']
    ).add_pvolumes({"/mnt/data": vop.volume}).after(prepare_task)
    
    # Train model
    train_task = train_dqn_model(
        features_path=engineer_task.outputs['features_path'],
        epochs=epochs,
        batch_size=batch_size
    ).add_pvolumes({
        "/mnt/data": vop.volume,
        "/mnt/models": vop.volume
    }).after(engineer_task)
    
    # Deploy model
    deploy_task = deploy_model(
        model_path=train_task.outputs['model_path'],
        model_name=model_name,
        replicas=serving_replicas
    ).add_pvolumes({
        "/mnt/models": vop.volume
    }).after(train_task)

if __name__ == "__main__":
    # Compile the pipeline
    kfp.compiler.Compiler().compile(
        pipeline_func=nimbusguard_pipeline,
        package_path='nimbusguard_pipeline.yaml'
    ) 