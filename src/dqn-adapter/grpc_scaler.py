"""
DQN External Scaler for KEDA - gRPC Service Implementation

This module implements the KEDA External Scaler gRPC interface for truly dynamic DQN-driven scaling.
Unlike the metrics-api approach, this provides direct replica control without artificial baselines.
"""

import grpc
import logging
import time
import asyncio
import threading
from concurrent import futures
from typing import Dict, Any, Optional
import os

import externalscaler_pb2
import externalscaler_pb2_grpc

# No hysteresis - instantaneous DQN scaling

class DQNExternalScaler(externalscaler_pb2_grpc.ExternalScalerServicer):
    """
    KEDA External Scaler for DQN-based autoscaling.
    
    This gRPC service bridges KEDA's scaling engine with our DQN decision system,
    providing intelligent, machine learning-driven scaling decisions.
    """
    
    def __init__(self, dqn_desired_gauge, current_replicas_gauge, logger):
        """
        Initialize the DQN External Scaler.
        
        Args:
            dqn_desired_gauge: Prometheus gauge tracking DQN's desired replica count
            current_replicas_gauge: Prometheus gauge tracking current replica count
            logger: Logger instance for debugging and monitoring
        """
        self.dqn_desired_gauge = dqn_desired_gauge
        self.current_replicas_gauge = current_replicas_gauge
        self.logger = logger
        
        # Streaming state management
        self._stream_active = False
        self._last_dqn_decision = None
        self._stream_clients = []
        
        self.logger.info("DQN_GRPC_SCALER: initialized for_truly_dynamic_scaling")
    
    def IsActive(self, request, context):
        """
        Determines if the scaler should be active.
        
        Called by KEDA every pollingInterval to check if scaling should occur.
        For DQN, we're always active when the system is running.
        
        Args:
            request: ScaledObjectRef containing scaler metadata
            context: gRPC context
            
        Returns:
            IsActiveResponse: Always True for DQN (we're always monitoring)
        """
        try:
            # DQN is always active when running - continuous intelligence
            self.logger.info(f"DQN_GRPC: IsActive_called name={request.name} namespace={request.namespace} always_active=True")
            return externalscaler_pb2.IsActiveResponse(result=True)
            
        except Exception as e:
            self.logger.error(f"DQN_GRPC: IsActive_error error={e}")
            import traceback
            self.logger.error(f"DQN_GRPC: IsActive_traceback {traceback.format_exc()}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return externalscaler_pb2.IsActiveResponse(result=False)
    
    def GetMetricSpec(self, request, context):
        """
        Returns metric specification for HPA.
        
        The target should be 1.0 since we return the absolute desired replica count.
        HPA will calculate: desired = current * (metric_value / target_value)
        
        Args:
            request: ScaledObjectRef containing scaler metadata
            context: gRPC context
            
        Returns:
            GetMetricSpecResponse: Metric specification for HPA
        """
        try:
            self.logger.info(f"DQN_GRPC: GetMetricSpec_called name={request.name} namespace={request.namespace}")
            
            metric_spec = externalscaler_pb2.MetricSpec(
                metricName="dqn-replica-need",
                targetSize=1000  # 1.0 in milli-units - we return absolute replica counts
            )
            
            response = externalscaler_pb2.GetMetricSpecResponse(
                metricSpecs=[metric_spec]
            )
            
            self.logger.info(f"DQN_GRPC: GetMetricSpec_success target=1.0 metric_name=dqn-replica-need")
            return response
            
        except Exception as e:
            self.logger.error(f"DQN_GRPC: GetMetricSpec_error error={e}")
            import traceback
            self.logger.error(f"DQN_GRPC: GetMetricSpec_traceback {traceback.format_exc()}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            
            # Return safe fallback
            fallback_spec = externalscaler_pb2.MetricSpec(
                metricName="dqn-replica-need",
                targetSize=1000  # 1.0
            )
            return externalscaler_pb2.GetMetricSpecResponse(metricSpecs=[fallback_spec])
    
    def GetMetrics(self, request, context):
        """
        Returns current metric values - DQN desired replica count as absolute value.
        
        FIXED: Return absolute DQN desired replica count, not ratios.
        HPA will calculate: desired = current * (dqn_desired / 1.0) = dqn_desired
        
        Args:
            request: GetMetricsRequest containing metric name and scaler metadata
            context: gRPC context
            
        Returns:
            GetMetricsResponse: Current replica need based on DQN decision (absolute value)
        """
        try:
            # Log the incoming request for debugging
            self.logger.info(f"DQN_GRPC: GetMetrics_called metric={request.metricName} "
                           f"name={request.scaledObjectRef.name} namespace={request.scaledObjectRef.namespace}")
            
            # Get DQN's desired replica count directly
            dqn_desired_replicas = int(self.dqn_desired_gauge._value._value)
            current_replicas = int(self.current_replicas_gauge._value._value)
            
            # Return the absolute DQN desired replica count
            # HPA will calculate: desired = current * (dqn_desired / 1.0) = dqn_desired
            metric_value = externalscaler_pb2.MetricValue(
                metricName="dqn-replica-need",
                metricValue=dqn_desired_replicas * 1000,  # Convert to milli-units
                metricValueFloat=float(dqn_desired_replicas)
            )
            
            response = externalscaler_pb2.GetMetricsResponse(
                metricValues=[metric_value]
            )
            
            self.logger.info(f"DQN_GRPC: GetMetrics_success dqn_wants={dqn_desired_replicas} "
                           f"current={current_replicas} "
                           f"returning_absolute_value={dqn_desired_replicas} "
                           f"hpa_will_calculate=current*({dqn_desired_replicas}/1.0)={dqn_desired_replicas}")
            return response
            
        except Exception as e:
            self.logger.error(f"DQN_GRPC: GetMetrics_error error={e}")
            import traceback
            self.logger.error(f"DQN_GRPC: GetMetrics_traceback {traceback.format_exc()}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            
            # Return safe fallback (current replica count to maintain status quo)
            current_replicas = int(self.current_replicas_gauge._value._value) if self.current_replicas_gauge else 2
            fallback_value = externalscaler_pb2.MetricValue(
                metricName="dqn-replica-need",
                metricValue=current_replicas * 1000,  # Maintain current state
                metricValueFloat=float(current_replicas)
            )
            return externalscaler_pb2.GetMetricsResponse(metricValues=[fallback_value])
    
    def StreamIsActive(self, request, response_iterator):
        """
        REAL-TIME Push-based scaling - sends immediate scaling signals when DQN decision changes.
        
        This enables truly instantaneous scaling reactions with sub-second detection of DQN changes.
        The stream remains open and pushes IsActiveResponse immediately when conditions change.
        
        Args:
            request: ScaledObjectRef containing scaler metadata
            response_iterator: gRPC stream for sending responses
            
        Yields:
            IsActiveResponse: Immediate scaling signals
        """
        try:
            self.logger.info(f"DQN_GRPC: StreamIsActive_started push_based_scaling=True "
                           f"name={request.name} namespace={request.namespace}")
            self._stream_active = True
            
            last_dqn_decision = None
            last_current_replicas = None
            last_signal_time = time.time()
            
            while self._stream_active:
                try:
                    # Get current DQN decision and replica state
                    current_dqn_decision = int(self.dqn_desired_gauge._value._value)
                    current_replicas = int(self.current_replicas_gauge._value._value)
                    current_time = time.time()
                    
                    # Send signal when DQN decision changes OR replica count changes OR heartbeat
                    decision_changed = current_dqn_decision != last_dqn_decision
                    replicas_changed = current_replicas != last_current_replicas
                    time_for_heartbeat = (current_time - last_signal_time) > 10  # Faster heartbeat
                    
                    if decision_changed or replicas_changed or time_for_heartbeat:
                        # Send active signal to trigger immediate scaling evaluation
                        yield externalscaler_pb2.IsActiveResponse(result=True)
                        
                        if decision_changed:
                            self.logger.info(f"DQN_GRPC: StreamIsActive_DQN_CHANGE "
                                           f"from={last_dqn_decision} to={current_dqn_decision} "
                                           f"current_replicas={current_replicas} PUSH_SIGNAL_SENT")
                        elif replicas_changed:
                            self.logger.info(f"DQN_GRPC: StreamIsActive_REPLICA_CHANGE "
                                           f"from={last_current_replicas} to={current_replicas} "
                                           f"dqn_wants={current_dqn_decision} PUSH_SIGNAL_SENT")
                        elif time_for_heartbeat:
                            self.logger.debug(f"DQN_GRPC: StreamIsActive_HEARTBEAT "
                                            f"dqn={current_dqn_decision} replicas={current_replicas}")
                        
                        last_dqn_decision = current_dqn_decision
                        last_current_replicas = current_replicas
                        last_signal_time = current_time
                    
                    # Ultra-fast checking for real-time DQN decision detection
                    time.sleep(0.5)  # Check every 500ms for sub-second responsiveness
                    
                except Exception as e:
                    self.logger.error(f"DQN_GRPC: StreamIsActive_loop_error error={e}")
                    time.sleep(1)  # Shorter sleep on error
            
            self.logger.info("DQN_GRPC: StreamIsActive_ended")
            
        except Exception as e:
            self.logger.error(f"DQN_GRPC: StreamIsActive_error error={e}")
            import traceback
            self.logger.error(f"DQN_GRPC: StreamIsActive_traceback {traceback.format_exc()}")
            return
    
    def stop_streaming(self):
        """Stop all streaming connections gracefully."""
        self._stream_active = False
        self.logger.info("DQN_GRPC: streaming_stopped")


def start_grpc_server(dqn_desired_gauge, current_replicas_gauge, logger, port=9091):
    """
    Start the gRPC server for KEDA External Scaler.
    
    Args:
        dqn_desired_gauge: Prometheus gauge for DQN decisions
        current_replicas_gauge: Prometheus gauge for current replicas
        logger: Logger instance
        port: Port to bind the gRPC server (default: 9090)
        
    Returns:
        grpc.Server: The started gRPC server instance
    """
    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Create and register the DQN scaler service
    dqn_scaler_service = DQNExternalScaler(
        dqn_desired_gauge=dqn_desired_gauge,
        current_replicas_gauge=current_replicas_gauge,
        logger=logger
    )
    
    externalscaler_pb2_grpc.add_ExternalScalerServicer_to_server(
        dqn_scaler_service, server
    )
    
    # Bind to all interfaces on the specified port
    server.add_insecure_port(f'[::]:{port}')
    
    # Start the server
    server.start()
    
    logger.info(f"DQN_GRPC_SERVER: started port={port} truly_dynamic_scaling=True")
    
    return server


def start_grpc_server_async(dqn_desired_gauge, current_replicas_gauge, logger, port=9091):
    """
    Start the gRPC server in a separate thread to avoid blocking.
    
    Args:
        dqn_desired_gauge: Prometheus gauge for DQN decisions
        current_replicas_gauge: Prometheus gauge for current replicas
        logger: Logger instance
        port: Port to bind the gRPC server (default: 9090)
        
    Returns:
        tuple: (grpc.Server, threading.Thread) - server instance and thread
    """
    def _run_server():
        server = start_grpc_server(dqn_desired_gauge, current_replicas_gauge, logger, port)
        try:
            # Keep server running
            while True:
                time.sleep(3600)  # Sleep for 1 hour
        except KeyboardInterrupt:
            logger.info("DQN_GRPC_SERVER: shutdown_requested")
            server.stop(0)
    
    # Start server in daemon thread
    grpc_thread = threading.Thread(target=_run_server, daemon=True)
    grpc_thread.start()
    
    logger.info(f"DQN_GRPC_SERVER: started_async port={port} thread={grpc_thread.name}")
    
    return grpc_thread 