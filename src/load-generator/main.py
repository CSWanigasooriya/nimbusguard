#!/usr/bin/env python3
"""
Load Generator Service for NimbusGuard

Generates configurable HTTP load and Kafka events to test scaling scenarios.
"""

import asyncio
import time
from datetime import datetime

import aiohttp
import psutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI(title="NimbusGuard Load Generator", version="1.0.0")


class LoadGenerationRequest(BaseModel):
    pattern: str  # "constant", "spike", "gradual", "burst"
    duration: int  # seconds
    target: str  # "http" or "kafka"
    intensity: int = 50  # 1-100 scale
    target_url: str = "http://consumer-workload:8080"


class LoadGenerationStatus(BaseModel):
    active: bool
    pattern: str
    elapsed: int
    remaining: int
    requests_sent: int


# Global state
load_generation_state = {
    "active": False,
    "task": None,
    "stats": {}
}


@app.get("/")
async def root():
    """Root endpoint with load generator dashboard"""
    import psutil
    import os
    
    # Get system information
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    cpu_percent = process.cpu_percent()
    
    # Get load generation status
    stats = load_generation_state.get("stats", {})
    is_active = load_generation_state["active"]
    current_pattern = stats.get("pattern", "none")
    requests_sent = stats.get("requests_sent", 0)
    
    # Calculate rates if active
    current_rate = 0
    elapsed_time = 0
    remaining_time = 0
    
    if is_active and stats:
        elapsed_time = int(time.time() - stats.get("start_time", time.time()))
        remaining_time = max(0, stats.get("duration", 0) - elapsed_time)
        current_rate = requests_sent / max(elapsed_time, 1)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>NimbusGuard Load Generator</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #ff7b7b 0%, #667eea 100%);
                color: white;
                min-height: 100vh;
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                background: rgba(255,255,255,0.1); 
                padding: 30px; 
                border-radius: 15px; 
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }}
            h1 {{ 
                color: #fff; 
                text-align: center; 
                margin-bottom: 30px; 
                font-size: 2.5em;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}
            .status-grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                gap: 20px; 
                margin-bottom: 30px; 
            }}
            .status-card {{ 
                background: rgba(255,255,255,0.15); 
                padding: 20px; 
                border-radius: 10px; 
                border: 1px solid rgba(255,255,255,0.2);
            }}
            .status-card h3 {{ 
                margin-top: 0; 
                color: #fff; 
                border-bottom: 2px solid rgba(255,255,255,0.3);
                padding-bottom: 10px;
            }}
            .metric {{ 
                display: flex; 
                justify-content: space-between; 
                margin: 10px 0; 
                padding: 5px 0;
            }}
            .metric-value {{ 
                font-weight: bold; 
                color: #4CAF50; 
            }}
            .metric-value.active {{ 
                color: #FF9800; 
            }}
            .endpoints {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                gap: 15px; 
                margin-top: 30px; 
            }}
            .endpoint-link {{ 
                display: block; 
                padding: 15px; 
                background: rgba(255,255,255,0.2); 
                color: white; 
                text-decoration: none; 
                border-radius: 8px; 
                text-align: center; 
                transition: all 0.3s ease;
                border: 1px solid rgba(255,255,255,0.3);
            }}
            .endpoint-link:hover {{ 
                background: rgba(255,255,255,0.3); 
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }}
            .status-indicator {{ 
                display: inline-block; 
                width: 12px; 
                height: 12px; 
                border-radius: 50%; 
                margin-right: 8px; 
            }}
            .status-active {{ background-color: #FF9800; }}
            .status-idle {{ background-color: #4CAF50; }}
            .status-running {{ background-color: #4CAF50; }}
            .refresh-btn {{
                background: rgba(255,255,255,0.2);
                color: white;
                border: 1px solid rgba(255,255,255,0.3);
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                margin: 10px 0;
            }}
            .refresh-btn:hover {{
                background: rgba(255,255,255,0.3);
            }}
            .load-controls {{
                background: rgba(255,255,255,0.15);
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                border: 1px solid rgba(255,255,255,0.2);
            }}
            .control-button {{
                background: rgba(255,255,255,0.2);
                color: white;
                border: 1px solid rgba(255,255,255,0.3);
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                margin: 5px;
                text-decoration: none;
                display: inline-block;
            }}
            .control-button:hover {{
                background: rgba(255,255,255,0.3);
            }}
        </style>
        <script>
            function refreshPage() {{
                location.reload();
            }}
            // Auto-refresh every 5 seconds when load generation is active
            {'setTimeout(refreshPage, 5000);' if is_active else 'setTimeout(refreshPage, 30000);'}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>⚡ NimbusGuard Load Generator</h1>
            
            <button class="refresh-btn" onclick="refreshPage()">🔄 Refresh Status</button>
            
            <div class="status-grid">
                <div class="status-card">
                    <h3>📊 Service Status</h3>
                    <div class="metric">
                        <span>Service:</span>
                        <span class="metric-value">
                            <span class="status-indicator status-running"></span>Running
                        </span>
                    </div>
                    <div class="metric">
                        <span>Version:</span>
                        <span class="metric-value">1.0.0</span>
                    </div>
                    <div class="metric">
                        <span>Load Generation:</span>
                        <span class="metric-value {'active' if is_active else ''}">
                            <span class="status-indicator status-{'active' if is_active else 'idle'}"></span>{'Active' if is_active else 'Idle'}
                        </span>
                    </div>
                    <div class="metric">
                        <span>Pattern:</span>
                        <span class="metric-value">{current_pattern.title()}</span>
                    </div>
                </div>
                
                <div class="status-card">
                    <h3>📈 Load Generation Metrics</h3>
                    <div class="metric">
                        <span>Requests Sent:</span>
                        <span class="metric-value">{requests_sent}</span>
                    </div>
                    <div class="metric">
                        <span>Current Rate:</span>
                        <span class="metric-value">{current_rate:.2f} req/s</span>
                    </div>
                    <div class="metric">
                        <span>Elapsed Time:</span>
                        <span class="metric-value">{elapsed_time}s</span>
                    </div>
                    <div class="metric">
                        <span>Remaining Time:</span>
                        <span class="metric-value">{remaining_time}s</span>
                    </div>
                </div>
                
                <div class="status-card">
                    <h3>💻 System Metrics</h3>
                    <div class="metric">
                        <span>CPU Usage:</span>
                        <span class="metric-value">{cpu_percent:.1f}%</span>
                    </div>
                    <div class="metric">
                        <span>Memory (RSS):</span>
                        <span class="metric-value">{memory_info.rss / 1024 / 1024:.1f} MB</span>
                    </div>
                    <div class="metric">
                        <span>Process ID:</span>
                        <span class="metric-value">{os.getpid()}</span>
                    </div>
                    <div class="metric">
                        <span>CPU Cores:</span>
                        <span class="metric-value">{os.cpu_count()}</span>
                    </div>
                </div>
            </div>
            
            <div class="load-controls">
                <h3>🎮 Load Generation Controls</h3>
                <p>Quick actions to control load generation:</p>
                <a href="/load/status" class="control-button">📊 View Detailed Status</a>
                <a href="/docs#/default/generate_load_load_generate_post" class="control-button">🚀 Start Load Generation</a>
                {'<a href="/load/stop" class="control-button">🛑 Stop Load Generation</a>' if is_active else ''}
            </div>
            
            <h2>🔗 Available Endpoints</h2>
            <div class="endpoints">
                <a href="/docs" class="endpoint-link">
                    📚 API Documentation
                    <br><small>Interactive Swagger UI</small>
                </a>
                <a href="/metrics" class="endpoint-link">
                    📈 Prometheus Metrics
                    <br><small>Load generation metrics</small>
                </a>
                <a href="/health" class="endpoint-link">
                    ❤️ Health Check
                    <br><small>Service health status</small>
                </a>
                <a href="/load/status" class="endpoint-link">
                    ⚡ Load Status
                    <br><small>Current generation status</small>
                </a>
            </div>
            
            <div style="text-align: center; margin-top: 30px; opacity: 0.7;">
                <small>Auto-refreshes every {'5' if is_active else '30'} seconds • Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
            </div>
        </div>
    </body>
    </html>
    """
    
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "load-generator"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    stats = load_generation_state.get("stats", {})

    # Current status metrics
    is_active = 1 if load_generation_state["active"] else 0
    requests_sent = stats.get("requests_sent", 0)

    # Calculate current rate if active
    current_rate = 0
    if load_generation_state["active"] and stats:
        elapsed = time.time() - stats.get("start_time", time.time())
        current_rate = requests_sent / max(elapsed, 1)

    # Generate Prometheus format metrics
    metrics_output = f"""# HELP load_generator_active Whether load generation is currently active
# TYPE load_generator_active gauge
load_generator_active {is_active}

# HELP load_generator_requests_sent_total Total number of requests sent
# TYPE load_generator_requests_sent_total counter
load_generator_requests_sent_total {requests_sent}

# HELP load_generator_current_rate Current request rate per second
# TYPE load_generator_current_rate gauge
load_generator_current_rate {current_rate:.2f}

# HELP load_generator_pattern_info Current load generation pattern
# TYPE load_generator_pattern_info gauge
load_generator_pattern_info{{pattern="{stats.get('pattern', 'none')}"}} {is_active}

# HELP load_generator_duration_seconds Total configured duration
# TYPE load_generator_duration_seconds gauge
load_generator_duration_seconds {stats.get('duration', 0)}

# HELP load_generator_elapsed_seconds Time elapsed since start
# TYPE load_generator_elapsed_seconds gauge
load_generator_elapsed_seconds {int(time.time() - stats.get('start_time', time.time())) if load_generation_state['active'] else 0}
"""

    return Response(content=metrics_output, media_type="text/plain")


@app.post("/load/generate")
async def generate_load(request: LoadGenerationRequest):
    """Start load generation"""
    if load_generation_state["active"]:
        raise HTTPException(status_code=400, detail="Load generation already active")

    # Start load generation task
    task = asyncio.create_task(run_load_generation(request))
    load_generation_state["active"] = True
    load_generation_state["task"] = task
    load_generation_state["stats"] = {
        "pattern": request.pattern,
        "start_time": time.time(),
        "duration": request.duration,
        "requests_sent": 0
    }

    return {"status": "started", "pattern": request.pattern, "duration": request.duration}


@app.get("/load/status")
async def get_load_status() -> LoadGenerationStatus:
    """Get current load generation status"""
    if not load_generation_state["active"]:
        return LoadGenerationStatus(
            active=False,
            pattern="none",
            elapsed=0,
            remaining=0,
            requests_sent=0
        )

    stats = load_generation_state["stats"]
    elapsed = int(time.time() - stats["start_time"])
    remaining = max(0, stats["duration"] - elapsed)

    return LoadGenerationStatus(
        active=True,
        pattern=stats["pattern"],
        elapsed=elapsed,
        remaining=remaining,
        requests_sent=stats["requests_sent"]
    )


@app.post("/load/stop")
async def stop_load():
    """Stop active load generation"""
    if not load_generation_state["active"]:
        raise HTTPException(status_code=400, detail="No active load generation")

    if load_generation_state["task"]:
        load_generation_state["task"].cancel()

    load_generation_state["active"] = False
    load_generation_state["task"] = None

    return {"status": "stopped"}


async def run_load_generation(request: LoadGenerationRequest):
    """Execute load generation based on pattern"""
    start_time = time.time()

    try:
        if request.target == "http":
            await generate_http_load(request, start_time)
        elif request.target == "kafka":
            await generate_kafka_events(request, start_time)
        else:
            raise ValueError(f"Unknown target: {request.target}")
    except asyncio.CancelledError:
        print("Load generation cancelled")
    except Exception as e:
        print(f"Load generation error: {e}")
    finally:
        load_generation_state["active"] = False
        load_generation_state["task"] = None


async def generate_http_load(request: LoadGenerationRequest, start_time: float):
    """Generate HTTP load to consumer workload"""
    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < request.duration:
            elapsed = time.time() - start_time
            progress = elapsed / request.duration

            # Calculate request rate based on pattern
            if request.pattern == "constant":
                requests_per_second = request.intensity / 10
            elif request.pattern == "spike":
                # Spike pattern: low, then high, then low
                if 0.2 <= progress <= 0.8:
                    requests_per_second = request.intensity / 5
                else:
                    requests_per_second = request.intensity / 20
            elif request.pattern == "gradual":
                # Gradual increase
                requests_per_second = (request.intensity / 10) * progress
            elif request.pattern == "burst":
                # Periodic bursts
                if int(elapsed) % 10 < 3:  # 3 seconds on, 7 seconds off
                    requests_per_second = request.intensity / 5
                else:
                    requests_per_second = request.intensity / 30
            else:
                requests_per_second = request.intensity / 10

            # Send requests
            delay = 1.0 / max(requests_per_second, 0.1)

            try:
                # First check if workload is already running
                status_response = await session.get(
                    f"{request.target_url}/api/v1/workload/cpu/status",
                    headers={"Content-Type": "application/json"}
                )

                if status_response.status == 200:
                    status_data = await status_response.json()

                    # If workload is idle, start a new one
                    if status_data.get("status") == "idle":
                        workload_data = {
                            "intensity": min(80, request.intensity),
                            "duration": 5  # Slightly longer duration for efficiency
                        }
                        response = await session.post(
                            f"{request.target_url}/api/v1/workload/cpu",
                            json=workload_data,
                            headers={"Content-Type": "application/json"}
                        )
                        if response.status == 200:
                            load_generation_state["stats"]["requests_sent"] += 1
                            print(f"Started new CPU workload (intensity: {workload_data['intensity']}%)")
                        else:
                            print(f"Failed to start workload: {response.status}")
                    else:
                        # Workload already running, count as successful
                        load_generation_state["stats"]["requests_sent"] += 1
                        # Only print occasionally to avoid log spam
                        if load_generation_state["stats"]["requests_sent"] % 10 == 0:
                            remaining = status_data.get("remaining_time", 0)
                            print(f"CPU workload active (remaining: {remaining:.1f}s)")
                else:
                    print(f"Failed to check workload status: {status_response.status}")
            except Exception as e:
                print(f"HTTP request failed: {e}")

            await asyncio.sleep(delay)


async def generate_kafka_events(request: LoadGenerationRequest, start_time: float):
    """Generate Kafka scaling events"""
    # This would integrate with Kafka producer
    # For now, send REST events to consumer workload
    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < request.duration:
            elapsed = time.time() - start_time

            # Generate scaling event
            event_data = {
                "event_type": "resource_pressure",
                "service": "load-generator",
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "cpu_utilization": min(90, request.intensity + (elapsed % 20)),
                    "memory_utilization": min(85, request.intensity + (elapsed % 15)),
                    "request_rate": request.intensity * 2
                }
            }

            try:
                await session.post(
                    f"{request.target_url}/api/v1/events/trigger",
                    json=event_data,
                    headers={"Content-Type": "application/json"}
                )
                load_generation_state["stats"]["requests_sent"] += 1
            except Exception as e:
                print(f"Kafka event failed: {e}")

            await asyncio.sleep(2)  # Event every 2 seconds


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081)
