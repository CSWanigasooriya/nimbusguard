#!/usr/bin/env python3
"""
Load Generator Service for NimbusGuard

Generates configurable HTTP load and Kafka events to test scaling scenarios.
"""

import asyncio
import time
from datetime import datetime

import aiohttp
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
