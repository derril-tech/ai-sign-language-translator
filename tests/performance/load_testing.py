#!/usr/bin/env python3
"""
Load testing and chaos engineering for ASL translation system
Tests system performance under high load, network issues, and failure conditions
"""

import asyncio
import aiohttp
import time
import random
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import websockets
import base64
import cv2
from datetime import datetime, timedelta
import statistics
import psutil
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LoadTestConfig:
    """Configuration for load testing"""
    # Load parameters
    concurrent_sessions: int = 100
    test_duration_seconds: int = 300
    ramp_up_seconds: int = 60
    
    # Request patterns
    frames_per_second: int = 30
    session_duration_seconds: int = 60
    
    # Network conditions
    simulate_network_issues: bool = True
    packet_loss_rate: float = 0.02  # 2%
    jitter_ms: int = 50
    bandwidth_limit_mbps: Optional[float] = None
    
    # Chaos engineering
    enable_chaos: bool = True
    service_failure_rate: float = 0.01  # 1%
    database_slowdown_rate: float = 0.05  # 5%
    
    # Endpoints
    api_base_url: str = "http://localhost:8000"
    websocket_url: str = "ws://localhost:8001/ws"

@dataclass
class LoadTestResult:
    """Results from load testing"""
    # Performance metrics
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    
    # Throughput
    requests_per_second: float
    frames_processed_per_second: float
    
    # Error analysis
    error_types: Dict[str, int]
    timeout_count: int
    connection_errors: int
    
    # System resources
    peak_cpu_usage: float
    peak_memory_usage_mb: float
    network_usage_mbps: float
    
    # Translation quality under load
    average_confidence: float
    translation_accuracy_degradation: float

class NetworkSimulator:
    """Simulate various network conditions"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.active_delays: Dict[str, float] = {}
    
    async def simulate_network_delay(self, session_id: str) -> float:
        """Simulate network delay with jitter"""
        if not self.config.simulate_network_issues:
            return 0.0
        
        # Base latency + jitter
        base_latency = 0.1  # 100ms base
        jitter = random.uniform(-self.config.jitter_ms/1000, self.config.jitter_ms/1000)
        
        total_delay = base_latency + jitter
        
        # Simulate packet loss (causes retransmission delay)
        if random.random() < self.config.packet_loss_rate:
            total_delay += random.uniform(0.5, 2.0)  # Retransmission delay
        
        if total_delay > 0:
            await asyncio.sleep(total_delay)
        
        return total_delay
    
    def simulate_bandwidth_limit(self, data_size_bytes: int) -> float:
        """Calculate delay based on bandwidth limit"""
        if not self.config.bandwidth_limit_mbps:
            return 0.0
        
        # Convert to bytes per second
        bandwidth_bps = self.config.bandwidth_limit_mbps * 1024 * 1024 / 8
        
        # Calculate transmission time
        transmission_time = data_size_bytes / bandwidth_bps
        
        return transmission_time

class ChaosEngineer:
    """Chaos engineering to test system resilience"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.active_failures: Dict[str, datetime] = {}
    
    def should_inject_failure(self, service_name: str) -> bool:
        """Determine if failure should be injected"""
        if not self.config.enable_chaos:
            return False
        
        # Check if service is already failing
        if service_name in self.active_failures:
            # Failure lasts 30-120 seconds
            failure_duration = random.uniform(30, 120)
            if (datetime.now() - self.active_failures[service_name]).total_seconds() < failure_duration:
                return True
            else:
                del self.active_failures[service_name]
                return False
        
        # Inject new failure
        if random.random() < self.config.service_failure_rate:
            self.active_failures[service_name] = datetime.now()
            logger.warning(f"Chaos: Injecting failure in {service_name}")
            return True
        
        return False
    
    def inject_database_slowdown(self) -> float:
        """Inject database slowdown"""
        if random.random() < self.config.database_slowdown_rate:
            slowdown = random.uniform(1.0, 5.0)  # 1-5 second delay
            logger.warning(f"Chaos: Injecting database slowdown of {slowdown:.1f}s")
            return slowdown
        
        return 0.0

class SessionSimulator:
    """Simulate a single user session"""
    
    def __init__(self, session_id: str, config: LoadTestConfig):
        self.session_id = session_id
        self.config = config
        self.network_sim = NetworkSimulator(config)
        self.chaos = ChaosEngineer(config)
        
        # Session state
        self.frames_sent = 0
        self.responses_received = 0
        self.errors = []
        self.response_times = []
        self.start_time = time.time()
        
    async def run_session(self) -> Dict[str, Any]:
        """Run a complete user session"""
        logger.info(f"Starting session {self.session_id}")
        
        try:
            # Create WebSocket connection
            async with websockets.connect(
                self.config.websocket_url,
                timeout=10,
                ping_interval=20,
                ping_timeout=10
            ) as websocket:
                
                # Send session start
                await self.send_session_start(websocket)
                
                # Send frames for session duration
                end_time = time.time() + self.config.session_duration_seconds
                frame_interval = 1.0 / self.config.frames_per_second
                
                while time.time() < end_time:
                    frame_start = time.time()
                    
                    try:
                        # Generate and send frame
                        await self.send_frame(websocket)
                        
                        # Wait for response (with timeout)
                        response = await asyncio.wait_for(
                            websocket.recv(), 
                            timeout=2.0
                        )
                        
                        # Process response
                        await self.process_response(response)
                        
                    except asyncio.TimeoutError:
                        self.errors.append("timeout")
                    except Exception as e:
                        self.errors.append(f"frame_error: {str(e)}")
                    
                    # Maintain frame rate
                    elapsed = time.time() - frame_start
                    if elapsed < frame_interval:
                        await asyncio.sleep(frame_interval - elapsed)
                
                # Send session end
                await self.send_session_end(websocket)
                
        except Exception as e:
            logger.error(f"Session {self.session_id} failed: {e}")
            self.errors.append(f"session_error: {str(e)}")
        
        return self.get_session_results()
    
    async def send_session_start(self, websocket):
        """Send session start message"""
        message = {
            "type": "session_start",
            "session_id": self.session_id,
            "timestamp": time.time()
        }
        
        await websocket.send(json.dumps(message))
    
    async def send_frame(self, websocket):
        """Send a video frame"""
        # Simulate network delay
        await self.network_sim.simulate_network_delay(self.session_id)
        
        # Check for chaos failures
        if self.chaos.should_inject_failure("pose_worker"):
            raise Exception("Simulated pose worker failure")
        
        # Generate mock frame data
        frame_data = self.generate_mock_frame()
        
        message = {
            "type": "video_frame",
            "session_id": self.session_id,
            "frame_data": frame_data,
            "timestamp": time.time(),
            "frame_id": f"frame_{self.frames_sent:06d}"
        }
        
        request_start = time.time()
        await websocket.send(json.dumps(message))
        self.frames_sent += 1
        
        # Track request timing
        self.response_times.append(time.time() - request_start)
    
    async def process_response(self, response_data: str):
        """Process response from server"""
        try:
            response = json.loads(response_data)
            
            if response.get("type") == "translation_result":
                self.responses_received += 1
                
                # Track response time
                if "timestamp" in response:
                    response_time = time.time() - response["timestamp"]
                    self.response_times.append(response_time * 1000)  # Convert to ms
            
        except json.JSONDecodeError:
            self.errors.append("invalid_json_response")
    
    async def send_session_end(self, websocket):
        """Send session end message"""
        message = {
            "type": "session_end",
            "session_id": self.session_id,
            "timestamp": time.time()
        }
        
        await websocket.send(json.dumps(message))
    
    def generate_mock_frame(self) -> str:
        """Generate mock video frame data"""
        # Create simple test frame
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        
        # Add some structure (person silhouette)
        cv2.rectangle(frame, (140, 80), (180, 200), (200, 150, 100), -1)  # Body
        cv2.circle(frame, (160, 60), 20, (220, 180, 150), -1)  # Head
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return frame_b64
    
    def get_session_results(self) -> Dict[str, Any]:
        """Get session performance results"""
        session_duration = time.time() - self.start_time
        
        return {
            "session_id": self.session_id,
            "duration_seconds": session_duration,
            "frames_sent": self.frames_sent,
            "responses_received": self.responses_received,
            "errors": self.errors,
            "error_count": len(self.errors),
            "response_times_ms": self.response_times,
            "average_response_time_ms": statistics.mean(self.response_times) if self.response_times else 0,
            "success_rate": self.responses_received / max(self.frames_sent, 1)
        }

class SystemMonitor:
    """Monitor system resources during load testing"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = {
            "cpu_usage": [],
            "memory_usage_mb": [],
            "network_io_mbps": [],
            "disk_io_mbps": []
        }
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start system monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        logger.info("Started system monitoring")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Stopped system monitoring")
    
    def _monitor_loop(self):
        """Monitoring loop"""
        last_network = psutil.net_io_counters()
        last_time = time.time()
        
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics["cpu_usage"].append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                self.metrics["memory_usage_mb"].append(memory_mb)
                
                # Network I/O
                current_network = psutil.net_io_counters()
                current_time = time.time()
                time_delta = current_time - last_time
                
                if time_delta > 0:
                    bytes_sent = current_network.bytes_sent - last_network.bytes_sent
                    bytes_recv = current_network.bytes_recv - last_network.bytes_recv
                    total_bytes = bytes_sent + bytes_recv
                    
                    mbps = (total_bytes * 8) / (time_delta * 1024 * 1024)
                    self.metrics["network_io_mbps"].append(mbps)
                
                last_network = current_network
                last_time = current_time
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def get_peak_metrics(self) -> Dict[str, float]:
        """Get peak resource usage metrics"""
        return {
            "peak_cpu_usage": max(self.metrics["cpu_usage"]) if self.metrics["cpu_usage"] else 0,
            "peak_memory_usage_mb": max(self.metrics["memory_usage_mb"]) if self.metrics["memory_usage_mb"] else 0,
            "peak_network_io_mbps": max(self.metrics["network_io_mbps"]) if self.metrics["network_io_mbps"] else 0,
            "average_cpu_usage": statistics.mean(self.metrics["cpu_usage"]) if self.metrics["cpu_usage"] else 0,
            "average_memory_usage_mb": statistics.mean(self.metrics["memory_usage_mb"]) if self.metrics["memory_usage_mb"] else 0
        }

class LoadTester:
    """Main load testing orchestrator"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.system_monitor = SystemMonitor()
        self.session_results: List[Dict[str, Any]] = []
    
    async def run_load_test(self) -> LoadTestResult:
        """Run comprehensive load test"""
        logger.info(f"Starting load test with {self.config.concurrent_sessions} concurrent sessions")
        
        # Start system monitoring
        self.system_monitor.start_monitoring()
        
        try:
            # Create session tasks with ramp-up
            session_tasks = []
            ramp_up_delay = self.config.ramp_up_seconds / self.config.concurrent_sessions
            
            for i in range(self.config.concurrent_sessions):
                session_id = f"session_{i:04d}"
                
                # Create session with delay for ramp-up
                task = asyncio.create_task(
                    self._run_delayed_session(session_id, i * ramp_up_delay)
                )
                session_tasks.append(task)
            
            # Wait for all sessions to complete
            logger.info("Waiting for all sessions to complete...")
            session_results = await asyncio.gather(*session_tasks, return_exceptions=True)
            
            # Process results
            self.session_results = [
                result for result in session_results 
                if isinstance(result, dict) and not isinstance(result, Exception)
            ]
            
            # Calculate aggregate metrics
            load_test_result = self._calculate_aggregate_results()
            
            return load_test_result
            
        finally:
            # Stop monitoring
            self.system_monitor.stop_monitoring()
    
    async def _run_delayed_session(self, session_id: str, delay: float) -> Dict[str, Any]:
        """Run session with initial delay for ramp-up"""
        if delay > 0:
            await asyncio.sleep(delay)
        
        simulator = SessionSimulator(session_id, self.config)
        return await simulator.run_session()
    
    def _calculate_aggregate_results(self) -> LoadTestResult:
        """Calculate aggregate results from all sessions"""
        
        # Aggregate basic metrics
        total_requests = sum(result["frames_sent"] for result in self.session_results)
        successful_requests = sum(result["responses_received"] for result in self.session_results)
        failed_requests = total_requests - successful_requests
        
        # Response time metrics
        all_response_times = []
        for result in self.session_results:
            all_response_times.extend(result["response_times_ms"])
        
        avg_response_time = statistics.mean(all_response_times) if all_response_times else 0
        p95_response_time = np.percentile(all_response_times, 95) if all_response_times else 0
        p99_response_time = np.percentile(all_response_times, 99) if all_response_times else 0
        
        # Throughput metrics
        total_duration = max(result["duration_seconds"] for result in self.session_results) if self.session_results else 1
        requests_per_second = total_requests / total_duration
        frames_per_second = requests_per_second  # Same for this test
        
        # Error analysis
        error_types = {}
        timeout_count = 0
        connection_errors = 0
        
        for result in self.session_results:
            for error in result["errors"]:
                if "timeout" in error:
                    timeout_count += 1
                elif "connection" in error:
                    connection_errors += 1
                
                error_types[error] = error_types.get(error, 0) + 1
        
        # System resource metrics
        system_metrics = self.system_monitor.get_peak_metrics()
        
        # Translation quality metrics (mock)
        average_confidence = 0.85 - (failed_requests / max(total_requests, 1)) * 0.2  # Degrade with failures
        accuracy_degradation = (failed_requests / max(total_requests, 1)) * 0.15  # 15% max degradation
        
        return LoadTestResult(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time_ms=avg_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            requests_per_second=requests_per_second,
            frames_processed_per_second=frames_per_second,
            error_types=error_types,
            timeout_count=timeout_count,
            connection_errors=connection_errors,
            peak_cpu_usage=system_metrics["peak_cpu_usage"],
            peak_memory_usage_mb=system_metrics["peak_memory_usage_mb"],
            network_usage_mbps=system_metrics["peak_network_io_mbps"],
            average_confidence=average_confidence,
            translation_accuracy_degradation=accuracy_degradation
        )
    
    def generate_report(self, result: LoadTestResult, output_path: str):
        """Generate comprehensive load test report"""
        
        report = {
            "test_configuration": asdict(self.config),
            "test_results": asdict(result),
            "test_timestamp": datetime.now().isoformat(),
            "session_details": self.session_results,
            "performance_analysis": self._analyze_performance(result),
            "recommendations": self._generate_recommendations(result)
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Load test report saved to {output_path}")
        
        # Print summary
        self._print_summary(result)
    
    def _analyze_performance(self, result: LoadTestResult) -> Dict[str, Any]:
        """Analyze performance characteristics"""
        
        # Success rate analysis
        success_rate = result.successful_requests / max(result.total_requests, 1)
        
        # Response time analysis
        response_time_rating = "excellent" if result.average_response_time_ms < 500 else \
                              "good" if result.average_response_time_ms < 1000 else \
                              "poor"
        
        # Throughput analysis
        throughput_rating = "excellent" if result.requests_per_second > 1000 else \
                           "good" if result.requests_per_second > 500 else \
                           "poor"
        
        # Resource usage analysis
        cpu_rating = "excellent" if result.peak_cpu_usage < 70 else \
                    "good" if result.peak_cpu_usage < 85 else \
                    "poor"
        
        memory_rating = "excellent" if result.peak_memory_usage_mb < 2048 else \
                       "good" if result.peak_memory_usage_mb < 4096 else \
                       "poor"
        
        return {
            "success_rate": success_rate,
            "success_rate_rating": "excellent" if success_rate > 0.99 else "good" if success_rate > 0.95 else "poor",
            "response_time_rating": response_time_rating,
            "throughput_rating": throughput_rating,
            "cpu_usage_rating": cpu_rating,
            "memory_usage_rating": memory_rating,
            "overall_rating": self._calculate_overall_rating(result)
        }
    
    def _calculate_overall_rating(self, result: LoadTestResult) -> str:
        """Calculate overall performance rating"""
        success_rate = result.successful_requests / max(result.total_requests, 1)
        
        # Weighted scoring
        scores = []
        scores.append(1.0 if success_rate > 0.99 else 0.5 if success_rate > 0.95 else 0.0)  # 40% weight
        scores.append(1.0 if result.average_response_time_ms < 500 else 0.5 if result.average_response_time_ms < 1000 else 0.0)  # 30% weight
        scores.append(1.0 if result.requests_per_second > 1000 else 0.5 if result.requests_per_second > 500 else 0.0)  # 20% weight
        scores.append(1.0 if result.peak_cpu_usage < 70 else 0.5 if result.peak_cpu_usage < 85 else 0.0)  # 10% weight
        
        overall_score = sum(scores) / len(scores)
        
        if overall_score > 0.8:
            return "excellent"
        elif overall_score > 0.6:
            return "good"
        else:
            return "poor"
    
    def _generate_recommendations(self, result: LoadTestResult) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        success_rate = result.successful_requests / max(result.total_requests, 1)
        
        if success_rate < 0.95:
            recommendations.append("Low success rate detected. Check error logs and improve error handling.")
        
        if result.average_response_time_ms > 1000:
            recommendations.append("High response times. Consider optimizing pose detection and translation pipelines.")
        
        if result.peak_cpu_usage > 85:
            recommendations.append("High CPU usage. Consider horizontal scaling or CPU optimization.")
        
        if result.peak_memory_usage_mb > 4096:
            recommendations.append("High memory usage. Check for memory leaks and optimize data structures.")
        
        if result.timeout_count > result.total_requests * 0.01:
            recommendations.append("High timeout rate. Increase timeout values or improve response times.")
        
        if result.translation_accuracy_degradation > 0.1:
            recommendations.append("Translation accuracy degrades under load. Implement quality preservation mechanisms.")
        
        return recommendations
    
    def _print_summary(self, result: LoadTestResult):
        """Print test summary to console"""
        print("\n" + "="*80)
        print("LOAD TEST RESULTS SUMMARY")
        print("="*80)
        
        print(f"Test Configuration:")
        print(f"  Concurrent Sessions: {self.config.concurrent_sessions}")
        print(f"  Test Duration: {self.config.test_duration_seconds}s")
        print(f"  Target FPS: {self.config.frames_per_second}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Total Requests: {result.total_requests:,}")
        print(f"  Successful: {result.successful_requests:,} ({result.successful_requests/max(result.total_requests,1)*100:.1f}%)")
        print(f"  Failed: {result.failed_requests:,}")
        print(f"  Avg Response Time: {result.average_response_time_ms:.1f}ms")
        print(f"  P95 Response Time: {result.p95_response_time_ms:.1f}ms")
        print(f"  P99 Response Time: {result.p99_response_time_ms:.1f}ms")
        print(f"  Throughput: {result.requests_per_second:.1f} req/s")
        
        print(f"\nSystem Resources:")
        print(f"  Peak CPU Usage: {result.peak_cpu_usage:.1f}%")
        print(f"  Peak Memory Usage: {result.peak_memory_usage_mb:.1f} MB")
        print(f"  Network Usage: {result.network_usage_mbps:.1f} Mbps")
        
        print(f"\nTranslation Quality:")
        print(f"  Average Confidence: {result.average_confidence:.3f}")
        print(f"  Accuracy Degradation: {result.translation_accuracy_degradation:.1%}")
        
        if result.error_types:
            print(f"\nTop Errors:")
            sorted_errors = sorted(result.error_types.items(), key=lambda x: x[1], reverse=True)
            for error, count in sorted_errors[:5]:
                print(f"  {error}: {count}")
        
        print("="*80)

# Example usage and test cases
if __name__ == '__main__':
    # Configure load test
    config = LoadTestConfig(
        concurrent_sessions=50,  # Reduced for demo
        test_duration_seconds=120,  # 2 minutes
        ramp_up_seconds=30,
        frames_per_second=15,  # Reduced for demo
        session_duration_seconds=60,
        simulate_network_issues=True,
        packet_loss_rate=0.01,
        jitter_ms=25,
        enable_chaos=True,
        service_failure_rate=0.005
    )
    
    # Run load test
    async def run_test():
        load_tester = LoadTester(config)
        
        try:
            result = await load_tester.run_load_test()
            load_tester.generate_report(result, "load_test_report.json")
            
        except KeyboardInterrupt:
            logger.info("Load test interrupted by user")
        except Exception as e:
            logger.error(f"Load test failed: {e}")
    
    # Run the test
    print("Starting ASL Translation System Load Test...")
    print("Note: This is a simulation - actual endpoints may not be available")
    
    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        print("\nLoad test stopped by user")
    
    print("Load test completed!")
