# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import tempfile
import time
import asyncio
from typing import List, Optional, Dict, Any
from unittest.mock import patch

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolverRegistry
from vllm.plugins.lora_resolvers.filesystem_resolver import FilesystemResolver


class StorageCatalogTester:
    """Test class for validating LoRA storage catalog functionality."""
    
    def __init__(self, storage_path: str = "/models/loras/"):
        self.storage_path = storage_path
        self.resolver = FilesystemResolver(storage_path)
        self.base_model_name = "meta-llama/Llama-2-7b-hf"
        
    def discover_adapters(self) -> List[str]:
        """Discover all LoRA adapters in the storage catalog."""
        if not os.path.exists(self.storage_path):
            return []
        
        adapters = []
        for item in os.listdir(self.storage_path):
            adapter_path = os.path.join(self.storage_path, item)
            if os.path.isdir(adapter_path):
                # Check if it has required files
                config_path = os.path.join(adapter_path, "adapter_config.json")
                if os.path.exists(config_path):
                    adapters.append(item)
        
        # Sort adapters numerically instead of alphabetically
        def extract_number(adapter_name):
            """Extract number from adapter name for numeric sorting."""
            import re
            match = re.search(r'(\d+)', adapter_name)
            return int(match.group(1)) if match else 0
        
        return sorted(adapters, key=extract_number)
    
    def validate_adapter_config(self, adapter_name: str) -> dict:
        """Validate adapter configuration and return config dict."""
        adapter_path = os.path.join(self.storage_path, adapter_name)
        config_path = os.path.join(adapter_path, "adapter_config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"adapter_config.json not found for {adapter_name}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = ["peft_type", "base_model_name_or_path", "r", "lora_alpha", "target_modules"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in {adapter_name}")
        
        if config["peft_type"] != "LORA":
            raise ValueError(f"Invalid peft_type '{config['peft_type']}' in {adapter_name}")
        
        return config
    
    async def test_resolver_discovery(self, adapter_name: str) -> Optional[LoRARequest]:
        """Test if resolver can discover and load a specific adapter."""
        return await self.resolver.resolve_lora(self.base_model_name, adapter_name)
    
    def test_all_adapters_discovery(self) -> dict:
        """Test discovery of all adapters in the catalog."""
        adapters = self.discover_adapters()
        results = {}
        
        for adapter in adapters:
            try:
                config = self.validate_adapter_config(adapter)
                results[adapter] = {
                    "status": "valid",
                    "config": config,
                    "base_model_match": config["base_model_name_or_path"] == self.base_model_name
                }
            except Exception as e:
                results[adapter] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results


class HTTPLoRATester:
    """Test class for HTTP-based LoRA testing with vLLM server."""
    
    def __init__(self, server_url: str = "http://localhost:8000", max_cpu_loras: int = 5):
        self.server_url = server_url
        self.session = requests.Session()
        self.max_cpu_loras = max_cpu_loras
        self.timing_results = {}  # Store timing results
        self.request_count = 0  # Track total requests made
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def test_server_health(self) -> bool:
        """Test if vLLM server is running and healthy."""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def test_lora_request(self, model_name: str, prompt: str = "Generate a SQL query for:", max_tokens: int = 50, test_case: str = "unknown") -> dict:
        """Test a LoRA adapter via HTTP request with timing measurements."""
        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        
        # Record start time
        start_time = time.time()
        self.request_count += 1
        
        try:
            response = self.session.post(
                f"{self.server_url}/v1/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            # Record end time
            end_time = time.time()
            completion_time = end_time - start_time
            
            if response.status_code == 200:
                result = {
                    "status": "success",
                    "response": response.json(),
                    "model": model_name,
                    "completion_time": completion_time,
                    "test_case": test_case,
                    "request_number": self.request_count,
                    "timestamp": start_time
                }
                
                # Store timing result
                if model_name not in self.timing_results:
                    self.timing_results[model_name] = []
                self.timing_results[model_name].append(result)
                
                return result
            else:
                return {
                    "status": "error",
                    "status_code": response.status_code,
                    "error": response.text,
                    "model": model_name,
                    "completion_time": completion_time,
                    "test_case": test_case,
                    "request_number": self.request_count,
                    "timestamp": start_time
                }
        except Exception as e:
            end_time = time.time()
            completion_time = end_time - start_time
            
            return {
                "status": "error",
                "error": str(e),
                "model": model_name,
                "completion_time": completion_time,
                "test_case": test_case,
                "request_number": self.request_count,
                "timestamp": start_time
            }
    
    def test_three_latency_scenarios(self, adapter_names: List[str]) -> dict:
        """
        Test 3 latency scenarios for LoRA adapters:
        1. Storage -> GPU (first load from disk)
        2. CPU -> GPU (evicted from GPU, reloaded from CPU memory)
        3. GPU cached (already in GPU memory)
        """
        results = {}
        
        print(f"Testing {len(adapter_names)} adapters for 3 latency scenarios...")
        print(f"Scenario 1: Storage -> GPU (first load)")
        print(f"Scenario 2: CPU -> GPU (after eviction - 5 adapter groups)")
        print(f"Scenario 3: GPU cached (already in GPU - single adapter)")
        print(f"Max CPU LoRAs: {self.max_cpu_loras} (will force eviction after {self.max_cpu_loras} adapters)")
        print(f"Max LoRAs: 1 (only one adapter in GPU at a time)")
        print(f"Note: CPU->GPU test uses 5 adapter groups, GPU cached test uses single adapter with 2 requests")
        
        # Phase 1: Load all adapters for the first time (Storage -> GPU)
        print(f"\n=== Phase 1: Storage -> GPU (First Load) ===")
        print(f"Loading all {len(adapter_names)} adapters from storage to GPU...")
        for i, adapter_name in enumerate(adapter_names):
            print(f"Loading {adapter_name} from storage to GPU...")
            result = self.test_lora_request(adapter_name, test_case="storage_to_gpu")
            
            if adapter_name not in results:
                results[adapter_name] = []
            results[adapter_name].append(result)
        
        # Phase 2: Test CPU -> GPU latency in 5 adapter groups
        print(f"\n=== Phase 2: CPU -> GPU (After Eviction) ===")
        print(f"Testing CPU->GPU latency in 5 adapter groups...")
        
        # Process adapters in groups of 5
        for group_start in range(0, len(adapter_names), 5):
            group_end = min(group_start + 5, len(adapter_names))
            group_adapters = adapter_names[group_start:group_end]
            
            print(f"\n--- Group {group_start//5 + 1}: Adapters {group_start+1}-{group_end} ---")
            
            # First, request the group adapters without recording (to load them to CPU)
            print(f"Loading adapters {group_start+1}-{group_end} to CPU memory...")
            for adapter_name in group_adapters:
                self.test_lora_request(adapter_name, test_case="cpu_warmup")
            
            # Then request them again and record the CPU->GPU latency
            print(f"Recording CPU->GPU latency for adapters {group_start+1}-{group_end}...")
            for adapter_name in group_adapters:
                print(f"Reloading {adapter_name} from CPU to GPU...")
                result = self.test_lora_request(adapter_name, test_case="cpu_to_gpu")
                results[adapter_name].append(result)
        
        # Phase 3: Test GPU cached performance (single adapter, 2 requests)
        print(f"\n=== Phase 3: GPU Cached (Already in GPU) ===")
        print(f"Testing GPU cached performance with single adapter...")
        print("Note: Request same adapter twice (ignore first, use second timing)")
        
        # Use the first adapter for GPU cached test
        test_adapter = adapter_names[0]
        print(f"Testing {test_adapter} from GPU cache...")
        
        # First request: loads adapter to GPU (ignore timing)
        self.test_lora_request(test_adapter, test_case="gpu_cache_warmup")
        # Second request: adapter already in GPU (use this timing)
        result = self.test_lora_request(test_adapter, test_case="gpu_cached")
        results[test_adapter].append(result)
        
        return results
    
    def analyze_timing_results(self) -> dict:
        """Analyze timing results for the 3 latency scenarios."""
        analysis = {
            "storage_to_gpu_times": {},
            "cpu_to_gpu_times": {},
            "gpu_cached_times": {},
            "performance_comparison": {},
            "summary": {}
        }
        
        for adapter_name, requests in self.timing_results.items():
            storage_times = []
            cpu_times = []
            gpu_times = []
            
            for request in requests:
                test_case = request.get("test_case", "unknown")
                if test_case == "storage_to_gpu":
                    storage_times.append(request["completion_time"])
                elif test_case == "cpu_to_gpu":
                    cpu_times.append(request["completion_time"])
                elif test_case == "gpu_cached":
                    gpu_times.append(request["completion_time"])
            
            # Store timing data for each scenario
            if storage_times:
                analysis["storage_to_gpu_times"][adapter_name] = {
                    "avg": sum(storage_times) / len(storage_times),
                    "min": min(storage_times),
                    "max": max(storage_times),
                    "count": len(storage_times)
                }
            
            if cpu_times:
                analysis["cpu_to_gpu_times"][adapter_name] = {
                    "avg": sum(cpu_times) / len(cpu_times),
                    "min": min(cpu_times),
                    "max": max(cpu_times),
                    "count": len(cpu_times)
                }
            
            if gpu_times:
                analysis["gpu_cached_times"][adapter_name] = {
                    "avg": sum(gpu_times) / len(gpu_times),
                    "min": min(gpu_times),
                    "max": max(gpu_times),
                    "count": len(gpu_times)
                }
            
            # Calculate performance comparisons
            if storage_times and cpu_times and gpu_times:
                storage_avg = sum(storage_times) / len(storage_times)
                cpu_avg = sum(cpu_times) / len(cpu_times)
                gpu_avg = sum(gpu_times) / len(gpu_times)
                
                analysis["performance_comparison"][adapter_name] = {
                    "storage_to_cpu_improvement": ((storage_avg - cpu_avg) / storage_avg) * 100 if storage_avg > 0 else 0,
                    "storage_to_gpu_improvement": ((storage_avg - gpu_avg) / storage_avg) * 100 if storage_avg > 0 else 0,
                    "cpu_to_gpu_improvement": ((cpu_avg - gpu_avg) / cpu_avg) * 100 if cpu_avg > 0 else 0,
                    "storage_to_cpu_speedup": storage_avg / cpu_avg if cpu_avg > 0 else float('inf'),
                    "storage_to_gpu_speedup": storage_avg / gpu_avg if gpu_avg > 0 else float('inf'),
                    "cpu_to_gpu_speedup": cpu_avg / gpu_avg if gpu_avg > 0 else float('inf')
                }
        
        # Summary statistics
        all_storage = [t["avg"] for t in analysis["storage_to_gpu_times"].values()]
        all_cpu = [t["avg"] for t in analysis["cpu_to_gpu_times"].values()]
        all_gpu = [t["avg"] for t in analysis["gpu_cached_times"].values()]
        
        if all_storage and all_cpu and all_gpu:
            analysis["summary"] = {
                "avg_storage_to_gpu_time": sum(all_storage) / len(all_storage),
                "avg_cpu_to_gpu_time": sum(all_cpu) / len(all_cpu),
                "avg_gpu_cached_time": sum(all_gpu) / len(all_gpu),
                "storage_vs_cpu_improvement": ((sum(all_storage) / len(all_storage) - sum(all_cpu) / len(all_cpu)) / (sum(all_storage) / len(all_storage))) * 100,
                "storage_vs_gpu_improvement": ((sum(all_storage) / len(all_storage) - sum(all_gpu) / len(all_gpu)) / (sum(all_storage) / len(all_storage))) * 100,
                "cpu_vs_gpu_improvement": ((sum(all_cpu) / len(all_cpu) - sum(all_gpu) / len(all_gpu)) / (sum(all_cpu) / len(all_cpu))) * 100,
                "total_adapters_tested": len(self.timing_results)
            }
        
        return analysis


# Test functions
def test_storage_catalog_discovery():
    """Test discovery of all LoRA adapters in storage catalog."""
    tester = StorageCatalogTester()
    results = tester.test_all_adapters_discovery()
    
    print(f"\n=== Storage Catalog Discovery Results ===")
    print(f"Total adapters found: {len(results)}")
    
    valid_count = 0
    error_count = 0
    
    for adapter, result in results.items():
        if result["status"] == "valid":
            valid_count += 1
            base_model_match = result["base_model_match"]
            print(f"✓ {adapter}: Valid config, base_model_match={base_model_match}")
        else:
            error_count += 1
            print(f"✗ {adapter}: {result['error']}")
    
    print(f"\nSummary: {valid_count} valid, {error_count} errors")
    
    # Check if we have results
    if len(results) == 0:
        print("Warning: No adapters found in storage catalog")
    if valid_count == 0:
        print("Warning: No valid adapters found")
    
    return results


async def test_resolver_discovery():
    """Test resolver discovery for all adapters."""
    tester = StorageCatalogTester()
    adapters = tester.discover_adapters()
    
    print(f"\n=== Resolver Discovery Test ===")
    print(f"Testing {len(adapters)} adapters...")
    
    results = {}
    successful_resolutions = 0
    
    for adapter in adapters:
        try:
            lora_request = await tester.test_resolver_discovery(adapter)
            if lora_request:
                results[adapter] = {
                    "status": "success",
                    "lora_name": lora_request.lora_name,
                    "lora_path": lora_request.lora_path,
                    "lora_int_id": lora_request.lora_int_id
                }
                successful_resolutions += 1
                print(f"✓ {adapter}: Successfully resolved")
            else:
                results[adapter] = {"status": "failed", "reason": "Resolver returned None"}
                print(f"✗ {adapter}: Resolver returned None")
        except Exception as e:
            results[adapter] = {"status": "error", "error": str(e)}
            print(f"✗ {adapter}: {str(e)}")
    
    print(f"\nSummary: {successful_resolutions}/{len(adapters)} successful resolutions")
    
    # Check if we have successful resolutions
    if successful_resolutions == 0:
        print("Warning: No successful resolver discoveries")
    
    return results


def test_http_integration():
    """Test HTTP integration with vLLM server for 3 latency scenarios."""
    http_tester = HTTPLoRATester(max_cpu_loras=5)  # Small cache to force eviction
    
    # Check if server is running
    if not http_tester.test_server_health():
        print("vLLM server is not running. Start server with LoRA resolver enabled to run this test.")
        return None
    
    # Get adapters from storage catalog
    catalog_tester = StorageCatalogTester()
    adapters = catalog_tester.discover_adapters()
    
    print(f"\n=== HTTP Integration Test - 3 Latency Scenarios ===")
    print(f"Testing {len(adapters)} adapters via HTTP...")
    print("Scenario 1: Storage -> GPU (first load from disk)")
    print("Scenario 2: CPU -> GPU (after eviction from GPU)")
    print("Scenario 3: GPU cached (already in GPU memory)")
    
    # Test the 3 latency scenarios
    results = http_tester.test_three_latency_scenarios(adapters)
    
    # Analyze timing results
    timing_analysis = http_tester.analyze_timing_results()
    
    # Print detailed timing results
    print(f"\n=== Timing Analysis Results ===")
    
    for adapter_name in adapters:
        if adapter_name in timing_analysis["storage_to_gpu_times"]:
            storage = timing_analysis["storage_to_gpu_times"][adapter_name]
            print(f"\n{adapter_name}:")
            print(f"  Storage->GPU: {storage['avg']:.3f}s (avg), {storage['min']:.3f}s (min), {storage['max']:.3f}s (max)")
            
            if adapter_name in timing_analysis["cpu_to_gpu_times"]:
                cpu = timing_analysis["cpu_to_gpu_times"][adapter_name]
                print(f"  CPU->GPU:     {cpu['avg']:.3f}s (avg), {cpu['min']:.3f}s (min), {cpu['max']:.3f}s (max)")
            
            if adapter_name in timing_analysis["gpu_cached_times"]:
                gpu = timing_analysis["gpu_cached_times"][adapter_name]
                print(f"  GPU Cached:   {gpu['avg']:.3f}s (avg), {gpu['min']:.3f}s (min), {gpu['max']:.3f}s (max)")
                
                if adapter_name in timing_analysis["performance_comparison"]:
                    comparison = timing_analysis["performance_comparison"][adapter_name]
                    print(f"  Performance:")
                    print(f"    Storage vs CPU: {comparison['storage_to_cpu_improvement']:.1f}% improvement ({comparison['storage_to_cpu_speedup']:.2f}x)")
                    print(f"    Storage vs GPU: {comparison['storage_to_gpu_improvement']:.1f}% improvement ({comparison['storage_to_gpu_speedup']:.2f}x)")
                    print(f"    CPU vs GPU:     {comparison['cpu_to_gpu_improvement']:.1f}% improvement ({comparison['cpu_to_gpu_speedup']:.2f}x)")
    
    # Print summary
    if timing_analysis["summary"]:
        summary = timing_analysis["summary"]
        print(f"\n=== Overall Summary ===")
        print(f"Total adapters tested: {summary['total_adapters_tested']}")
        print(f"Average Storage->GPU time: {summary['avg_storage_to_gpu_time']:.3f}s")
        print(f"Average CPU->GPU time:     {summary['avg_cpu_to_gpu_time']:.3f}s")
        print(f"Average GPU cached time:   {summary['avg_gpu_cached_time']:.3f}s")
        print(f"Performance improvements:")
        print(f"  Storage vs CPU: {summary['storage_vs_cpu_improvement']:.1f}%")
        print(f"  Storage vs GPU: {summary['storage_vs_gpu_improvement']:.1f}%")
        print(f"  CPU vs GPU:     {summary['cpu_vs_gpu_improvement']:.1f}%")
    
    successful_requests = 0
    failed_requests = 0
    
    for adapter, adapter_results in results.items():
        for result in adapter_results:
            if result["status"] == "success":
                successful_requests += 1
            else:
                failed_requests += 1
    
    print(f"\nSummary: {successful_requests} successful, {failed_requests} failed HTTP requests")
    
    return {
        "results": results,
        "timing_analysis": timing_analysis
    }


def test_environment_setup():
    """Test that environment variables are properly set for LoRA resolver."""
    print(f"\n=== Environment Setup Test ===")
    
    # Check required environment variables
    required_vars = {
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": os.getenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING"),
        "VLLM_PLUGINS": os.getenv("VLLM_PLUGINS"),
        "VLLM_LORA_RESOLVER_CACHE_DIR": os.getenv("VLLM_LORA_RESOLVER_CACHE_DIR")
    }
    
    for var_name, var_value in required_vars.items():
        if var_value:
            print(f"✓ {var_name}: {var_value}")
        else:
            print(f"✗ {var_name}: Not set")
    
    # Check if cache directory exists
    cache_dir = os.getenv("VLLM_LORA_RESOLVER_CACHE_DIR")
    if cache_dir:
        if os.path.exists(cache_dir) and os.path.isdir(cache_dir):
            print(f"✓ Cache directory exists: {cache_dir}")
        else:
            print(f"✗ Cache directory does not exist: {cache_dir}")
    
    return required_vars


def run_comprehensive_test():
    """Run all tests in sequence."""
    print("=" * 60)
    print("LoRA Storage Catalog Comprehensive Test - 3 Latency Scenarios")
    print("=" * 60)
    print("This test measures 3 different latency scenarios:")
    print("1. Storage -> GPU: First load from disk to GPU memory")
    print("2. CPU -> GPU: Reload from CPU memory to GPU (after eviction)")
    print("3. GPU Cached: Already loaded in GPU memory")
    print("Expected behavior: Storage > CPU > GPU (slowest to fastest)")
    print("=" * 60)
    
    # Test 1: Environment setup
    env_results = test_environment_setup()
    
    # Test 2: Storage catalog discovery
    discovery_results = test_storage_catalog_discovery()
    
    # Test 3: Resolver discovery (async)
    resolver_results = asyncio.run(test_resolver_discovery())
    
    # Test 4: HTTP integration with timing (if server is running)
    try:
        http_results = test_http_integration()
    except Exception as e:
        print(f"Skipping HTTP test: {e}")
        http_results = None
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_adapters = len(discovery_results)
    valid_adapters = sum(1 for r in discovery_results.values() if r["status"] == "valid")
    successful_resolutions = sum(1 for r in resolver_results.values() if r["status"] == "success")
    
    print(f"Total adapters in catalog: {total_adapters}")
    print(f"Valid adapter configs: {valid_adapters}")
    print(f"Successful resolver discoveries: {successful_resolutions}")
    
    if http_results and "timing_analysis" in http_results:
        timing_analysis = http_results["timing_analysis"]
        if timing_analysis["summary"]:
            summary = timing_analysis["summary"]
            print(f"\n=== Performance Summary ===")
            print(f"Average Storage->GPU time: {summary['avg_storage_to_gpu_time']:.3f}s")
            print(f"Average CPU->GPU time:     {summary['avg_cpu_to_gpu_time']:.3f}s")
            print(f"Average GPU cached time:   {summary['avg_gpu_cached_time']:.3f}s")
            print(f"Performance improvements:")
            print(f"  Storage vs CPU: {summary['storage_vs_cpu_improvement']:.1f}%")
            print(f"  Storage vs GPU: {summary['storage_vs_gpu_improvement']:.1f}%")
            print(f"  CPU vs GPU:     {summary['cpu_vs_gpu_improvement']:.1f}%")
        
        # Count successful HTTP requests
        if "results" in http_results:
            successful_http = 0
            for adapter_results in http_results["results"].values():
                for result in adapter_results:
                    if result["status"] == "success":
                        successful_http += 1
            print(f"Successful HTTP requests: {successful_http}")
    
    return {
        "environment": env_results,
        "discovery": discovery_results,
        "resolver": resolver_results,
        "http": http_results
    }


if __name__ == "__main__":
    # Run comprehensive test when executed directly
    results = run_comprehensive_test()
