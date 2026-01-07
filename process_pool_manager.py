"""
Controlled Process Pool Manager
Manages multiprocessing with resource limits to avoid system overload
"""

import os
import psutil
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Callable
import time


class ProcessPoolManager:
    """
    Manages process pool with intelligent resource control.
    
    Key Features:
    - Memory-aware worker count calculation
    - Prevents system overload
    - Graceful error handling
    - Result re-ordering for determinism
    
    Why Multiprocessing (Not Asyncio):
    1. OCR is CPU-bound (95% CPU, 5% I/O)
    2. Asyncio does NOT bypass Python GIL
    3. Multiprocessing = true parallelism
    4. 3-4x speedup on 4-core CPU
    
    Performance Comparison (10-page document, 4 cores):
    - Sequential: 10 pages × 5s = 50s
    - Asyncio: 10 pages × 5s = 50s (NO improvement)
    - Multiprocessing: 10 pages ÷ 4 cores = 12.5s (4x faster)
    """
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 memory_limit_gb: Optional[float] = None,
                 worker_memory_mb: float = 500):
        """
        Initialize process pool manager.
        
        Args:
            max_workers: Maximum number of workers (None = auto-calculate)
            memory_limit_gb: Maximum memory to use in GB (None = 80% of available)
            worker_memory_mb: Expected memory per worker in MB (default: 500 for EasyOCR)
        """
        self.max_workers = max_workers
        self.memory_limit_gb = memory_limit_gb
        self.worker_memory_mb = worker_memory_mb
        
    def calculate_optimal_workers(self, total_pages: int) -> int:
        """
        Calculate optimal number of workers based on resources.
        
        Constraints:
        - Each EasyOCR process: ~500MB
        - Each Tesseract process: ~50MB
        - Assume worst case (all EasyOCR)
        
        Formula:
        max_workers = min(
            cpu_count(),
            total_pages,
            available_memory_gb / worker_memory_gb
        )
        
        Args:
            total_pages: Total number of pages to process
            
        Returns:
            Optimal worker count (1 to cpu_count)
        """
        
        # Get CPU count
        cpu_cores = cpu_count()
        
        # Get available memory
        if self.memory_limit_gb is not None:
            available_memory_gb = self.memory_limit_gb
        else:
            # Use 80% of available memory to leave headroom
            mem = psutil.virtual_memory()
            available_memory_gb = (mem.available / (1024**3)) * 0.8
        
        # Calculate memory-limited workers
        worker_memory_gb = self.worker_memory_mb / 1024
        memory_limited_workers = int(available_memory_gb / worker_memory_gb)
        
        # If user specified max_workers, use it as upper bound
        if self.max_workers is not None:
            optimal = min(cpu_cores, total_pages, memory_limited_workers, self.max_workers)
        else:
            optimal = min(cpu_cores, total_pages, memory_limited_workers)
        
        # Minimum 1, maximum cpu_cores
        return max(1, min(optimal, cpu_cores))
    
    def process_pages_parallel(self,
                               worker_func: Callable,
                               page_args: List[Tuple],
                               total_pages: int,
                               verbose: bool = True) -> List:
        """
        Process pages in parallel using controlled process pool.
        
        Parallelization Strategy:
        - Granularity: Page-level (optimal for PDF processing)
        - Pool size: min(CPU_CORES, total_pages, memory_limit)
        - Ordering: Results re-ordered by page_number
        - Error handling: Failed pages don't crash entire job
        
        Args:
            worker_func: Function to process single page (must be picklable)
            page_args: List of arguments for each page
            total_pages: Total number of pages
            verbose: Whether to print progress
            
        Returns:
            List of results ordered by page number
        """
        
        # Calculate optimal worker count
        max_workers = self.calculate_optimal_workers(total_pages)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Multiprocessing Configuration:")
            print(f"{'='*60}")
            print(f"Total pages: {total_pages}")
            print(f"CPU cores: {cpu_count()}")
            print(f"Workers: {max_workers}")
            mem = psutil.virtual_memory()
            print(f"Available memory: {mem.available / (1024**3):.1f} GB")
            print(f"Expected memory per worker: {self.worker_memory_mb} MB")
            print(f"{'='*60}\n")
        
        results = []
        start_time = time.time()
        
        # Process pages in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all pages
            future_to_page = {
                executor.submit(worker_func, args): args[1]  # args[1] is page_num
                for args in page_args
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if verbose:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / completed
                        remaining = (total_pages - completed) * avg_time
                        print(f"✓ Page {result.page_number}/{total_pages} "
                              f"({completed}/{total_pages}) "
                              f"[{elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining]")
                
                except Exception as e:
                    completed += 1
                    if verbose:
                        print(f"✗ Page {page_num + 1} failed: {str(e)}")
                    
                    # Create error result
                    from rag_models import PageChunk
                    error_result = PageChunk(
                        page_number=page_num + 1,
                        text="",
                        language="UNKNOWN",
                        text_direction="LTR",
                        average_confidence=0.0,
                        word_count=0,
                        char_count=0,
                        status="failed",
                        error=str(e)
                    )
                    results.append(error_result)
        
        # Re-order by page number (important for determinism)
        results.sort(key=lambda x: x.page_number)
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing Complete:")
            print(f"{'='*60}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Average time per page: {total_time/total_pages:.2f}s")
            print(f"Throughput: {total_pages/total_time*60:.1f} pages/min")
            successful = sum(1 for r in results if r.status == 'success')
            print(f"Successful pages: {successful}/{total_pages}")
            print(f"{'='*60}\n")
        
        return results
    
    def get_system_info(self) -> Dict:
        """
        Get current system resource information.
        
        Returns:
            Dictionary with CPU, memory, and disk info
        """
        mem = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            'cpu': {
                'cores': cpu_count(),
                'usage_percent': cpu_percent
            },
            'memory': {
                'total_gb': mem.total / (1024**3),
                'available_gb': mem.available / (1024**3),
                'used_gb': mem.used / (1024**3),
                'percent': mem.percent
            },
            'recommended_workers': self.calculate_optimal_workers(100)  # Estimate
        }


def estimate_processing_time(total_pages: int, 
                            arabic_ratio: float = 0.5,
                            workers: int = 4) -> Dict[str, float]:
    """
    Estimate processing time based on document characteristics.
    
    Assumptions:
    - Arabic pages with EasyOCR: 2.5s per page
    - English pages with Tesseract: 0.7s per page
    - Overhead: 10% for process management
    
    Args:
        total_pages: Total number of pages
        arabic_ratio: Estimated ratio of Arabic pages (0.0 - 1.0)
        workers: Number of parallel workers
        
    Returns:
        Dictionary with time estimates
    """
    
    arabic_pages = int(total_pages * arabic_ratio)
    english_pages = total_pages - arabic_pages
    
    # Sequential time
    sequential_time = (arabic_pages * 2.5) + (english_pages * 0.7)
    
    # Parallel time (with overhead)
    parallel_time = (sequential_time / workers) * 1.1
    
    # Throughput
    throughput = total_pages / parallel_time * 60  # pages per minute
    
    return {
        'sequential_time': round(sequential_time, 1),
        'parallel_time': round(parallel_time, 1),
        'speedup': round(sequential_time / parallel_time, 1),
        'throughput_pages_per_min': round(throughput, 1),
        'arabic_pages': arabic_pages,
        'english_pages': english_pages,
        'workers': workers
    }


# Example usage
if __name__ == "__main__":
    # Test system info
    manager = ProcessPoolManager()
    info = manager.get_system_info()
    
    print("System Information:")
    print(f"CPU Cores: {info['cpu']['cores']}")
    print(f"CPU Usage: {info['cpu']['usage_percent']}%")
    print(f"Total Memory: {info['memory']['total_gb']:.1f} GB")
    print(f"Available Memory: {info['memory']['available_gb']:.1f} GB")
    print(f"Recommended Workers: {info['recommended_workers']}")
    
    # Test time estimation
    print("\nTime Estimation (20-page mixed document):")
    estimate = estimate_processing_time(20, arabic_ratio=0.5, workers=4)
    print(f"Sequential: {estimate['sequential_time']}s")
    print(f"Parallel (4 workers): {estimate['parallel_time']}s")
    print(f"Speedup: {estimate['speedup']}x")
    print(f"Throughput: {estimate['throughput_pages_per_min']} pages/min")
