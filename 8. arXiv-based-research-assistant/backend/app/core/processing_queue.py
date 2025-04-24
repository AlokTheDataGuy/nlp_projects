import logging
import threading
import queue
import time
from typing import Dict, Any, Callable, List
import psutil

logger = logging.getLogger(__name__)

class ResourceMonitor:
    def __init__(self, ram_threshold_gb: float = 2.0, gpu_threshold_gb: float = 1.0):
        """
        Initialize the resource monitor.
        
        Args:
            ram_threshold_gb: RAM threshold in GB
            gpu_threshold_gb: GPU memory threshold in GB
        """
        self.ram_threshold_gb = ram_threshold_gb
        self.gpu_threshold_gb = gpu_threshold_gb
    
    def check_resources(self) -> Dict[str, Any]:
        """
        Check system resources.
        
        Returns:
            Dictionary with resource information
        """
        # Check system RAM
        available_ram = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
        
        # Check GPU memory if available
        gpu_usage = None
        total_gpu = None
        available_gpu = None
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
                total_gpu = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
                available_gpu = total_gpu - gpu_usage
        except (ImportError, Exception):
            pass
        
        # Determine if resources are constrained
        ram_constrained = available_ram < self.ram_threshold_gb
        gpu_constrained = available_gpu is not None and available_gpu < self.gpu_threshold_gb
        
        return {
            "available_ram_gb": available_ram,
            "gpu_usage_gb": gpu_usage,
            "total_gpu_gb": total_gpu,
            "available_gpu_gb": available_gpu,
            "ram_constrained": ram_constrained,
            "gpu_constrained": gpu_constrained,
            "is_constrained": ram_constrained or gpu_constrained
        }

# Create a resource monitor instance
resource_monitor = ResourceMonitor()

class ProcessingQueue:
    def __init__(self):
        """
        Initialize the processing queue.
        """
        self.queue = queue.Queue()
        self.running = False
        self.worker_thread = None
    
    def start(self):
        """
        Start the processing queue worker.
        """
        if self.running:
            logger.warning("Processing queue is already running")
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        logger.info("Processing queue started")
    
    def stop(self):
        """
        Stop the processing queue worker.
        """
        if not self.running:
            logger.warning("Processing queue is not running")
            return
        
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        
        logger.info("Processing queue stopped")
    
    def add_task(self, task_type: str, task_func: Callable, task_args: Dict[str, Any]):
        """
        Add a task to the processing queue.
        
        Args:
            task_type: Type of task
            task_func: Function to execute
            task_args: Arguments for the function
        """
        self.queue.put({
            'type': task_type,
            'func': task_func,
            'args': task_args,
            'added_time': time.time()
        })
        
        logger.info(f"Added {task_type} task to processing queue")
    
    def _worker(self):
        """
        Worker thread function.
        """
        while self.running:
            try:
                # Check if there are tasks in the queue
                if self.queue.empty():
                    time.sleep(1.0)
                    continue
                
                # Check system resources
                resources = resource_monitor.check_resources()
                if resources['is_constrained']:
                    logger.warning("System resources constrained, waiting before processing next task")
                    time.sleep(5.0)
                    continue
                
                # Get the next task
                task = self.queue.get(block=False)
                
                # Process the task
                logger.info(f"Processing {task['type']} task")
                try:
                    task['func'](**task['args'])
                    logger.info(f"Completed {task['type']} task")
                except Exception as e:
                    logger.error(f"Error processing {task['type']} task: {e}")
                
                # Mark the task as done
                self.queue.task_done()
                
            except queue.Empty:
                # No tasks in the queue
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"Error in processing queue worker: {e}")
                time.sleep(1.0)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get the status of the processing queue.
        
        Returns:
            Dictionary with queue status
        """
        return {
            'running': self.running,
            'queue_size': self.queue.qsize(),
            'resources': resource_monitor.check_resources()
        }

# Create a singleton instance
processing_queue = ProcessingQueue()
