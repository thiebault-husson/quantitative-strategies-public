
import time
import psutil
import logging

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.memory_usage = []
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.memory_usage = []
        
    def record_memory_usage(self):
        """Record current memory usage"""
        if self.start_time is not None:
            memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            self.memory_usage.append(memory)
            
    def get_performance_summary(self):
        """Get performance summary"""
        if self.start_time is None:
            return {}
            
        end_time = time.time()
        duration = end_time - self.start_time
        max_memory = max(self.memory_usage) if self.memory_usage else 0
        
        return {
            'duration_seconds': duration,
            'max_memory_mb': max_memory,
            'memory_samples': len(self.memory_usage)
        }
