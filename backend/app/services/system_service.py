from __future__ import annotations

import os
import platform
import sys
from typing import Any

import torch


class SystemService:
    def get_device_info(self) -> dict[str, Any]:
        cuda_available = torch.cuda.is_available()
        gpu_devices = []
        total_gpu_memory_gb = 0.0
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                memory_gb = round(properties.total_memory / (1024**3), 2)
                total_gpu_memory_gb += memory_gb
                gpu_devices.append(
                    {
                        "index": index,
                        "name": properties.name,
                        "memory_gb": memory_gb,
                        "multi_processor_count": properties.multi_processor_count,
                    }
                )
        gpu_name = None
        if gpu_devices:
            unique_names = sorted({device["name"] for device in gpu_devices})
            if len(unique_names) == 1:
                gpu_name = f"{len(gpu_devices)}x {unique_names[0]}"
            else:
                gpu_name = ", ".join(device["name"] for device in gpu_devices)
        return {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "torch_available": True,
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "gpu_count": len(gpu_devices),
            "gpu_devices": gpu_devices,
            "total_gpu_memory_gb": round(total_gpu_memory_gb, 2),
            "preferred_training_device": "cuda" if cuda_available else "cpu",
        }
