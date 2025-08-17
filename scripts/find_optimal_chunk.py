import psutil
import pandas as pd

def suggest_chunk_size():
    # Get available RAM in GB
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    
    # Conservative chunk size recommendations
    if available_ram_gb < 4:
        return 50000, "Low RAM detected"
    elif available_ram_gb < 8:
        return 100000, "Moderate RAM"
    elif available_ram_gb < 16:
        return 250000, "Good RAM"
    else:
        return 500000, "High RAM"

chunk_size, ram_status = suggest_chunk_size()
print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
print(f"Recommended chunk size: {chunk_size:,} rows ({ram_status})")
print(f"Estimated chunks needed: {15891523 // chunk_size + 1}")