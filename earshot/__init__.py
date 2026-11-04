"""Earshot: a live conversation assistant for deaf and mute users."""
import os
import platform

if platform.system() == "Darwin":
    # On macOS, PyTorch and FAISS each ship their own OpenMP runtime, and loading both
    # aborts the process. One thread keeps the two from running in parallel.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
