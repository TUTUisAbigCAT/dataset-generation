"""Host BLAS/OpenMP startup policy; deliberately imports no numerical library.

Configure before importing NumPy/PyTorch: BLIS reads its environment during
initialization. Explicit loop settings also cover older BLIS versions without
BLIS_THREAD_IMPL support. Model training and sampling remain CUDA-only.
"""
from __future__ import annotations

import os

BLIS_ENVIRONMENT = {
    'BLIS_THREAD_IMPL': 'single',
    'BLIS_NUM_THREADS': '1',
    'BLIS_JC_NT': '1',
    'BLIS_PC_NT': '1',
    'BLIS_IC_NT': '1',
    'BLIS_JR_NT': '1',
    'BLIS_IR_NT': '1',
    'OMP_DYNAMIC': 'FALSE',
    'MKL_DYNAMIC': 'FALSE',
    'PYTHONFAULTHANDLER': '1',
}


def configure_host_threads(threads=None, environ=None):
    """Apply before numerical imports; None preserves the caller's OMP/MKL count."""
    if threads is not None and (type(threads) is not int or threads < 1):
        raise ValueError('Host thread count must be a positive integer')
    environment = os.environ if environ is None else environ
    environment.update(BLIS_ENVIRONMENT)
    if threads is not None:
        environment['OMP_NUM_THREADS'] = str(threads)
        environment['MKL_NUM_THREADS'] = str(threads)
    keys = (*BLIS_ENVIRONMENT, 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
    return dict(policy='single_thread_blis', requested_host_threads=threads,
                environment={key: environment.get(key) for key in keys})
