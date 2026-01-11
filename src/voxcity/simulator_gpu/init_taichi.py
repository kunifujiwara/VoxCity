"""
Taichi initialization for simulator_gpu.

This module provides centralized Taichi initialization to ensure ti.init()
is called before any Taichi fields or kernels are used.
"""

import taichi as ti
import os
from typing import Optional

# Track initialization state
_TAICHI_INITIALIZED = False


def init_taichi(
    arch: Optional[str] = None,
    default_fp: type = ti.f32,
    default_ip: type = ti.i32,
    debug: bool = False,
    **kwargs
) -> bool:
    """
    Initialize Taichi runtime if not already initialized.
    
    This function is idempotent - calling it multiple times is safe.
    The first call will initialize Taichi, subsequent calls will be no-ops.
    
    Args:
        arch: Architecture to use. Options:
            - None (default): Auto-detect best available (GPU preferred)
            - 'gpu': Use GPU (CUDA, Vulkan, Metal, etc.)
            - 'cuda': Use CUDA specifically
            - 'vulkan': Use Vulkan
            - 'metal': Use Metal (macOS)
            - 'cpu': Use CPU
        default_fp: Default floating point type (ti.f32 or ti.f64)
        default_ip: Default integer type (ti.i32 or ti.i64)
        debug: Enable debug mode for better error messages
        **kwargs: Additional arguments passed to ti.init()
        
    Returns:
        True if initialization was performed, False if already initialized.
    """
    global _TAICHI_INITIALIZED
    
    if _TAICHI_INITIALIZED:
        return False
    
    # Determine architecture
    if arch is None:
        # Auto-detect: prefer GPU, fall back to CPU
        ti_arch = ti.gpu
    elif arch == 'gpu':
        ti_arch = ti.gpu
    elif arch == 'cuda':
        ti_arch = ti.cuda
    elif arch == 'vulkan':
        ti_arch = ti.vulkan
    elif arch == 'metal':
        ti_arch = ti.metal
    elif arch == 'cpu':
        ti_arch = ti.cpu
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    # Check environment variable for override
    env_arch = os.environ.get('TAICHI_ARCH', '').lower()
    if env_arch == 'cpu':
        ti_arch = ti.cpu
    elif env_arch == 'cuda':
        ti_arch = ti.cuda
    elif env_arch == 'vulkan':
        ti_arch = ti.vulkan
    elif env_arch == 'gpu':
        ti_arch = ti.gpu
    
    # Initialize Taichi
    ti.init(
        arch=ti_arch,
        default_fp=default_fp,
        default_ip=default_ip,
        debug=debug,
        **kwargs
    )
    
    _TAICHI_INITIALIZED = True
    return True


def ensure_initialized():
    """
    Ensure Taichi is initialized with default settings.
    
    This is a convenience function for lazy initialization.
    Call this before any Taichi operations if you're not sure
    whether init_taichi() has been called.
    """
    if not _TAICHI_INITIALIZED:
        init_taichi()


def is_initialized() -> bool:
    """Check if Taichi has been initialized."""
    return _TAICHI_INITIALIZED


def reset():
    """
    Reset initialization state.
    
    Note: This does NOT reset Taichi itself (which cannot be reset once initialized).
    This only resets the tracking flag for testing purposes.
    """
    global _TAICHI_INITIALIZED
    _TAICHI_INITIALIZED = False


# Convenience: expose ti for direct access
taichi = ti

__all__ = [
    'init_taichi',
    'ensure_initialized', 
    'is_initialized',
    'reset',
    'taichi',
    'ti',
]
