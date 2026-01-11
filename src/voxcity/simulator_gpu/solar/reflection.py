"""
Optimized Radiation Computation for GPU

This module provides optimized GPU kernels for radiation computation
that minimize kernel launches and synchronization overhead.

Key optimizations:
1. Fused kernels - combine multiple operations into single kernel launches
2. Reduced synchronization - batch operations to minimize ti.sync() calls
3. Better memory access patterns - coalesced memory access
4. Reduced atomic operations - use local accumulation where possible
"""

import taichi as ti
import numpy as np
from typing import Optional

# Vector type
Vector3 = ti.math.vec3


@ti.data_oriented
class OptimizedReflectionSolver:
    """
    Optimized GPU solver for multi-bounce radiation reflections.
    
    This replaces the per-step kernel launches with fused operations
    that run the entire reflection loop on GPU without CPU intervention.
    """
    
    def __init__(
        self,
        n_surfaces: int,
        max_svf_entries: int,
        n_reflection_steps: int = 3
    ):
        self.n_surfaces = n_surfaces
        self.max_svf_entries = max_svf_entries
        self.n_reflection_steps = n_reflection_steps
        
        # Preallocate ping-pong buffers for reflection iterations
        # This avoids memory allocation during computation
        self._surfins_a = ti.field(dtype=ti.f32, shape=(n_surfaces,))
        self._surfins_b = ti.field(dtype=ti.f32, shape=(n_surfaces,))
        self._surfout = ti.field(dtype=ti.f32, shape=(n_surfaces,))
        
        # Accumulated totals
        self._total_incoming = ti.field(dtype=ti.f32, shape=(n_surfaces,))
        self._total_outgoing = ti.field(dtype=ti.f32, shape=(n_surfaces,))
    
    @ti.kernel
    def solve_reflections_fused(
        self,
        # Initial radiation
        initial_sw: ti.template(),
        # Surface properties
        albedo: ti.template(),
        svf: ti.template(),
        # Cached SVF matrix (sparse COO)
        svf_source: ti.template(),
        svf_target: ti.template(),
        svf_vf: ti.template(),
        svf_trans: ti.template(),
        svf_nnz: ti.i32,
        # Number of reflection steps
        n_steps: ti.i32
    ):
        """
        Fused kernel for complete multi-bounce reflection computation.
        
        This runs all reflection iterations in a single kernel launch,
        eliminating CPU-GPU synchronization overhead between steps.
        
        Uses ping-pong buffers to avoid race conditions between iterations.
        """
        n_surf = self.n_surfaces
        
        # Initialize - copy initial radiation to buffer A and totals
        for i in range(n_surf):
            self._surfins_a[i] = initial_sw[i]
            self._total_incoming[i] = initial_sw[i]
            self._total_outgoing[i] = 0.0
        
        # Sync after initialization
        ti.sync()
        
        # Reflection loop - alternate between buffers A and B
        for step in range(n_steps):
            # Determine which buffer is input and which is output
            use_a_as_input = (step % 2 == 0)
            
            # Phase 1: Compute outgoing = albedo * incoming
            for i in range(n_surf):
                if use_a_as_input:
                    self._surfout[i] = albedo[i] * self._surfins_a[i]
                else:
                    self._surfout[i] = albedo[i] * self._surfins_b[i]
                self._total_outgoing[i] += self._surfout[i]
            
            ti.sync()
            
            # Phase 2: Reset output buffer
            for i in range(n_surf):
                if use_a_as_input:
                    self._surfins_b[i] = 0.0
                else:
                    self._surfins_a[i] = 0.0
            
            ti.sync()
            
            # Phase 3: Sparse matrix-vector multiply for reflection distribution
            for idx in range(svf_nnz):
                source = svf_source[idx]
                target = svf_target[idx]
                vf = svf_vf[idx]
                trans = svf_trans[idx]
                
                outgoing = self._surfout[source]
                if outgoing > 0.01:
                    contribution = outgoing * vf * trans
                    if use_a_as_input:
                        ti.atomic_add(self._surfins_b[target], contribution)
                    else:
                        ti.atomic_add(self._surfins_a[target], contribution)
            
            ti.sync()
            
            # Phase 4: Apply urban view factor scaling and accumulate
            for i in range(n_surf):
                urban_vf = 1.0 - svf[i]
                if urban_vf < 0.01:
                    if use_a_as_input:
                        self._surfins_b[i] = 0.0
                    else:
                        self._surfins_a[i] = 0.0
                else:
                    if use_a_as_input:
                        self._surfins_b[i] *= urban_vf
                        self._total_incoming[i] += self._surfins_b[i]
                    else:
                        self._surfins_a[i] *= urban_vf
                        self._total_incoming[i] += self._surfins_a[i]
            
            ti.sync()
    
    def get_results(self):
        """Get accumulated totals as numpy arrays."""
        return {
            'total_incoming': self._total_incoming.to_numpy(),
            'total_outgoing': self._total_outgoing.to_numpy()
        }


@ti.kernel
def fused_reflection_step_kernel(
    # Current incoming radiation (input)
    surfins_in: ti.template(),
    # Next incoming radiation (output, will be accumulated)
    surfins_out: ti.template(),
    # Outgoing buffer (temporary)
    surfout: ti.template(),
    # Surface properties
    albedo: ti.template(),
    svf: ti.template(),
    # Accumulated totals
    total_incoming: ti.template(),
    total_outgoing: ti.template(),
    # Cached SVF matrix
    svf_source: ti.template(),
    svf_target: ti.template(),
    svf_vf: ti.template(),
    svf_trans: ti.template(),
    svf_nnz: ti.i32,
    n_surfaces: ti.i32
):
    """
    Single fused kernel for one reflection step.
    
    Combines: outgoing computation + distribution + accumulation
    into fewer synchronization points.
    """
    # Phase 1: Compute outgoing and reset output buffer
    for i in range(n_surfaces):
        out = albedo[i] * surfins_in[i]
        surfout[i] = out
        total_outgoing[i] += out
        surfins_out[i] = 0.0
    
    ti.sync()
    
    # Phase 2: Sparse matrix-vector multiply
    for idx in range(svf_nnz):
        source = svf_source[idx]
        target = svf_target[idx]
        vf = svf_vf[idx]
        trans = svf_trans[idx]
        
        outgoing = surfout[source]
        if outgoing > 0.01:
            ti.atomic_add(surfins_out[target], outgoing * vf * trans)
    
    ti.sync()
    
    # Phase 3: Apply urban VF scaling and accumulate
    for i in range(n_surfaces):
        urban_vf = 1.0 - svf[i]
        if urban_vf < 0.01:
            surfins_out[i] = 0.0
        else:
            surfins_out[i] *= urban_vf
            total_incoming[i] += surfins_out[i]


@ti.kernel
def compute_initial_and_reflections_fused(
    # Surface properties
    surf_direction: ti.template(),
    surf_svf: ti.template(),
    surf_shadow: ti.template(),
    surf_canopy_trans: ti.template(),
    surf_albedo: ti.template(),
    surf_normal: ti.template(),
    # Sun properties
    sun_dir_x: ti.f32,
    sun_dir_y: ti.f32,
    sun_dir_z: ti.f32,
    cos_zenith: ti.f32,
    # Radiation inputs
    sw_direct: ti.f32,
    sw_diffuse: ti.f32,
    # SVF matrix
    svf_source: ti.template(),
    svf_target: ti.template(),
    svf_vf: ti.template(),
    svf_trans: ti.template(),
    svf_nnz: ti.i32,
    # Number of surfaces and reflection steps
    n_surfaces: ti.i32,
    n_ref_steps: ti.i32,
    # Output arrays (preallocated)
    sw_in_direct: ti.template(),
    sw_in_diffuse: ti.template(),
    sw_in_reflected: ti.template(),
    sw_out_total: ti.template(),
    # Temporary buffers (ping-pong)
    surfins_a: ti.template(),
    surfins_b: ti.template(),
    surfout: ti.template()
):
    """
    Fully fused kernel: initial radiation + all reflection iterations.
    
    This is the most optimized version that runs everything in one kernel.
    """
    min_stable_coszen = 0.0262
    
    # ========== Phase 1: Initial radiation pass ==========
    for i in range(n_surfaces):
        direction = surf_direction[i]
        svf_val = surf_svf[i]
        shadow = surf_shadow[i]
        canopy_trans = surf_canopy_trans[i]
        
        # Get surface normal
        normal_x, normal_y, normal_z = 0.0, 0.0, 0.0
        if direction == 0:  # Up
            normal_z = 1.0
        elif direction == 1:  # Down
            normal_z = -1.0
        elif direction == 2:  # North
            normal_y = 1.0
        elif direction == 3:  # South
            normal_y = -1.0
        elif direction == 4:  # East
            normal_x = 1.0
        elif direction == 5:  # West
            normal_x = -1.0
        
        # Cosine of incidence
        cos_inc = sun_dir_x * normal_x + sun_dir_y * normal_y + sun_dir_z * normal_z
        cos_inc = ti.max(0.0, cos_inc)
        
        # Direct radiation
        sw_dir = 0.0
        if cos_zenith > min_stable_coszen and shadow < 0.5:
            sw_dir = sw_direct * cos_inc * canopy_trans
        
        # Diffuse radiation
        sw_dif = 0.0
        if direction != 1:  # Not downward
            sw_dif = sw_diffuse * svf_val
        
        # Store results
        sw_in_direct[i] = sw_dir
        sw_in_diffuse[i] = sw_dif
        sw_in_reflected[i] = 0.0
        sw_out_total[i] = 0.0
        
        # Initialize reflection buffer
        surfins_a[i] = sw_dir + sw_dif
    
    ti.sync()
    
    # ========== Phase 2: Reflection iterations ==========
    for step in range(n_ref_steps):
        use_a = (step % 2 == 0)
        
        # Compute outgoing and reset next buffer
        for i in range(n_surfaces):
            if use_a:
                surfout[i] = surf_albedo[i] * surfins_a[i]
                surfins_b[i] = 0.0
            else:
                surfout[i] = surf_albedo[i] * surfins_b[i]
                surfins_a[i] = 0.0
            sw_out_total[i] += surfout[i]
        
        ti.sync()
        
        # Sparse matmul for reflection distribution
        for idx in range(svf_nnz):
            src = svf_source[idx]
            tgt = svf_target[idx]
            vf = svf_vf[idx]
            trans = svf_trans[idx]
            
            out_val = surfout[src]
            if out_val > 0.01:
                contrib = out_val * vf * trans
                if use_a:
                    ti.atomic_add(surfins_b[tgt], contrib)
                else:
                    ti.atomic_add(surfins_a[tgt], contrib)
        
        ti.sync()
        
        # Apply urban VF and accumulate to reflected
        for i in range(n_surfaces):
            urban_vf = 1.0 - surf_svf[i]
            if urban_vf < 0.01:
                if use_a:
                    surfins_b[i] = 0.0
                else:
                    surfins_a[i] = 0.0
            else:
                if use_a:
                    surfins_b[i] *= urban_vf
                    sw_in_reflected[i] += surfins_b[i]
                else:
                    surfins_a[i] *= urban_vf
                    sw_in_reflected[i] += surfins_a[i]
        
        ti.sync()


def benchmark_reflections():
    """Benchmark the reflection solver."""
    import time
    
    print("Creating test data...")
    n_surfaces = 10000
    svf_nnz = 500000  # 5% sparse
    n_ref_steps = 3
    
    # Create test arrays
    initial_sw = ti.field(dtype=ti.f32, shape=(n_surfaces,))
    albedo = ti.field(dtype=ti.f32, shape=(n_surfaces,))
    svf = ti.field(dtype=ti.f32, shape=(n_surfaces,))
    
    svf_source = ti.field(dtype=ti.i32, shape=(svf_nnz,))
    svf_target = ti.field(dtype=ti.i32, shape=(svf_nnz,))
    svf_vf = ti.field(dtype=ti.f32, shape=(svf_nnz,))
    svf_trans = ti.field(dtype=ti.f32, shape=(svf_nnz,))
    
    # Output buffers
    surfins_a = ti.field(dtype=ti.f32, shape=(n_surfaces,))
    surfins_b = ti.field(dtype=ti.f32, shape=(n_surfaces,))
    surfout = ti.field(dtype=ti.f32, shape=(n_surfaces,))
    total_incoming = ti.field(dtype=ti.f32, shape=(n_surfaces,))
    total_outgoing = ti.field(dtype=ti.f32, shape=(n_surfaces,))
    
    # Initialize with random data
    np.random.seed(42)
    initial_sw.from_numpy(np.random.rand(n_surfaces).astype(np.float32) * 500)
    albedo.from_numpy(np.random.rand(n_surfaces).astype(np.float32) * 0.3 + 0.1)
    svf.from_numpy(np.random.rand(n_surfaces).astype(np.float32) * 0.5 + 0.3)
    
    svf_source.from_numpy(np.random.randint(0, n_surfaces, svf_nnz).astype(np.int32))
    svf_target.from_numpy(np.random.randint(0, n_surfaces, svf_nnz).astype(np.int32))
    svf_vf.from_numpy(np.random.rand(svf_nnz).astype(np.float32) * 0.1)
    svf_trans.from_numpy(np.random.rand(svf_nnz).astype(np.float32) * 0.5 + 0.5)
    
    # Warmup with separate kernel approach
    print("Warming up...")
    
    @ti.kernel
    def init_step(initial: ti.template(), ins_a: ti.template(), tot_in: ti.template(), tot_out: ti.template(), n: ti.i32):
        for i in range(n):
            ins_a[i] = initial[i]
            tot_in[i] = initial[i]
            tot_out[i] = 0.0
    
    @ti.kernel 
    def compute_outgoing_step(ins: ti.template(), out: ti.template(), alb: ti.template(), tot_out: ti.template(), n: ti.i32):
        for i in range(n):
            o = alb[i] * ins[i]
            out[i] = o
            tot_out[i] += o
    
    @ti.kernel
    def reset_buffer(buf: ti.template(), n: ti.i32):
        for i in range(n):
            buf[i] = 0.0
    
    @ti.kernel
    def sparse_matmul_step(
        out: ti.template(),
        ins_next: ti.template(),
        src: ti.template(),
        tgt: ti.template(),
        vf: ti.template(),
        trans: ti.template(),
        nnz: ti.i32
    ):
        for idx in range(nnz):
            s = src[idx]
            t = tgt[idx]
            v = vf[idx]
            tr = trans[idx]
            o = out[s]
            if o > 0.01:
                ti.atomic_add(ins_next[t], o * v * tr)
    
    @ti.kernel
    def scale_and_accumulate(ins: ti.template(), svf_arr: ti.template(), tot_in: ti.template(), n: ti.i32):
        for i in range(n):
            urban_vf = 1.0 - svf_arr[i]
            if urban_vf < 0.01:
                ins[i] = 0.0
            else:
                ins[i] *= urban_vf
                tot_in[i] += ins[i]
    
    # Warmup run
    init_step(initial_sw, surfins_a, total_incoming, total_outgoing, n_surfaces)
    for step in range(n_ref_steps):
        if step % 2 == 0:
            compute_outgoing_step(surfins_a, surfout, albedo, total_outgoing, n_surfaces)
            reset_buffer(surfins_b, n_surfaces)
            sparse_matmul_step(surfout, surfins_b, svf_source, svf_target, svf_vf, svf_trans, svf_nnz)
            scale_and_accumulate(surfins_b, svf, total_incoming, n_surfaces)
        else:
            compute_outgoing_step(surfins_b, surfout, albedo, total_outgoing, n_surfaces)
            reset_buffer(surfins_a, n_surfaces)
            sparse_matmul_step(surfout, surfins_a, svf_source, svf_target, svf_vf, svf_trans, svf_nnz)
            scale_and_accumulate(surfins_a, svf, total_incoming, n_surfaces)
    ti.sync()
    
    # Benchmark with separate kernel launches (like current implementation)
    print(f"\nBenchmarking SEPARATE KERNELS ({n_ref_steps} reflection steps)...")
    n_iterations = 20
    times_separate = []
    
    for i in range(n_iterations):
        t0 = time.perf_counter()
        init_step(initial_sw, surfins_a, total_incoming, total_outgoing, n_surfaces)
        for step in range(n_ref_steps):
            if step % 2 == 0:
                compute_outgoing_step(surfins_a, surfout, albedo, total_outgoing, n_surfaces)
                reset_buffer(surfins_b, n_surfaces)
                sparse_matmul_step(surfout, surfins_b, svf_source, svf_target, svf_vf, svf_trans, svf_nnz)
                scale_and_accumulate(surfins_b, svf, total_incoming, n_surfaces)
            else:
                compute_outgoing_step(surfins_b, surfout, albedo, total_outgoing, n_surfaces)
                reset_buffer(surfins_a, n_surfaces)
                sparse_matmul_step(surfout, surfins_a, svf_source, svf_target, svf_vf, svf_trans, svf_nnz)
                scale_and_accumulate(surfins_a, svf, total_incoming, n_surfaces)
        ti.sync()
        times_separate.append(time.perf_counter() - t0)
    
    mean_sep = np.mean(times_separate) * 1000
    min_sep = np.min(times_separate) * 1000
    print(f"  Mean time: {mean_sep:.2f}ms")
    print(f"  Min time: {min_sep:.2f}ms")
    
    # Compare with fused version
    print(f"\nBenchmarking FUSED KERNEL ({n_ref_steps} reflection steps)...")
    solver = OptimizedReflectionSolver(n_surfaces, svf_nnz, n_ref_steps)
    
    # Warmup fused
    solver.solve_reflections_fused(
        initial_sw, albedo, svf,
        svf_source, svf_target, svf_vf, svf_trans,
        svf_nnz, n_ref_steps
    )
    ti.sync()
    
    times_fused = []
    for i in range(n_iterations):
        t0 = time.perf_counter()
        solver.solve_reflections_fused(
            initial_sw, albedo, svf,
            svf_source, svf_target, svf_vf, svf_trans,
            svf_nnz, n_ref_steps
        )
        ti.sync()
        times_fused.append(time.perf_counter() - t0)
    
    mean_fused = np.mean(times_fused) * 1000
    min_fused = np.min(times_fused) * 1000
    print(f"  Mean time: {mean_fused:.2f}ms")
    print(f"  Min time: {min_fused:.2f}ms")
    
    print(f"\n  Surfaces: {n_surfaces}, SVF entries: {svf_nnz}")
    print(f"\n  Comparison: Separate={min_sep:.2f}ms, Fused={min_fused:.2f}ms")
    if min_fused < min_sep:
        print(f"  Fused is {min_sep/min_fused:.2f}x faster")
    else:
        print(f"  Separate is {min_fused/min_sep:.2f}x faster")
    
    return times_separate, times_fused


if __name__ == "__main__":
    # Test with GPU
    print("="*60)
    print("Testing Reflection Solver on GPU")
    print("="*60)
    ti.init(arch=ti.gpu, default_fp=ti.f32)
    gpu_times = benchmark_reflections()
    
    # Note: Can't reinitialize Taichi in same process for CPU comparison
    print("\nNote: To compare with CPU, run with ti.init(arch=ti.cpu)")
