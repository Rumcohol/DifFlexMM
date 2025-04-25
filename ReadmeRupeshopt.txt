"""
Detailed Description of run_quads_focusing_k_rot_only.py
=====================================================


Core Classes:
------------
1. RotationalStiffnessForwardProblem:
   - Inherits from ForwardProblem
   - Manages physics simulation of quad structure
   - Focuses on rotational stiffness (k_rot) optimization
   - Maintains fixed geometry while varying stiffness

2. RotationalStiffnessOptimizationProblem:
   - Inherits from OptimizationProblem
   - Controls optimization process
   - Generates wave-like stiffness patterns
   - Manages design variables for k_rot multipliers

Main Components:
--------------
1. Initialization:
   - Creates quad grid (n1_blocks × n2_blocks)
   - Sets mechanical parameters:
     * spacing: Inter-quad distance
     * bond_length: Connection length
     * k_stretch, k_shear: Fixed stiffness values
     * k_rot: Variable rotational stiffness

2. Optimization Strategy:
   - Pattern-based stiffness distribution:
     * Higher stiffness near target region
     * Alternating positive/negative values in outer regions
   - Wave propagation simulation:
     * Input waves from left side
     * Guided propagation toward target
   - Distance-based modulation:
     * Exponential decay in central region
     * Periodic variation in outer areas

Key Parameters:
-------------
Structure Configuration:
- n1_blocks: Width (default: 20)
- n2_blocks: Height (default: 10)
- spacing: Inter-quad distance
- bond_length: Connection length

Target Settings:
- target_size: Focus region dimensions (default: 2×2)
- target_shift: Position offset

Mechanical Properties:
- k_stretch: Fixed stretching stiffness
- k_shear: Fixed shear stiffness
- k_rot: Optimized rotational stiffness
- density: Material density
- damping: System damping coefficients

Pattern Implementation:
--------------------
1. Wave Pattern Design:
   - Concentric patterns around target
   - Central region: Positive stiffness (≈3.0)
   - Outer regions: Oscillating stiffness

2. Bond Management:
   - Separate horizontal/vertical bond handling
   - Distance-based calculations
   - Wave phase implementation

Output Generation:
----------------
1. File Structure:
   - PKL format output
   - Includes:
     * Optimized k_rot multipliers
     * Configuration parameters
     * Optimization results
   - Filename contains grid size and target parameters

2. Visualization Integration:
   - Compatible with visualize_k_rot_optimization.py
   - Generates:
     * Static structure visualizations
     * Wave propagation animations
     * Color-coded stiffness patterns

Usage Example:
------------
def main():
    # Run optimization with default parameters
    result_file = run_optimization(
        n1_blocks=20,          # Width of structure
        n2_blocks=10,          # Height of structure
        target_size=(2, 2),    # Size of target region
        target_shift=(2, 0),   # Target position offset
        max_iterations=50,     # Optimization iterations
        analysis_time=20.0     # Simulation duration
    )
VISUALISATION
- Results can be visualized using visualize_k_rot_optimization.py
- Creates static visualizations and animations
- Shows wave propagation through the structure
- Highlights stiffness patterns with color coding

Note: This implementation is part of a larger metamaterial optimization framework
and requires companion visualization tools for result analysis.


"""