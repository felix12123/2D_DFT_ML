import math
import numpy as np
import json
import matplotlib.pyplot as plt

def load_potential_params(json_file):
    """
    Load potential parameters from a JSON file and fill in default values for missing keys.
    
    Parameters:
    json_file (str): Path to the JSON file
    
    Returns:
    dict: Dictionary with all necessary parameters, using defaults for missing values
    """
    # Define default values for all possible parameters
    try:
        # Load the JSON file
        with open(json_file, 'r') as f:
            params = json.load(f)
        
        num_plat = len(params['plat_heights'])
        defaults = {
        "L": 10.0,                    # System size
        "dx": 0.01,                   # Grid spacing
        "mu": 0.0,                    # Chemical potential
        "beta": 1.0,                  # Inverse temperature
        "wall": "n",                  # Wall type (n: none, b: box, w: walls)
        "wall_thickness": 0.0,        # Thickness of walls
        
        # Sinusoidal components
        "amplitudes": [],          # Default single wave
        "periods_x": [],
        "periods_y": [],
        "phases": [],
        
        # Plateau components
        "plat_heights": [],           # Empty by default
        "plat_position_x": [],
        "plat_position_y": [],
        "plat_size_x": [],
        "plat_size_y": [],
        "plat_phi": np.full(num_plat, 0).tolist(),
        "plat_theta": np.full(num_plat, np.pi/2).tolist(),
    }
    
        
        # Update defaults with provided values
        for key, default_value in defaults.items():
            if key not in params:
                params[key] = default_value
                
                
        # Verify that array lengths match within components
        sin_arrays = ['amplitudes', 'periods_x', 'periods_y', 'phases']
        sin_lengths = [len(params[key]) for key in sin_arrays]
        if len(set(sin_lengths)) > 1:
            raise ValueError(f"Sinusoidal component arrays must have same length. Found lengths: {dict(zip(sin_arrays, sin_lengths))}")
            
        plat_arrays = ['plat_heights', 'plat_phi', 'plat_position_x', 'plat_position_y',
                      'plat_size_x', 'plat_size_y', 'plat_theta']
        plat_lengths = [len(params[key]) for key in plat_arrays]
        if len(set(plat_lengths)) > 1:
            raise ValueError(f"Plateau component arrays must have same length. Found lengths: {dict(zip(plat_arrays, plat_lengths))}")
            
        return params
        
    except FileNotFoundError:
        raise KeyError(f"Warning: File '{json_file}' not found. Using all default values.")
        # return defaults
    except json.JSONDecodeError:
        raise KeyError(f"Warning: File '{json_file}' is not a valid JSON file.")
    except Exception as e:
        raise KeyError(f"Warning: An error occurred while loading the JSON file: {e}")


def adjust_periodic(value, L):
    """Adjust value for periodic boundary conditions."""
    value = (value + L/2) % L
    value[value < 0] += L
    return value - L/2

def in_par_helper(v1_x:float, v1_y:float, v2_x:float, v2_y:float, dx:np.ndarray, dy:np.ndarray):
    """Helper function to check if points are inside a parallelogram."""
    assert dx.shape == dy.shape
    
    det:float = v1_x * v2_y - v2_x * v1_y
    a:np.ndarray = v2_y * dx - v2_x * dy
    b:np.ndarray = -v1_y * dx + v1_x * dy

    if det > 0:
        return np.logical_and(np.logical_and(0 <= a, a <= det), 
                              np.logical_and(0 <= b, b <= det))
    else:
        return np.logical_and(np.logical_and(det <= a, a <= 0), 
                              np.logical_and(det <= b, b <= 0))


def point_in_parallelogram(L, pos_x, pos_y, phi, theta, size_x, size_y, x, y):
    """Check if point (x,y) is inside the parallelogram."""
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    
    dx = adjust_periodic(x - pos_x, L)
    dy = adjust_periodic(y - pos_y, L)

    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    cos_phi_theta = math.cos(phi + theta)
    sin_phi_theta = math.sin(phi + theta)

    v1_x = size_x * cos_phi
    v1_y = size_x * sin_phi
    v2_x = size_y * cos_phi_theta
    v2_y = size_y * sin_phi_theta

    return in_par_helper(v1_x, v1_y, v2_x, v2_y, dx, dy)

def eval_wall_component(wall_type, thickness, L, x, y):
    """Evaluate wall component of the potential."""
    if wall_type == 'b':
        mask = (x < thickness) | (x > L - thickness) | (y < thickness) | (y > L - thickness)
    elif wall_type == 'w':
        mask = (x < thickness) | (x > L - thickness)
    else:
        mask = np.zeros_like(x, dtype=bool)
    
    return mask


def evaluate_potential(x, y, params):
    """
    Evaluate the potential at position (x,y) given the parameters from the JSON file.
    
    params should be the loaded JSON dictionary containing:
    - L: system size
    - wall: wall type ('n', 'b', or 'w')
    - wall_thickness: thickness of the wall
    - amplitudes: list of sine wave amplitudes
    - periods_x, periods_y: lists of sine wave periods
    - phases: list of sine wave phases
    - plat_heights, plat_phi, plat_position_x, plat_position_y, 
      plat_size_x, plat_size_y, plat_theta: lists of plateau parameters
    """
    assert isinstance(x, np.ndarray), "x must be a numpy array"
    assert isinstance(y, np.ndarray), "x must be a numpy array"
    assert len(x) == len(y), "x and y must have the same length"
    # Wrap coordinates for periodic boundary conditions
    L = params['L']
    y = y % L
    x = x % L

    # Check wall
    wall_mask = eval_wall_component(params['wall'], params['wall_thickness'], L, y, x)

    potential = np.zeros_like(x, dtype=float)
    potential[wall_mask] = np.inf

    # Sum all sinusoidal components
    for i in range(len(params['amplitudes'])):
        potential += params['amplitudes'][i] * np.sin(
            2 * np.pi * params['periods_x'][i] / L * y + 
            2 * np.pi * params['periods_y'][i] / L * x + 
            params['phases'][i]
        )

    # Add all plateau components
    for i in range(len(params['plat_heights'])):
        plat_mask = point_in_parallelogram(
            L,
            params['plat_position_x'][i],
            params['plat_position_y'][i],
            params['plat_phi'][i],
            params['plat_theta'][i],
            params['plat_size_x'][i],
            params['plat_size_y'][i],
            y, x
        )
        potential += params['plat_heights'][i] * plat_mask

    return potential


# Example of how to use:
"""
# Load parameters
params = load_potential_params('potential.json')

# Evaluate potential at a point
potential = evaluate_potential(2.5, 3.5, params)
"""