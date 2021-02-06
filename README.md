# GPU-accelerated DFL viscoacoustic wave equation modeling
It is a python package for DFL viscoacoustic wave equation modeling based on GPU parallelization.  
&emsp;&emsp;The package provides balances between accuracy and efficiency:  
&emsp;&emsp;&emsp;&emsp;**for accuracy**: it provides two-step extrapolation scheme with time-stepping error compensation (2nd-order $k$-space compensators and their simplified version);  
&emsp;&emsp;&emsp;&emsp;**for efficiency**: the package is utilizing CuPy library to perform 3-D FFTs and elementwise computations; it also provides more efficient natural-attenuation ABCs (naABCs) besides the conventional hybrid ABCs.

## The naABC-related functions and classes are separately documented in "gvopt.py", which is readily applicable in other viscoacoustic modeling schemes.

# New branch (hnaABC): we are testing the idea of "adding one layer OWWE ABC outside the naABCs".
Since a single layer of OWWE ABC does not influence the GPU efficiency much, and a single layer of the OWWE ABC could reduce the outtermost reflectivity to typically 0.1. Revise the naABC accordingly could significantly improve the absorbing efficiency with similar number of absorbing layers.
