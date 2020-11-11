# GPU-accelerated DFL viscoacoustic wave equation modeling
It is a python package for DFL viscoacoustic wave equation modeling based on GPU parallelization.  
&emsp;&emsp;The package provides balances between accuracy and efficiency:  
&emsp;&emsp;&emsp;&emsp;**for accuracy**: it provides two-step extrapolation scheme with time-stepping error compensation (2nd-order $k$-space compensators and their simplified version);  
&emsp;&emsp;&emsp;&emsp;**for efficiency**: the package is utilizing CuPy library to perform 3-D FFTs and elementwise computations; it also provides more efficient natural-attenuation ABCs (naABCs) besides the conventional hybrid ABCs.

## The naABC-related functions and classes are separately documented in "gvopt.py", making it convenient to be applied in other viscoacoustic modeling schemes.
