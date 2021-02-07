# GPU-accelerated DFL viscoacoustic wave equation modeling
It is a python package for DFL viscoacoustic wave equation modeling based on GPU parallelization.  
&emsp;&emsp;The package provides balances between accuracy and efficiency:  
&emsp;&emsp;&emsp;&emsp;**for accuracy**: it provides two-step extrapolation scheme with time-stepping error compensation (2nd-order $k$-space compensators and their simplified version);  
&emsp;&emsp;&emsp;&emsp;**for efficiency**: the package is utilizing CuPy library to perform 3-D FFTs and elementwise computations; it also provides more efficient natural-attenuation ABCs (naABCs) besides the conventional hybrid ABCs.

## The naABC-related functions and classes are separately documented in "gvopt.py", which is readily applicable in other viscoacoustic modeling schemes.

# New branch (hnaABC): we are testing the idea of "adding a few layers of hABC outside the naABCs".
**Logic and expectation**: Since a few layers (<=2) of hABC does not slow down the GPU efficiency much, while they could reduce the outtermost reflectivity from original 1 to <0.2. Such revised hnaABC could significantly improve the absorbing efficiency with similar number of absorbing layers.

**Testing results**: Unfortunately, the hABC implemented outside the naABC tends to be unstable. The reason could be the relatively abrupt change (due to smaller number, as 1 or 2, of hABC layers) from TWWE (with very low-velocity, but very high attenuation (Q~0)) to OWWE. In addition, experiments show that with the outtermost layer reflectivity reduced from 1 to 0.1, the number of naABC layers does not decrease much, since most of the absorbing layers are constructed to fufill the initial $\epsilon_0=\varepsilon/10$ in our methodology.
