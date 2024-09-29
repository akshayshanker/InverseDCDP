# Using Inverse Euler Equations to Solve Multidimensional Discrete-Continuous Dynamic Models: A General Method

This repository contains the code for the paper *Using Inverse Euler Equations to Solve Multidimensional Discrete-Continuous Dynamic Models: A General Method* by Isabella Dobrescu and Akshay Shanker.

## Contents

The repository includes:

- `RFC.RFCSimple`: The roof-top cut (RFC) algorithm that searches for sub-optimal points by examining **each** neighboring point.
- `RFC.RFC`: An optimized version of the RFC algorithm that focuses on **likely regions** where sub-optimal points may occur, potentially reducing runtime when the number of points is exponentially large.
- Applications that implement the results from the paper on inverse constrained Euler equations.

## Abstract

We develop a general inverse Euler equation method to efficiently solve multidimensional stochastic dynamic optimization problems where agents face discrete-continuous choices and occasionally binding constraints. First, we provide a discrete-time analogue of the Hamiltonian to recover the generalized necessary Euler equation. We then establish necessary and sufficient conditions to determine the existence of an Euler equation inverse and the structure of the exogenous computational grid where the inverse is well-defined.

To handle applications with non-convexities arising from discrete choices, we present an off-the-shelf *rooftop-cut* algorithm that guarantees the asymptotic recovery of the optimal policy function under standard economic assumptions. Our approach is demonstrated using two workhorse applications from the literature, showcasing the computational speed and accuracy improvements generated by our method.

## Application 1: 2D Pension Saving and Retirement Choice Model by Dreudhal and Jorgensen (2017)

To run the pension model:

```bash
python3 plotPension.py
```

### Four Constrained Regions

Performance comparison with four constraint regions, as in Dreudhal and Jorgensen (2017).

- **RFC+ Delaunay**: Our RFC implemented with SciPy's [Delaunay triangulation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html) and [LinearNDInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html).
- **G2EGM**: Upper-envelope algorithm using the [Druedhal and Jorgensen (2017)](https://www.sciencedirect.com/science/article/abs/pii/S0165188916301920) approach.

The RFC implementation also removes points that do not satisfy complementary slackness conditions on the exogenous grid, eliminating the need for an upper-envelope step over the constrained regions. 


|                   | RFC+ Delaunay | G2EGM      |
|-------------------|---------------|------------|
| **Total time (min.)** | 2.61          | 2.64       |
| **Euler errors**   |               |            |
| **All (average)**  | -6.660        | -6.572     |
| **5th percentile** | -7.772        | -7.631     |
| **95th percentile**| -4.423        | -4.964     |
| **Median**         | -6.930        | -6.713     |

### Six Constrained Regions

Introducing an additional upper-bound constraint on pension deposits increases the number of constraint regions to six (a 50% increase). 

Under G2EGM, the total number of interpolants increases accordingly, leading to a 50% increase in computation time.

By applying Theorem 1 to eliminate sub-optimal points that violate complementary slackness in the exogenous grid, the number of grid points in the six-region case can be reduced to approximately match that of the four-region case. This is because, in the RFC implementation, the optimal endogenous grid points cover approximately the same area in the endogenous grid space.

|                     | RFC+ Delaunay | G2EGM      |
|---------------------|---------------|------------|
| **Total time (min.)**| 2.32          | 3.73       |
| **Euler errors**     |               |            |
| **All (average)**    | -6.542        | -5.872     |
| **5th percentile**   | -7.855        | -7.593     |
| **95th percentile**  | -3.982        | -1.807     |
| **Median**           | -6.844        | -6.531     |

## Acknowledgements

This application applies the RFC algorithm and our paper's results to an implementation of the pension and retirement choice model originally developed by [Druedhal and Jorgensen (2017)](https://www.sciencedirect.com/science/article/abs/pii/S0165188916301920), available under the MIT License at [this repository](https://github.com/NumEconCopenhagen/ConsumptionSavingNotebooks).

In our project, the [Druedhal and Jorgensen (2017)](https://www.sciencedirect.com/science/article/abs/pii/S0165188916301920) model has been adapted to:

1. Invert the Euler equation according to the necessary conditions described in Dobrescu and Shanker (2024).
2. Implement the rooftop-cut algorithm for comparison with the G2EGM upper-envelope algorithm.

We gratefully acknowledge the [G2EGM](https://github.com/NumEconCopenhagen/ConsumptionSavingNotebooks) modules developed by Druedhal, with modifications for this project noted in the corresponding code files.
