# Discrete-continuous High Dimensional Dynamic Optimization
Repository for `Discrete-continuous High Dimensional Dynamic Optimization' by Dobrescu and Shanker (2024)


## Benchmark application: pension saving and retirement by Dreudhal and Jorgensen (2017)

With four constraints regions as in Dreudhal and Jorgensen (2017). 

```
python3 plotPension.py
``` 

<center>

|                   | RFC+ Delaunay | G2EGM |
|-------------------|---------------|-------|
| Total time (min)  | 2.61          | 2.64  |
| Euler errors      |               |       |
| All (average)     | -6.660        |-6.572 |
| 5th percentile    | -7.772        | -7.631|
| 95th percentile   | -4.423        | -4.964|
| Median            | -6.930        | -6.713|

</center>


Additional constraint upper-bound on pension deposits and six constrained regions (50\% increase in constrained regions).

Under G2EGM, number of total interpolants over which to calculate grid points is now six, leading to 50% increase in time. 

Using Theorem 1 to eliminate sub-optimal points, the number of grid points in the six region case can be reduced to be approximately equal to the number of points in the four region case; this is because in the RFC implmentation, the optimal endogenous grid points cover approximately the same measure in the endogenous grid space. 

|                     | RFC+ Delaunay | G2EGM      |
|---------------------|---------------|------------|
| All (average)       | -6.231        | -5.130     |
| 5th percentile      | -7.467        | -7.149     |
| 95th percentile     | -4.156        | -1.433     |
| Median              | -6.472        | -6.103     |
| Total time (min)    | 1.07          | 1.67       |
| Inversion time (min)| 0.08          | 0.00       |
| RFC time (min)      | 0.29          | 0.00       |


