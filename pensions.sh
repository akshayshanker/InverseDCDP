#!/bin/bash
#PBS -l ncpus=1
#PBS -l mem=5GB
#PBS -l jobfs=400GB
#PBS -q expresssr
#PBS -P tp66
#PBS -l walltime=1:00:00
#PBS -l storage=scratch/pv33+gdata/pv33
#PBS -l wd

module load python3/3.12.1
module load openmpi/4.1.5


mpiexec -n 24 --map-by ppr:6:numa python3 timingPensions.py draft_v6