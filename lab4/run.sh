#!/bin/bash
#PBS -l walltime=00:10:00
#PBS -l select=2:ncpus=8:mpiprocs=8:mem=4gb
#PBS -m n

cd $PBS_O_WORKDIR
MPI_NP=$(wc -l $PBS_NODEFILE | awk '{ print $1 }')
echo "Number of MPI processes: $MPI_NP"
echo "Node file:o:"
cat $PBS_NODEFILE
echo
size=2000
iters=100
mpirun -machinefile $PBS_NODEFILE -np 1 ./lab4 $size $size $iters
mpirun -machinefile $PBS_NODEFILE -np 2 ./lab4 $size $size $iters
mpirun -machinefile $PBS_NODEFILE -np 4 ./lab4 $size $size $iters
mpirun -machinefile $PBS_NODEFILE -np 8 ./lab4 $size $size $iters
mpirun -machinefile $PBS_NODEFILE -np 16 ./lab4 $size $size $iters