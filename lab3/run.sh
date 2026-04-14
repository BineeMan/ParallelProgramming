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

mpirun -machinefile $PBS_NODEFILE -np 16 ./lab3 2048 2048 2048 2048 0 4 4
mpirun -machinefile $PBS_NODEFILE -np 16 ./lab3 4096 1024 1024 1024 0 8 2
mpirun -machinefile $PBS_NODEFILE -np 16 ./lab3 1024 1024 1024 4096 0 2 8
mpirun -machinefile $PBS_NODEFILE -np 16 ./lab3 1024 4096 4096 1024 0 4 4
mpirun -machinefile $PBS_NODEFILE -np 16 ./lab3 2048 2048 2048 2048 0 16 1
mpirun -machinefile $PBS_NODEFILE -np 16 ./lab3 2048 2048 2048 2048 0 1 16
mpirun -machinefile $PBS_NODEFILE -np 16 ./lab3 4096 4096 4096 4096 0 4 4
mpirun -machinefile $PBS_NODEFILE -np 9 ./lab3 1800 1800 1800 1800 0 3 3
mpirun -machinefile $PBS_NODEFILE -np 16 ./lab3 4096 2048 2048 1024 0 8 2

mpirun -machinefile $PBS_NODEFILE -np 1  ./lab3 2400 2400 2400 2400 1 0 0
mpirun -machinefile $PBS_NODEFILE -np 4  ./lab3 2400 2400 2400 2400 1 0 0
mpirun -machinefile $PBS_NODEFILE -np 12 ./lab3 2400 2400 2400 2400 1 0 0
mpirun -machinefile $PBS_NODEFILE -np 16 ./lab3 2400 2400 2400 2400 1 0 0