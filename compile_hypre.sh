#!/bin/bash

source /opt/nec/ve/nlc/2.3.0/bin/nlcvars.sh
source /opt/nec/ve/mpi/2.18.0/bin/necmpivars.sh

HYPRE_ROOT=`pwd`
PREFIX=`pwd`/../build
COMPILER_ROOT=/opt/nec/ve/mpi/2.18.0/bin64

FC_FLAGS="-fopenmp -O2 "
CC_EXTRAFLAGS=" -fopenmp  -O2 -report-all -fdiag-vector=3  "
DEBUG_FLAGS="-g -traceback=verbose "
INLINE_FLAGS="-finline-functions -finline-max-depth=3 -finline-max-function-size=500 "
# INLINE_FLAGS="-finline-functions -finline-max-depth=3 -finline-max-function-size=500 -finline-directory=${HYPRE_ROOT}/src/FEI_mv:${HYPRE_ROOT}/src/FEI_mv/fei-hypre:${HYPRE_ROOT}/src/FEI_mv/femli:${HYPRE_ROOT}/src/IJ_mv:${HYPRE_ROOT}/src/blas:${HYPRE_ROOT}/src/distributed_ls:${HYPRE_ROOT}/src/distributed_ls/Euclid:${HYPRE_ROOT}/src/distributed_ls/ParaSails:${HYPRE_ROOT}/src/distributed_ls/pilut:${HYPRE_ROOT}/src/distributed_matrix:${HYPRE_ROOT}/src/krylov:${HYPRE_ROOT}/src/lapack:${HYPRE_ROOT}/src/matrix_matrix:${HYPRE_ROOT}/src/multivector:${HYPRE_ROOT}/src/parcsr_block_mv:${HYPRE_ROOT}/src/parcsr_ls:${HYPRE_ROOT}/src/parcsr_mv:${HYPRE_ROOT}/src/seq_mv:${HYPRE_ROOT}/src/sstruct_ls:${HYPRE_ROOT}/src/sstruct_mv:${HYPRE_ROOT}/src/struct_ls:${HYPRE_ROOT}/src/struct_mv "
MPI_PERF_FLAG="-mpiprof "
FTRACE_FLAG="-ftrace "
CXXFLAGS+="-fno-defer-inline-template-instantiation " 

BLAS="blas_openmp sblas_openmp"

FTRACE=0
DEBUG=1
MPI_PERF=0
INLINE=1

if [[ ${INLINE} -eq 1 ]]
then
    FC_FLAGS+=${INLINE_FLAGS}
    CC_EXTRAFLAGS+=${INLINE_FLAGS}
fi

if [[ ${FTRACE} -eq 1 ]]
then
    FC_FLAGS+=${FTRACE_FLAG}
    CC_EXTRAFLAGS+=${FTRACE_FLAG}

    PREFIX+="_ftrace"
fi

if [[ ${DEBUG} -eq 1 ]]
then
    FC_FLAGS+=${DEBUG_FLAGS}
    CC_EXTRAFLAGS+=${DEBUG_FLAGS}

fi

if [[ ${MPI_PERF} -eq 1 ]]
then
    FC_FLAGS+=${MPI_PERF_FLAG}
    CC_EXTRAFLAGS+=${MPI_PERF_FLAG}

fi

${COMPILER_ROOT}/mpincc --version

pushd ${HYPRE_ROOT}/src
rm -rf ${PREFIX} >/dev/null

[ -f config/Makefile.config ] && make clean
CC=${COMPILER_ROOT}/mpincc ./configure \
                --disable-fortran \
                --with-MPI \
                --with-openmp \
                --with-extra-CFLAGS="${CC_EXTRAFLAGS}" \
                --with-extra-CXXFLAGS="${CC_EXTRAFLAGS}" \
                --with-blas-libs="${BLAS}" \
                --with-blas-lib-dirs="/opt/nec/ve/nlc/2.3.0/lib/" \
                --prefix="${PREFIX}"

make -j
make install

popd
