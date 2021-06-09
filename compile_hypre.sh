#!/bin/bash

source /opt/nec/ve/nlc/2.3.0/bin/nlcvars.sh
source /opt/nec/ve/mpi/2.16.0/bin/necmpivars.sh

HYPRE_ROOT=`pwd`
PREFIX=`pwd`/build
COMPILER_ROOT=/opt/nec/ve/mpi/2.16.0/bin64

FC_FLAGS="-fopenmp -O3 "
CC_EXTRAFLAGS=" -fopenmp  -O3 -report-all -fdiag-vector=3  "
DEBUG_FLAGS="-g -traceback=verbose "
INLINE_FLAGS="-finline-functions -finline-max-depth=5 -finline-max-function-size=400"
MPI_PERF_FLAG="-mpiprof "
FTRACE_FLAG="-ftrace "
CXXFLAGS +="-fno-defer-inline-template-instantiation" 

BLAS="blas_openmp sblas_openmp"

FTRACE=0
DEBUG=0
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

make clean
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
