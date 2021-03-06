/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for CSR Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef hypre_CSR_MATRIX_HEADER
#define hypre_CSR_MATRIX_HEADER
#ifdef __ve__
#include <sblas.h>
#endif
/*--------------------------------------------------------------------------
 * CSR Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int     *i;
   HYPRE_Int     *j;
   HYPRE_BigInt  *big_j;
   HYPRE_Int      num_rows;
   HYPRE_Int      num_cols;
   HYPRE_Int      num_nonzeros;
   hypre_int     *i_short;
   hypre_int     *j_short;

   /* Does the CSRMatrix create/destroy `data', `i', `j'? */
   HYPRE_Int      owns_data;

   HYPRE_Complex *data;

   /* for compressing rows in matrix multiplication  */
   HYPRE_Int     *rownnz;
   HYPRE_Int      num_rownnz;

   /* memory location of arrays i, j, data */
   HYPRE_MemoryLocation      memory_location;
#ifdef __ve__
   int flag=0;
   sblas_handle_t hnd;
   int *level;
   int *Ulevel;
   int *act_rows;
   int n_act_rows;
#endif

} hypre_CSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_CSRMatrixData(matrix)           ((matrix) -> data)
#define hypre_CSRMatrixI(matrix)              ((matrix) -> i)
#define hypre_CSRMatrixJ(matrix)              ((matrix) -> j)
#define hypre_CSRMatrixBigJ(matrix)           ((matrix) -> big_j)
#define hypre_CSRMatrixNumRows(matrix)        ((matrix) -> num_rows)
#define hypre_CSRMatrixNumCols(matrix)        ((matrix) -> num_cols)
#define hypre_CSRMatrixNumNonzeros(matrix)    ((matrix) -> num_nonzeros)
#define hypre_CSRMatrixRownnz(matrix)         ((matrix) -> rownnz)
#define hypre_CSRMatrixNumRownnz(matrix)      ((matrix) -> num_rownnz)
#define hypre_CSRMatrixOwnsData(matrix)       ((matrix) -> owns_data)
#define hypre_CSRMatrixMemoryLocation(matrix) ((matrix) -> memory_location)

HYPRE_Int hypre_CSRMatrixGetLoadBalancedPartitionBegin( hypre_CSRMatrix *A );
HYPRE_Int hypre_CSRMatrixGetLoadBalancedPartitionEnd( hypre_CSRMatrix *A );

/*--------------------------------------------------------------------------
 * CSR Boolean Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int    *i;
   HYPRE_Int    *j;
   HYPRE_BigInt *big_j;
   HYPRE_Int     num_rows;
   HYPRE_Int     num_cols;
   HYPRE_Int     num_nonzeros;
   HYPRE_Int     owns_data;

} hypre_CSRBooleanMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the CSR Boolean Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_CSRBooleanMatrix_Get_I(matrix)        ((matrix)->i)
#define hypre_CSRBooleanMatrix_Get_J(matrix)        ((matrix)->j)
#define hypre_CSRBooleanMatrix_Get_BigJ(matrix)     ((matrix)->big_j)
#define hypre_CSRBooleanMatrix_Get_NRows(matrix)    ((matrix)->num_rows)
#define hypre_CSRBooleanMatrix_Get_NCols(matrix)    ((matrix)->num_cols)
#define hypre_CSRBooleanMatrix_Get_NNZ(matrix)      ((matrix)->num_nonzeros)
#define hypre_CSRBooleanMatrix_Get_OwnsData(matrix) ((matrix)->owns_data)

#endif

