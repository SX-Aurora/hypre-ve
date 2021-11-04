/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Relaxation scheme
 *
 *****************************************************************************/

#include "../sstruct_ls/gselim.h"
#include "Common.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_lapack.h"
#include "_hypre_parcsr_ls.h"

#ifdef __ve__
#include <asl.h>
#include <ftrace.h>
#include <sblas.h>
#endif

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_BoomerAMGRelax(hypre_ParCSRMatrix *A, hypre_ParVector *f,
                               HYPRE_Int *cf_marker, HYPRE_Int relax_type,
                               HYPRE_Int relax_points, HYPRE_Real relax_weight,
                               HYPRE_Real omega, HYPRE_Real *l1_norms,
                               hypre_ParVector *u, hypre_ParVector *Vtemp,
                               hypre_ParVector *Ztemp) {
  MPI_Comm comm = hypre_ParCSRMatrixComm(A);
  hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
  HYPRE_Real *A_diag_data = hypre_CSRMatrixData(A_diag);
  HYPRE_Int *A_diag_i = hypre_CSRMatrixI(A_diag);
  HYPRE_Int *A_diag_j = hypre_CSRMatrixJ(A_diag);
  hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
  HYPRE_Int *A_offd_i = hypre_CSRMatrixI(A_offd);
  HYPRE_Real *A_offd_data = hypre_CSRMatrixData(A_offd);
  HYPRE_Int *A_offd_j = hypre_CSRMatrixJ(A_offd);

  hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
  hypre_ParCSRCommHandle *comm_handle;

  HYPRE_BigInt global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
  HYPRE_Int n = hypre_CSRMatrixNumRows(A_diag);
  HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
  HYPRE_BigInt first_ind = hypre_ParVectorFirstIndex(u);

  hypre_Vector *u_local = hypre_ParVectorLocalVector(u);
  HYPRE_Real *u_data = hypre_VectorData(u_local);

  hypre_Vector *f_local = hypre_ParVectorLocalVector(f);
  HYPRE_Real *f_data = hypre_VectorData(f_local);

  hypre_Vector *Vtemp_local;
  HYPRE_Real *Vtemp_data;
  if (relax_type != 10) {
    Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
    Vtemp_data = hypre_VectorData(Vtemp_local);
  }
  HYPRE_Real *Vext_data = NULL;
  HYPRE_Real *v_buf_data = NULL;
  HYPRE_Real *tmp_data;

  hypre_Vector *Ztemp_local;
  HYPRE_Real *Ztemp_data;

  hypre_CSRMatrix *A_CSR;
  HYPRE_Int *A_CSR_i;
  HYPRE_Int *A_CSR_j;
  HYPRE_Real *A_CSR_data;

  hypre_Vector *f_vector;
  HYPRE_Real *f_vector_data;

  HYPRE_Int i, j, jr;
  HYPRE_Int ii, jj;
  HYPRE_Int ns, ne, size, rest;
  HYPRE_Int column;
  HYPRE_Int relax_error = 0;
  HYPRE_Int num_sends;
  HYPRE_Int num_recvs;
  HYPRE_Int index, start;
  HYPRE_Int num_procs, num_threads, my_id, ip, p;
  HYPRE_Int vec_start, vec_len;
  hypre_MPI_Status *status;
  hypre_MPI_Request *requests;

  HYPRE_Real *A_mat;
  HYPRE_Real *b_vec;

  HYPRE_Real zero = 0.0;
  HYPRE_Real res, res0, res2;
  HYPRE_Real one_minus_weight;
  HYPRE_Real one_minus_omega;
  HYPRE_Real prod;

  one_minus_weight = 1.0 - relax_weight;
  one_minus_omega = 1.0 - omega;
  hypre_MPI_Comm_size(comm, &num_procs);
  hypre_MPI_Comm_rank(comm, &my_id);
  num_threads = hypre_NumThreads();
  /*-----------------------------------------------------------------------
   * Switch statement to direct control based on relax_type:
   *     relax_type = 0 -> Jacobi or CF-Jacobi
   *     relax_type = 1 -> Gauss-Seidel <--- very slow, sequential
   *     relax_type = 2 -> Gauss_Seidel: interior points in parallel ,
   *                                     boundary sequential
   *     relax_type = 3 -> hybrid: SOR-J mix off-processor, SOR on-processor
   *                               with outer relaxation parameters (forward
   *solve) relax_type = 4 -> hybrid: SOR-J mix off-processor, SOR on-processor
   *                               with outer relaxation parameters (backward
   *solve) relax_type = 5 -> hybrid: GS-J mix off-processor, chaotic GS on-node
   *     relax_type = 6 -> hybrid: SSOR-J mix off-processor, SSOR on-processor
   *                               with outer relaxation parameters
   *     relax_type = 7 -> Jacobi (uses Matvec), only needed in CGNR
   *     relax_type = 8 -> hybrid L1 Symm. Gauss-Seidel
   *     relax_type = 10 -> On-processor direct forward solve for matrices with
   *                        triangular structure (indices need not be ordered
   *                        triangular)
   *     relax_type = 13 -> hybrid L1 Gauss-Seidel forward solve
   *     relax_type = 14 -> hybrid L1 Gauss-Seidel backward solve
   *     relax_type = 15 -> CG
   *     relax_type = 16 -> Scaled Chebyshev
   *     relax_type = 17 -> FCF-Jacobi
   *     relax_type = 18 -> L1-Jacobi
   *     relax_type = 9, 99, 98 -> Direct solve, Gaussian elimination
   *     relax_type = 19-> Direct Solve, (old version)
   *     relax_type = 29-> Direct solve: use gaussian elimination & BLAS
   *                       (with pivoting) (old version)
   *-----------------------------------------------------------------------*/

  switch (relax_type) {
  case 0: /* Weighted Jacobi */
  {
    if (num_procs > 1) {
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

      /* printf("!! Proc %d: n %d,  num_sends %d, num_cols_offd %d\n", my_id, n,
       * num_sends, num_cols_offd); */

      v_buf_data = hypre_CTAlloc(
          HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
          HYPRE_MEMORY_HOST);

      Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);

      if (num_cols_offd) {
        A_offd_j = hypre_CSRMatrixJ(A_offd);
        A_offd_data = hypre_CSRMatrixData(A_offd);
      }

      index = 0;
      for (i = 0; i < num_sends; i++) {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);
             j++) {
          v_buf_data[index++] =
              u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
        }
      }

      comm_handle =
          hypre_ParCSRCommHandleCreate(1, comm_pkg, v_buf_data, Vext_data);
    }
    /*-----------------------------------------------------------------
     * Copy current approximation into temporary vector.
     *-----------------------------------------------------------------*/

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < n; i++) {
      Vtemp_data[i] = u_data[i];
    }
    if (num_procs > 1) {
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
    }

    /*-----------------------------------------------------------------
     * Relax all points.
     *-----------------------------------------------------------------*/

    if (relax_points == 0) {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, jj, res) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < n; i++) {

        /*-----------------------------------------------------------
         * If diagonal is nonzero, relax point i; otherwise, skip it.
         *-----------------------------------------------------------*/

        if (A_diag_data[A_diag_i[i]] != zero) {
          res = f_data[i];
          for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
            ii = A_diag_j[jj];
            res -= A_diag_data[jj] * Vtemp_data[ii];
          }
          for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
            ii = A_offd_j[jj];
            res -= A_offd_data[jj] * Vext_data[ii];
          }
          u_data[i] *= one_minus_weight;
          u_data[i] += relax_weight * res / A_diag_data[A_diag_i[i]];
        }
      }
    }
    /*-----------------------------------------------------------------
     * Relax only C or F points as determined by relax_points.
     *-----------------------------------------------------------------*/
    else {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, jj, res) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < n; i++) {

        /*-----------------------------------------------------------
         * If i is of the right type ( C or F ) and diagonal is
         * nonzero, relax point i; otherwise, skip it.
         *-----------------------------------------------------------*/

        if (cf_marker[i] == relax_points && A_diag_data[A_diag_i[i]] != zero) {
          res = f_data[i];
          for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
            ii = A_diag_j[jj];
            res -= A_diag_data[jj] * Vtemp_data[ii];
          }
          for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
            ii = A_offd_j[jj];
            res -= A_offd_data[jj] * Vext_data[ii];
          }
          u_data[i] *= one_minus_weight;
          u_data[i] += relax_weight * res / A_diag_data[A_diag_i[i]];
        }
      }
    }
    if (num_procs > 1) {
      hypre_TFree(Vext_data, HYPRE_MEMORY_HOST);
      hypre_TFree(v_buf_data, HYPRE_MEMORY_HOST);
    }
  } break;

  case 5: /* Hybrid: Jacobi off-processor,
                     chaotic Gauss-Seidel on-processor       */
  {
    if (num_procs > 1) {
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

      v_buf_data = hypre_CTAlloc(
          HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
          HYPRE_MEMORY_HOST);

      Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);

      if (num_cols_offd) {
        A_offd_j = hypre_CSRMatrixJ(A_offd);
        A_offd_data = hypre_CSRMatrixData(A_offd);
      }

      index = 0;
      for (i = 0; i < num_sends; i++) {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);
             j++) {
          v_buf_data[index++] =
              u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
        }
      }

      comm_handle =
          hypre_ParCSRCommHandleCreate(1, comm_pkg, v_buf_data, Vext_data);

      /*-----------------------------------------------------------------
       * Copy current approximation into temporary vector.
       *-----------------------------------------------------------------*/
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
    }

    /*-----------------------------------------------------------------
     * Relax all points.
     *-----------------------------------------------------------------*/

    if (relax_points == 0) {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, jj, res) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < n; i++) /* interior points first */
      {

        /*-----------------------------------------------------------
         * If diagonal is nonzero, relax point i; otherwise, skip it.
         *-----------------------------------------------------------*/

        if (A_diag_data[A_diag_i[i]] != zero) {
          res = f_data[i];
          for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
            ii = A_diag_j[jj];
            res -= A_diag_data[jj] * u_data[ii];
          }
          for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
            ii = A_offd_j[jj];
            res -= A_offd_data[jj] * Vext_data[ii];
          }
          u_data[i] = res / A_diag_data[A_diag_i[i]];
        }
      }
    }

    /*-----------------------------------------------------------------
     * Relax only C or F points as determined by relax_points.
     *-----------------------------------------------------------------*/

    else {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, jj, res) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < n; i++) /* relax interior points */
      {

        /*-----------------------------------------------------------
         * If i is of the right type ( C or F ) and diagonal is
         * nonzero, relax point i; otherwise, skip it.
         *-----------------------------------------------------------*/

        if (cf_marker[i] == relax_points && A_diag_data[A_diag_i[i]] != zero) {
          res = f_data[i];
          for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
            ii = A_diag_j[jj];
            res -= A_diag_data[jj] * u_data[ii];
          }
          for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
            ii = A_offd_j[jj];
            res -= A_offd_data[jj] * Vext_data[ii];
          }
          u_data[i] = res / A_diag_data[A_diag_i[i]];
        }
      }
    }
    if (num_procs > 1) {
      hypre_TFree(Vext_data, HYPRE_MEMORY_HOST);
      hypre_TFree(v_buf_data, HYPRE_MEMORY_HOST);
    }
  } break;

  /* Hybrid: Jacobi off-processor, Gauss-Seidel on-processor (forward loop) */
  case 3: {
    if (num_threads > 1) {
      Ztemp_local = hypre_ParVectorLocalVector(Ztemp);
      Ztemp_data = hypre_VectorData(Ztemp_local);
    }

#if defined(HYPRE_USING_PERSISTENT_COMM)
    // JSP: persistent comm can be similarly used for other smoothers
    hypre_ParCSRPersistentCommHandle *persistent_comm_handle;
#endif

    if (num_procs > 1) {
#ifdef HYPRE_PROFILE
      hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] -= hypre_MPI_Wtime();
#endif

      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

#if defined(HYPRE_USING_PERSISTENT_COMM)
      persistent_comm_handle =
          hypre_ParCSRCommPkgGetPersistentCommHandle(1, comm_pkg);
      v_buf_data = (HYPRE_Real *)hypre_ParCSRCommHandleSendDataBuffer(
          persistent_comm_handle);
      Vext_data = (HYPRE_Real *)hypre_ParCSRCommHandleRecvDataBuffer(
          persistent_comm_handle);
#else
      v_buf_data = hypre_CTAlloc(
          HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
          HYPRE_MEMORY_HOST);

      Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);
#endif

      if (num_cols_offd) {
        A_offd_j = hypre_CSRMatrixJ(A_offd);
        A_offd_data = hypre_CSRMatrixData(A_offd);
      }

      HYPRE_Int begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
      HYPRE_Int end = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
      for (i = begin; i < end; i++) {
        v_buf_data[i - begin] =
            u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)];
      }

#ifdef HYPRE_PROFILE
      hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] += hypre_MPI_Wtime();
      hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] -= hypre_MPI_Wtime();
#endif

#if defined(HYPRE_USING_PERSISTENT_COMM)
      hypre_ParCSRPersistentCommHandleStart(persistent_comm_handle,
                                            HYPRE_MEMORY_HOST, v_buf_data);
#else
      comm_handle =
          hypre_ParCSRCommHandleCreate(1, comm_pkg, v_buf_data, Vext_data);
#endif

      /*-----------------------------------------------------------------
       * Copy current approximation into temporary vector.
       *-----------------------------------------------------------------*/
#if defined(HYPRE_USING_PERSISTENT_COMM)
      hypre_ParCSRPersistentCommHandleWait(persistent_comm_handle,
                                           HYPRE_MEMORY_HOST, Vext_data);
#else
      hypre_ParCSRCommHandleDestroy(comm_handle);
#endif
      comm_handle = NULL;

#ifdef HYPRE_PROFILE
      hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
#endif
    }

    /*-----------------------------------------------------------------
     * ESSAM: replace A_offd_i SpMV routine
     *-----------------------------------------------------------------*/
#ifdef _FTRACE
    ftrace_region_begin("RELAX_CF_POINT_DATA");
#endif

    sblas_int_t s_ierr;
    HYPRE_Int jmp = 1, t_id, inc, nn;
    HYPRE_Real A_offd_res[global_num_rows];
    // fprintf(stderr, "Data: %d\n", n);
    if (!A_offd->hnd) {
      // A_offd->flag=1;
      sblas_int_t mrow = (sblas_int_t)n;
      sblas_int_t ncol =
          (sblas_int_t)num_cols_offd; // always passed as zero ..!

      sblas_int_t *iaptr = A_offd_i;
      sblas_int_t *iaind = A_offd_j;
      double *avals = A_offd_data;

      s_ierr = sblas_create_matrix_handle_from_csr_rd(
          mrow, mrow, iaptr, iaind, avals, SBLAS_INDEXING_0, SBLAS_GENERAL,
          &A_offd->hnd); // handler

      s_ierr = sblas_analyze_mv_rd(SBLAS_NON_TRANSPOSE, A_offd->hnd);
// multi-level scheduling
#if 1

      // Essam: relaxation points can't be used in checking
      //        according to the algirthm it can be -1/1 or 0 ?!!

      asl_sort_t sort;
      asl_library_initialize();
      asl_sort_create_i32(&sort, ASL_SORTORDER_ASCENDING,
                          ASL_SORTALGORITHM_AUTO_STABLE);
      asl_sort_preallocate(sort, n);
      HYPRE_Int m;

      if (relax_points == 0) {
        int *rows;
        int *level;

        int max_nnz_row = 0;
#pragma omp parallel for private(i, j) reduction(max : max_nnz_row)
        for (i = 0; i < n; i++) {
          j = A_diag_i[i + 1] - A_diag_i[i] /*- 1*/;
          max_nnz_row = (max_nnz_row < j) ? j : max_nnz_row;
        }
        int nnz = n * max_nnz_row;
        // int nnz = n * (max_nnz_row - 1);
        // nnz = (nnz > (A_diag_i[n] - A_diag_i[0])) ? nnz : (A_diag_i[n] -
        // A_diag_i[0]);

        A_diag->max_nnz_row = max_nnz_row;

        int t_levels_idx[n];
        int t_nnz_rows[n];

        A_diag->ms_rows_freq = (int *)malloc(sizeof(int) * 2 * n);
        A_diag->ms_i = (int *)malloc(sizeof(int) * (n + num_threads));

        A_diag->ms_j = (int *)malloc(sizeof(int) * nnz);
        A_diag->ms_data = (double *)malloc(sizeof(double) * nnz);

#pragma omp parallel for if (nnz > 4069)
        for (i = 0; i < nnz; i++) {
          A_diag->ms_j[i] = 0;
          A_diag->ms_data[i] = 0;
        }

        level = (int *)malloc(sizeof(int) * n);
        rows = (int *)malloc(sizeof(int) * n);

        A_diag->level_idx = (int *)malloc(sizeof(int) * n);
        // A_diag->ms_rhs_idx = (int *)malloc(sizeof(int) * nnz);

        A_diag->f_act_rows = (int *)malloc(sizeof(int) * nnz);
        A_diag->ms_vdata = (double *)malloc(sizeof(double) * n);

#pragma omp parallel private(i, j, jj, ii, t_id, ns, ne, rest, size, m)
        {

          size = n / num_threads;
          rest = n - size * num_threads;
          t_id = omp_get_thread_num();

          if (t_id < rest) {
            ns = t_id * size + t_id;
            ne = (t_id + 1) * size + t_id + 1;
          } else {
            ns = t_id * size + rest;
            ne = (t_id + 1) * size + rest;
          }

          for (i = ns; i < ne; i++)
            level[i] = -1;

          for (i = ns; i < ne; i++) {
            m = -1;
            rows[i] = -1;
            A_diag->ms_vdata[i] = A_diag_data[A_diag_i[i]];
            t_nnz_rows[i] = A_diag_i[i + 1] - A_diag_i[i] - 1;
            if (A_diag->ms_vdata[i] != zero) {
              rows[i] = i;
#pragma _NEC novector
              for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                if (ii >= ns && ii < ne) {
                  if (ii < i && m < level[ii])
                    m = level[ii];
                  // if (ii > i)
                  //   level[ii] = m + 1;
                }
              }
              // Check if some previous rows needs my value.
              if (level[i] > m + 1) {
                m = level[i];
              } else {
                level[i] = m + 1;
              }
// Set all columns after my diagonal to my level.
#pragma _NEC novector
              for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                if (ii >= ns && ii < ne)
                  if (ii > i && A_diag->ms_vdata[ii] != zero)
                    level[ii] = m + 1;
              }
            }
          }
          int nelem = ne - ns;

          asl_sort_execute_i32(sort, nelem, &level[ns], &rows[ns], &level[ns],
                               &t_levels_idx[ns]);

          for (i = A_diag_i[ns]; i < A_diag_i[ne]; i++)
            A_diag->f_act_rows[i] = 0;

          int _ns, _ne;

          int idx_ia, idx_rhs, idx, k, ik, prev, prev_i, max_nnz;

          _ns = ns + t_id;
          _ne = ne + t_id;
          idx_ia = 0;
          idx = 0;

          int *ms_i = A_diag->ms_i + _ns;
          ms_i[0] = ns * max_nnz_row; // A_diag_i[ns];

          int *ms_j = A_diag->ms_j + ms_i[0];
          double *ms_data = A_diag->ms_data + ms_i[0];
          int *f_act_rows = A_diag->f_act_rows + ms_i[0];
          int *ms_rows_freq = A_diag->ms_rows_freq;
          int *level_idx = A_diag->level_idx;

          for (i = ns; i < ne; i++) {
            ms_rows_freq[2 * i] = 0;
            ms_rows_freq[2 * i + 1] = 0;
          }

          prev = -1; // level[ns];
          prev_i = ns;
          max_nnz = 0;
          ms_rows_freq[2 * prev_i] = 0;
          // if (prev != -1)
          //   ms_rows_freq[2 * prev_i + 1] = 1;
          for (i = ns; i < ne; i++) {
            if (level[i] != -1) {
              if (level[i] == prev) {
                ms_rows_freq[2 * prev_i + 1]++;
                if (ms_rows_freq[2 * prev_i] < t_nnz_rows[t_levels_idx[i]])
                  ms_rows_freq[2 * prev_i] = t_nnz_rows[t_levels_idx[i]];
              } else {
                // set the max nnz for all rows with a level
                for (j = prev_i + 1; j < ns + idx_ia; j++)
                  ms_rows_freq[2 * j] = ms_rows_freq[2 * prev_i];

                prev_i = ns + idx_ia;
                prev = level[i];
                ms_rows_freq[2 * prev_i + 1] = 1;
                ms_rows_freq[2 * prev_i] = t_nnz_rows[t_levels_idx[i]];
              }
              idx_ia++;
            }
          }
          // the last set of rows
          for (j = prev_i + 1; j < ne; j++)
            ms_rows_freq[2 * j] = ms_rows_freq[2 * prev_i];

          idx_ia = 0;

          for (i = ns; i < ne; i++) {

            level_idx[i] = -1;
            ii = t_levels_idx[i];
            if (ii != -1) {
              level_idx[ns + idx_ia] = ii;

              for (jj = (A_diag_i[ii] + 1); jj < A_diag_i[ii + 1]; jj++) {
                k = jj - (A_diag_i[ii] + 1);
                ik = A_diag_j[jj];

                ms_j[idx + k] = ik;
                ms_data[idx + k] = A_diag_data[jj];

                if (ik >= ns && ik < ne)
                  f_act_rows[idx + k] = 1;
              }
              idx += ms_rows_freq[2 * (ns + idx_ia)]; // max_nnz_row;

              ms_i[idx_ia + 1] =
                  ms_i[idx_ia] +
                  (ms_rows_freq[2 * (ns + idx_ia)] /*max_nnz_row*/);
              idx_ia++;
            }
          }

          for (i = idx_ia + 1; i < (ne - ns); i++) {
            ms_i[i] = ms_i[idx_ia];
          }
        }
        free(rows);
        free(level);

      } else {

        int *rows;
        int *level;

        int max_nnz_row = 0;

#if 0
#include <stdio.h>
#include <string.h>
        int t = 0;
        for (i = 0; i < n; i++)
          t += A_diag_i[i + 1] - A_diag_i[i];

        int length = snprintf(NULL, 0, "%d", t);
        char nnz_str[50]; //= malloc(length + 1);
        snprintf(nnz_str, length + 1, "%d", t);
        FILE *fstat;
        char *ext = "_amg_level_stat.out";
        char *fname = strcat(nnz_str, ext);
        fprintf(stderr, "%s\n", fname);

        if ((fstat = fopen(fname, "w")) == NULL)
          fprintf(stderr, "can't open file");
        fprintf(fstat, "%d,%d\n", n,t);

        int histo[n];
        int histo_nnz_cnt[n];

        for (i = 0; i < n; i++) {
          histo[i] = 0;
          histo_nnz_cnt[i] = -1;
        }

        for (i = 0; i < n; i++) {
          int cols = (A_diag_i[i + 1] - A_diag_i[i] - 1);
          j = 0;
          for (; j < n; j++) {
            if (histo_nnz_cnt[j] == cols || histo_nnz_cnt[j] == -1) {
              break;
            }
          }
          histo_nnz_cnt[j] = cols;
          histo[j]++;
        }

        for (i = 0; i < n; i++)
          if (histo_nnz_cnt[i] != -1)
            fprintf(fstat, "%d,%d\n", histo_nnz_cnt[i], histo[i]);

        fclose(fstat);

#endif

        // fprintf(stderr, "AMG N: %d \t NNZ: %d\n", n, A_diag_i[n] -
        // A_diag_i[0]);

#pragma omp parallel for private(i, j) reduction(max : max_nnz_row)
        for (i = 0; i < n; i++) {
          j = A_diag_i[i + 1] - A_diag_i[i] /*- 1*/;
          max_nnz_row = (max_nnz_row < j) ? j : max_nnz_row;
        }
        int nnz = n * max_nnz_row;
        // int nnz = n * (max_nnz_row - 1);
        // nnz = (nnz > (A_diag_i[n] - A_diag_i[0])) ? nnz : (A_diag_i[n] -
        // A_diag_i[0]);

        A_diag->max_nnz_row = max_nnz_row;

        int t_levels_idx[n];
        int t_nnz_rows[n];

        A_diag->ms_rows_freq = (int *)malloc(sizeof(int) * 4 * n);
        A_diag->ms_i = (int *)malloc(sizeof(int) * 2 * (n + num_threads));

        A_diag->ms_j = (int *)malloc(sizeof(int) * 2 * nnz);
        A_diag->ms_data = (double *)malloc(sizeof(double) * 2 * nnz);

#pragma omp parallel for if (nnz > 4069)
        for (i = 0; i < 2 * nnz; i++) {
          A_diag->ms_j[i] = 0;
          A_diag->ms_data[i] = 0;
        }

        level = (int *)malloc(sizeof(int) * n);
        rows = (int *)malloc(sizeof(int) * n);

        A_diag->level_idx = (int *)malloc(sizeof(int) * 2 * n);
        // A_diag->ms_rhs_idx = (int *)malloc(sizeof(int) * nnz);

        A_diag->f_act_rows = (int *)malloc(sizeof(int) * 2 * nnz);
        A_diag->ms_vdata = (double *)malloc(sizeof(double) * n);

#pragma omp parallel private(i, j, jj, ii, t_id, ns, ne, rest, size, m)
        {

          size = n / num_threads;
          rest = n - size * num_threads;
          t_id = omp_get_thread_num();

          if (t_id < rest) {
            ns = t_id * size + t_id;
            ne = (t_id + 1) * size + t_id + 1;
          } else {
            ns = t_id * size + rest;
            ne = (t_id + 1) * size + rest;
          }

          for (i = ns; i < ne; i++)
            level[i] = -1;

          for (i = ns; i < ne; i++) {
            m = -1;
            rows[i] = -1;
            A_diag->ms_vdata[i] = A_diag_data[A_diag_i[i]];
            t_nnz_rows[i] = A_diag_i[i + 1] - A_diag_i[i] - 1;
            if (cf_marker[i] == -1 && A_diag->ms_vdata[i] != zero) {
              rows[i] = i;
#pragma _NEC novector
              for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                if (ii >= ns && ii < ne) {
                  if (ii < i && m < level[ii])
                    m = level[ii];
                  // if (ii > i)
                  //   level[ii] = m + 1;
                }
              }
              // Check if some previous rows needs my value.
              if (level[i] > m + 1) {
                m = level[i];
              } else {
                level[i] = m + 1;
              }
// Set all columns after my diagonal to my level.
#pragma _NEC novector
              for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                if (ii >= ns && ii < ne)
                  if (ii > i && cf_marker[ii] == -1 &&
                      A_diag->ms_vdata[ii] != zero)
                    level[ii] = m + 1;
              }
            }
          }
          int nelem = ne - ns;

          asl_sort_execute_i32(sort, nelem, &level[ns], &rows[ns], &level[ns],
                               &t_levels_idx[ns]);

          for (i = A_diag_i[ns]; i < A_diag_i[ne]; i++)
            A_diag->f_act_rows[i] = 0;

          int _ns, _ne;

          int idx_ia, idx_rhs, idx, k, ik, prev, prev_i, max_nnz;

          _ns = ns + t_id;
          _ne = ne + t_id;
          idx_ia = 0;
          idx = 0;

          int *ms_i = A_diag->ms_i + _ns;
          ms_i[0] = ns * max_nnz_row; // A_diag_i[ns];

          int *ms_j = A_diag->ms_j + ms_i[0];
          double *ms_data = A_diag->ms_data + ms_i[0];
          int *f_act_rows = A_diag->f_act_rows + ms_i[0];
          int *ms_rows_freq = A_diag->ms_rows_freq;
          int *level_idx = A_diag->level_idx;

          for (i = ns; i < ne; i++) {
            ms_rows_freq[2 * i] = 0;
            ms_rows_freq[2 * i + 1] = 0;
          }

          prev = -1; // level[ns];
          prev_i = ns;
          max_nnz = 0;
          ms_rows_freq[2 * prev_i] = 0;
          // if (prev != -1)
          //   ms_rows_freq[2 * prev_i + 1] = 1;
          for (i = ns; i < ne; i++) {
            if (level[i] != -1) {
              if (level[i] == prev) {
                ms_rows_freq[2 * prev_i + 1]++;
                if (ms_rows_freq[2 * prev_i] < t_nnz_rows[t_levels_idx[i]])
                  ms_rows_freq[2 * prev_i] = t_nnz_rows[t_levels_idx[i]];
              } else {
                // set the max nnz for all rows with a level
                for (j = prev_i + 1; j < ns + idx_ia; j++)
                  ms_rows_freq[2 * j] = ms_rows_freq[2 * prev_i];

                prev_i = ns + idx_ia;
                prev = level[i];
                ms_rows_freq[2 * prev_i + 1] = 1;
                ms_rows_freq[2 * prev_i] = t_nnz_rows[t_levels_idx[i]];
              }
              idx_ia++;
            }
          }
          // the last set of rows
          for (j = prev_i + 1; j < ne; j++)
            ms_rows_freq[2 * j] = ms_rows_freq[2 * prev_i];

          idx_ia = 0;

          for (i = ns; i < ne; i++) {

            level_idx[i] = -1;
            ii = t_levels_idx[i];
            if (ii != -1) {
              level_idx[ns + idx_ia] = ii;

              for (jj = (A_diag_i[ii] + 1); jj < A_diag_i[ii + 1]; jj++) {
                k = jj - (A_diag_i[ii] + 1);
                ik = A_diag_j[jj];

                ms_j[idx + k] = ik;
                ms_data[idx + k] = A_diag_data[jj];

                if (ik >= ns && ik < ne)
                  f_act_rows[idx + k] = 1;
              }
              idx += ms_rows_freq[2 * (ns + idx_ia)]; // max_nnz_row;

              ms_i[idx_ia + 1] =
                  ms_i[idx_ia] +
                  (ms_rows_freq[2 * (ns + idx_ia)] /*max_nnz_row*/);
              idx_ia++;
            }
          }

          for (i = idx_ia + 1; i < (ne - ns); i++) {
            ms_i[i] = ms_i[idx_ia];
          }
          // forward substitution (relax point 1) multi-level scheduling
          for (i = ns; i < ne; i++)
            level[i] = -1;

          for (i = ns; i < ne; i++) {
            m = -1;
            rows[i] = -1;
            if (cf_marker[i] == 1 && A_diag->ms_vdata[i] != zero) {
              rows[i] = i;
#pragma _NEC novector
              for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                if (ii >= ns && ii < ne) {
                  if (ii < i && m < level[ii])
                    m = level[ii];
                }
              }
              // Check if some previous rows needs my value.
              if (level[i] > m + 1) {
                m = level[i];
              } else {
                level[i] = m + 1;
              }
// Set all columns after my diagonal to my level.
#pragma _NEC novector
              for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                if (ii >= ns && ii < ne)
                  if (ii > i && cf_marker[ii] == 1 &&
                      A_diag->ms_vdata[ii] != zero)
                    level[ii] = m + 1;
              }
            }
          }

          asl_sort_execute_i32(sort, nelem, &level[ns], &rows[ns], &level[ns],
                               &t_levels_idx[ns]);
          // copy data..
          _ns = ns + t_id;
          _ne = ne + t_id;
          idx_ia = 0;
          idx = 0;

          ms_i = A_diag->ms_i + (n + num_threads) + _ns;
          ms_i[0] = ns * max_nnz_row; // A_diag_i[ns];

          ms_j = A_diag->ms_j + nnz + ms_i[0];
          ms_data = A_diag->ms_data + nnz + ms_i[0];
          f_act_rows = A_diag->f_act_rows + nnz + ms_i[0];
          ms_rows_freq = A_diag->ms_rows_freq + 2 * n;
          level_idx = A_diag->level_idx + n;

          for (i = ns; i < ne; i++) {
            ms_rows_freq[2 * i] = 0;
            ms_rows_freq[2 * i + 1] = 0;
          }

          prev = -1; // level[ns];
          prev_i = ns;
          max_nnz = 0;
          ms_rows_freq[2 * prev_i] = 0;
          // if (prev != -1)
          //   ms_rows_freq[2 * prev_i + 1] = 1;

          for (i = ns; i < ne; i++) {
            if (level[i] != -1) {
              if (level[i] == prev) {
                ms_rows_freq[2 * prev_i + 1]++;
                if (ms_rows_freq[2 * prev_i] < t_nnz_rows[t_levels_idx[i]])
                  ms_rows_freq[2 * prev_i] = t_nnz_rows[t_levels_idx[i]];
              } else {
                // set the max nnz for all rows with a level
                for (j = prev_i + 1; j < ns + idx_ia; j++)
                  ms_rows_freq[2 * j] = ms_rows_freq[2 * prev_i];

                prev_i = ns + idx_ia;
                prev = (level[i] != -1) ? level[i] : level[++i];
                ms_rows_freq[2 * prev_i + 1] = 1;
                ms_rows_freq[2 * prev_i] = t_nnz_rows[t_levels_idx[i]];
              }
              idx_ia++;
            }
          }
          // set the last level
          for (j = prev_i + 1; j < ne; j++)
            ms_rows_freq[2 * j] = ms_rows_freq[2 * prev_i];

          idx_ia = 0;
          for (i = ns; i < ne; i++) {
            level_idx[i] = -1;
            ii = t_levels_idx[i];
            if (ii != -1) {
              level_idx[ns + idx_ia] = ii;

              for (jj = (A_diag_i[ii] + 1); jj < A_diag_i[ii + 1]; jj++) {
                k = jj - (A_diag_i[ii] + 1);
                ik = A_diag_j[jj];

                ms_j[idx + k] = ik;
                ms_data[idx + k] = A_diag_data[jj];

                if (ik >= ns && ik < ne)
                  f_act_rows[idx + k] = 1;
              }
              idx += /*max_nnz_row*/ ms_rows_freq[2 * (ns + idx_ia)];

              ms_i[idx_ia + 1] =
                  ms_i[idx_ia] +
                  (/*max_nnz_row*/ ms_rows_freq[2 * (ns + idx_ia)]);
              idx_ia++;
            }
          }

          for (i = idx_ia + 1; i < (ne - ns); i++) {
            ms_i[i] = ms_i[idx_ia];
          }
        }
        free(level);
        free(rows);
      }
      /* Sorting Finalization */
      asl_sort_destroy(sort);
      /* Library Finalization */
      asl_library_finalize();
#endif
    }

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i)
#endif
    for (i = 0; i < n; i++) {
      A_offd_res[i] = f_data[i];
    }
    s_ierr = sblas_execute_mv_rd(SBLAS_NON_TRANSPOSE, A_offd->hnd, -1.0,
                                 Vext_data, 1.0, A_offd_res);
#ifdef _FTRACE
    ftrace_region_end("RELAX_CF_POINT_DATA");
#endif

    /*-----------------------------------------------------------------
     * Relax all points.
     *-----------------------------------------------------------------*/
#ifdef HYPRE_PROFILE
    hypre_profile_times[HYPRE_TIMER_ID_RELAX] -= hypre_MPI_Wtime();
#endif
    // fprintf(stderr, "relax_weight %d \t omega %d \t relax_points %d\n",
    //         relax_weight, omega, relax_points);
    if (relax_weight == 1 && omega == 1) {
      if (relax_points == 0) {
        // Essam: testing
        if (num_threads > 1) {
          tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, rest, size)        \
    HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ns; i < ne; i++) /* interior points first */
            {
              /*-----------------------------------------------------------
               * If diagonal is nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/
              if (A_diag_data[A_diag_i[i]] != zero) {
                //  res = f_data[i];
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne)
                    A_offd_res[i] -= A_diag_data[jj] * u_data[ii];
                  else
                    A_offd_res[i] -= A_diag_data[jj] * tmp_data[ii];
                }
                //  for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                //    ii = A_offd_j[jj];
                //    res -= A_offd_data[jj] * Vext_data[ii];
                //  }
                u_data[i] = A_offd_res[i] / A_diag_data[A_diag_i[i]];
              }
            }
          }
        } else {
          for (i = 0; i < n; i++) /* interior points first */
          {

            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (A_diag_data[A_diag_i[i]] != zero) {
              //   res = f_data[i];
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                //  fprintf(stderr, "i: %d \n ii: %d\n", i, ii);
                A_offd_res[i] -= A_diag_data[jj] * u_data[ii];
              }
              //   for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
              //     ii = A_offd_j[jj];
              //     res -= A_offd_data[jj] * Vext_data[ii];
              //   }
              u_data[i] = A_offd_res[i] / A_diag_data[A_diag_i[i]];
            }
          }
        }
      }

      /*-----------------------------------------------------------------
       * Relax only C or F points as determined by relax_points.
       *-----------------------------------------------------------------*/
      else {
        // Essam: target AMG 0
        // fprintf(stderr, "RELAX_CF_POINT_0");

#ifdef _FTRACE
        ftrace_region_begin("RELAX_CF_POINT_0");
#endif
        if (num_threads >= 1) {
#if 1 // optimized implementation
          jmp = 1;
          if (relax_points == -1)
            jmp = 0;

          HYPRE_Int nnz = A_diag->max_nnz_row * n; // A_diag_i[n] - A_diag_i[0];

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel private(i, ii, j, jj, t_id, ns, ne, res, rest, size)
#endif
          {
            size = n / omp_get_num_threads();
            rest = n - size * omp_get_num_threads();
            t_id = omp_get_thread_num();

            if (t_id < rest) {
              ns = t_id * size + t_id;
              ne = (t_id + 1) * size + t_id + 1;
            } else {
              ns = t_id * size + rest;
              ne = (t_id + 1) * size + rest;
            }

            double t_data[n];
            double t_res[ne - ns];
            for (i = 0; i < n; i++)
              t_data[i] = u_data[i];

            // alias sorted data
            int *level_idx = A_diag->level_idx + n * jmp;
            int *ms_i = A_diag->ms_i + (n + num_threads) * jmp;
            int *ms_j = A_diag->ms_j + nnz * jmp;
            int *ms_rows_freq = A_diag->ms_rows_freq + 2 * n * jmp;
            double *ms_data = A_diag->ms_data + nnz * jmp;

            HYPRE_Int MAX_NNZ, freq, ik, jk;

            jk = 0;
            freq = 0;
            for (ik = ns; ik < ne; ik += MAX(freq, 1)) {
              i = level_idx[ik];
              if (i != -1) {
                freq = ms_rows_freq[2 * ik + 1];
                MAX_NNZ = ms_rows_freq[2 * ik];
                for (jk = 0; jk < freq; jk++) {
                  i = level_idx[ik + jk];
                  t_res[jk] = A_offd_res[i];
                }
                if (freq < MAX_NNZ) {
                  for (jk = 0; jk < freq; jk++) {
                    const int strt = ms_i[t_id + ik] + jk * MAX_NNZ;
                    const int end = ms_i[t_id + ik] + (jk + 1) * MAX_NNZ;

                    for (jj = strt; jj < end; jj++) {
                      ii = ms_j[jj];
                      t_res[jk] -= ms_data[jj] * t_data[ii];
                    }
                  }
                } else {
                  for (jj = 0; jj < MAX_NNZ; jj++) {
                    for (jk = 0; jk < freq; jk++) {
                      int jmp = ms_i[t_id + ik] + jk * MAX_NNZ;
                      ii = ms_j[jmp + jj];
                      t_res[jk] -= ms_data[jmp + jj] * t_data[ii];
                    }
                  }
                }

                for (jk = 0; jk < freq; jk++) {
                  i = level_idx[ik + jk];
                  t_data[i] = t_res[jk] / A_diag->ms_vdata[i];
                }
              }
            }
            for (i = ns; i < ne; i++)
              u_data[i] = t_data[i];
          }
#else
          tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, rest, size)        \
    HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ns; i < ne; i++) /* relax interior points */
            {
              /*-----------------------------------------------------------
               * If i is of the right type ( C or F ) and diagonal is
               * nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/
              if (cf_marker[i] == relax_points &&
                  A_diag_data[A_diag_i[i]] != zero) {
                //  res = f_data[i];
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne)
                    A_offd_res[i] -= A_diag_data[jj] * u_data[ii];
                  else
                    A_offd_res[i] -= A_diag_data[jj] * tmp_data[ii];
                }
                //  for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                //    ii = A_offd_j[jj];
                //    res -= A_offd_data[jj] * Vext_data[ii];
                //  }
                u_data[i] = A_offd_res[i] / A_diag_data[A_diag_i[i]];
              }
            }
          }
#endif

        } else {
          for (i = 0; i < n; i++) /* relax interior points */
          {
            /*-----------------------------------------------------------
             * If i is of the right type ( C or F ) and diagonal is
             * nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/
            if (cf_marker[i] == relax_points &&
                A_diag_data[A_diag_i[i]] != zero) {
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                A_offd_res[i] -= A_diag_data[jj] * u_data[ii];
              }
              u_data[i] = A_offd_res[i] / A_diag_data[A_diag_i[i]];
            }
          }
        }
#ifdef _FTRACE
        ftrace_region_end("RELAX_CF_POINT_0");
#endif
      }
    } else {
      // fprintf(stderr,"AMG RELAX from else\n");
      // Essam; target for the 1st benchmark amg 62
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < n; i++) {
        Vtemp_data[i] = u_data[i];
      }
      prod = (1.0 - relax_weight * omega);
      if (relax_points == 0) {
#ifdef _FTRACE
        ftrace_region_begin("RELAX_POINT_0");
#endif
        if (num_threads > 1) {
          tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel private(i, ii, j, jj, t_id, ns, ne, res, rest, size)      \
    firstprivate(u_data)
#endif
          {
            HYPRE_Int ik, jk, freq;

            size = n / num_threads;
            rest = n - size * num_threads;
            t_id = omp_get_thread_num();
            if (t_id < rest) {
              ns = t_id * size + t_id;
              ne = (t_id + 1) * size + t_id + 1;
            } else {
              ns = t_id * size + rest;
              ne = (t_id + 1) * size + rest;
            }

#if 0
            for (ik = ns; ik < ne; ik++) /* interior points first */
            {
              i = A_diag->level_idx[ik];
              if (i != -1) {
                res0 = 0.0;
                res2 = 0.0;
                for (jj = A_diag->ms_i[ik]; jj < A_diag->ms_i[ik + 1]; jj++) {
                  ii = A_diag->ms_j[jj];
                  res0 -=
                      A_diag->f_act_rows[jj] * A_diag->ms_data[jj] * u_data[ii];
                  res2 += A_diag->f_act_rows[jj] * A_diag->ms_data[jj] *
                          Vtemp_data[ii];
                  A_offd_res[i] -= (A_diag->f_act_rows[jj] ^ 1) *
                                   A_diag->ms_data[jj] * tmp_data[ii];
                }

                u_data[i] *= prod;
                u_data[i] +=
                    relax_weight *
                    (omega * A_offd_res[i] + res0 + one_minus_omega * res2) /
                    A_diag->ms_vdata[i];
              }
            }
#elif 1
            double t_res[(ne - ns) * 3];
            int *level_idx = A_diag->level_idx;
            int *ms_i = A_diag->ms_i;
            int *ms_j = A_diag->ms_j;
            int *ms_rows_freq = A_diag->ms_rows_freq;
            double *ms_data = A_diag->ms_data;
            // debug_print(t_id, "\noptimized implementation\n");
            // const int MAX_NNZ = A_diag->max_nnz_row;
            int MAX_NNZ;
            double t_data[n];
            for (i = 0; i < n; i++)
              t_data[i] = u_data[i];

            for (ik = ns; ik < ne; ik += MAX(1, freq)) {
              i = level_idx[ik];
              if (i != -1) {
                freq = ms_rows_freq[2 * ik + 1];
                MAX_NNZ = ms_rows_freq[2 * ik];
                for (jk = 0; jk < freq; jk++) {
                  t_res[3 * jk] = 0;
                  t_res[3 * jk + 1] = 0;
                  t_res[3 * jk + 2] = 0;
                  // i = level_idx[ik + jk];
                  // t_res[jk] = A_offd_res[i];
                }

                if (freq < MAX_NNZ) {
                  for (jk = 0; jk < freq; jk++) {
                    const int strt = ms_i[t_id + ik] + jk * MAX_NNZ;
                    const int end = ms_i[t_id + ik] + (jk + 1) * MAX_NNZ;

                    for (jj = strt; jj < end; jj++) {
                      ii = ms_j[jj];
                      // t_res[jk] -= ms_data[jj] * t_data[ii];
                      t_res[3 * jk] -= A_diag->f_act_rows[jj] *
                                       A_diag->ms_data[jj] * t_data[ii];
                      t_res[3 * jk + 1] += A_diag->f_act_rows[jj] *
                                           A_diag->ms_data[jj] * Vtemp_data[ii];
                      t_res[3 * jk + 2] += (A_diag->f_act_rows[jj] ^ 1) *
                                           A_diag->ms_data[jj] * t_data[ii];
                    }
                  }
                } else {
                  for (jj = 0; jj < MAX_NNZ; jj++) {
                    for (jk = 0; jk < freq; jk++) {
                      int jmp = ms_i[t_id + ik] + jk * MAX_NNZ;
                      ii = ms_j[jmp + jj];
                      // t_res[jk] -= ms_data[jmp + jj] * t_data[ii];
                      t_res[3 * jk] -= A_diag->f_act_rows[jj] *
                                       A_diag->ms_data[jj] * t_data[ii];
                      t_res[3 * jk + 1] += A_diag->f_act_rows[jj] *
                                           A_diag->ms_data[jj] * Vtemp_data[ii];
                      t_res[3 * jk + 2] += (A_diag->f_act_rows[jj] ^ 1) *
                                           A_diag->ms_data[jj] * t_data[ii];
                    }
                  }
                }
                for (jk = 0; jk < freq; jk++) {
                  i = level_idx[ik + jk];
                  // t_data[i] = t_res[jk] / A_diag->ms_vdata[i];
                  A_offd_res[i] -= t_res[3 * jk + 2];

                  t_data[i] *= prod;
                  t_data[i] += relax_weight *
                               (omega * A_offd_res[i] + t_res[3 * jk] +
                                one_minus_omega * t_res[3 * jk + 1]) /
                               A_diag->ms_vdata[i];
                }
              }
            }
#else
            for (i = ns; i < ne; i++) /* interior points first */
            {
              /*-----------------------------------------------------------
               * If diagonal is nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/
              if (A_diag_data[A_diag_i[i]] != zero) {
                res = f_data[i];
                res0 = 0.0;
                res2 = 0.0;
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res0 -= A_diag_data[jj] * u_data[ii];
                    res2 += A_diag_data[jj] * Vtemp_data[ii];
                  } else {
                    res -= A_diag_data[jj] * tmp_data[ii];
                  }
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] *= prod;
                u_data[i] +=
                    omega *
                    (relax_weight * res + res0 + one_minus_weight * res2) /
                    A_diag_data[A_diag_i[i]];
              }
            }
#endif
          }
        } else {
          for (i = 0; i < n; i++) /* interior points first */
          {
            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/
            if (A_diag_data[A_diag_i[i]] != zero) {
              res0 = 0.0;
              res2 = 0.0;
              //   res = f_data[i];
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res0 -= A_diag_data[jj] * u_data[ii];
                res2 += A_diag_data[jj] * Vtemp_data[ii];
              }
              //   for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
              //     ii = A_offd_j[jj];
              //     res -= A_offd_data[jj] * Vext_data[ii];
              //   }
              u_data[i] *= prod;
              u_data[i] +=
                  relax_weight *
                  (omega * A_offd_res[i] + res0 + one_minus_omega * res2) /
                  A_diag_data[A_diag_i[i]];
              /*u_data[i] += omega*(relax_weight*res + res0 +
                one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
            }
          }
        }
#ifdef _FTRACE
        ftrace_region_end("RELAX_POINT_0");
#endif
      }

      /*-----------------------------------------------------------------
       * Relax only C or F points as determined by relax_points.
       *-----------------------------------------------------------------*/
      else {
#ifdef _FTRACE
        ftrace_region_begin("RELAX_C/F");
#endif
        if (num_threads > 1) {
          tmp_data = Ztemp_data;

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, rest, size)        \
    HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ns; i < ne; i++) /* relax interior points */
            {

              /*-----------------------------------------------------------
               * If i is of the right type ( C or F ) and diagonal is
               * nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (cf_marker[i] == relax_points &&
                  A_diag_data[A_diag_i[i]] != zero) {
                res0 = 0.0;
                res2 = 0.0;
                //  res = f_data[i];
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res0 -= A_diag_data[jj] * u_data[ii];
                    res2 += A_diag_data[jj] * Vtemp_data[ii];
                  } else
                    A_offd_res[i] -= A_diag_data[jj] * tmp_data[ii];
                }
                //  for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                //    ii = A_offd_j[jj];
                //    res -= A_offd_data[jj] * Vext_data[ii];
                //  }
                u_data[i] *= prod;
                u_data[i] +=
                    relax_weight *
                    (omega * A_offd_res[i] + res0 + one_minus_omega * res2) /
                    A_diag_data[A_diag_i[i]];
                /*u_data[i] += omega*(relax_weight*res + res0 +
                  one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
              }
            }
          }

        } else {
          for (i = 0; i < n; i++) /* relax interior points */
          {

            /*-----------------------------------------------------------
             * If i is of the right type ( C or F ) and diagonal is

             * nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (cf_marker[i] == relax_points &&
                A_diag_data[A_diag_i[i]] != zero) {
              //   res = f_data[i];
              res0 = 0.0;
              res2 = 0.0;
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res0 -= A_diag_data[jj] * u_data[ii];
                res2 += A_diag_data[jj] * Vtemp_data[ii];
              }
              //   for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
              //     ii = A_offd_j[jj];
              //     res -= A_offd_data[jj] * Vext_data[ii];
              //   }
              u_data[i] *= prod;
              u_data[i] +=
                  relax_weight *
                  (omega * A_offd_res[i] + res0 + one_minus_omega * res2) /
                  A_diag_data[A_diag_i[i]];
              /*u_data[i] += omega*(relax_weight*res + res0 +
                one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
            }
          }
        }
#ifdef _FTRACE
        ftrace_region_end("RELAX_C/F");
#endif
      }
    }
#ifndef HYPRE_USING_PERSISTENT_COMM
    if (num_procs > 1) {
      hypre_TFree(Vext_data, HYPRE_MEMORY_HOST);
      hypre_TFree(v_buf_data, HYPRE_MEMORY_HOST);
    }
#endif
#ifdef HYPRE_PROFILE
    hypre_profile_times[HYPRE_TIMER_ID_RELAX] += hypre_MPI_Wtime();
#endif
  } break;

  case 1: /* Gauss-Seidel VERY SLOW */
  {
    if (num_procs > 1) {
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);

      v_buf_data = hypre_CTAlloc(
          HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
          HYPRE_MEMORY_HOST);

      Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);

      status = hypre_CTAlloc(hypre_MPI_Status, num_recvs + num_sends,
                             HYPRE_MEMORY_HOST);
      requests = hypre_CTAlloc(hypre_MPI_Request, num_recvs + num_sends,
                               HYPRE_MEMORY_HOST);

      if (num_cols_offd) {
        A_offd_j = hypre_CSRMatrixJ(A_offd);
        A_offd_data = hypre_CSRMatrixData(A_offd);
      }

      /*-----------------------------------------------------------------
       * Copy current approximation into temporary vector.
       *-----------------------------------------------------------------*/
      /*
         for (i = 0; i < n; i++)
         {
         Vtemp_data[i] = u_data[i];
         } */
    }
    /*-----------------------------------------------------------------
     * Relax all points.
     *-----------------------------------------------------------------*/
    for (p = 0; p < num_procs; p++) {
      jr = 0;
      if (p != my_id) {
        for (i = 0; i < num_sends; i++) {
          ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
          if (ip == p) {
            vec_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            vec_len =
                hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - vec_start;
            for (j = vec_start; j < vec_start + vec_len; j++)
              v_buf_data[j] =
                  u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
            hypre_MPI_Isend(&v_buf_data[vec_start], vec_len, HYPRE_MPI_REAL, ip,
                            0, comm, &requests[jr++]);
          }
        }
        hypre_MPI_Waitall(jr, requests, status);
        hypre_MPI_Barrier(comm);
      } else {
        if (num_procs > 1) {
          for (i = 0; i < num_recvs; i++) {
            ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
            vec_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
            vec_len =
                hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i + 1) - vec_start;
            hypre_MPI_Irecv(&Vext_data[vec_start], vec_len, HYPRE_MPI_REAL, ip,
                            0, comm, &requests[jr++]);
          }
          hypre_MPI_Waitall(jr, requests, status);
        }
        if (relax_points == 0) {
          for (i = 0; i < n; i++) {

            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (A_diag_data[A_diag_i[i]] != zero) {
              res = f_data[i];
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res -= A_diag_data[jj] * u_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] = res / A_diag_data[A_diag_i[i]];
            }
          }
        }

        /*-----------------------------------------------------------------
         * Relax only C or F points as determined by relax_points.
         *-----------------------------------------------------------------*/

        else {
          for (i = 0; i < n; i++) {

            /*-----------------------------------------------------------
             * If i is of the right type ( C or F ) and diagonal is
             * nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (cf_marker[i] == relax_points &&
                A_diag_data[A_diag_i[i]] != zero) {
              res = f_data[i];
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res -= A_diag_data[jj] * u_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] = res / A_diag_data[A_diag_i[i]];
            }
          }
        }
        if (num_procs > 1)
          hypre_MPI_Barrier(comm);
      }
    }
    if (num_procs > 1) {
      hypre_TFree(Vext_data, HYPRE_MEMORY_HOST);
      hypre_TFree(v_buf_data, HYPRE_MEMORY_HOST);
      hypre_TFree(status, HYPRE_MEMORY_HOST);
      hypre_TFree(requests, HYPRE_MEMORY_HOST);
    }
  } break;

  case 2: /* Gauss-Seidel: relax interior points in parallel, boundary
             sequentially */
  {
    if (num_procs > 1) {
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);

      v_buf_data = hypre_CTAlloc(
          HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
          HYPRE_MEMORY_HOST);

      Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);

      status = hypre_CTAlloc(hypre_MPI_Status, num_recvs + num_sends,
                             HYPRE_MEMORY_HOST);
      requests = hypre_CTAlloc(hypre_MPI_Request, num_recvs + num_sends,
                               HYPRE_MEMORY_HOST);

      if (num_cols_offd) {
        A_offd_j = hypre_CSRMatrixJ(A_offd);
        A_offd_data = hypre_CSRMatrixData(A_offd);
      }
    }

    /*-----------------------------------------------------------------
     * Copy current approximation into temporary vector.
     *-----------------------------------------------------------------*/
    /*
       for (i = 0; i < n; i++)
       {
       Vtemp_data[i] = u_data[i];
       } */

    /*-----------------------------------------------------------------
     * Relax interior points first
     *-----------------------------------------------------------------*/
    if (relax_points == 0) {
      for (i = 0; i < n; i++) {

        /*-----------------------------------------------------------
         * If diagonal is nonzero, relax point i; otherwise, skip it.
         *-----------------------------------------------------------*/

        if ((A_offd_i[i + 1] - A_offd_i[i]) == zero &&
            A_diag_data[A_diag_i[i]] != zero) {
          res = f_data[i];
          for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
            ii = A_diag_j[jj];
            res -= A_diag_data[jj] * u_data[ii];
          }
          u_data[i] = res / A_diag_data[A_diag_i[i]];
        }
      }
    } else {
      for (i = 0; i < n; i++) {

        /*-----------------------------------------------------------
         * If i is of the right type ( C or F ) and diagonal is
         * nonzero, relax point i; otherwise, skip it.
         *-----------------------------------------------------------*/

        if (cf_marker[i] == relax_points &&
            (A_offd_i[i + 1] - A_offd_i[i]) == zero &&
            A_diag_data[A_diag_i[i]] != zero) {
          res = f_data[i];
          for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
            ii = A_diag_j[jj];
            res -= A_diag_data[jj] * u_data[ii];
          }
          u_data[i] = res / A_diag_data[A_diag_i[i]];
        }
      }
    }
    for (p = 0; p < num_procs; p++) {
      jr = 0;
      if (p != my_id) {
        for (i = 0; i < num_sends; i++) {
          ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
          if (ip == p) {
            vec_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            vec_len =
                hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - vec_start;
            for (j = vec_start; j < vec_start + vec_len; j++)
              v_buf_data[j] =
                  u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
            hypre_MPI_Isend(&v_buf_data[vec_start], vec_len, HYPRE_MPI_REAL, ip,
                            0, comm, &requests[jr++]);
          }
        }
        hypre_MPI_Waitall(jr, requests, status);
        hypre_MPI_Barrier(comm);
      } else {
        if (num_procs > 1) {
          for (i = 0; i < num_recvs; i++) {
            ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
            vec_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
            vec_len =
                hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i + 1) - vec_start;
            hypre_MPI_Irecv(&Vext_data[vec_start], vec_len, HYPRE_MPI_REAL, ip,
                            0, comm, &requests[jr++]);
          }
          hypre_MPI_Waitall(jr, requests, status);
        }
        if (relax_points == 0) {
          for (i = 0; i < n; i++) {

            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if ((A_offd_i[i + 1] - A_offd_i[i]) != zero &&
                A_diag_data[A_diag_i[i]] != zero) {
              res = f_data[i];
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res -= A_diag_data[jj] * u_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] = res / A_diag_data[A_diag_i[i]];
            }
          }
        }

        /*-----------------------------------------------------------------
         * Relax only C or F points as determined by relax_points.
         *-----------------------------------------------------------------*/

        else {
          for (i = 0; i < n; i++) {

            /*-----------------------------------------------------------
             * If i is of the right type ( C or F ) and diagonal is
             * nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (cf_marker[i] == relax_points &&
                (A_offd_i[i + 1] - A_offd_i[i]) != zero &&
                A_diag_data[A_diag_i[i]] != zero) {
              res = f_data[i];
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res -= A_diag_data[jj] * u_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] = res / A_diag_data[A_diag_i[i]];
            }
          }
        }
        if (num_procs > 1)
          hypre_MPI_Barrier(comm);
      }
    }
    if (num_procs > 1) {
      hypre_TFree(Vext_data, HYPRE_MEMORY_HOST);
      hypre_TFree(v_buf_data, HYPRE_MEMORY_HOST);
      hypre_TFree(status, HYPRE_MEMORY_HOST);
      hypre_TFree(requests, HYPRE_MEMORY_HOST);
    }
  } break;

  case 4: /* Hybrid: Jacobi off-processor,
             Gauss-Seidel/SOR on-processor
             (backward loop) */
  {
    if (num_procs > 1) {
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

      v_buf_data = hypre_CTAlloc(
          HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
          HYPRE_MEMORY_HOST);

      Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);

      if (num_cols_offd) {
        A_offd_j = hypre_CSRMatrixJ(A_offd);
        A_offd_data = hypre_CSRMatrixData(A_offd);
      }

      index = 0;
      for (i = 0; i < num_sends; i++) {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);
             j++)
          v_buf_data[index++] =
              u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }

      comm_handle =
          hypre_ParCSRCommHandleCreate(1, comm_pkg, v_buf_data, Vext_data);

      /*-----------------------------------------------------------------
       * Copy current approximation into temporary vector.
       *-----------------------------------------------------------------*/
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
    }

    /*-----------------------------------------------------------------
     * Relax all points.
     *-----------------------------------------------------------------*/

    if (relax_weight == 1 && omega == 1) {
      if (relax_points == 0) {
        if (num_threads > 1) {
          tmp_data = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, rest, size)        \
    HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ne - 1; i > ns - 1; i--) /* interior points first */
            {

              /*-----------------------------------------------------------
               * If diagonal is nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (A_diag_data[A_diag_i[i]] != zero) {
                res = f_data[i];
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne)
                    res -= A_diag_data[jj] * u_data[ii];
                  else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] = res / A_diag_data[A_diag_i[i]];
              }
            }
          }
          hypre_TFree(tmp_data, HYPRE_MEMORY_HOST);
        } else {
          for (i = n - 1; i > -1; i--) /* interior points first */
          {

            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (A_diag_data[A_diag_i[i]] != zero) {
              res = f_data[i];
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res -= A_diag_data[jj] * u_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] = res / A_diag_data[A_diag_i[i]];
            }
          }
        }
      }

      /*-----------------------------------------------------------------
       * Relax only C or F points as determined by relax_points.
       *-----------------------------------------------------------------*/

      else {
        if (num_threads > 1) {
          tmp_data = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, rest, size)        \
    HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ne - 1; i > ns - 1; i--) /* relax interior points */
            {

              /*-----------------------------------------------------------
               * If i is of the right type ( C or F ) and diagonal is
               * nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (cf_marker[i] == relax_points &&
                  A_diag_data[A_diag_i[i]] != zero) {
                res = f_data[i];
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne)
                    res -= A_diag_data[jj] * u_data[ii];
                  else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] = res / A_diag_data[A_diag_i[i]];
              }
            }
          }
          hypre_TFree(tmp_data, HYPRE_MEMORY_HOST);

        } else {
          for (i = n - 1; i > -1; i--) /* relax interior points */
          {

            /*-----------------------------------------------------------
             * If i is of the right type ( C or F ) and diagonal is

             * nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (cf_marker[i] == relax_points &&
                A_diag_data[A_diag_i[i]] != zero) {
              res = f_data[i];
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res -= A_diag_data[jj] * u_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] = res / A_diag_data[A_diag_i[i]];
            }
          }
        }
      }
    } else {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < n; i++) {
        Vtemp_data[i] = u_data[i];
      }
      prod = (1.0 - relax_weight * omega);
      if (relax_points == 0) {
        if (num_threads > 1) {
          tmp_data = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, rest, size)        \
    HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ne - 1; i > ns - 1; i--) /* interior points first */
            {

              /*-----------------------------------------------------------
               * If diagonal is nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (A_diag_data[A_diag_i[i]] != zero) {
                res = f_data[i];
                res0 = 0.0;
                res2 = 0.0;
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res0 -= A_diag_data[jj] * u_data[ii];
                    res2 += A_diag_data[jj] * Vtemp_data[ii];
                  } else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] *= prod;
                u_data[i] += relax_weight *
                             (omega * res + res0 + one_minus_omega * res2) /
                             A_diag_data[A_diag_i[i]];
                /*u_data[i] += omega*(relax_weight*res + res0 +
                  one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
              }
            }
          }
          hypre_TFree(tmp_data, HYPRE_MEMORY_HOST);

        } else {
          for (i = n - 1; i > -1; i--) /* interior points first */
          {

            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (A_diag_data[A_diag_i[i]] != zero) {
              res0 = 0.0;
              res2 = 0.0;
              res = f_data[i];
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res0 -= A_diag_data[jj] * u_data[ii];
                res2 += A_diag_data[jj] * Vtemp_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] *= prod;
              u_data[i] += relax_weight *
                           (omega * res + res0 + one_minus_omega * res2) /
                           A_diag_data[A_diag_i[i]];
              /*u_data[i] += omega*(relax_weight*res + res0 +
                one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
            }
          }
        }
      }

      /*-----------------------------------------------------------------
       * Relax only C or F points as determined by relax_points.
       *-----------------------------------------------------------------*/

      else {
        if (num_threads > 1) {
          tmp_data = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, res0, res2, rest,  \
                                 size) HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ne - 1; i > ns - 1; i--) /* relax interior points */
            {

              /*-----------------------------------------------------------
               * If i is of the right type ( C or F ) and diagonal is
               * nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (cf_marker[i] == relax_points &&
                  A_diag_data[A_diag_i[i]] != zero) {
                res0 = 0.0;
                res2 = 0.0;
                res = f_data[i];
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res0 -= A_diag_data[jj] * u_data[ii];
                    res2 += A_diag_data[jj] * Vtemp_data[ii];
                  } else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] *= prod;
                u_data[i] += relax_weight *
                             (omega * res + res0 + one_minus_omega * res2) /
                             A_diag_data[A_diag_i[i]];
                /*u_data[i] += omega*(relax_weight*res + res0 +
                  one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
              }
            }
          }
          hypre_TFree(tmp_data, HYPRE_MEMORY_HOST);
        } else {
          for (i = n - 1; i > -1; i--) /* relax interior points */
          {

            /*-----------------------------------------------------------
             * If i is of the right type ( C or F ) and diagonal is

             * nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (cf_marker[i] == relax_points &&
                A_diag_data[A_diag_i[i]] != zero) {
              res = f_data[i];
              res0 = 0.0;
              res2 = 0.0;
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res0 -= A_diag_data[jj] * u_data[ii];
                res2 += A_diag_data[jj] * Vtemp_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] *= prod;
              u_data[i] += relax_weight *
                           (omega * res + res0 + one_minus_omega * res2) /
                           A_diag_data[A_diag_i[i]];
              /*u_data[i] += omega*(relax_weight*res + res0 +
                one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
            }
          }
        }
      }
    }
    if (num_procs > 1) {
      hypre_TFree(Vext_data, HYPRE_MEMORY_HOST);
      hypre_TFree(v_buf_data, HYPRE_MEMORY_HOST);
    }
  } break;

  case 6: /* Hybrid: Jacobi off-processor,
             Symm. Gauss-Seidel/ SSOR on-processor
             with outer relaxation parameter */
  {

    if (num_threads > 1) {
      Ztemp_local = hypre_ParVectorLocalVector(Ztemp);
      Ztemp_data = hypre_VectorData(Ztemp_local);
    }

    /*-----------------------------------------------------------------
     * Copy current approximation into temporary vector.
     *-----------------------------------------------------------------*/
    if (num_procs > 1) {
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

      v_buf_data = hypre_CTAlloc(
          HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
          HYPRE_MEMORY_HOST);

      Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);

      if (num_cols_offd) {
        A_offd_j = hypre_CSRMatrixJ(A_offd);
        A_offd_data = hypre_CSRMatrixData(A_offd);
      }

      index = 0;
      for (i = 0; i < num_sends; i++) {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);
             j++)
          v_buf_data[index++] =
              u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }

      comm_handle =
          hypre_ParCSRCommHandleCreate(1, comm_pkg, v_buf_data, Vext_data);

      /*-----------------------------------------------------------------
       * Copy current approximation into temporary vector.
       *-----------------------------------------------------------------*/
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
    }

    /*-----------------------------------------------------------------
     * Relax all points.
     *-----------------------------------------------------------------*/

    /*-----------------------------------------------------------------
     * ESSAM: replace A_offd_i SpMV routine
     *-----------------------------------------------------------------*/
#ifdef _FTRACE
    ftrace_region_begin("RELAX_CF_CASE_6_DATA");
#endif
    sblas_int_t s_ierr;
    HYPRE_Int jmp = 1, t_id, inc, nn;
    HYPRE_Real A_offd_res[global_num_rows];
    if (!A_offd->hnd) {
      // A_offd->flag=1;
      sblas_int_t mrow = (sblas_int_t)n;
      sblas_int_t ncol =
          (sblas_int_t)num_cols_offd; // always passed as zero ..!

      sblas_int_t *iaptr = A_offd_i;
      sblas_int_t *iaind = A_offd_j;
      double *avals = A_offd_data;

      s_ierr = sblas_create_matrix_handle_from_csr_rd(
          mrow, mrow, iaptr, iaind, avals, SBLAS_INDEXING_0, SBLAS_GENERAL,
          &A_offd->hnd); // handler

      s_ierr = sblas_analyze_mv_rd(SBLAS_NON_TRANSPOSE, A_offd->hnd);
// multi-level scheduling
#if 1

      int *rows;
      int *level;
      int m;

      int max_nnz_row = 0;
#pragma omp parallel for private(i, j) reduction(max : max_nnz_row)
      for (i = 0; i < n; i++) {
        j = A_diag_i[i + 1] - A_diag_i[i] /*- 1*/;
        max_nnz_row = (max_nnz_row < j) ? j : max_nnz_row;
      }
      int nnz = n * max_nnz_row;

      A_diag->max_nnz_row = max_nnz_row;

      // d_fprintf("The max num of nnz per rows: %d\n", max_nnz_row);

      int t_levels_idx[n];
      int t_nnz_rows[n];

      A_diag->ms_rows_freq = (int *)malloc(sizeof(int) * 4 * n);
      A_diag->ms_i = (int *)malloc(sizeof(int) * 2 * (n + num_threads));

      A_diag->ms_j = (int *)malloc(sizeof(int) * 2 * nnz);
      A_diag->ms_data = (double *)malloc(sizeof(double) * 2 * nnz);

#pragma omp parallel for if (nnz > 4069)
      for (i = 0; i < 2 * nnz; i++) {
        A_diag->ms_j[i] = 0;
        A_diag->ms_data[i] = 0;
      }

      level = (int *)malloc(sizeof(int) * n);
      rows = (int *)malloc(sizeof(int) * n);

      A_diag->level_idx = (int *)malloc(sizeof(int) * 2 * n);
      // A_diag->ms_rhs_idx = (int *)malloc(sizeof(int) * nnz);

      A_diag->f_act_rows = (int *)malloc(sizeof(int) * 2 * nnz);
      A_diag->ms_vdata = (double *)malloc(sizeof(double) * n);

      asl_sort_t sort, usort;
      asl_library_initialize();
      asl_sort_create_i32(&sort, ASL_SORTORDER_ASCENDING,
                          ASL_SORTALGORITHM_AUTO_STABLE);
      asl_sort_preallocate(sort, n);

#pragma omp parallel private(i, j, jj, ii, t_id, ns, ne, rest, size, m)
      {

        size = n / num_threads;
        rest = n - size * num_threads;
        t_id = omp_get_thread_num();

        if (t_id < rest) {
          ns = t_id * size + t_id;
          ne = (t_id + 1) * size + t_id + 1;
        } else {
          ns = t_id * size + rest;
          ne = (t_id + 1) * size + rest;
        }

        for (i = ns; i < ne; i++)
          level[i] = -1;

        for (i = ns; i < ne; i++) {
          m = -1;
          rows[i] = -1;
          A_diag->ms_vdata[i] = A_diag_data[A_diag_i[i]];
          t_nnz_rows[i] = A_diag_i[i + 1] - A_diag_i[i] - 1;
          if (A_diag->ms_vdata[i] != zero) {
            rows[i] = i;
#pragma _NEC novector
            for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
              ii = A_diag_j[jj];
              if (ii >= ns && ii < ne) {
                if (ii < i && m < level[ii])
                  m = level[ii];
                // if (ii > i)
                //   level[ii] = m + 1;
              }
            }
            // Check if some previous rows needs my value.
            if (level[i] > m + 1) {
              m = level[i];
            } else {
              level[i] = m + 1;
            }
// Set all columns after my diagonal to my level.
#pragma _NEC novector
            for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
              ii = A_diag_j[jj];
              if (ii >= ns && ii < ne)
                if (ii > i && A_diag->ms_vdata[ii] != zero)
                  level[ii] = m + 1;
            }
          }
        }
        int nelem = ne - ns;

        asl_sort_execute_i32(sort, nelem, &level[ns], &rows[ns], &level[ns],
                             &t_levels_idx[ns]);

        for (i = A_diag_i[ns]; i < A_diag_i[ne]; i++)
          A_diag->f_act_rows[i] = 0;

        int _ns, _ne;

        int idx_ia, idx_rhs, idx, k, ik, prev, prev_i, max_nnz;

        _ns = ns + t_id;
        _ne = ne + t_id;
        idx_ia = 0;
        idx = 0;

        int *ms_i = A_diag->ms_i + _ns;
        ms_i[0] = ns * max_nnz_row; // A_diag_i[ns];

        int *ms_j = A_diag->ms_j + ms_i[0];
        double *ms_data = A_diag->ms_data + ms_i[0];
        int *f_act_rows = A_diag->f_act_rows + ms_i[0];
        int *ms_rows_freq = A_diag->ms_rows_freq;
        int *level_idx = A_diag->level_idx;

        for (i = ns; i < ne; i++) {
          ms_rows_freq[2 * i] = 0;
          ms_rows_freq[2 * i + 1] = 0;
        }

        prev = -1; // level[ns];
        prev_i = ns;
        max_nnz = 0;
        ms_rows_freq[2 * prev_i] = 0;
        // if (prev != -1)
        //   ms_rows_freq[2 * prev_i + 1] = 1;
        for (i = ns; i < ne; i++) {
          if (level[i] != -1) {
            if (level[i] == prev) {
              ms_rows_freq[2 * prev_i + 1]++;
              if (ms_rows_freq[2 * prev_i] < t_nnz_rows[t_levels_idx[i]])
                ms_rows_freq[2 * prev_i] = t_nnz_rows[t_levels_idx[i]];
            } else {
              // set the max nnz for all rows with a level
              for (j = prev_i + 1; j < ns + idx_ia; j++)
                ms_rows_freq[2 * j] = ms_rows_freq[2 * prev_i];

              prev_i = ns + idx_ia;
              prev = level[i];
              ms_rows_freq[2 * prev_i + 1] = 1;
              ms_rows_freq[2 * prev_i] = t_nnz_rows[t_levels_idx[i]];
            }
            idx_ia++;
          }
        }
        // the last set of rows
        for (j = prev_i + 1; j < ne; j++)
          ms_rows_freq[2 * j] = ms_rows_freq[2 * prev_i];

        idx_ia = 0;

        for (i = ns; i < ne; i++) {

          level_idx[i] = -1;
          ii = t_levels_idx[i];
          if (ii != -1) {
            level_idx[ns + idx_ia] = ii;

            for (jj = (A_diag_i[ii] + 1); jj < A_diag_i[ii + 1]; jj++) {
              k = jj - (A_diag_i[ii] + 1);
              ik = A_diag_j[jj];

              ms_j[idx + k] = ik;
              ms_data[idx + k] = A_diag_data[jj];

              if (ik >= ns && ik < ne)
                f_act_rows[idx + k] = 1;
            }
            idx += ms_rows_freq[2 * (ns + idx_ia)]; // max_nnz_row;

            ms_i[idx_ia + 1] =
                ms_i[idx_ia] +
                (ms_rows_freq[2 * (ns + idx_ia)] /*max_nnz_row*/);
            idx_ia++;
          }
        }

        for (i = idx_ia + 1; i < (ne - ns); i++) {
          ms_i[i] = ms_i[idx_ia];
        }

        // back substitution multi-level scheduling
        for (i = ns; i < ne; i++)
          level[i] = -1;

        for (i = ne - 1; i > ns - 1; i--) {
          m = -1;
          rows[i] = -1;
          if (A_diag->ms_vdata[i] != zero) {
            rows[i] = i;
#pragma _NEC novector
            for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
              ii = A_diag_j[jj];
              if (ii >= ns && ii < ne) {
                if (ii > i && m < level[ii])
                  m = level[ii];
                // if (ii < i)
                //   level[ii] = m + 1;
              }
            }
            if (level[i] > m + 1)
              m = level[i];
            else
              level[i] = m + 1;
#pragma _NEC novector
            for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
              ii = A_diag_j[jj];
              if (ii >= ns && ii < ne) {
                // if (ii > i && m < level[ii])
                //   m = level[ii];
                if (ii < i && A_diag->ms_vdata[ii] != zero)
                  level[ii] = m + 1;
              }
            }
          }
        }

        asl_sort_execute_i32(sort, nelem, &level[ns], &rows[ns], &level[ns],
                             &t_levels_idx[ns]);
        // copy data..
        _ns = ns + t_id;
        _ne = ne + t_id;
        idx_ia = 0;
        idx = 0;

        ms_i = A_diag->ms_i + (n + num_threads) + _ns;
        ms_i[0] = ns * max_nnz_row; // A_diag_i[ns];

        ms_j = A_diag->ms_j + nnz + ms_i[0];
        ms_data = A_diag->ms_data + nnz + ms_i[0];
        f_act_rows = A_diag->f_act_rows + nnz + ms_i[0];
        ms_rows_freq = A_diag->ms_rows_freq + 2 * n;
        level_idx = A_diag->level_idx + n;

        for (i = ns; i < ne; i++) {
          ms_rows_freq[2 * i] = 0;
          ms_rows_freq[2 * i + 1] = 0;
        }

        prev = -1; // level[ns];
        prev_i = ns;
        max_nnz = 0;
        ms_rows_freq[2 * prev_i] = 0;
        // if (prev != -1)
        //   ms_rows_freq[2 * prev_i + 1] = 1;

        for (i = ns; i < ne; i++) {
          if (level[i] != -1) {
            if (level[i] == prev) {
              ms_rows_freq[2 * prev_i + 1]++;
              if (ms_rows_freq[2 * prev_i] < t_nnz_rows[t_levels_idx[i]])
                ms_rows_freq[2 * prev_i] = t_nnz_rows[t_levels_idx[i]];
            } else {
              // set the max nnz for all rows with a level
              for (j = prev_i + 1; j < ns + idx_ia; j++)
                ms_rows_freq[2 * j] = ms_rows_freq[2 * prev_i];

              prev_i = ns + idx_ia;
              prev = (level[i] != -1) ? level[i] : level[++i];
              ms_rows_freq[2 * prev_i + 1] = 1;
              ms_rows_freq[2 * prev_i] = t_nnz_rows[t_levels_idx[i]];
            }
            idx_ia++;
          }
        }
        // set the last level
        for (j = prev_i + 1; j < ne; j++)
          ms_rows_freq[2 * j] = ms_rows_freq[2 * prev_i];

        idx_ia = 0;
        for (i = ns; i < ne; i++) {
          level_idx[i] = -1;
          ii = t_levels_idx[i];
          if (ii != -1) {
            level_idx[ns + idx_ia] = ii;

            for (jj = (A_diag_i[ii] + 1); jj < A_diag_i[ii + 1]; jj++) {
              k = jj - (A_diag_i[ii] + 1);
              ik = A_diag_j[jj];

              ms_j[idx + k] = ik;
              ms_data[idx + k] = A_diag_data[jj];

              if (ik >= ns && ik < ne)
                f_act_rows[idx + k] = 1;
            }
            idx += /*max_nnz_row*/ ms_rows_freq[2 * (ns + idx_ia)];

            ms_i[idx_ia + 1] =
                ms_i[idx_ia] +
                (/*max_nnz_row*/ ms_rows_freq[2 * (ns + idx_ia)]);
            idx_ia++;
          }
        }

        for (i = idx_ia + 1; i < (ne - ns); i++) {
          ms_i[i] = ms_i[idx_ia];
        }
        //
      }

      /* Sorting Finalization */
      asl_sort_destroy(sort);
      /* Library Finalization */
      asl_library_finalize();

      // free aux arrays
      free(level);
      free(rows);

#endif
    }
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i)
#endif
    for (i = 0; i < n; i++) {
      A_offd_res[i] = f_data[i];
    }
    s_ierr = sblas_execute_mv_rd(SBLAS_NON_TRANSPOSE, A_offd->hnd, -1.0,
                                 Vext_data, 1.0, A_offd_res);
#ifdef _FTRACE
    ftrace_region_end("RELAX_CF_CASE_6_DATA");
#endif

    /*-----------------------------------------------------------------
     * End of level scheduling.
     *-----------------------------------------------------------------*/

    if (relax_weight == 1 && omega == 1) {
      if (relax_points == 0) {

        // Essam: solver 61 & 1
        if (num_threads >= 1) {

#if 0 // original implementation

          tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, rest, size)        \
    HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ns; i < ne; i++) /* interior points first */
            {

              /*-----------------------------------------------------------
               * If diagonal is nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (A_diag_data[A_diag_i[i]] != zero) {
                res = f_data[i];
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res -= A_diag_data[jj] * u_data[ii];
                  } else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] = res / A_diag_data[A_diag_i[i]];
              }
            }
            for (i = ne - 1; i > ns - 1; i--) /* interior points first */
            {

              /*-----------------------------------------------------------
               * If diagonal is nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (A_diag_data[A_diag_i[i]] != zero) {
                res = f_data[i];
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res -= A_diag_data[jj] * u_data[ii];
                  } else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] = res / A_diag_data[A_diag_i[i]];
              }
            }
          }
#else // modified implmentation
      // #ifdef HYPRE_USING_OPENMP
      // #pragma omp parallel private(i, ii, j, jj, t_id, ns, ne, res, rest,
      // size)      \
//     firstprivate(u_data)
      // #endif
      //           {
      //             HYPRE_Int ik, jk, t_id;
      //             t_id = omp_get_thread_num();
      //             size = n / num_threads;
      //             rest = n - size * num_threads;
      //             if (t_id < rest) {
      //               ns = t_id * size + t_id;
      //               ne = (t_id + 1) * size + t_id + 1;
      //             } else {
      //               ns = t_id * size + rest;
      //               ne = (t_id + 1) * size + rest;
      //             }
      // #if 0
      //             for (i = ns; i < ne; i++) /* interior points first */
      //             {

          //               /*-----------------------------------------------------------
          //                * If diagonal is nonzero, relax point i; otherwise,
          //                skip it.
          //                *-----------------------------------------------------------*/

          //               if (A_diag_data[A_diag_i[i]] != zero) {
          //                 res = A_offd_res[i];
          //                 for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1];
          //                 jj++) {
          //                   ii = A_diag_j[jj];
          //                   res -= A_diag_data[jj] * u_data[ii];
          //                 }
          //                 u_data[i] = res / A_diag_data[A_diag_i[i]];
          //               }
          //             }
          // #else
          //             // forward updated with multi-level
          //             HYPRE_Int freq;
          //             HYPRE_Real freq_res[ne - ns];
          //             for (ik = ns; ik < ne; ik++) {
          //               i = A_diag->level_idx[ik];
          //               if (i != -1) {
          //                 freq = A_diag->ms_rows_freq[2 * i + 1];
          //                 for (jk = 0; jk < freq; jk++) {
          //                   i = A_diag->level_idx[ik + jk];
          //                   freq_res[jk] = A_offd_res[i];
          //                 }
          //                 // #pragma _NEC collapse
          //                 for (jk = 0; jk < freq; jk++) {
          //                   // #pragma _NEC outerloop_unroll(16)
          //                   for (jj = A_diag->ms_i[t_id + ik + jk];
          //                        jj < A_diag->ms_i[t_id + ik + jk + 1]; jj++)
          //                        {
          //                     ii = A_diag->ms_j[jj];
          //                     freq_res[jk] -= A_diag->ms_data[jj] *
          //                     u_data[ii];
          //                   }
          //                 }
          //                 // #pragma _NEC ivdep
          //                 for (jk = 0; jk < freq; jk++) {
          //                   i = A_diag->level_idx[ik + jk];
          //                   u_data[i] = freq_res[jk] / A_diag->ms_vdata[i];
          //                 }

          //                 ik += freq;
          //               }
          //             }
          // #endif
          //             for (i = ne - 1; i > ns - 1; i--) /* interior points
          //             first */
          //             {

          //               /*-----------------------------------------------------------
          //                * If diagonal is nonzero, relax point i; otherwise,
          //                skip it.
          //                *-----------------------------------------------------------*/

          //               if (A_diag_data[A_diag_i[i]] != zero) {
          //                 res = A_offd_res[i];
          //                 for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1];
          //                 jj++) {
          //                   ii = A_diag_j[jj];
          //                   res -= A_diag_data[jj] * u_data[ii];
          //                 }
          //                 u_data[i] = res / A_diag_data[A_diag_i[i]];
          //               }
          //             }
          //           }
          const int nnz = A_diag->max_nnz_row * n;
          // fprintf(stderr, "from case 6\n");
#ifdef _FTRACE
    ftrace_region_begin("RELAX_CF_CASE_6");
#endif
#pragma omp parallel private(i, ii, j, jj, t_id, ns, ne, res, rest, size)
          {
            double t_data[n];
            for (i = 0; i < n; i++)
              t_data[i] = u_data[i];
            int freq;

            int ik, jk;

            t_id = omp_get_thread_num();
            size = n / num_threads;
            rest = n - size * num_threads;
            if (t_id < rest) {
              ns = t_id * size + t_id;
              ne = (t_id + 1) * size + t_id + 1;
            } else {
              ns = t_id * size + rest;
              ne = (t_id + 1) * size + rest;
            }
            double t_res[ne - ns];
            int *level_idx = A_diag->level_idx;
            int *ms_i = A_diag->ms_i;
            int *ms_j = A_diag->ms_j;
            int *ms_rows_freq = A_diag->ms_rows_freq;
            double *ms_data = A_diag->ms_data;
            // debug_print(t_id, "\noptimized implementation\n");
            // const int MAX_NNZ = A_diag->max_nnz_row;
            int MAX_NNZ;

#if 1
            // forward updated with multi-level
            for (ik = ns; ik < ne; ik += MAX(1, freq)) {
              i = level_idx[ik];
              if (i != -1) {
                freq = ms_rows_freq[2 * ik + 1];
                MAX_NNZ = ms_rows_freq[2 * ik];
                for (jk = 0; jk < freq; jk++) {
                  i = level_idx[ik + jk];
                  t_res[jk] = A_offd_res[i];
                }

                if (freq < MAX_NNZ) {
                  for (jk = 0; jk < freq; jk++) {
                    const int strt = ms_i[t_id + ik] + jk * MAX_NNZ;
                    const int end = ms_i[t_id + ik] + (jk + 1) * MAX_NNZ;

                    for (jj = strt; jj < end; jj++) {
                      ii = ms_j[jj];
                      t_res[jk] -= ms_data[jj] * t_data[ii];
                    }
                  }
                } else {
                  for (jj = 0; jj < MAX_NNZ; jj++) {
                    for (jk = 0; jk < freq; jk++) {
                      int jmp = ms_i[t_id + ik] + jk * MAX_NNZ;
                      ii = ms_j[jmp + jj];
                      t_res[jk] -= ms_data[jmp + jj] * t_data[ii];
                    }
                  }
                }
                for (jk = 0; jk < freq; jk++) {
                  i = level_idx[ik + jk];
                  t_data[i] = t_res[jk] / A_diag->ms_vdata[i];
                }
              }
            }
#endif

// backward substitution
#if 1

            level_idx = A_diag->level_idx + n;
            ms_i = A_diag->ms_i + (n + num_threads);
            ms_j = A_diag->ms_j + nnz;
            ms_rows_freq = A_diag->ms_rows_freq + 2 * n;
            ms_data = A_diag->ms_data + nnz;
            jk = 0;
            freq = 0;
            for (ik = ns; ik < ne; ik += MAX(freq, 1)) {
              i = level_idx[ik];
              if (i != -1) {
                freq = ms_rows_freq[2 * ik + 1];
                MAX_NNZ = ms_rows_freq[2 * ik];
                for (jk = 0; jk < freq; jk++) {
                  i = level_idx[ik + jk];
                  t_res[jk] = A_offd_res[i];
                }
                if (freq < MAX_NNZ) {
                  for (jk = 0; jk < freq; jk++) {
                    const int strt = ms_i[t_id + ik] + jk * MAX_NNZ;
                    const int end = ms_i[t_id + ik] + (jk + 1) * MAX_NNZ;

                    for (jj = strt; jj < end; jj++) {
                      ii = ms_j[jj];
                      t_res[jk] -= ms_data[jj] * t_data[ii];
                    }
                  }
                } else {
                  for (jj = 0; jj < MAX_NNZ; jj++) {
                    for (jk = 0; jk < freq; jk++) {
                      int jmp = ms_i[t_id + ik] + jk * MAX_NNZ;
                      ii = ms_j[jmp + jj];
                      t_res[jk] -= ms_data[jmp + jj] * t_data[ii];
                    }
                  }
                }

                for (jk = 0; jk < freq; jk++) {
                  i = level_idx[ik + jk];
                  t_data[i] = t_res[jk] / A_diag->ms_vdata[i];
                }
              }
            }
#endif

            for (i = ns; i < ne; i++)
              u_data[i] = t_data[i];
          }

#endif
#ifdef _FTRACE
    ftrace_region_end("RELAX_CF_CASE_6");
#endif
        } else {
          for (i = 0; i < n; i++) /* interior points first */
          {

            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (A_diag_data[A_diag_i[i]] != zero) {
              res = f_data[i];
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res -= A_diag_data[jj] * u_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] = res / A_diag_data[A_diag_i[i]];
            }
          }
          for (i = n - 1; i > -1; i--) /* interior points first */
          {

            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (A_diag_data[A_diag_i[i]] != zero) {
              res = f_data[i];
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res -= A_diag_data[jj] * u_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] = res / A_diag_data[A_diag_i[i]];
            }
          }
        }
      }

      /*-----------------------------------------------------------------
       * Relax only C or F points as determined by relax_points.
       *-----------------------------------------------------------------*/

      else {
        if (num_threads > 1) {
          tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, rest, size)        \
    HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ns; i < ne; i++) /* relax interior points */
            {

              /*-----------------------------------------------------------
               * If i is of the right type ( C or F ) and diagonal is
               * nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (cf_marker[i] == relax_points &&
                  A_diag_data[A_diag_i[i]] != zero) {
                res = f_data[i];
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res -= A_diag_data[jj] * u_data[ii];
                  } else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] = res / A_diag_data[A_diag_i[i]];
              }
            }
            for (i = ne - 1; i > ns - 1; i--) /* relax interior points */
            {

              /*-----------------------------------------------------------
               * If i is of the right type ( C or F ) and diagonal is
               * nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (cf_marker[i] == relax_points &&
                  A_diag_data[A_diag_i[i]] != zero) {
                res = f_data[i];
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res -= A_diag_data[jj] * u_data[ii];
                  } else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] = res / A_diag_data[A_diag_i[i]];
              }
            }
          }

        } else {
          for (i = 0; i < n; i++) /* relax interior points */
          {

            /*-----------------------------------------------------------
             * If i is of the right type ( C or F ) and diagonal is

             * nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (cf_marker[i] == relax_points &&
                A_diag_data[A_diag_i[i]] != zero) {
              res = f_data[i];
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res -= A_diag_data[jj] * u_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] = res / A_diag_data[A_diag_i[i]];
            }
          }
          for (i = n - 1; i > -1; i--) /* relax interior points */
          {

            /*-----------------------------------------------------------
             * If i is of the right type ( C or F ) and diagonal is

             * nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (cf_marker[i] == relax_points &&
                A_diag_data[A_diag_i[i]] != zero) {
              res = f_data[i];
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res -= A_diag_data[jj] * u_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] = res / A_diag_data[A_diag_i[i]];
            }
          }
        }
      }
    } else {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < n; i++) {
        Vtemp_data[i] = u_data[i];
      }
      prod = (1.0 - relax_weight * omega);
      if (relax_points == 0) {
        if (num_threads > 1) {
          tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, res0, res2, rest,  \
                                 size) HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ns; i < ne; i++) /* interior points first */
            {

              /*-----------------------------------------------------------
               * If diagonal is nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (A_diag_data[A_diag_i[i]] != zero) {
                res0 = 0.0;
                res2 = 0.0;
                res = f_data[i];
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res0 -= A_diag_data[jj] * u_data[ii];
                    res2 += A_diag_data[jj] * Vtemp_data[ii];
                  } else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] *= prod;
                u_data[i] += relax_weight *
                             (omega * res + res0 + one_minus_omega * res2) /
                             A_diag_data[A_diag_i[i]];
                /*u_data[i] += omega*(relax_weight*res + res0 +
                  one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
              }
            }
            for (i = ne - 1; i > ns - 1; i--) /* interior points first */
            {

              /*-----------------------------------------------------------
               * If diagonal is nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (A_diag_data[A_diag_i[i]] != zero) {
                res0 = 0.0;
                res2 = 0.0;
                res = f_data[i];
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res0 -= A_diag_data[jj] * u_data[ii];
                    res2 += A_diag_data[jj] * Vtemp_data[ii];
                  } else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] *= prod;
                u_data[i] += relax_weight *
                             (omega * res + res0 + one_minus_omega * res2) /
                             A_diag_data[A_diag_i[i]];
                /*u_data[i] += omega*(relax_weight*res + res0 +
                  one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
              }
            }
          }

        } else {
          for (i = 0; i < n; i++) /* interior points first */
          {

            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (A_diag_data[A_diag_i[i]] != zero) {
              res0 = 0.0;
              res = f_data[i];
              res2 = 0.0;
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res0 -= A_diag_data[jj] * u_data[ii];
                res2 += A_diag_data[jj] * Vtemp_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] *= prod;
              u_data[i] += relax_weight *
                           (omega * res + res0 + one_minus_omega * res2) /
                           A_diag_data[A_diag_i[i]];
              /*u_data[i] += omega*(relax_weight*res + res0 +
                one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
            }
          }
          for (i = n - 1; i > -1; i--) /* interior points first */
          {

            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (A_diag_data[A_diag_i[i]] != zero) {
              res0 = 0.0;
              res = f_data[i];
              res2 = 0.0;
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res0 -= A_diag_data[jj] * u_data[ii];
                res2 += A_diag_data[jj] * Vtemp_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] *= prod;
              u_data[i] += relax_weight *
                           (omega * res + res0 + one_minus_omega * res2) /
                           A_diag_data[A_diag_i[i]];
              /*u_data[i] += omega*(relax_weight*res + res0 +
                one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
            }
          }
        }
      }

      /*-----------------------------------------------------------------
       * Relax only C or F points as determined by relax_points.
       *-----------------------------------------------------------------*/

      else {
        if (num_threads > 1) {
          tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, res0, res2, rest,  \
                                 size) HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ns; i < ne; i++) /* relax interior points */
            {

              /*-----------------------------------------------------------
               * If i is of the right type ( C or F ) and diagonal is
               * nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (cf_marker[i] == relax_points &&
                  A_diag_data[A_diag_i[i]] != zero) {
                res0 = 0.0;
                res2 = 0.0;
                res = f_data[i];
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res2 += A_diag_data[jj] * Vtemp_data[ii];
                    res0 -= A_diag_data[jj] * u_data[ii];
                  } else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] *= prod;
                u_data[i] += relax_weight *
                             (omega * res + res0 + one_minus_omega * res2) /
                             A_diag_data[A_diag_i[i]];
                /*u_data[i] += omega*(relax_weight*res + res0 +
                  one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
              }
            }
            for (i = ne - 1; i > ns - 1; i--) /* relax interior points */
            {

              /*-----------------------------------------------------------
               * If i is of the right type ( C or F ) and diagonal is
               * nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (cf_marker[i] == relax_points &&
                  A_diag_data[A_diag_i[i]] != zero) {
                res0 = 0.0;
                res2 = 0.0;
                res = f_data[i];
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res2 += A_diag_data[jj] * Vtemp_data[ii];
                    res0 -= A_diag_data[jj] * u_data[ii];
                  } else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] *= prod;
                u_data[i] += relax_weight *
                             (omega * res + res0 + one_minus_omega * res2) /
                             A_diag_data[A_diag_i[i]];
                /*u_data[i] += omega*(relax_weight*res + res0 +
                  one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
              }
            }
          }

        } else {
          for (i = 0; i < n; i++) /* relax interior points */
          {

            /*-----------------------------------------------------------
             * If i is of the right type ( C or F ) and diagonal is

             * nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (cf_marker[i] == relax_points &&
                A_diag_data[A_diag_i[i]] != zero) {
              res = f_data[i];
              res0 = 0.0;
              res2 = 0.0;
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res0 -= A_diag_data[jj] * u_data[ii];
                res2 += A_diag_data[jj] * Vtemp_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] *= prod;
              u_data[i] += relax_weight *
                           (omega * res + res0 + one_minus_omega * res2) /
                           A_diag_data[A_diag_i[i]];
              /*u_data[i] += omega*(relax_weight*res + res0 +
                one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
            }
          }
          for (i = n - 1; i > -1; i--) /* relax interior points */
          {

            /*-----------------------------------------------------------
             * If i is of the right type ( C or F ) and diagonal is

             * nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (cf_marker[i] == relax_points &&
                A_diag_data[A_diag_i[i]] != zero) {
              res = f_data[i];
              res0 = 0.0;
              res2 = 0.0;
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res0 -= A_diag_data[jj] * u_data[ii];
                res2 += A_diag_data[jj] * Vtemp_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] *= prod;
              u_data[i] += relax_weight *
                           (omega * res + res0 + one_minus_omega * res2) /
                           A_diag_data[A_diag_i[i]];
              /*u_data[i] += omega*(relax_weight*res + res0 +
                one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
            }
          }
        }
      }
    }
    if (num_procs > 1) {
      hypre_TFree(Vext_data, HYPRE_MEMORY_HOST);
      hypre_TFree(v_buf_data, HYPRE_MEMORY_HOST);
    }
  } break;

  case 7: /* Jacobi (uses ParMatvec) */
  {

    /*-----------------------------------------------------------------
     * Copy f into temporary vector.
     *-----------------------------------------------------------------*/
    // hypre_SeqVectorPrefetch(hypre_ParVectorLocalVector(Vtemp),
    // HYPRE_MEMORY_DEVICE);
    // hypre_SeqVectorPrefetch(hypre_ParVectorLocalVector(f),
    // HYPRE_MEMORY_DEVICE);
    hypre_ParVectorCopy(f, Vtemp);

    /*-----------------------------------------------------------------
     * Perform Matvec Vtemp=f-Au
     *-----------------------------------------------------------------*/

    hypre_ParCSRMatrixMatvec(-relax_weight, A, u, relax_weight, Vtemp);
#if defined(HYPRE_USING_CUDA)
    hypreDevice_IVAXPY(n, l1_norms, Vtemp_data, u_data);
#else
    for (i = 0; i < n; i++) {
      /*-----------------------------------------------------------
       * If diagonal is nonzero, relax point i; otherwise, skip it.
       *-----------------------------------------------------------*/
      u_data[i] += Vtemp_data[i] / l1_norms[i];
    }
#endif
  } break;

  case 8: /* hybrid L1 Symm. Gauss-Seidel */
  {
    if (num_threads > 1) {
      Ztemp_local = hypre_ParVectorLocalVector(Ztemp);
      Ztemp_data = hypre_VectorData(Ztemp_local);
    }

    /*-----------------------------------------------------------------
     * Copy current approximation into temporary vector.
     *-----------------------------------------------------------------*/
    if (num_procs > 1) {
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

      v_buf_data = hypre_CTAlloc(
          HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
          HYPRE_MEMORY_HOST);

      Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);

      if (num_cols_offd) {
        A_offd_j = hypre_CSRMatrixJ(A_offd);
        A_offd_data = hypre_CSRMatrixData(A_offd);
      }

      index = 0;
      for (i = 0; i < num_sends; i++) {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);
             j++) {
          v_buf_data[index++] =
              u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
        }
      }

      comm_handle =
          hypre_ParCSRCommHandleCreate(1, comm_pkg, v_buf_data, Vext_data);

      /*-----------------------------------------------------------------
       * Copy current approximation into temporary vector.
       *-----------------------------------------------------------------*/
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
    }

    /*-----------------------------------------------------------------
     * Relax all points.
     *-----------------------------------------------------------------*/

    if (relax_weight == 1 && omega == 1) {
      if (relax_points == 0) {
        if (num_threads > 1) {
          tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, rest, size)        \
    HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ns; i < ne; i++) /* interior points first */
            {

              /*-----------------------------------------------------------
               * If diagonal is nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (l1_norms[i] != zero) {
                res = f_data[i];
                for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res -= A_diag_data[jj] * u_data[ii];
                  } else {
                    res -= A_diag_data[jj] * tmp_data[ii];
                  }
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] += res / l1_norms[i];
              }
            }
            for (i = ne - 1; i > ns - 1; i--) /* interior points first */
            {

              /*-----------------------------------------------------------
               * If diagonal is nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (l1_norms[i] != zero) {
                res = f_data[i];
                for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res -= A_diag_data[jj] * u_data[ii];
                  } else {
                    res -= A_diag_data[jj] * tmp_data[ii];
                  }
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] += res / l1_norms[i];
              }
            }
          }

        } else {
          for (i = 0; i < n; i++) /* interior points first */
          {

            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (l1_norms[i] != zero) {
              res = f_data[i];
              for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res -= A_diag_data[jj] * u_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] += res / l1_norms[i];
            }
          }
          for (i = n - 1; i > -1; i--) /* interior points first */
          {

            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (l1_norms[i] != zero) {
              res = f_data[i];
              for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res -= A_diag_data[jj] * u_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] += res / l1_norms[i];
            }
          }
        }
      }
      /*-----------------------------------------------------------------
       * Relax only C or F points as determined by relax_points.
       *-----------------------------------------------------------------*/
      else {
        if (num_threads > 1) {
          tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, rest, size)        \
    HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ns; i < ne; i++) /* relax interior points */
            {

              /*-----------------------------------------------------------
               * If i is of the right type ( C or F ) and diagonal is
               * nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (cf_marker[i] == relax_points && l1_norms[i] != zero) {
                res = f_data[i];
                for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res -= A_diag_data[jj] * u_data[ii];
                  } else {
                    res -= A_diag_data[jj] * tmp_data[ii];
                  }
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] += res / l1_norms[i];
              }
            }
            for (i = ne - 1; i > ns - 1; i--) /* relax interior points */
            {

              /*-----------------------------------------------------------
               * If i is of the right type ( C or F ) and diagonal is
               * nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (cf_marker[i] == relax_points && l1_norms[i] != zero) {
                res = f_data[i];
                for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res -= A_diag_data[jj] * u_data[ii];
                  } else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] += res / l1_norms[i];
              }
            }
          }
        } else {
          for (i = 0; i < n; i++) /* relax interior points */
          {

            /*-----------------------------------------------------------
             * If i is of the right type ( C or F ) and diagonal is

             * nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (cf_marker[i] == relax_points && l1_norms[i] != zero) {
              res = f_data[i];
              for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res -= A_diag_data[jj] * u_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] += res / l1_norms[i];
            }
          }
          for (i = n - 1; i > -1; i--) /* relax interior points */
          {

            /*-----------------------------------------------------------
             * If i is of the right type ( C or F ) and diagonal is

             * nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (cf_marker[i] == relax_points && l1_norms[i] != zero) {
              res = f_data[i];
              for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res -= A_diag_data[jj] * u_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] += res / l1_norms[i];
            }
          }
        }
      }
    } else {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < n; i++) {
        Vtemp_data[i] = u_data[i];
      }
      prod = (1.0 - relax_weight * omega);
      if (relax_points == 0) {
        if (num_threads > 1) {
          tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, rest, size)        \
    HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ns; i < ne; i++) /* interior points first */
            {

              /*-----------------------------------------------------------
               * If diagonal is nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (l1_norms[i] != zero) {
                res0 = 0.0;
                res2 = 0.0;
                res = f_data[i];
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res0 -= A_diag_data[jj] * u_data[ii];
                    res2 += A_diag_data[jj] * Vtemp_data[ii];
                  } else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] *= prod;
                u_data[i] += relax_weight *
                             (omega * res + res0 + one_minus_omega * res2) /
                             l1_norms[i];
                /*u_data[i] += omega*(relax_weight*res + res0 +
                  one_minus_weight*res2) / l1_norms[i];*/
              }
            }
            for (i = ne - 1; i > ns - 1; i--) /* interior points first */
            {

              /*-----------------------------------------------------------
               * If diagonal is nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (l1_norms[i] != zero) {
                res0 = 0.0;
                res2 = 0.0;
                res = f_data[i];
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res0 -= A_diag_data[jj] * u_data[ii];
                    res2 += A_diag_data[jj] * Vtemp_data[ii];
                  } else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] *= prod;
                u_data[i] += relax_weight *
                             (omega * res + res0 + one_minus_omega * res2) /
                             l1_norms[i];
                /*u_data[i] += omega*(relax_weight*res + res0 +
                  one_minus_weight*res2) / l1_norms[i];*/
              }
            }
          }
        } else {
          for (i = 0; i < n; i++) /* interior points first */
          {

            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (l1_norms[i] != zero) {
              res0 = 0.0;
              res = f_data[i];
              res2 = 0.0;
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res0 -= A_diag_data[jj] * u_data[ii];
                res2 += A_diag_data[jj] * Vtemp_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] *= prod;
              u_data[i] += relax_weight *
                           (omega * res + res0 + one_minus_omega * res2) /
                           l1_norms[i];
              /*u_data[i] += omega*(relax_weight*res + res0 +
                one_minus_weight*res2) / l1_norms[i];*/
            }
          }
          for (i = n - 1; i > -1; i--) /* interior points first */
          {

            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (l1_norms[i] != zero) {
              res0 = 0.0;
              res = f_data[i];
              res2 = 0.0;
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res0 -= A_diag_data[jj] * u_data[ii];
                res2 += A_diag_data[jj] * Vtemp_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] *= prod;
              u_data[i] += relax_weight *
                           (omega * res + res0 + one_minus_omega * res2) /
                           l1_norms[i];
              /*u_data[i] += omega*(relax_weight*res + res0 +
                one_minus_weight*res2) / l1_norms[i];*/
            }
          }
        }
      }
      /*-----------------------------------------------------------------
       * Relax only C or F points as determined by relax_points.
       *-----------------------------------------------------------------*/
      else {
        if (num_threads > 1) {
          tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, rest, size)        \
    HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ns; i < ne; i++) /* relax interior points */
            {

              /*-----------------------------------------------------------
               * If i is of the right type ( C or F ) and diagonal is
               * nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (cf_marker[i] == relax_points && l1_norms[i] != zero) {
                res0 = 0.0;
                res2 = 0.0;
                res = f_data[i];
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res2 += A_diag_data[jj] * Vtemp_data[ii];
                    res0 -= A_diag_data[jj] * u_data[ii];
                  } else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] *= prod;
                u_data[i] += relax_weight *
                             (omega * res + res0 + one_minus_omega * res2) /
                             l1_norms[i];
                /*u_data[i] += omega*(relax_weight*res + res0 +
                  one_minus_weight*res2) / l1_norms[i];*/
              }
            }
            for (i = ne - 1; i > ns - 1; i--) /* relax interior points */
            {

              /*-----------------------------------------------------------
               * If i is of the right type ( C or F ) and diagonal is
               * nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (cf_marker[i] == relax_points && l1_norms[i] != zero) {
                res0 = 0.0;
                res2 = 0.0;
                res = f_data[i];
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res2 += A_diag_data[jj] * Vtemp_data[ii];
                    res0 -= A_diag_data[jj] * u_data[ii];
                  } else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] *= prod;
                u_data[i] += relax_weight *
                             (omega * res + res0 + one_minus_omega * res2) /
                             l1_norms[i];
                /*u_data[i] += omega*(relax_weight*res + res0 +
                  one_minus_weight*res2) / l1_norms[i];*/
              }
            }
          }

        } else {
          for (i = 0; i < n; i++) /* relax interior points */
          {

            /*-----------------------------------------------------------
             * If i is of the right type ( C or F ) and diagonal is

             * nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (cf_marker[i] == relax_points && l1_norms[i] != zero) {
              res = f_data[i];
              res0 = 0.0;
              res2 = 0.0;
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res0 -= A_diag_data[jj] * u_data[ii];
                res2 += A_diag_data[jj] * Vtemp_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] *= prod;
              u_data[i] += relax_weight *
                           (omega * res + res0 + one_minus_omega * res2) /
                           l1_norms[i];
              /*u_data[i] += omega*(relax_weight*res + res0 +
                one_minus_weight*res2) / l1_norms[i];*/
            }
          }
          for (i = n - 1; i > -1; i--) /* relax interior points */
          {

            /*-----------------------------------------------------------
             * If i is of the right type ( C or F ) and diagonal is

             * nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (cf_marker[i] == relax_points && l1_norms[i] != zero) {
              res = f_data[i];
              res0 = 0.0;
              res2 = 0.0;
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res0 -= A_diag_data[jj] * u_data[ii];
                res2 += A_diag_data[jj] * Vtemp_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] *= prod;
              u_data[i] += relax_weight *
                           (omega * res + res0 + one_minus_omega * res2) /
                           l1_norms[i];
              /*u_data[i] += omega*(relax_weight*res + res0 +
                one_minus_weight*res2) / l1_norms[i];*/
            }
          }
        }
      }
    }
    if (num_procs > 1) {
      hypre_TFree(Vext_data, HYPRE_MEMORY_HOST);
      hypre_TFree(v_buf_data, HYPRE_MEMORY_HOST);
    }
  } break;

  /* Hybrid: Jacobi off-processor, ordered Gauss-Seidel on-processor */
  case 10: {
    if (num_threads > 1) {
      Ztemp_local = hypre_ParVectorLocalVector(Ztemp);
      Ztemp_data = hypre_VectorData(Ztemp_local);
    }

#ifdef HYPRE_USING_PERSISTENT_COMM
    // JSP: persistent comm can be similarly used for other smoothers
    hypre_ParCSRPersistentCommHandle *persistent_comm_handle;
#endif

    if (num_procs > 1) {
#ifdef HYPRE_PROFILE
      hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] -= hypre_MPI_Wtime();
#endif
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

#ifdef HYPRE_USING_PERSISTENT_COMM
      persistent_comm_handle =
          hypre_ParCSRCommPkgGetPersistentCommHandle(1, comm_pkg);
      v_buf_data = (HYPRE_Real *)hypre_ParCSRCommHandleSendDataBuffer(
          persistent_comm_handle);
      Vext_data = (HYPRE_Real *)hypre_ParCSRCommHandleRecvDataBuffer(
          persistent_comm_handle);
#else
      v_buf_data = hypre_CTAlloc(
          HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
          HYPRE_MEMORY_HOST);

      Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);
#endif

      if (num_cols_offd) {
        A_offd_j = hypre_CSRMatrixJ(A_offd);
        A_offd_data = hypre_CSRMatrixData(A_offd);
      }

      HYPRE_Int begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
      HYPRE_Int end = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
      for (i = begin; i < end; i++) {
        v_buf_data[i - begin] =
            u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)];
      }

#ifdef HYPRE_PROFILE
      hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] += hypre_MPI_Wtime();
      hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] -= hypre_MPI_Wtime();
#endif

#ifdef HYPRE_USING_PERSISTENT_COMM
      hypre_ParCSRPersistentCommHandleStart(persistent_comm_handle,
                                            HYPRE_MEMORY_HOST, v_buf_data);
#else
      comm_handle =
          hypre_ParCSRCommHandleCreate(1, comm_pkg, v_buf_data, Vext_data);
#endif

      /*-----------------------------------------------------------------
       * Copy current approximation into temporary vector.
       *-----------------------------------------------------------------*/
#ifdef HYPRE_USING_PERSISTENT_COMM
      hypre_ParCSRPersistentCommHandleWait(persistent_comm_handle,
                                           HYPRE_MEMORY_HOST, Vext_data);
#else
      hypre_ParCSRCommHandleDestroy(comm_handle);
#endif
      comm_handle = NULL;

#ifdef HYPRE_PROFILE
      hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
#endif
    }

    // Check for ordering of matrix. If stored, get pointer, otherwise
    // compute ordering and point matrix variable to array.
    HYPRE_Int *proc_ordering;
    if (!hypre_ParCSRMatrixProcOrdering(A)) {
      proc_ordering = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
      hypre_topo_sort(A_diag_i, A_diag_j, A_diag_data, proc_ordering, n);
      hypre_ParCSRMatrixProcOrdering(A) = proc_ordering;
    } else {
      proc_ordering = hypre_ParCSRMatrixProcOrdering(A);
    }

    /*-----------------------------------------------------------------
     * Relax all points.
     *-----------------------------------------------------------------*/
#ifdef HYPRE_PROFILE
    hypre_profile_times[HYPRE_TIMER_ID_RELAX] -= hypre_MPI_Wtime();
#endif

    if (relax_points == 0) {
      if (num_threads > 1) {
        tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
        for (i = 0; i < n; i++)
          tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, rest, size)        \
    HYPRE_SMP_SCHEDULE
#endif
        for (j = 0; j < num_threads; j++) {
          size = n / num_threads;
          rest = n - size * num_threads;
          if (j < rest) {
            ns = j * size + j;
            ne = (j + 1) * size + j + 1;
          } else {
            ns = j * size + rest;
            ne = (j + 1) * size + rest;
          }
          for (i = ns; i < ne; i++) /* interior points first */
          {
            HYPRE_Int row = proc_ordering[i];
            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point row; otherwise, skip it.
             *-----------------------------------------------------------*/
            if (A_diag_data[A_diag_i[row]] != zero) {
              res = f_data[row];
              for (jj = A_diag_i[row] + 1; jj < A_diag_i[row + 1]; jj++) {
                ii = A_diag_j[jj];
                if (ii >= ns && ii < ne)
                  res -= A_diag_data[jj] * u_data[ii];
                else
                  res -= A_diag_data[jj] * tmp_data[ii];
              }
              for (jj = A_offd_i[row]; jj < A_offd_i[row + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[row] = res / A_diag_data[A_diag_i[row]];
            }
          }
        }
      } else {
        for (i = 0; i < n; i++) /* interior points first */
        {
          HYPRE_Int row = proc_ordering[i];
          /*-----------------------------------------------------------
           * If diagonal is nonzero, relax point i; otherwise, skip it.
           *-----------------------------------------------------------*/
          if (A_diag_data[A_diag_i[row]] != zero) {
            res = f_data[row];
            for (jj = A_diag_i[row] + 1; jj < A_diag_i[row + 1]; jj++) {
              ii = A_diag_j[jj];
              res -= A_diag_data[jj] * u_data[ii];
            }
            for (jj = A_offd_i[row]; jj < A_offd_i[row + 1]; jj++) {
              ii = A_offd_j[jj];
              res -= A_offd_data[jj] * Vext_data[ii];
            }
            u_data[row] = res / A_diag_data[A_diag_i[row]];
          }
        }
      }
    }

    /*-----------------------------------------------------------------
     * Relax only C or F points as determined by relax_points.
     *-----------------------------------------------------------------*/
    else {
      if (num_threads > 1) {
        tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
        for (i = 0; i < n; i++)
          tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, rest, size)        \
    HYPRE_SMP_SCHEDULE
#endif
        for (j = 0; j < num_threads; j++) {
          size = n / num_threads;
          rest = n - size * num_threads;
          if (j < rest) {
            ns = j * size + j;
            ne = (j + 1) * size + j + 1;
          } else {
            ns = j * size + rest;
            ne = (j + 1) * size + rest;
          }
          for (i = ns; i < ne; i++) /* relax interior points */
          {
            HYPRE_Int row = proc_ordering[i];
            /*-----------------------------------------------------------
             * If row is of the right type ( C or F ) and diagonal is
             * nonzero, relax point row; otherwise, skip it.
             *-----------------------------------------------------------*/
            if (cf_marker[row] == relax_points &&
                A_diag_data[A_diag_i[row]] != zero) {
              res = f_data[row];
              for (jj = A_diag_i[row] + 1; jj < A_diag_i[row + 1]; jj++) {
                ii = A_diag_j[jj];
                if (ii >= ns && ii < ne)
                  res -= A_diag_data[jj] * u_data[ii];
                else
                  res -= A_diag_data[jj] * tmp_data[ii];
              }
              for (jj = A_offd_i[row]; jj < A_offd_i[row + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[row] = res / A_diag_data[A_diag_i[row]];
            }
          }
        }
      } else {
        for (i = 0; i < n; i++) /* relax interior points */
        {
          HYPRE_Int row = proc_ordering[i];
          /*-----------------------------------------------------------
           * If row is of the right type ( C or F ) and diagonal is
           * nonzero, relax point row; otherwise, skip it.
           *-----------------------------------------------------------*/
          if (cf_marker[row] == relax_points &&
              A_diag_data[A_diag_i[row]] != zero) {
            res = f_data[row];
            for (jj = A_diag_i[row] + 1; jj < A_diag_i[row + 1]; jj++) {
              ii = A_diag_j[jj];
              res -= A_diag_data[jj] * u_data[ii];
            }
            for (jj = A_offd_i[row]; jj < A_offd_i[row + 1]; jj++) {
              ii = A_offd_j[jj];
              res -= A_offd_data[jj] * Vext_data[ii];
            }
            u_data[row] = res / A_diag_data[A_diag_i[row]];
          }
        }
      }
    }

#ifndef HYPRE_USING_PERSISTENT_COMM
    if (num_procs > 1) {
      hypre_TFree(Vext_data, HYPRE_MEMORY_HOST);
      hypre_TFree(v_buf_data, HYPRE_MEMORY_HOST);
    }
#endif
#ifdef HYPRE_PROFILE
    hypre_profile_times[HYPRE_TIMER_ID_RELAX] += hypre_MPI_Wtime();
#endif
  } break;

  case 13: /* hybrid L1 Gauss-Seidel forward solve */
  {
    if (num_threads > 1) {
      Ztemp_local = hypre_ParVectorLocalVector(Ztemp);
      Ztemp_data = hypre_VectorData(Ztemp_local);
    }

    /*-----------------------------------------------------------------
     * Copy current approximation into temporary vector.
     *-----------------------------------------------------------------*/
    if (num_procs > 1) {
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

      v_buf_data = hypre_CTAlloc(
          HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
          HYPRE_MEMORY_HOST);

      Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);

      if (num_cols_offd) {
        A_offd_j = hypre_CSRMatrixJ(A_offd);
        A_offd_data = hypre_CSRMatrixData(A_offd);
      }

      index = 0;
      for (i = 0; i < num_sends; i++) {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);
             j++)
          v_buf_data[index++] =
              u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }

      comm_handle =
          hypre_ParCSRCommHandleCreate(1, comm_pkg, v_buf_data, Vext_data);

      /*-----------------------------------------------------------------
       * Copy current approximation into temporary vector.
       *-----------------------------------------------------------------*/
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
    }

    /*-----------------------------------------------------------------
     * Relax all points.
     *-----------------------------------------------------------------*/

    if (relax_weight == 1 && omega == 1) {
      if (relax_points == 0) {
        if (num_threads > 1) {
          tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++) {
            tmp_data[i] = u_data[i];
          }
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, rest, size)        \
    HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ns; i < ne; i++) /* interior points first */
            {

              /*-----------------------------------------------------------
               * If diagonal is nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (l1_norms[i] != zero) {
                res = f_data[i];
                for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res -= A_diag_data[jj] * u_data[ii];
                  } else {
                    res -= A_diag_data[jj] * tmp_data[ii];
                  }
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] += res / l1_norms[i];
              }
            }
          }
        } else {
          for (i = 0; i < n; i++) /* interior points first */
          {

            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (l1_norms[i] != zero) {
              res = f_data[i];
              for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res -= A_diag_data[jj] * u_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] += res / l1_norms[i];
            }
          }
        }
      }

      /*-----------------------------------------------------------------
       * Relax only C or F points as determined by relax_points.
       *-----------------------------------------------------------------*/

      else {
        if (num_threads > 1) {
          tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, rest, size)        \
    HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ns; i < ne; i++) /* relax interior points */
            {

              /*-----------------------------------------------------------
               * If i is of the right type ( C or F ) and diagonal is
               * nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (cf_marker[i] == relax_points && l1_norms[i] != zero) {
                res = f_data[i];
                for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res -= A_diag_data[jj] * u_data[ii];
                  } else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] += res / l1_norms[i];
              }
            }
          }

        } else {
          for (i = 0; i < n; i++) /* relax interior points */
          {

            /*-----------------------------------------------------------
             * If i is of the right type ( C or F ) and diagonal is

             * nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (cf_marker[i] == relax_points && l1_norms[i] != zero) {
              res = f_data[i];
              for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res -= A_diag_data[jj] * u_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] += res / l1_norms[i];
            }
          }
        }
      }
    } else {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < n; i++) {
        Vtemp_data[i] = u_data[i];
      }
      prod = (1.0 - relax_weight * omega);
      if (relax_points == 0) {
        if (num_threads > 1) {
          tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, rest, size)        \
    HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ns; i < ne; i++) /* interior points first */
            {

              /*-----------------------------------------------------------
               * If diagonal is nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (l1_norms[i] != zero) {
                res0 = 0.0;
                res2 = 0.0;
                res = f_data[i];
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res0 -= A_diag_data[jj] * u_data[ii];
                    res2 += A_diag_data[jj] * Vtemp_data[ii];
                  } else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] *= prod;
                u_data[i] += relax_weight *
                             (omega * res + res0 + one_minus_omega * res2) /
                             l1_norms[i];
                /*u_data[i] += omega*(relax_weight*res + res0 +
                  one_minus_weight*res2) / l1_norms[i];*/
              }
            }
          }

        } else {
          for (i = 0; i < n; i++) /* interior points first */
          {

            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (l1_norms[i] != zero) {
              res0 = 0.0;
              res = f_data[i];
              res2 = 0.0;
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res0 -= A_diag_data[jj] * u_data[ii];
                res2 += A_diag_data[jj] * Vtemp_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] *= prod;
              u_data[i] += relax_weight *
                           (omega * res + res0 + one_minus_omega * res2) /
                           l1_norms[i];
              /*u_data[i] += omega*(relax_weight*res + res0 +
                one_minus_weight*res2) / l1_norms[i];*/
            }
          }
        }
      }

      /*-----------------------------------------------------------------
       * Relax only C or F points as determined by relax_points.
       *-----------------------------------------------------------------*/

      else {
        if (num_threads > 1) {
          tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, rest, size)        \
    HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ns; i < ne; i++) /* relax interior points */
            {

              /*-----------------------------------------------------------
               * If i is of the right type ( C or F ) and diagonal is
               * nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (cf_marker[i] == relax_points && l1_norms[i] != zero) {
                res0 = 0.0;
                res2 = 0.0;
                res = f_data[i];
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res2 += A_diag_data[jj] * Vtemp_data[ii];
                    res0 -= A_diag_data[jj] * u_data[ii];
                  } else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] *= prod;
                u_data[i] += relax_weight *
                             (omega * res + res0 + one_minus_omega * res2) /
                             l1_norms[i];
                /*u_data[i] += omega*(relax_weight*res + res0 +
                  one_minus_weight*res2) / l1_norms[i];*/
              }
            }
          }

        } else {
          for (i = 0; i < n; i++) /* relax interior points */
          {

            /*-----------------------------------------------------------
             * If i is of the right type ( C or F ) and diagonal is

             * nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (cf_marker[i] == relax_points && l1_norms[i] != zero) {
              res = f_data[i];
              res0 = 0.0;
              res2 = 0.0;
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res0 -= A_diag_data[jj] * u_data[ii];
                res2 += A_diag_data[jj] * Vtemp_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] *= prod;
              u_data[i] += relax_weight *
                           (omega * res + res0 + one_minus_omega * res2) /
                           l1_norms[i];
              /*u_data[i] += omega*(relax_weight*res + res0 +
                one_minus_weight*res2) / l1_norms[i];*/
            }
          }
        }
      }
    }
    if (num_procs > 1) {
      hypre_TFree(Vext_data, HYPRE_MEMORY_HOST);
      hypre_TFree(v_buf_data, HYPRE_MEMORY_HOST);
    }
  } break;

  case 14: /* hybrid L1 Gauss-Seidel backward solve */
  {

    if (num_threads > 1) {
      Ztemp_local = hypre_ParVectorLocalVector(Ztemp);
      Ztemp_data = hypre_VectorData(Ztemp_local);
    }

    /*-----------------------------------------------------------------
     * Copy current approximation into temporary vector.
     *-----------------------------------------------------------------*/
    if (num_procs > 1) {
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

      v_buf_data = hypre_CTAlloc(
          HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
          HYPRE_MEMORY_HOST);

      Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);

      if (num_cols_offd) {
        A_offd_j = hypre_CSRMatrixJ(A_offd);
        A_offd_data = hypre_CSRMatrixData(A_offd);
      }

      index = 0;
      for (i = 0; i < num_sends; i++) {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);
             j++) {
          v_buf_data[index++] =
              u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
        }
      }

      comm_handle =
          hypre_ParCSRCommHandleCreate(1, comm_pkg, v_buf_data, Vext_data);

      /*-----------------------------------------------------------------
       * Copy current approximation into temporary vector.
       *-----------------------------------------------------------------*/
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
    }

    /*-----------------------------------------------------------------
     * Relax all points.
     *-----------------------------------------------------------------*/

    if (relax_weight == 1 && omega == 1) {
      if (relax_points == 0) {
        if (num_threads > 1) {
          tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, rest, size)        \
    HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ne - 1; i > ns - 1; i--) /* interior points first */
            {

              /*-----------------------------------------------------------
               * If diagonal is nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (l1_norms[i] != zero) {
                res = f_data[i];
                for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res -= A_diag_data[jj] * u_data[ii];
                  } else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] += res / l1_norms[i];
              }
            }
          }

        } else {
          for (i = n - 1; i > -1; i--) /* interior points first */
          {

            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (l1_norms[i] != zero) {
              res = f_data[i];
              for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res -= A_diag_data[jj] * u_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] += res / l1_norms[i];
            }
          }
        }
      }

      /*-----------------------------------------------------------------
       * Relax only C or F points as determined by relax_points.
       *-----------------------------------------------------------------*/

      else {
        if (num_threads > 1) {
          tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, rest, size)        \
    HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ne - 1; i > ns - 1; i--) /* relax interior points */
            {

              /*-----------------------------------------------------------
               * If i is of the right type ( C or F ) and diagonal is
               * nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (cf_marker[i] == relax_points && l1_norms[i] != zero) {
                res = f_data[i];
                for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res -= A_diag_data[jj] * u_data[ii];
                  } else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] += res / l1_norms[i];
              }
            }
          }

        } else {
          for (i = n - 1; i > -1; i--) /* relax interior points */
          {

            /*-----------------------------------------------------------
             * If i is of the right type ( C or F ) and diagonal is

             * nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (cf_marker[i] == relax_points && l1_norms[i] != zero) {
              res = f_data[i];
              for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res -= A_diag_data[jj] * u_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] += res / l1_norms[i];
            }
          }
        }
      }
    } else {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < n; i++) {
        Vtemp_data[i] = u_data[i];
      }
      prod = (1.0 - relax_weight * omega);
      if (relax_points == 0) {
        if (num_threads > 1) {
          tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, rest, size)        \
    HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ne - 1; i > ns - 1; i--) /* interior points first */
            {

              /*-----------------------------------------------------------
               * If diagonal is nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (l1_norms[i] != zero) {
                res0 = 0.0;
                res2 = 0.0;
                res = f_data[i];
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res0 -= A_diag_data[jj] * u_data[ii];
                    res2 += A_diag_data[jj] * Vtemp_data[ii];
                  } else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] *= prod;
                u_data[i] += relax_weight *
                             (omega * res + res0 + one_minus_omega * res2) /
                             l1_norms[i];
                /*u_data[i] += omega*(relax_weight*res + res0 +
                  one_minus_weight*res2) / l1_norms[i];*/
              }
            }
          }

        } else {
          for (i = n - 1; i > -1; i--) /* interior points first */
          {

            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (l1_norms[i] != zero) {
              res0 = 0.0;
              res = f_data[i];
              res2 = 0.0;
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res0 -= A_diag_data[jj] * u_data[ii];
                res2 += A_diag_data[jj] * Vtemp_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] *= prod;
              u_data[i] += relax_weight *
                           (omega * res + res0 + one_minus_omega * res2) /
                           l1_norms[i];
              /*u_data[i] += omega*(relax_weight*res + res0 +
                one_minus_weight*res2) / l1_norms[i];*/
            }
          }
        }
      }

      /*-----------------------------------------------------------------
       * Relax only C or F points as determined by relax_points.
       *-----------------------------------------------------------------*/

      else {
        if (num_threads > 1) {
          tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
          for (i = 0; i < n; i++)
            tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j, jj, ns, ne, res, rest, size)        \
    HYPRE_SMP_SCHEDULE
#endif
          for (j = 0; j < num_threads; j++) {
            size = n / num_threads;
            rest = n - size * num_threads;
            if (j < rest) {
              ns = j * size + j;
              ne = (j + 1) * size + j + 1;
            } else {
              ns = j * size + rest;
              ne = (j + 1) * size + rest;
            }
            for (i = ne - 1; i > ns - 1; i--) /* relax interior points */
            {

              /*-----------------------------------------------------------
               * If i is of the right type ( C or F ) and diagonal is
               * nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/

              if (cf_marker[i] == relax_points && l1_norms[i] != zero) {
                res0 = 0.0;
                res2 = 0.0;
                res = f_data[i];
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                  ii = A_diag_j[jj];
                  if (ii >= ns && ii < ne) {
                    res2 += A_diag_data[jj] * Vtemp_data[ii];
                    res0 -= A_diag_data[jj] * u_data[ii];
                  } else
                    res -= A_diag_data[jj] * tmp_data[ii];
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * Vext_data[ii];
                }
                u_data[i] *= prod;
                u_data[i] += relax_weight *
                             (omega * res + res0 + one_minus_omega * res2) /
                             l1_norms[i];
                /*u_data[i] += omega*(relax_weight*res + res0 +
                  one_minus_weight*res2) / l1_norms[i];*/
              }
            }
          }

        } else {
          for (i = n - 1; i > -1; i--) /* relax interior points */
          {

            /*-----------------------------------------------------------
             * If i is of the right type ( C or F ) and diagonal is

             * nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (cf_marker[i] == relax_points && l1_norms[i] != zero) {
              res = f_data[i];
              res0 = 0.0;
              res2 = 0.0;
              for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                ii = A_diag_j[jj];
                res0 -= A_diag_data[jj] * u_data[ii];
                res2 += A_diag_data[jj] * Vtemp_data[ii];
              }
              for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
              }
              u_data[i] *= prod;
              u_data[i] += relax_weight *
                           (omega * res + res0 + one_minus_omega * res2) /
                           l1_norms[i];
              /*u_data[i] += omega*(relax_weight*res + res0 +
                one_minus_weight*res2) / l1_norms[i];*/
            }
          }
        }
      }
    }
    if (num_procs > 1) {
      hypre_TFree(Vext_data, HYPRE_MEMORY_HOST);
      hypre_TFree(v_buf_data, HYPRE_MEMORY_HOST);
    }
  } break;

  case 19: /* Direct solve: use gaussian elimination */
  {
    HYPRE_Int n_global = (HYPRE_Int)global_num_rows;
    HYPRE_Int first_index = (HYPRE_Int)first_ind;
    /*-----------------------------------------------------------------
     *  Generate CSR matrix from ParCSRMatrix A
     *-----------------------------------------------------------------*/
#ifdef HYPRE_NO_GLOBAL_PARTITION
    /* all processors are needed for these routines */
    A_CSR = hypre_ParCSRMatrixToCSRMatrixAll(A);
    f_vector = hypre_ParVectorToVectorAll(f);
#endif
    if (n) {
#ifndef HYPRE_NO_GLOBAL_PARTITION
      A_CSR = hypre_ParCSRMatrixToCSRMatrixAll(A);
      f_vector = hypre_ParVectorToVectorAll(f);
#endif
      A_CSR_i = hypre_CSRMatrixI(A_CSR);
      A_CSR_j = hypre_CSRMatrixJ(A_CSR);
      A_CSR_data = hypre_CSRMatrixData(A_CSR);
      f_vector_data = hypre_VectorData(f_vector);

      A_mat = hypre_CTAlloc(HYPRE_Real, n_global * n_global, HYPRE_MEMORY_HOST);
      b_vec = hypre_CTAlloc(HYPRE_Real, n_global, HYPRE_MEMORY_HOST);

      /*---------------------------------------------------------------
       *  Load CSR matrix into A_mat.
       *---------------------------------------------------------------*/

      for (i = 0; i < n_global; i++) {
        for (jj = A_CSR_i[i]; jj < A_CSR_i[i + 1]; jj++) {
          column = A_CSR_j[jj];
          A_mat[i * n_global + column] = A_CSR_data[jj];
        }
        b_vec[i] = f_vector_data[i];
      }

      hypre_gselim(A_mat, b_vec, n_global, relax_error);

      for (i = 0; i < n; i++) {
        u_data[i] = b_vec[first_index + i];
      }

      hypre_TFree(A_mat, HYPRE_MEMORY_HOST);
      hypre_TFree(b_vec, HYPRE_MEMORY_HOST);
      hypre_CSRMatrixDestroy(A_CSR);
      A_CSR = NULL;
      hypre_SeqVectorDestroy(f_vector);
      f_vector = NULL;
    }
#ifdef HYPRE_NO_GLOBAL_PARTITION
    else {

      hypre_CSRMatrixDestroy(A_CSR);
      A_CSR = NULL;
      hypre_SeqVectorDestroy(f_vector);
      f_vector = NULL;
    }
#endif

  } break;
  case 98: /* Direct solve: use gaussian elimination & BLAS (with pivoting) */
  {

    HYPRE_Int n_global = (HYPRE_Int)global_num_rows;
    HYPRE_Int first_index = (HYPRE_Int)first_ind;
    HYPRE_Int info;
    HYPRE_Int one_i = 1;
    HYPRE_Int *piv;
    /*-----------------------------------------------------------------
     *  Generate CSR matrix from ParCSRMatrix A
     *-----------------------------------------------------------------*/
#ifdef HYPRE_NO_GLOBAL_PARTITION
    /* all processors are needed for these routines */
    A_CSR = hypre_ParCSRMatrixToCSRMatrixAll(A);
    f_vector = hypre_ParVectorToVectorAll(f);
#endif
    if (n) {
#ifndef HYPRE_NO_GLOBAL_PARTITION
      A_CSR = hypre_ParCSRMatrixToCSRMatrixAll(A);
      f_vector = hypre_ParVectorToVectorAll(f);
#endif
      A_CSR_i = hypre_CSRMatrixI(A_CSR);
      A_CSR_j = hypre_CSRMatrixJ(A_CSR);
      A_CSR_data = hypre_CSRMatrixData(A_CSR);
      f_vector_data = hypre_VectorData(f_vector);

      A_mat = hypre_CTAlloc(HYPRE_Real, n_global * n_global, HYPRE_MEMORY_HOST);
      b_vec = hypre_CTAlloc(HYPRE_Real, n_global, HYPRE_MEMORY_HOST);

      /*---------------------------------------------------------------
       *  Load CSR matrix into A_mat.
       *---------------------------------------------------------------*/

      for (i = 0; i < n_global; i++) {
        for (jj = A_CSR_i[i]; jj < A_CSR_i[i + 1]; jj++) {

          /* need col major */
          column = A_CSR_j[jj];
          A_mat[i + n_global * column] = A_CSR_data[jj];
        }
        b_vec[i] = f_vector_data[i];
      }

      piv = hypre_CTAlloc(HYPRE_Int, n_global, HYPRE_MEMORY_HOST);

      /* write over A with LU */
      hypre_dgetrf(&n_global, &n_global, A_mat, &n_global, piv, &info);

      /*now b_vec = inv(A)*b_vec  */
      hypre_dgetrs("N", &n_global, &one_i, A_mat, &n_global, piv, b_vec,
                   &n_global, &info);

      hypre_TFree(piv, HYPRE_MEMORY_HOST);

      for (i = 0; i < n; i++) {
        u_data[i] = b_vec[first_index + i];
      }

      hypre_TFree(A_mat, HYPRE_MEMORY_HOST);
      hypre_TFree(b_vec, HYPRE_MEMORY_HOST);
      hypre_CSRMatrixDestroy(A_CSR);
      A_CSR = NULL;
      hypre_SeqVectorDestroy(f_vector);
      f_vector = NULL;
    }
#ifdef HYPRE_NO_GLOBAL_PARTITION
    else {

      hypre_CSRMatrixDestroy(A_CSR);
      A_CSR = NULL;
      hypre_SeqVectorDestroy(f_vector);
      f_vector = NULL;
    }
#endif
  } break;
  }

  return (relax_error);
}
