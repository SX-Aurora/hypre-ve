/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Member functions for zzz_StructMatrix class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_NewStructMatrix
 *--------------------------------------------------------------------------*/

zzz_StructMatrix *
zzz_NewStructMatrix( zzz_StructGrid    *grid,
                     zzz_StructStencil *user_stencil )
{
   zzz_StructMatrix  *matrix;

   int                i;

   matrix = ctalloc(zzz_StructMatrix, 1);

   zzz_StructMatrixGrid(matrix)        = grid;
   zzz_StructMatrixUserStencil(matrix) = user_stencil;

   /* set defaults */
   zzz_StructMatrixStencilSpace(matrix) =
      zzz_ConvertToSBoxArray(zzz_DuplicateBoxArray(zzz_StructGridBoxes(grid)));
   zzz_StructMatrixSymmetric(matrix) = 0;
   for (i = 0; i < 6; i++)
      zzz_StructMatrixNumGhost(matrix)[i] = 0;

   return matrix;
}

/*--------------------------------------------------------------------------
 * zzz_FreeStructMatrix
 *--------------------------------------------------------------------------*/

int 
zzz_FreeStructMatrix( zzz_StructMatrix *matrix )
{
   int  ierr;

   int  i;

   if (matrix)
   {
      zzz_FreeCommPkg(zzz_StructMatrixCommPkg(matrix));

      zzz_ForBoxI(i, zzz_StructMatrixDataSpace(matrix))
         tfree(zzz_StructMatrixDataIndices(matrix)[i]);
      tfree(zzz_StructMatrixDataIndices(matrix));
      tfree(zzz_StructMatrixData(matrix));

      zzz_FreeBoxArray(zzz_StructMatrixDataSpace(matrix));
      zzz_FreeSBoxArray(zzz_StructMatrixStencilSpace(matrix));

      if (zzz_StructMatrixSymmetric(matrix))
      {
         tfree(zzz_StructMatrixSymmCoeff(matrix));
         zzz_FreeStructStencil(zzz_StructMatrixStencil(matrix));
      }

      tfree(matrix);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_InitializeStructMatrixShell
 *--------------------------------------------------------------------------*/

int 
zzz_InitializeStructMatrixShell( zzz_StructMatrix *matrix )
{
   int    ierr;

   zzz_StructGrid     *grid;

   zzz_StructStencil  *stencil;
   zzz_Index         **stencil_shape;
   int                 stencil_size;
   zzz_StructStencil  *user_stencil;
   zzz_Index         **user_stencil_shape;
   int                 user_stencil_size;
   int                *symm_coeff;
   int                 no_symmetric_stencil_element;

   int                *num_ghost;
 
   zzz_BoxArray       *data_space;
   zzz_BoxArray       *boxes;
   zzz_Box            *box;
   zzz_Box            *data_box;

   int               **data_indices;
   int                 data_size;
   int                 data_box_volume;

   int                 i, j, d;
 
   grid = zzz_StructMatrixGrid(matrix);

   /*-----------------------------------------------------------------------
    * Set up stencil:
    *
    *------------------
    * Non-symmetric case:
    *    Just set pointer to user's stencil, `user_stencil'.
    *
    *------------------
    * Symmetric case:
    *    Copy the user's stencil elements into the new stencil, and
    *    create symmetric stencil elements if needed.
    *
    *    Also set up an array called `symm_coeff'.  A non-zero value
    *    of `symm_coeff[i]' indicates that the `i'th stencil element
    *    is a "symmetric element".  That is, the data associated with
    *    stencil element `i' is not explicitely stored, but is instead
    *    stored as the transpose coefficient at a neighboring grid point.
    *    The value of `symm_coeff[i]' is also the index of the transpose
    *    stencil element.
    *-----------------------------------------------------------------------*/

   user_stencil = zzz_StructMatrixUserStencil(matrix);

   /* non-symmetric case */
   if (!zzz_StructMatrixSymmetric(matrix))
   {
      stencil = user_stencil;
      stencil_size  = zzz_StructStencilSize(stencil);
      stencil_shape = zzz_StructStencilShape(stencil);
   }

   /* symmetric case */
   else
   {
      user_stencil_shape = zzz_StructStencilShape(user_stencil);
      user_stencil_size = zzz_StructStencilSize(user_stencil);

      /* copy user's stencil elements into `stencil_shape' */
      stencil_shape = ctalloc(zzz_Index *, 2*user_stencil_size);
      for (i = 0; i < user_stencil_size; i++)
      {
         for (d = 0; d < 3; d++)
            zzz_IndexD(stencil_shape[i], d) =
               zzz_IndexD(user_stencil_shape[i], d);
      }

      /* create symmetric stencil elements and `symm_coeff' */
      symm_coeff = ctalloc(int, zzz_StructStencilSize(stencil));
      stencil_size = user_stencil_size;
      for (i = 0; i < user_stencil_size; i++)
      {
	 if (!symm_coeff[i])
	 {
            no_symmetric_stencil_element = 1;
            for (j = (i + 1); j < user_stencil_size; j++)
            {
	       if ( (zzz_IndexX(stencil_shape[j]) ==
                     -zzz_IndexX(stencil_shape[i])  ) &&
                    (zzz_IndexY(stencil_shape[j]) ==
                     -zzz_IndexY(stencil_shape[i])  ) &&
                    (zzz_IndexZ(stencil_shape[j]) ==
                     -zzz_IndexZ(stencil_shape[i])  )   )
	       {
		  symm_coeff[j] = i;
                  no_symmetric_stencil_element = 0;
	       }
            }

            if (no_symmetric_stencil_element)
            {
               /* add symmetric stencil element to `stencil' */
               for (d = 0; d < 3; d++)
               {
                  zzz_IndexD(stencil_shape[stencil_size], d) =
                     -zzz_IndexD(stencil_shape[i], d);
               }
	       
               symm_coeff[stencil_size] = i;
               stencil_size++;
	    }
	 }
      }

      stencil = zzz_NewStructStencil(zzz_StructStencilDim(user_stencil),
                                     stencil_size, stencil_shape);
      zzz_StructMatrixSymmCoeff(matrix) = symm_coeff;
   }

   zzz_StructMatrixStencil(matrix) = stencil;

   /*-----------------------------------------------------------------------
    * Set ghost-layer size for symmetric storage
    *-----------------------------------------------------------------------*/

   num_ghost = zzz_StructMatrixNumGhost(matrix);

   if (zzz_StructMatrixSymmetric(matrix))
   {
      for (i = 0; i < stencil_size; i++)
      {
         if (symm_coeff[i])
         {
            j = 0;
            for (d = 0; d < 3; d++)
            {
               num_ghost[j] =
                  max(num_ghost[  j], -zzz_IndexD(stencil_shape[i], d));
               num_ghost[j+1] =
                  max(num_ghost[j+1],  zzz_IndexD(stencil_shape[i], d));
               j += 2;
            }
         }
      }
   }

   /*-----------------------------------------------------------------------
    * Set up data_space
    *-----------------------------------------------------------------------*/

   boxes = zzz_StructGridBoxes(grid);
   data_space = zzz_NewBoxArray();

   zzz_ForBoxI(i, boxes)
   {
      box = zzz_BoxArrayBox(boxes, i);

      data_box = zzz_DuplicateBox(box);
      if (zzz_BoxVolume(data_box))
      {
         for (d = 0; d < 3; d++)
         {
            zzz_BoxIMinD(data_box, d) -= num_ghost[2*d];
            zzz_BoxIMinD(data_box, d) += num_ghost[2*d + 1];
         }
      }

      zzz_AppendBox(data_box, data_space);
   }

   zzz_StructMatrixDataSpace(matrix) = data_space;

   /*-----------------------------------------------------------------------
    * Set up data_indices array and data-size
    *-----------------------------------------------------------------------*/

   data_indices = ctalloc(int *, zzz_BoxArraySize(data_space));

   data_size = 0;
   zzz_ForBoxI(i, data_space)
   {
      data_box = zzz_BoxArrayBox(data_space, i);
      data_box_volume  = zzz_BoxVolume(data_box);

      data_indices[i] = ctalloc(int, stencil_size);

      /* non-symmetric case */
      if (!zzz_StructMatrixSymmetric(matrix))
      {
         for (j = 0; j < stencil_size; j++)
	 {
	    data_indices[i][j] = data_size;
	    data_size += data_box_volume;
	 }
      }

      /* symmetric case */
      else
      {
         /* set pointers for "stored" coefficients */
         for (j = 0; j < stencil_size; j++)
         {
            if (!symm_coeff[j])
            {
               data_indices[i][j] = data_size;
               data_size += data_box_volume;
            }
         }

         /* set pointers for "symmetric" coefficients */
         for (j = 0; j < stencil_size; j++)
         {
            if (symm_coeff[j])
            {
               data_indices[i][j] = data_indices[i][symm_coeff[j]] +
                  zzz_BoxOffsetDistance(data_box, stencil_shape[j]);
            }
         }
      }
   }

   zzz_StructMatrixDataIndices(matrix) = data_indices;
   zzz_StructMatrixDataSize(matrix)    = data_size;

   /*-----------------------------------------------------------------------
    * Set total number of nonzero coefficients
    *-----------------------------------------------------------------------*/

   zzz_StructMatrixGlobalSize(matrix) =
      zzz_StructGridGlobalSize(grid) * stencil_size;

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_InitializeStructMatrixData
 *--------------------------------------------------------------------------*/

void
zzz_InitializeStructMatrixData( zzz_StructMatrix *matrix,
                                double           *data   )
{
   zzz_StructMatrixData(matrix) = data;
}

/*--------------------------------------------------------------------------
 * zzz_InitializeStructMatrix
 *--------------------------------------------------------------------------*/

int 
zzz_InitializeStructMatrix( zzz_StructMatrix *matrix )
{
   int    ierr;

   double *data;

   ierr = zzz_InitializeStructMatrixShell(matrix);

   data = ctalloc(double, zzz_StructMatrixDataSize(matrix));
   zzz_InitializeStructMatrixData(matrix, data);

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SetStructMatrixValues
 *--------------------------------------------------------------------------*/

int 
zzz_SetStructMatrixValues( zzz_StructMatrix *matrix,
                           zzz_Index        *grid_index,
                           int               num_stencil_indices,
                           int              *stencil_indices,
                           double           *values              )
{
   int    ierr;

   zzz_BoxArray     *boxes;
   zzz_Box          *box;
   zzz_Index        *imin;
   zzz_Index        *imax;

   double           *matp;

   int               i, s;

   boxes = zzz_StructGridBoxes(zzz_StructMatrixGrid(matrix));

   zzz_ForBoxI(i, boxes)
   {
      box = zzz_BoxArrayBox(boxes, i);
      imin = zzz_BoxIMin(box);
      imax = zzz_BoxIMax(box);

      if ((zzz_IndexX(grid_index) >= zzz_IndexX(imin)) &&
          (zzz_IndexX(grid_index) <= zzz_IndexX(imax)) &&
          (zzz_IndexY(grid_index) >= zzz_IndexY(imin)) &&
          (zzz_IndexY(grid_index) <= zzz_IndexY(imax)) &&
          (zzz_IndexZ(grid_index) >= zzz_IndexZ(imin)) &&
          (zzz_IndexZ(grid_index) <= zzz_IndexZ(imax))   )
      {
         for (s = 0; s < num_stencil_indices; s++)
         {
            matp = zzz_StructMatrixBoxDataValue(matrix, i, stencil_indices[s],
                                                grid_index);
            *matp = values[s];
         }
      }
   }

   return(ierr);
}

/*--------------------------------------------------------------------------
 * zzz_SetStructMatrixBoxValues
 *--------------------------------------------------------------------------*/

int 
zzz_SetStructMatrixBoxValues( zzz_StructMatrix *matrix,
                              zzz_Box          *value_box,
                              int               num_stencil_indices,
                              int              *stencil_indices,
                              double           *values              )
{
   int    ierr;

   zzz_BoxArray     *box_array;
   zzz_Box          *box;
   zzz_BoxArray     *box_a0;
   zzz_BoxArray     *box_a1;

   zzz_BoxArray     *data_space;
   zzz_Box          *data_box;
   zzz_Index        *index;
   zzz_Index        *stride;

   double           *matp;
   int               mati;

   int               value_index;

   int               i, s, d;

   /*-----------------------------------------------------------------------
    * Set up `box_array' by intersecting `box' with the grid boxes
    *-----------------------------------------------------------------------*/

   box_a0 = zzz_NewBoxArray();
   zzz_AppendBox(value_box, box_a0);
   box_a1 = zzz_StructGridBoxes(zzz_StructMatrixGrid(matrix));

   box_array = zzz_IntersectBoxArrays(box_a0, box_a1);

   zzz_FreeBoxArrayShell(box_a0);

   /*-----------------------------------------------------------------------
    * Set the matrix coefficients
    *-----------------------------------------------------------------------*/

   if (box_array)
   {
      data_space = zzz_StructMatrixDataSpace(matrix);

      index = zzz_NewIndex();

      stride = zzz_NewIndex();
      for (d = 0; d < 3; d++)
         zzz_IndexD(stride, d) = 1;

      zzz_ForBoxI(i, box_array)
      {
         box      = zzz_BoxArrayBox(box_array, i);
         data_box = zzz_BoxArrayBox(data_space, i);
 
         for (s = 0; s < num_stencil_indices; s++)
         {
            matp = zzz_StructMatrixBoxData(matrix, i, stencil_indices[s]);

            value_index = s;
            zzz_BoxLoop1(box, index,
                         data_box, zzz_BoxIMin(box), stride, mati,
                         {
                            matp[mati] = values[value_index];
                            value_index += num_stencil_indices;
                         });
         }
      }

      zzz_FreeIndex(stride);
      zzz_FreeIndex(index);

      zzz_FreeBoxArray(box_array);
   }

   return(ierr);
}

/*--------------------------------------------------------------------------
 * zzz_AssembleStructMatrix
 *--------------------------------------------------------------------------*/

int 
zzz_AssembleStructMatrix( zzz_StructMatrix *matrix )
{
   int    ierr;

   zzz_StructStencil   *comm_stencil;
   zzz_Index          **comm_stencil_shape;
   int                 *num_ghost;
   zzz_Box             *box;
   zzz_Index           *imin;
   zzz_Index           *imax;
   zzz_Index           *index;

   zzz_BoxArrayArray   *send_boxes;
   zzz_BoxArrayArray   *recv_boxes;
   int                **send_processes;
   int                **recv_processes;

   zzz_SBoxArrayArray  *send_sboxes;
   zzz_SBoxArrayArray  *recv_sboxes;
   int                  num_values;
   zzz_CommPkg         *comm_pkg;

   int                  i, d;

   zzz_CommHandle      *comm_handle;

   /*-----------------------------------------------------------------------
    * If the CommPkg has not been set up, set it up
    *-----------------------------------------------------------------------*/

   comm_pkg = zzz_StructMatrixCommPkg(matrix);

   if (!comm_pkg)
   {
      /* Set up the stencil describing communications, `comm_stencil' */

      num_ghost = zzz_StructMatrixNumGhost(matrix);
      imin = zzz_NewIndex();
      imax = zzz_NewIndex();
      for (d = 0; d < 3; d++)
      {
         zzz_IndexD(imin, d) = -num_ghost[2*d];
         zzz_IndexD(imax, d) =  num_ghost[2*d + 1];
      }
      box = zzz_NewBox(imin, imax);

      comm_stencil_shape = ctalloc(zzz_Index *, zzz_BoxVolume(box));
      index = zzz_NewIndex();
      i = 0;
      zzz_BoxLoop0(box, index,
                   {
                      comm_stencil_shape[i] = zzz_NewIndex();
                      for (d = 0; d < 3; d++)
                         zzz_IndexD(comm_stencil_shape[i], d) =
                            zzz_IndexD(index, d);
                      i++;
                   });
      comm_stencil = zzz_NewStructStencil(3, zzz_BoxVolume(box),
                                          comm_stencil_shape);

      zzz_FreeIndex(index);
      zzz_FreeBox(box);

      /* Set up the CommPkg */

      zzz_GetCommInfo(&send_boxes, &recv_boxes,
                      &send_processes, &recv_processes,
                      zzz_StructMatrixGrid(matrix),
                      comm_stencil);

      send_sboxes = zzz_ConvertToSBoxArrayArray(send_boxes);
      recv_sboxes = zzz_ConvertToSBoxArrayArray(recv_boxes);

      num_values = zzz_StructStencilSize(zzz_StructMatrixStencil(matrix));
      if (zzz_StructMatrixSymmetric(matrix))
         num_values = (num_values + 1) / 2;
      comm_pkg = zzz_NewCommPkg(send_sboxes, recv_sboxes,
                                send_processes, recv_processes,
                                zzz_StructMatrixDataSpace(matrix),
                                num_values);

      zzz_StructMatrixCommPkg(matrix) = comm_pkg;

      zzz_ForSBoxArrayI(i, send_sboxes)
         tfree(send_processes[i]);
      tfree(send_processes);
      zzz_FreeSBoxArrayArray(send_sboxes);

      zzz_ForSBoxArrayI(i, recv_sboxes)
         tfree(recv_processes[i]);
      tfree(recv_processes);
      zzz_FreeSBoxArrayArray(recv_sboxes);

      zzz_FreeStructStencil(comm_stencil);
   }

   /*-----------------------------------------------------------------------
    * Update the ghost data
    *-----------------------------------------------------------------------*/

   comm_handle =
      zzz_InitializeCommunication(comm_pkg, zzz_StructMatrixData(matrix));
   zzz_FinalizeCommunication(comm_handle);

   return(ierr);
}

/*--------------------------------------------------------------------------
 * zzz_SetStructMatrixStencilSpace
 *--------------------------------------------------------------------------*/

void
zzz_SetStructMatrixStencilSpace( zzz_StructMatrix *matrix,
                                 zzz_SBoxArray    *stencil_space )
{
   zzz_FreeSBoxArray(zzz_StructMatrixStencilSpace(matrix));
   zzz_StructMatrixStencilSpace(matrix) = stencil_space;
}

/*--------------------------------------------------------------------------
 * zzz_SetStructMatrixNumGhost
 *--------------------------------------------------------------------------*/

void
zzz_SetStructMatrixNumGhost( zzz_StructMatrix *matrix,
                             int              *num_ghost )
{
   int  i;

   for (i = 0; i < 6; i++)
      zzz_StructMatrixNumGhost(matrix)[i] = num_ghost[i];
}

/*--------------------------------------------------------------------------
 * zzz_PrintStructMatrix
 *--------------------------------------------------------------------------*/

void
zzz_PrintStructMatrix( char             *filename,
                       zzz_StructMatrix *matrix,
                       int               all      )
{
   FILE                *file;
   char                 new_filename[255];
                      
   zzz_BoxArray        *boxes;
   zzz_BoxArray        *data_space;

   zzz_StructStencil  *stencil;
   zzz_Index         **shape;
   int                *symm_coeff;

   int                 num_values;

   int                 i, j;

   int                 myid;
 
   /*----------------------------------------
    * Open file
    *----------------------------------------*/
 
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );
   sprintf(new_filename, "%s.%05d", filename, myid);
 
   if ((file = fopen(new_filename, "w")) == NULL)
   {
      printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   /*----------------------------------------
    * Print header info
    *----------------------------------------*/

   fprintf(file, "StructMatrix\n");

   fprintf(file, "\nStencil:\n");
   stencil = zzz_StructMatrixStencil(matrix);
   shape = zzz_StructStencilShape(stencil);

   /* symmetric case */
   num_values = zzz_StructStencilSize(stencil);
   if (zzz_StructMatrixSymmetric(matrix))
   {
      j = 0;
      for (i = 0; i < zzz_StructStencilSize(stencil); i++)
      {
         if (!symm_coeff)
         {
            fprintf(file, "%d: %d %d %d\n", j++,
                    zzz_IndexX(shape[i]),
                    zzz_IndexY(shape[i]),
                    zzz_IndexZ(shape[i]));
         }
      }
      num_values = (num_values + 1) / 2;
   }

   /* non-symmetric case */
   else
   {
      for (i = 0; i < zzz_StructStencilSize(stencil); i++)
      {
         fprintf(file, "%d: %d %d %d\n", i,
                 zzz_IndexX(shape[i]),
                 zzz_IndexY(shape[i]),
                 zzz_IndexZ(shape[i]));
      }
   }

   /*----------------------------------------
    * Print data
    *----------------------------------------*/

   data_space = zzz_StructMatrixDataSpace(matrix);
 
   if (all)
      boxes = zzz_StructGridBoxes(zzz_StructMatrixGrid(matrix));
   else
      boxes = data_space;
 
   fprintf(file, "\nData:\n");
   zzz_PrintBoxArrayData(file, boxes, data_space, num_values,
                         zzz_StructMatrixData(matrix));

   /*----------------------------------------
    * Close file
    *----------------------------------------*/
 
   fflush(file);
   fclose(file);
}
