/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/




#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_Maxwell_Grad.c
 *   Forms a node-to-edge gradient operator. Looping over the
 *   edge grid so that each processor fills up only its own rows. Each 
 *   processor will have its processor interface nodal ranks.
 *   Loops over two types of boxes, interior of grid boxes and boundary
 *   of boxes. Algo:
 *       find all nodal and edge physical boundary points and set 
 *       the appropriate flag to be 0 at a boundary dof.
 *       set -1's in value array
 *       for each edge box, 
 *       for interior 
 *       {
 *          connect edge ijk (row) to nodes (col) connected to this edge
 *          and change -1 to 1 if needed;
 *       }
 *       for boundary layers
 *       {
 *          if edge not on the physical boundary connect only the nodes
 *          that are not on the physical boundary
 *       }
 *       set parcsr matrix with values;
 *
 * Note that the nodes that are on the processor interface can be 
 * on the physical boundary. But the off-proc edges connected to this
 * type of node will be a physical boundary edge.
 *
 *--------------------------------------------------------------------------*/
hypre_ParCSRMatrix *
hypre_Maxwell_Grad(hypre_SStructGrid    *grid)
{
   MPI_Comm               comm = (grid ->  comm);

   HYPRE_IJMatrix         T_grad;
   hypre_ParCSRMatrix    *parcsr_grad;
   int                    matrix_type= HYPRE_PARCSR;

   hypre_SStructGrid     *node_grid, *edge_grid;

   hypre_SStructPGrid    *pgrid;
   hypre_StructGrid      *var_grid;
   hypre_BoxArray        *boxes, *tmp_box_array1, *tmp_box_array2;
   hypre_BoxArray        *node_boxes, *edge_boxes, *cell_boxes;
   hypre_Box             *box, *cell_box;
   hypre_Box              layer, interior_box;
   hypre_Box             *box_piece;

   hypre_BoxMap          *map;
   hypre_BoxMapEntry     *entry;

   int                   *inode, *jedge;
   int                    nrows, nnodes, *nflag, *eflag, *ncols;
   double                *vals;

   hypre_Index            index;
   hypre_Index            loop_size, start;
   hypre_Index            shift, shift2;
   hypre_Index           *offsets, *varoffsets;
   int                    loopi, loopj, loopk;

   int                    nparts= hypre_SStructGridNParts(grid);
   int                    ndim  = hypre_SStructGridNDim(grid);

   HYPRE_SStructVariable  vartype_node, *vartype_edges;
   HYPRE_SStructVariable *vartypes;

   int                    nvars, part;

   int                    i, j, k, m, n, d;
   int                   *direction, ndirection;

   int                    ilower, iupper;
   int                    jlower, jupper;

   int                    start_rank1, start_rank2, rank;

   int                    myproc;
   int                    ierr;

   MPI_Comm_rank(comm, &myproc);

   hypre_ClearIndex(shift);
   for (i= 0; i< ndim; i++)
   {
      hypre_IndexD(shift, i)= -1;
   }

  /* To get the correct ranks, separate node & edge grids must be formed. 
     Note that the edge vars must be ordered the same way as is in grid.*/
   HYPRE_SStructGridCreate(comm, ndim, nparts, &node_grid);
   HYPRE_SStructGridCreate(comm, ndim, nparts, &edge_grid);

   vartype_node = HYPRE_SSTRUCT_VARIABLE_NODE;
   vartype_edges= hypre_TAlloc(HYPRE_SStructVariable, ndim);

  /* Assuming the same edge variable types on all parts */
   pgrid   = hypre_SStructGridPGrid(grid, 0);
   vartypes= hypre_SStructPGridVarTypes(pgrid);
   nvars   = hypre_SStructPGridNVars(pgrid);

   k= 0;
   for (i= 0; i< nvars; i++)
   {
      j= vartypes[i];
      switch(j)
      {
         case 2:
         {
            vartype_edges[k]= HYPRE_SSTRUCT_VARIABLE_XFACE;
            k++;
            break;
         }

         case 3:
         {
            vartype_edges[k]= HYPRE_SSTRUCT_VARIABLE_YFACE;
            k++;
            break;
         }
                                                                                                                       
         case 5:
         {
            vartype_edges[k]= HYPRE_SSTRUCT_VARIABLE_XEDGE;
            k++;
            break;
         }
                                                                                                                       
         case 6:
         {
            vartype_edges[k]= HYPRE_SSTRUCT_VARIABLE_YEDGE;
            k++;
            break;
         }
                                                                                                                       
         case 7:
         {
            vartype_edges[k]= HYPRE_SSTRUCT_VARIABLE_ZEDGE;
            k++;
            break;
         }

      }  /* switch(j) */
   }     /* for (i= 0; i< nvars; i++) */

   for (part= 0; part< nparts; part++)
   {
      pgrid= hypre_SStructGridPGrid(grid, part);  
      var_grid= hypre_SStructPGridCellSGrid(pgrid) ;

      boxes= hypre_StructGridBoxes(var_grid);
      hypre_ForBoxI(j, boxes)
      {
          box= hypre_BoxArrayBox(boxes, j);
          HYPRE_SStructGridSetExtents(node_grid, part,
                                      hypre_BoxIMin(box), hypre_BoxIMax(box));
          HYPRE_SStructGridSetExtents(edge_grid, part,
                                      hypre_BoxIMin(box), hypre_BoxIMax(box));
      }
      HYPRE_SStructGridSetVariables(node_grid, part, 1, &vartype_node);
      HYPRE_SStructGridSetVariables(edge_grid, part, ndim, vartype_edges);
   }
   HYPRE_SStructGridAssemble(node_grid);
   HYPRE_SStructGridAssemble(edge_grid);

  /* CREATE IJ_MATRICES- need to find the size of each one. Notice that the row
     and col ranks of these matrices can be created using only grid information. 
     Grab the first part, first variable, first box, and lower index (lower rank);
     Grab the last part, last variable, last box, and upper index (upper rank). */
 
  /* Grad: node(col) -> edge(row). Same for 2-d and 3-d */
  /* lower rank */
   part= 0;
   i   = 0;

   hypre_SStructGridBoxProcFindMapEntry(edge_grid, part, 0, i, myproc, &entry);
   pgrid   = hypre_SStructGridPGrid(edge_grid, part);
   var_grid= hypre_SStructPGridSGrid(pgrid, 0);
   boxes   = hypre_StructGridBoxes(var_grid);
   box     = hypre_BoxArrayBox(boxes, 0);
   hypre_SStructMapEntryGetGlobalCSRank(entry, hypre_BoxIMin(box), &ilower);

   hypre_SStructGridBoxProcFindMapEntry(node_grid, part, 0, i, myproc, &entry);
   pgrid   = hypre_SStructGridPGrid(node_grid, part);
   var_grid= hypre_SStructPGridSGrid(pgrid, 0);
   boxes   = hypre_StructGridBoxes(var_grid);
   box     = hypre_BoxArrayBox(boxes, 0);
   hypre_SStructMapEntryGetGlobalCSRank(entry, hypre_BoxIMin(box), &jlower);

  /* upper rank */
   part= nparts-1;

   pgrid   = hypre_SStructGridPGrid(edge_grid, part);
   nvars   = hypre_SStructPGridNVars(pgrid);
   var_grid= hypre_SStructPGridSGrid(pgrid, nvars-1);
   boxes   = hypre_StructGridBoxes(var_grid);
   box     = hypre_BoxArrayBox(boxes, hypre_BoxArraySize(boxes)-1);

   hypre_SStructGridBoxProcFindMapEntry(edge_grid, part, nvars-1, 
                                        hypre_BoxArraySize(boxes)-1, myproc,
                                       &entry);
   hypre_SStructMapEntryGetGlobalCSRank(entry, hypre_BoxIMax(box), &iupper);

   pgrid   = hypre_SStructGridPGrid(node_grid, part);
   nvars   = hypre_SStructPGridNVars(pgrid);
   var_grid= hypre_SStructPGridSGrid(pgrid, nvars-1);
   boxes   = hypre_StructGridBoxes(var_grid);
   box     = hypre_BoxArrayBox(boxes, hypre_BoxArraySize(boxes)-1);

   hypre_SStructGridBoxProcFindMapEntry(node_grid, part, nvars-1, 
                                        hypre_BoxArraySize(boxes)-1, myproc,
                                       &entry);
   hypre_SStructMapEntryGetGlobalCSRank(entry, hypre_BoxIMax(box), &jupper);

   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &T_grad);
   HYPRE_IJMatrixSetObjectType(T_grad, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(T_grad);

  /*------------------------------------------------------------------------------
   * fill up the parcsr matrix.
   *------------------------------------------------------------------------------*/

  /* count the no. of rows. Make sure repeated nodes along the boundaries are counted.*/
   nrows = 0;
   nnodes= 0;
   for (part= 0; part< nparts; part++)
   {
      pgrid= hypre_SStructGridPGrid(edge_grid, part);  
      nvars= hypre_SStructPGridNVars(pgrid);
      for (m= 0; m< nvars; m++)
      {
         var_grid= hypre_SStructPGridSGrid(pgrid, m);
         boxes   = hypre_StructGridBoxes(var_grid);
         hypre_ForBoxI(j, boxes)
         {
            box= hypre_BoxArrayBox(boxes, j);
           /* make slightly bigger to handle any shared nodes */
            hypre_CopyBox(box, &layer);
            hypre_AddIndex(hypre_BoxIMin(&layer), shift, hypre_BoxIMin(&layer));
            hypre_SubtractIndex(hypre_BoxIMax(&layer), shift, hypre_BoxIMax(&layer));
            nrows+= hypre_BoxVolume(&layer);
         }
      }

      pgrid= hypre_SStructGridPGrid(node_grid, part);
      var_grid= hypre_SStructPGridSGrid(pgrid, 0); /* only one variable grid */
      boxes   = hypre_StructGridBoxes(var_grid);
      hypre_ForBoxI(j, boxes)
      {
         box= hypre_BoxArrayBox(boxes, j);
        /* make slightly bigger to handle any shared nodes */
         hypre_CopyBox(box, &layer);
         hypre_AddIndex(hypre_BoxIMin(&layer), shift, hypre_BoxIMin(&layer));
         hypre_SubtractIndex(hypre_BoxIMax(&layer), shift, hypre_BoxIMax(&layer));
         nnodes+= hypre_BoxVolume(&layer);
      }
   }

   eflag = hypre_CTAlloc(int, nrows);
   nflag = hypre_CTAlloc(int, nnodes);

  /* Set eflag to have the number of nodes connected to an edge (2) and
     nflag to have the number of edges connect to a node. */
   for (i= 0; i< nrows; i++)
   {
      eflag[i]= 2;
   }
   j= 2*ndim;
   for (i= 0; i< nnodes; i++)
   {
      nflag[i]= j;
   }

  /* Determine physical boundary points. Get the rank and set flag[rank]= 0.
     This will boundary dof, i.e., flag[rank]= 0 will flag a boundary dof. */

   start_rank1= hypre_SStructGridStartRank(node_grid);
   start_rank2= hypre_SStructGridStartRank(edge_grid);
   for (part= 0; part< nparts; part++)
   {
     /* node flag */
      pgrid   = hypre_SStructGridPGrid(node_grid, part);  
      var_grid= hypre_SStructPGridSGrid(pgrid, 0);
      boxes   = hypre_StructGridBoxes(var_grid);
      map     = hypre_SStructGridMap(node_grid, part, 0);

      hypre_ForBoxI(j, boxes)
      {
         box= hypre_BoxArrayBox(boxes, j);
         hypre_BoxMapFindBoxProcEntry(map, j, myproc, &entry);
         i= hypre_BoxVolume(box);

         tmp_box_array1= hypre_BoxArrayCreate(0);
         ierr         += hypre_BoxBoundaryG(box, var_grid, tmp_box_array1);

         for (m= 0; m< hypre_BoxArraySize(tmp_box_array1); m++)
         {
            box_piece= hypre_BoxArrayBox(tmp_box_array1, m);
            if (hypre_BoxVolume(box_piece) < i)
            {
               hypre_BoxGetSize(box_piece, loop_size);
               hypre_CopyIndex(hypre_BoxIMin(box_piece), start);
         
               hypre_BoxLoop0Begin(loop_size)
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj
#include "hypre_box_smp_forloop.h"
               hypre_BoxLoop0For(loopi, loopj, loopk)
               {
                   hypre_SetIndex(index, loopi, loopj, loopk);
                   hypre_AddIndex(index, start, index);

                   hypre_SStructMapEntryGetGlobalRank(entry, index,
                                                     &rank, matrix_type);
                   nflag[rank-start_rank1]= 0; 
               }
               hypre_BoxLoop0End();
            }  /* if (hypre_BoxVolume(box_piece) < i) */

         }  /* for (m= 0; m< hypre_BoxArraySize(tmp_box_array1); m++) */
         hypre_BoxArrayDestroy(tmp_box_array1);

      }  /* hypre_ForBoxI(j, boxes) */

     /*-----------------------------------------------------------------
      * edge flag. Since we want only the edges that completely lie
      * on a boundary, whereas the boundary extraction routines mark
      * edges that touch the boundary, we need to call the boundary
      * routines in appropriate directions:
      *    2-d horizontal edges (y faces)- search in j directions
      *    2-d vertical edges (x faces)  - search in i directions
      *    3-d x edges                   - search in j,k directions
      *    3-d y edges                   - search in i,k directions
      *    3-d z edges                   - search in i,j directions
      *-----------------------------------------------------------------*/
      pgrid    = hypre_SStructGridPGrid(edge_grid, part);
      nvars    = hypre_SStructPGridNVars(pgrid);
      direction= hypre_TAlloc(int, 2); /* only two directions at most */
      for (m= 0; m< nvars; m++)
      {
         var_grid= hypre_SStructPGridSGrid(pgrid, m);
         boxes   = hypre_StructGridBoxes(var_grid);
         map     = hypre_SStructGridMap(edge_grid, part, m);

         j= vartype_edges[m];
         switch(j)
         {
            case 2: /* x faces, 2d */
            {
               ndirection  = 1;
               direction[0]= 0;
               break;
            }
         
            case 3: /* y faces, 2d */
            {
               ndirection  = 1;
               direction[0]= 1;
               break;
            }

            case 5: /* x edges, 3d */
            {
               ndirection  = 2;
               direction[0]= 1;
               direction[1]= 2;
               break;
            }

            case 6: /* y edges, 3d */
            {
               ndirection  = 2;
               direction[0]= 0;
               direction[1]= 2;
               break;
            }
         
            case 7: /* z edges, 3d */
            {
               ndirection  = 2;
               direction[0]= 0;
               direction[1]= 1;
               break;
            }
         }  /* switch(j) */
                                                                                                                        
         hypre_ForBoxI(j, boxes)
         {
            box= hypre_BoxArrayBox(boxes, j);
            hypre_BoxMapFindBoxProcEntry(map, j, myproc, &entry);
            i= hypre_BoxVolume(box);
                                                                                                                        
            for (d= 0; d< ndirection; d++)
            {
               tmp_box_array1= hypre_BoxArrayCreate(0);
               tmp_box_array2= hypre_BoxArrayCreate(0);
               ierr+= hypre_BoxBoundaryDG(box, var_grid, tmp_box_array1,
                                          tmp_box_array2, direction[d]);
             
               for (k= 0; k< hypre_BoxArraySize(tmp_box_array1); k++)
               {
                  box_piece= hypre_BoxArrayBox(tmp_box_array1, k);
                  if (hypre_BoxVolume(box_piece) < i)
                  {
                     hypre_BoxGetSize(box_piece, loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(box_piece), start);
                                                                                                                        
                     hypre_BoxLoop0Begin(loop_size)
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj
#include "hypre_box_smp_forloop.h"
                     hypre_BoxLoop0For(loopi, loopj, loopk)
                     {
                        hypre_SetIndex(index, loopi, loopj, loopk);
                        hypre_AddIndex(index, start, index);

                        hypre_SStructMapEntryGetGlobalRank(entry, index,
                                                          &rank, matrix_type);
                        eflag[rank-start_rank2]= 0;
                     }
                     hypre_BoxLoop0End();
                  }  /* if (hypre_BoxVolume(box_piece) < i) */
               }     /* for (k= 0; k< hypre_BoxArraySize(tmp_box_array1); k++) */

               hypre_BoxArrayDestroy(tmp_box_array1);

               for (k= 0; k< hypre_BoxArraySize(tmp_box_array2); k++)
               {
                  box_piece= hypre_BoxArrayBox(tmp_box_array2, k);
                  if (hypre_BoxVolume(box_piece) < i)
                  {
                     hypre_BoxGetSize(box_piece, loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(box_piece), start);
                                                                                                                        
                     hypre_BoxLoop0Begin(loop_size)
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj
#include "hypre_box_smp_forloop.h"
                     hypre_BoxLoop0For(loopi, loopj, loopk)
                     {
                        hypre_SetIndex(index, loopi, loopj, loopk);
                        hypre_AddIndex(index, start, index);
                                                                                                                        
                        hypre_SStructMapEntryGetGlobalRank(entry, index,
                                                          &rank, matrix_type);
                        eflag[rank-start_rank2]= 0;
                     }
                     hypre_BoxLoop0End();
                  }  /* if (hypre_BoxVolume(box_piece) < i) */
               }     /* for (k= 0; k< hypre_BoxArraySize(tmp_box_array2); k++) */
                                                                                                                        
               hypre_BoxArrayDestroy(tmp_box_array2);
            }  /* for (d= 0; d< ndirection; d++) */

         }  /* hypre_ForBoxI(j, boxes) */
      }     /* for (m= 0; m< nvars; m++) */

      hypre_TFree(direction); 
   }  /* for (part= 0; part< nparts; part++) */

  /* set vals. Will have more memory than is needed- extra allotted
     for repeated nodes. */
   inode= hypre_CTAlloc(int, nrows);
   ncols= hypre_CTAlloc(int, nrows);

  /* each row can have at most two columns */
   k= 2*nrows;
   jedge= hypre_CTAlloc(int, k);
   vals = hypre_TAlloc(double, k);
   for (i= 0; i< k; i++)
   {
      vals[i]=-1.0;
   }

  /* to get the correct col connection to each node, we need to offset
     index ijk. Determine these. Assuming the same var ordering for each 
     part. Note that these are not the variable offsets. */
   offsets   = hypre_TAlloc(hypre_Index, ndim);
   varoffsets= hypre_TAlloc(hypre_Index, ndim);
   for (i= 0; i< ndim; i++)
   {
      j= vartype_edges[i];
      hypre_SStructVariableGetOffset(vartype_edges[i], ndim, varoffsets[i]);
      switch(j)
      {
         case 2:
         {
            hypre_SetIndex(offsets[i], 0, 1, 0);
            break;
         }

         case 3:
         {
            hypre_SetIndex(offsets[i], 1, 0, 0);
            break;
         }
                                                                                                             
         case 5:
         {
            hypre_SetIndex(offsets[i], 1, 0, 0);
            break;
         }
                                                                                                             
         case 6:
         {
            hypre_SetIndex(offsets[i], 0, 1, 0);
            break;
         }

         case 7:
         {
            hypre_SetIndex(offsets[i], 0, 0, 1);
            break;
         }
     }   /*  switch(j) */
   }     /* for (i= 0; i< ndim; i++) */

   nrows= 0; i= 0;
   for (part= 0; part< nparts; part++)
   {
     /* grab boxarray for node rank extracting later */
      pgrid       = hypre_SStructGridPGrid(node_grid, part);  
      var_grid    = hypre_SStructPGridSGrid(pgrid, 0);
      node_boxes  = hypre_StructGridBoxes(var_grid);

     /* grab edge structures */
      pgrid     = hypre_SStructGridPGrid(edge_grid, part);  

     /* the cell-centred reference box is used to get the correct 
        interior edge box. For parallel distribution of the edge
        grid, simple contraction of the edge box does not get the
        correct interior edge box. Need to contract the cell box. */
      var_grid= hypre_SStructPGridCellSGrid(pgrid);
      cell_boxes= hypre_StructGridBoxes(var_grid);

      nvars     = hypre_SStructPGridNVars(pgrid);
      for (n= 0; n< nvars; n++)
      {
         var_grid  = hypre_SStructPGridSGrid(pgrid, n);
         edge_boxes= hypre_StructGridBoxes(var_grid);

         hypre_ForBoxI(j, edge_boxes)
         {
            box= hypre_BoxArrayBox(edge_boxes, j);
            cell_box= hypre_BoxArrayBox(cell_boxes, j);

            hypre_CopyBox(cell_box, &interior_box);

           /* shrink the cell_box to get the interior cell_box. All
              edges in the interior box should be on this proc. */ 
            hypre_SubtractIndex(hypre_BoxIMin(&interior_box), shift, 
                                hypre_BoxIMin(&interior_box));

            hypre_AddIndex(hypre_BoxIMax(&interior_box), shift, 
                           hypre_BoxIMax(&interior_box));

           /* offset this to the variable interior box */
            hypre_CopyBox(&interior_box, &layer);
            hypre_SubtractIndex(hypre_BoxIMin(&layer), varoffsets[n], 
                                hypre_BoxIMin(&layer));

            hypre_BoxGetSize(&layer, loop_size);
            hypre_CopyIndex(hypre_BoxIMin(&layer), start);

           /* Interior box- loop over each edge and find the row rank and 
              then the column ranks for the connected nodes. Change the 
              appropriate values to 1. */
            hypre_BoxLoop0Begin(loop_size)
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj
#include "hypre_box_smp_forloop.h"
            hypre_BoxLoop0For(loopi, loopj, loopk)
            {
               hypre_SetIndex(index, loopi, loopj, loopk);
               hypre_AddIndex(index, start, index);

              /* edge ijk connected to nodes ijk & ijk-offsets. Interior edges
                 and so no boundary edges to consider. */ 
               hypre_SStructGridFindMapEntry(edge_grid, part, index, n,
                                            &entry);
               hypre_SStructMapEntryGetGlobalRank(entry, index, &m, matrix_type);
               inode[nrows]= m;

               hypre_SStructGridFindMapEntry(node_grid, part, index, 0,
                                            &entry);
               hypre_SStructMapEntryGetGlobalRank(entry, index, &m, matrix_type);
               jedge[i]= m;
               vals[i] = 1.0; /* change only this connection */
               i++;

               hypre_SubtractIndex(index, offsets[n], index);
               hypre_SStructGridFindMapEntry(node_grid, part, index, 0,
                                            &entry);
               hypre_SStructMapEntryGetGlobalRank(entry, index, &m, matrix_type);
               jedge[i]= m;
               i++;
 
               ncols[nrows]= 2;
               nrows++;
            }
            hypre_BoxLoop0End();

           /* now the boundary layers. To cases to consider: is the
              edge totally on the boundary or is the edge connected
              to the boundary. Need to check eflag & nflag. */
            for (d= 0; d< ndim; d++)
            {
              /*shift the layer box in the correct direction and distance.
                distance= hypre_BoxIMax(box)[d]-hypre_BoxIMin(box)[d]+1-1
                        = hypre_BoxIMax(box)[d]-hypre_BoxIMin(box)[d] */
               hypre_ClearIndex(shift2);
               shift2[d]= hypre_BoxIMax(box)[d]-hypre_BoxIMin(box)[d];

              /* ndirection= 0 negative; ndirection= 1 positive */
               for (ndirection= 0; ndirection< 2; ndirection++)
               {
                  hypre_CopyBox(box, &layer);

                  if (ndirection)
                  {
                     hypre_BoxShiftPos(&layer, shift2);
                  }
                  else
                  {
                     hypre_BoxShiftNeg(&layer, shift2);
                  }

                  hypre_IntersectBoxes(box, &layer, &layer);
                  hypre_BoxGetSize(&layer, loop_size);
                  hypre_CopyIndex(hypre_BoxIMin(&layer), start);

                  hypre_BoxLoop0Begin(loop_size)
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj
#include "hypre_box_smp_forloop.h"
                  hypre_BoxLoop0For(loopi, loopj, loopk)
                  {
                     hypre_SetIndex(index, loopi, loopj, loopk);
                     hypre_AddIndex(index, start, index);

                    /* edge ijk connects to nodes ijk & ijk+offsets. */
                     hypre_SStructGridFindMapEntry(edge_grid, part, index, n,
                                                  &entry);
                     hypre_SStructMapEntryGetGlobalRank(entry, index, &m, 
                                                        matrix_type);

                    /* check if the edge lies on the boundary & if not
                       check if the connecting node is on the boundary. */
                     if (eflag[m-start_rank2])
                     {
                        inode[nrows]= m;
                       /* edge not completely on the boundary. One connecting
                          node must be in the interior. */
                        hypre_SStructGridFindMapEntry(node_grid, part, index, 0,
                                                     &entry);
                        hypre_SStructMapEntryGetGlobalRank(entry, index, &m, 
                                                           matrix_type);

                       /* check if node on my processor. If not, the node must
                          be in the interior (draw a diagram to see this). */
                        if (m >= start_rank1 && m <= jupper)
                        {
                           /* node on proc. Now check if on the boundary. */
                           if (nflag[m-start_rank1]) /* interior node */
                           {
                              jedge[i]= m;
                              vals[i] = 1.0; 
                              i++;

                              ncols[nrows]++;
                           }
                        }
                        else  /* node off-proc */
                        {
                           jedge[i]= m;
                           vals[i] = 1.0; 
                           i++;

                           ncols[nrows]++;
                        }

                       /* ijk+offsets */
                        hypre_SubtractIndex(index, offsets[n], index);
                        hypre_SStructGridFindMapEntry(node_grid, part, index, 0,
                                                     &entry);
                        hypre_SStructMapEntryGetGlobalRank(entry, index, &m, 
                                                           matrix_type);
                       /* boundary checks again */
                        if (m >= start_rank1 && m <= jupper)
                        {
                           /* node on proc. Now check if on the boundary. */
                           if (nflag[m-start_rank1]) /* interior node */
                           {
                              jedge[i]= m;
                              i++;
                                                                                                                        
                              ncols[nrows]++;
                           }
                        }
                        else  /* node off-proc */
                        {
                           jedge[i]= m;
                           i++;
                                                                                                                        
                           ncols[nrows]++;
                        }
                                                                                                                        
                        nrows++; /* must have at least one node connection */
                     }  /* if (eflag[m-start_rank2]) */

                  }
                  hypre_BoxLoop0End();
               }  /* for (ndirection= 0; ndirection< 2; ndirection++) */
            }     /* for (d= 0; d< ndim; d++) */

         }  /* hypre_ForBoxI(j, boxes) */
      }     /* for (n= 0; n< nvars; n++) */
   }        /* for (part= 0; part< nparts; part++) */

   hypre_TFree(offsets);
   hypre_TFree(varoffsets);
   hypre_TFree(vartype_edges);
   HYPRE_SStructGridDestroy(node_grid);
   HYPRE_SStructGridDestroy(edge_grid);

   HYPRE_IJMatrixSetValues(T_grad, nrows, ncols,
                          (const int*) inode, (const int*) jedge,
                          (const double*) vals);
   HYPRE_IJMatrixAssemble(T_grad);

   hypre_TFree(eflag);
   hypre_TFree(nflag);
   hypre_TFree(ncols);
   hypre_TFree(inode);
   hypre_TFree(jedge);
   hypre_TFree(vals);

   parcsr_grad= (hypre_ParCSRMatrix *) hypre_IJMatrixObject(T_grad);
   HYPRE_IJMatrixSetObjectType(T_grad, -1);
   HYPRE_IJMatrixDestroy(T_grad);

   return  parcsr_grad;
}
