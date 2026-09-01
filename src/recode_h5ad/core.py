#!/usr/bin/env python3

import sys
import argparse
import anndata as ad
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import importlib
from packaging.version import parse
import hdf5plugin

def main():

  # Define arguments
  #-------------------

  parser = argparse.ArgumentParser(
      description="Convert an AnnData .h5ad file so that X (and raw/X) is stored in CSC sparse format (v aug.26.2026)"
  )

  parser.add_argument("--input",
      type=Path, 
      required=True,
      help="Input .h5ad file")

  parser.add_argument("--output",      
      type=Path, 
      required=True,
      help="Output .h5ad file")

  parser.add_argument("--ondisk", 
      action="store_true",
      help="Use file-backed mode to reduce memory usage")

  parser.add_argument("--sortBy", 
      default=None,
      help="Cols to sort by in _decreasing_ order of importance")

  parser.add_argument(
      "--compression",
      default=None,
      choices=[None, "None", "gzip", "lzf", "zstd"],
      help="Optional compression for output file (gzip, lzf, zstd). Default: None")

  parser.add_argument(
      "--format",
      default="CSR",
      choices=["CSR", "CSC"],
      help="Store sparse count matrix (X and raw/X) in CSR or CSC format.  CSR allows faster access to cells, CSC gives faster access to genes. Default: CSR")

  parser.add_argument("--noLibSize", 
      action="store_true",
      help="Skip computing libSize for each cell")

  # parse args
  args = parser.parse_args()

  # if --ondisk set backed to "r"
  backed = None
  if args.ondisk:
    backed = "r"

  if args.compression == "None":
    args.compression = None

  print("Script version:", "aug.26.2026")

  version = importlib.metadata.version("anndata")
  print("AnnData version:", version) 

  if parse(version) >= parse("0.13.0"):
    # for compatibility with anndataR
    ad.settings.allow_write_nullable_strings = False

  # read file
  print("Read file...") 
  adata = ad.read_h5ad(args.input, backed=backed) 

  # Replace forward slashes in column names of data.obs
  adata.obs.columns = adata.obs.columns.str.replace("/", "_", regex=False)

  # Find counts entry
  if adata.X is not None:
      print("Using AnnData X matrix...")
  else:
    if 'X' in adata.layers: 
      print("Using AnnData layers/X matrix...")
      adata.X = adata.layers['X']
    elif 'counts' in adata.layers: 
      print("Using AnnData layers/counts matrix...")
      adata.X = adata.layers['counts']

  # sort cells by type  
  if args.sortBy != None:

    print("Sorting...") 
    fields = args.sortBy.split(",")

    # check if all fields are present
    missing = set(fields) - set(adata.obs.columns)

    if missing:
      print(f"Missing columns: {missing}")
      sys.exit(2)     

    # get sorted order
    # reverse since 1st sorted index is the last one for lexsort
    idx = np.lexsort( 
      keys = tuple(adata.obs[c].to_numpy() for c in fields[::-1]) )

    # apply reordering
    adata = adata[idx,:]

  if args.format == "CSC":

    if not args.noLibSize or not sp.isspmatrix_csc(adata.X):
      if backed is None:
        adata = adata.copy()
      else:
        adata = adata.to_memory()

    if not args.noLibSize:
      # compute library size for each cell
      print("Compute libSize...")
      if adata.raw is not None:
        adata.obs['libSize'] = adata.raw.X.sum(axis=1)
      else:
        adata.obs['libSize'] = adata.X.sum(axis=1)

    # if matrix is not a CSC of doubles
    if not (sp.isspmatrix_csc(adata.X) and adata.X.dtype == np.float64):
      print("Converting .X to CSC sparse format...")
      # convert matrix type
      adata.X = sp.csc_matrix(adata.X, dtype=np.float64)

      if adata.raw is not None:
        print("Converting .row.X to CSC sparse format...")
        raw = adata.raw.to_adata()
        raw.X = sp.csc_matrix(raw.X, dtype=np.float64)
        adata.raw = raw

  if args.format == "CSR":
    # X
    if not sp.isspmatrix_csr(adata.X):
      print("Converting .X to CSR sparse format...")
      if backed is None:
        adata = adata.copy()
      else:
        adata = adata.to_memory()
      
      # convert matrix type
      adata.X = sp.csr_matrix(adata.X, dtype=np.float64)

      if adata.raw is not None:
        print("Converting .row.X to CSR sparse format...")
        raw = adata.raw.to_adata()
        raw.X = sp.csr_matrix(raw.X, dtype=np.float64)
        adata.raw = raw

      if not args.noLibSize:
        # compute library size for each cell
        print("Compute libSize...")
        if adata.raw is not None:
          adata.obs['libSize'] = adata.raw.X.sum(axis=1)
        else:
          adata.obs['libSize'] = adata.X.sum(axis=1)

  if args.output.is_file():
    args.output.unlink(missing_ok=True)

  print("Writing H5AD...") 

  compressMethod = args.compression

  if compressMethod == "zstd":
    compressMethod = hdf5plugin.FILTERS["zstd"]

  adata.write_h5ad( args.output, compression=compressMethod )



if __name__ == "__main__":
  main()
