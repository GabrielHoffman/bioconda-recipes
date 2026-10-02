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

# import additional code from utils/utils.py
from .utils import _is_format, _row_sum, _streaming_convert

def main():

  # Define arguments
  #-------------------

  parser = argparse.ArgumentParser(
      description="Convert an AnnData .h5ad file so that X (and raw/X) is stored in CSC sparse format (v 0.1.3)"
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
      help="Use file-backed mode to stream data into memory in chunks to reduce maximum memory usage")

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

  parser.add_argument("--colBlock",
      type=int, default=64,
      help="(streaming mode only) Number of destination-major columns processed "
           "per pass. Lower = less peak memory, more passes. For very large "
           "h5ads tune down (e.g. 16 or 8). Default: 64")

  parser.add_argument("--valueChunk",
      type=int, default=8_000_000,
      help="(streaming mode only) Number of nonzeros per HDF5 read in pass 1. "
           "Default: 8000000")

  # parse args
  args = parser.parse_args()

  # if --ondisk set backed to "r"
  backed = None
  if args.ondisk:
    backed = "r"

  if args.compression == "None":
    args.compression = None

  print("Script version:", "0.1.3")

  version = importlib.metadata.version("anndata")
  print("AnnData version:", version) 

  if parse(version) >= parse("0.13.0"):
    # for compatibility with anndataR
    ad.settings.allow_write_nullable_strings = False

  # read file
  if backed:
    print("Stream file...") 
  else:
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

 # ---- streaming conversion path: --ondisk + format change needed ----
  # We have to decide this BEFORE doing format-change in memory below.
  needs_x_swap = (adata.X is not None) and (not _is_format(adata.X, args.format.lower()))
  needs_raw_swap = (adata.raw is not None) and (not _is_format(adata.raw.X, args.format.lower()))
  use_streaming = backed is not None and (needs_x_swap or needs_raw_swap)

  if use_streaming:
    if args.sortBy is not None:
      sys.exit("ERROR: --sortBy is not supported with --ondisk + format conversion. "
               "Re-run without --ondisk, or pre-sort and re-run.")
    if args.compression is not None:
      print(f"NOTE: --compression={args.compression} ignored in streaming mode "
            f"(output preserves input file's compression).")

    if args.output.is_file():
      args.output.unlink(missing_ok=True)

    libsize_values = None
    if not args.noLibSize:
      print("Compute libSize...")
      if adata.raw is not None:
        libsize_values = _row_sum(adata.raw.X)
      else:
        libsize_values = _row_sum(adata.X)

    # Close the backed input before copying the file (avoid HDF5 lock conflicts)
    in_path = str(args.input)
    out_path = str(args.output)
    adata.file.close()
    del adata

    print(f"Streaming format conversion to {args.format} (no full matrix load)...")
    _streaming_convert(in_path, out_path, args.format,
                       libsize_values=libsize_values,
                       skip_libsize=args.noLibSize,
                       col_block=args.colBlock,
                       value_chunk=args.valueChunk,
                       verbose=True)
    print("Done.")
    return

  # compute libSize (works in backed mode via chunked row-sum)
  if not args.noLibSize:
    print("Compute libSize...")
    if adata.raw is not None:
      adata.obs['libSize'] = _row_sum(adata.raw.X)
    else:
      adata.obs['libSize'] = _row_sum(adata.X)

  if args.format == "CSC":

    if not _is_format(adata.X, "csc"):
      if backed is None:
        adata = adata.copy()
      else:
        sys.exit("ERROR: --ondisk mode is strict: converting .X to CSC requires loading into memory. Re-run without --ondisk.")

    if not _is_format(adata.X, "csc"):
      print("Converting .X to CSC sparse format...")
      # convert matrix type
      adata.X = sp.csc_matrix(adata.X)

      if adata.raw is not None and not _is_format(adata.raw.X, "csc"):
        print("Converting .raw.X to CSC sparse format...")
        raw = adata.raw.to_adata()
        raw.X = sp.csc_matrix(raw.X)
        adata.raw = raw

  if args.format == "CSR":
    # X
    if not _is_format(adata.X, "csr"):
      print("Converting .X to CSR sparse format...")
      if backed is None:
        adata = adata.copy()
      else:
        sys.exit("ERROR: --ondisk mode is strict: converting .X to CSR requires loading into memory. Re-run without --ondisk.")
      
      # convert matrix type
      adata.X = sp.csr_matrix(adata.X)

      if adata.raw is not None and not _is_format(adata.raw.X, "csr"):
        print("Converting .raw.X to CSR sparse format...")
        raw = adata.raw.to_adata()
        raw.X = sp.csr_matrix(raw.X)
        adata.raw = raw

  # if args.format == "CSC":

  #   if not args.noLibSize or not sp.isspmatrix_csc(adata.X):
  #     if backed is None:
  #       adata = adata.copy()
  #     else:
  #       adata = adata.to_memory()

  #   if not args.noLibSize:
  #     # compute library size for each cell
  #     print("Compute libSize...")
  #     if adata.raw is not None:
  #       adata.obs['libSize'] = adata.raw.X.sum(axis=1)
  #     else:
  #       adata.obs['libSize'] = adata.X.sum(axis=1)

  #   # if matrix is not a CSC of doubles
  #   if not (sp.isspmatrix_csc(adata.X) and adata.X.dtype == np.float64):
  #     print("Converting .X to CSC sparse format...")
  #     # convert matrix type
  #     adata.X = sp.csc_matrix(adata.X, dtype=np.float64)

  #     if adata.raw is not None:
  #       print("Converting .row.X to CSC sparse format...")
  #       raw = adata.raw.to_adata()
  #       raw.X = sp.csc_matrix(raw.X, dtype=np.float64)
  #       adata.raw = raw

  # if args.format == "CSR":
  #   # X
  #   if not sp.isspmatrix_csr(adata.X):
  #     print("Converting .X to CSR sparse format...")
  #     if backed is None:
  #       adata = adata.copy()
  #     else:
  #       adata = adata.to_memory()
      
  #     # convert matrix type
  #     adata.X = sp.csr_matrix(adata.X, dtype=np.float64)

  #     if adata.raw is not None:
  #       print("Converting .row.X to CSR sparse format...")
  #       raw = adata.raw.to_adata()
  #       raw.X = sp.csr_matrix(raw.X, dtype=np.float64)
  #       adata.raw = raw

  #     if not args.noLibSize:
  #       # compute library size for each cell
  #       print("Compute libSize...")
  #       if adata.raw is not None:
  #         adata.obs['libSize'] = adata.raw.X.sum(axis=1)
  #       else:
  #         adata.obs['libSize'] = adata.X.sum(axis=1)

  if args.output.is_file():
    args.output.unlink(missing_ok=True)

  print("Writing H5AD...") 

  compressMethod = args.compression

  if compressMethod == "zstd":
    compressMethod = hdf5plugin.FILTERS["zstd"]

  adata.write_h5ad( args.output, compression=compressMethod )



if __name__ == "__main__":
  main()
