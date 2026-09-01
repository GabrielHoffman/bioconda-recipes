
We developed the recode_h5ad script that uses the AnnData library in order to:

- Convert read count matrix to gene-major format
- Sort cells by annotated class and biological sample identifier
- Create variable libSize storing the number of reads for each cell. If raw/X exists, computed from this. Otherwise computed from X
- Write to new H5AD file using LZF compression

The result is a valid H5AD file supported by standard tools, but just optimized for gene-centric access.

Here, we download the script and examine the arguments:

```sh
recode_h5ad -h
usage: recode_h5ad [-h] --input INPUT --output OUTPUT [--ondisk]
                   [--sortBy SORTBY] [--compression {None,None,gzip,lzf,zstd}]
                   [--format {CSR,CSC}] [--noLibSize]

Convert an AnnData .h5ad file so that X (and raw/X) is stored in CSC sparse
format (v aug.26.2026)

options:
  -h, --help            show this help message and exit
  --input INPUT         Input .h5ad file
  --output OUTPUT       Output .h5ad file
  --ondisk              Use file-backed mode to reduce memory usage
  --sortBy SORTBY       Cols to sort by in _decreasing_ order of importance
  --compression {None,None,gzip,lzf,zstd}
                        Optional compression for output file (gzip, lzf,
                        zstd). Default: None
  --format {CSR,CSC}    Store sparse count matrix (X and raw/X) in CSR or CSC
                        format. CSR allows faster access to cells, CSC gives
                        faster access to genes. Default: CSR
  --noLibSize           Skip computing libSize for each cell
```

### Example
Now, we convert an H5AD file to CSC (i.e. gene-major) format, sorting the cells by class, subclass and SampleID, and saving using LZF compression.

```bash
# H5AD=(Original H5AD file)
# OUTFILE=(New H5AD file)

recode_h5ad \
  --input $H5AD \
  --sortBy class,subclass,SampleID \
  --format CSC \
  --compression lzf \
  --out $OUTFILE
```
