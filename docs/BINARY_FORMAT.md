# HIGGS Binary Matrix Format

The processed matrix is deliberately headerless so an MPI rank can calculate
its byte range without parsing a file header.

## Representation

- rows: 11,000,000;
- features per row: 28;
- scalar type: IEEE-754 `float32`;
- byte order: little-endian;
- layout: contiguous row-major;
- labels: excluded; and
- values: z-score normalized with population standard deviation.

The sidecar JSON file records the dimensions, preprocessing method, feature
means, and feature standard deviations.

## MPI-IO offset

For a rank whose first global row is `start_row`, the byte offset is:

```text
offset = start_row * 28 * sizeof(float)
```

The rank reads `local_rows * 28` consecutive floats. Phase 5 will calculate
`start_row` and `local_rows` so the final remainder rows are distributed
correctly when the dataset size is not divisible by the number of ranks.

## Validation

The complete matrix must be exactly:

```text
11,000,000 * 28 * 4 = 1,232,000,000 bytes
```

The preprocessing program checks this size and compares the first 1,000 binary
rows with values independently parsed and normalized from the source archive.
