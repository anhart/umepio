# Generating UMEP inputs

This package automatically generates the neccesary input data for running UMEP solvers SOLWEIG and URock in the Netherlands.
It needs to be given a bounding box and an output folder and generates the files needed to run SOLWEIG_GPU.

Generate all the files at once.

```Python
 inputs = umep_inputs.generate_solweig_inputs(bbox, "test_output")

Generate some of the files as needed.
