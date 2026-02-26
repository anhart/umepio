# Generating UMEP inputs

This package automatically generates the neccesary input data for running UMEP solver SOLWEIG in the Netherlands.
It needs to be given a bounding box and an output folder and generates the files needed to run SOLWEIG_GPU.

Generate all the files at once.

```Python
    import umep_inputs
    bbox = (94040,437614, 95230, 439399) #bounding box in Amersfoort / RD new
    inputs = umep_inputs.generate_solweig_inputs(bbox, "test_output")
```
Generate some of the files as needed.

```Python
    # bounding box, path to output folder
    buildings = umep_inputs.load_buildings(bbox, "test_output")

    # bounding box, buildings, path to output folder
    dtm, dsm, _ = umep_inputs.load_dems(bbox, buildings, "test_output")

    # bounding box, dtm, dsm, trunk height, path to output folder
    chm = umep_inputs.load_chm(bbox, dtm, dsm, 25, "test_output")

    # bounding box, path to output folder
    landcover = umep_inputs.load_landcover(bbox, "output_folder")
```

The repository builds on the work done by Jessica Monahan, [SOLFD](https://github.com/jsscmnhn/SOLWEIG_SOLFD) (2025). The logic is the same but the code is reformatted and polished to work as a standalone pip package. 
