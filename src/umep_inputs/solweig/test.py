import cProfile
import pstats
from io import StringIO
from umep_inputs.solweig.processing.chm import CHM
from umep_inputs.solweig.api import load_buildings,load_dems

bbox = (94040,437614,94251,437797)
b = load_buildings(bbox, "output_folder")
dtm, dsm, _ = load_dems(bbox, b,  "output_folder")

def initialize_chm_class():
    bbox = (94040,437614,94251,437797)  # Example bounding box values
    b = load_buildings(bbox, "output_folder")
    dtm, dsm, _ = load_dems(bbox, b,  "output_folder")
    trunk_height = 25  # Your desired trunk height percentage
    output_folder_chm = "output_folder"
    output_folder_las = "temp"
    merged_output = "pointcloud.las"
    resolution = 0.5
    ndvi_threshold = 0.05
    
    # Initialize the CHM class
    chm = CHM(bbox, dtm, dsm, trunk_height, output_folder_chm, output_folder_las, resolution, merged_output, ndvi_threshold)
    
# Use cProfile to profile the initialization process

pr = cProfile.Profile()
pr.enable()

initialize_chm_class()

pr.disable()

# Capture the stats and format them to show only the CHM methods
s = StringIO()
ps = pstats.Stats(pr, stream=s)
ps.strip_dirs()  # Clean up file paths to make the output more readable
ps.sort_stats('cumulative')  # Sort by cumulative time

# Filter out methods that are not part of the CHM class
ps.print_stats("chm.py")  # Only show methods in CHM class

# Print the filtered output
print(s.getvalue())