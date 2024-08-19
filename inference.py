"""
This is a simple example algorithm meant to run within a container.

To run it locally, call the following bash script:

  ./test_run.sh

This will start the inference and read from ./test/input and output to ./test/output.

To save the container and prepare it for upload to Grand-Challenge.org, call:

  ./save.sh

Any container that shows the same behavior will do; this is purely an example of how one could do it.

Happy programming!
"""

from pathlib import Path
from utils import (
    show_torch_cuda_info,
    mha_to_nii,
    nii_to_mha,
    create_mapping,
    rename_nii_files,
    rename_mha_files,
)
from predict import run_prediction
from batchgenerators.utilities.file_and_folder_operations import join
# from nnunetv2.paths import nnUNet_results, nnUNet_raw
import os

import time
start_time = time.time()

from docker_utils import get_volume_info

# os.environ["nnUNet_def_n_proc"] = "2"

# Print the current working directory
print("Current working directory:", Path.cwd())

# Print all subdirectories and files in the current working directory
print("All subdirectories and files in the current working directory:")
for x in Path.cwd().iterdir():
    print(x)

# Set the paths for input, output, and resources
INPUT_PATH = Path("test/input")
PROC_PATH = Path("proc")
PRED_PATH = Path("pred")
OUTPUT_PATH = Path("test/output")

# Create folders pred and proc if they do not exist
if not PRED_PATH.exists():
    PRED_PATH.mkdir()
if not PROC_PATH.exists():
    PROC_PATH.mkdir()

# Print all subdirectories and files in the current working directory
print("All subdirectories and files in the current working directory:")
for x in Path.cwd().iterdir():
    print(x)

# Get all mha files in current working directory and its subdirectories
all_mha_files = list(Path.cwd().rglob("*.mha"))
print("All mha files in current working directory and its subdirectories:", all_mha_files)

# # Print the paths
# try:
#     container_name = "mtoan65-pengwin-mtoan65-submission:mtoan65-submission-v5"
#     input_vol, output_vol = get_volume_info(container_name)
#     print("Input volume:", input_vol)
#     print("Output volume:", output_vol)
# except Exception as e:
#     print("Error getting volume information:", e)



def main():
    # Display CUDA information
    show_torch_cuda_info()

    # Get all mha files in input folder
    mha_files = list((INPUT_PATH).rglob("*.mha"))
    print("mha files:", mha_files)

    # print(mha_files[0].name)

    # Create mapping between mha and nii files
    create_mapping(mha_files)
    print("Mapping created")

    # Loop for prediction
    for mha_file in mha_files:
        # Convert mha files to nii files
        mha_to_nii([mha_file])
        print("mha files converted to nii")

        # Get all nii files in processed folder
        nii_files = list((PROC_PATH).rglob("*.nii"))
        print("nii files:", nii_files)

        # Rename nii files in processed folder
        rename_nii_files(nii_files)
        print("nii files renamed")
        nii_files = list((PROC_PATH).rglob("*.nii"))
        print("nii files:", nii_files)

        # Run prediction
        run_prediction()
        print("Prediction done")

        # Delete nii files in processed folder
        nii_files = list((PROC_PATH).rglob("*.nii"))
        for nii_file in nii_files:
            nii_file.unlink()

    # Get all nii files in pred folder
    nii_files = list((PRED_PATH).rglob("*.nii"))
    print("nii files:", nii_files)

    # Convert nii files to mha files
    nii_to_mha(nii_files)
    print("nii files converted to mha")

    # Get all output mha files in output folder
    output_mha_files = list((OUTPUT_PATH).rglob("*.mha"))
    print("mha files:", output_mha_files)

    # Rename mha files
    rename_mha_files(output_mha_files)
    print("mha files renamed")

    # Print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:",execution_time)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
