import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
import os
import json
import torch
from pathlib import Path


def show_torch_cuda_info():
    """Display information about the Torch CUDA configuration."""
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: {(current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


def mha_to_nii(mha_files):
    # Function to convert multiple files
    for i in tqdm(range(len(mha_files))):
        if ".mha" in str(mha_files[i].resolve()):
            # Read the .mha file
            img = sitk.ReadImage(mha_files[i])
            # Write the image as .nii
            nii_filename = (
                str(mha_files[i].resolve())
                .replace("mha", "nii")
                .replace("test/input", "proc")
            )
            sitk.WriteImage(img, nii_filename)


# nii to mha
def nii_to_mha(nii_files):
    # Function to convert multiple files
    for i in tqdm(range(len(nii_files))):
        if ".nii" in str(nii_files[i].resolve()):
            # Read the .nii file
            img = sitk.ReadImage(nii_files[i])
            # Write the image as .mha
            mha_filename = (
                str(nii_files[i].resolve())
                .replace("nii", "mha")
                .replace("pred", "test/output")
            )
            sitk.WriteImage(img, mha_filename)


# Create and save json to mapping all filename in mha to id in nii with format: abc.mha -> xxx_0000.nii (xxx is the integer id)
def create_mapping(mha_files):
    mapping = {}
    for i in range(len(mha_files)):
        mapping[str(mha_files[i].name).replace(".mha", "")] = str(i).zfill(3) + "_0000"
    with open("mapping.json", "w") as f:
        json.dump(mapping, f)


# Base on above json, rename all nii files
def rename_nii_files(nii_files):
    with open("mapping.json", "r") as f:
        mapping = json.load(f)

    for i in range(len(nii_files)):
        # os.rename(key, mapping[key])
        key = str(nii_files[i].name).replace(".nii", "")
        val = mapping[key]
        print(key, val)
        print(str(nii_files[i].resolve()))
        if nii_files[i].exists():
            print("File exists")
        else:
            print("File does not exist")

        destination_path = nii_files[i].with_name(nii_files[i].name.replace(key, val))
        nii_files[i].rename(destination_path)


# Base on above json, rename reversely all mha files in output folder
def rename_mha_files(mha_files: list[Path]):
    with open("mapping.json", "r") as f:
        mapping = json.load(f)

    for i in range(len(mha_files)):
        for key, val in mapping.items():
            if val == str(mha_files[i].name).replace(".mha", "") + "_0000":
                print(key, val, str(mha_files[i].name).replace(".mha", "") + "_0000")
                destination_path = mha_files[i].with_name(
                    mha_files[i].name.replace(
                        str(mha_files[i].name).replace(".mha", ""), key
                    )
                )
                print(destination_path)
                mha_files[i].rename(destination_path)
                break
