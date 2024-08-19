import torch
from batchgenerators.utilities.file_and_folder_operations import join

from predict_from_raw_data_new import nnUNetPredictor
from nnunetv2.paths import nnUNet_results, nnUNet_raw


def run_prediction():
    """Run the prediction using nnUNetPredictor."""
    # Instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    print("Predictor initialized")

    # Initialize the network architecture and load the checkpoint
    predictor.initialize_from_trained_model_folder(
        "./Model/exp_3_200_epoch/Dataset001_PengwinTask01/nnUNetTrainer__nnUNetPlans__3d_lowres",
        use_folds=(0,),
        checkpoint_name="checkpoint_best.pth",
    )
    print("Network architecture initialized")

    # Predict from input files and save the results
    predictor.predict_from_files(
        "./proc",
        "./pred",
        save_probabilities=False,
        overwrite=False,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )
    print("Prediction done")
