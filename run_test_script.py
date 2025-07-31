import torch
from pytorch_lightning import seed_everything

from model_2.model_builder import ijepa_model_builders
from data_loader import RGBDDataModule

from configs import (
    get_image_experiment_config,
    get_image_model_config,
    get_image_dataset_config,
)

def main():
    # Make reproducible
    seed_everything(42)
    print("import complete")
    print('__________________________________________________________________________')

    # Load and override configs
    experiment_config = get_image_experiment_config()
    model_config = get_image_model_config()
    dataset_config = get_image_dataset_config()

    experiment_config["BATCH_SIZE"] = 1
    dataset_dir = dataset_config["DATASET_DIR"]

    print("Done getting config")
    print('__________________________________________________________________________')

    # Load datamodule with 1-worker for Windows safety
    datamodule = RGBDDataModule(
        dataset_dir=dataset_dir,
        batch_size=1,
        img_size=model_config["IMAGE_SIZE"],
        num_workers=0,  # Avoid multiprocessing crash in test
    )
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()

    print(">>>>> Finish loading data")
    print('__________________________________________________________________________')

    # Load the smallest model ("nano")
    model = ijepa_model_builders["nano"]()
    print(">>>>> Instantiate model Completed")
    print('__________________________________________________________________________')

    # Pull one batch
    batch = next(iter(loader))
    x_img, x_dep = batch
    print("Image shape:", x_img.shape)
    print("Depth shape:", x_dep.shape)

    print(">>>>> Start training ......")
    print('__________________________________________________________________________')

    # Forward pass
    model.train()
    print(">>>>> Forward pass ....")
    print('__________________________________________________________________________')

    y_student, y_teacher = model(
        x_img=x_img,
        x_dep=x_dep,
        target_aspect_ratio=0.75,
        target_scale=0.15,
        context_aspect_ratio=1.0,
        context_scale=0.9
    )

    print("Student prediction shape:", y_student[0].shape)
    print("Teacher target shape:", y_teacher[0].shape)
    print('__________________________________________________________________________')

    # Loss computation
    loss = model.criterion(y_student, y_teacher)
    print("Loss:", loss.item())

    # Backward pass
    loss.backward()
    print("✅ Backward pass successful!")

if __name__ == "__main__":
    main()
