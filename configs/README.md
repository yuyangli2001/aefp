# Configuration Notes

Some defaults in this directory are intended for training and parameter sweeps and do not fully match the settings used in the paper.

- **Paper models**: MNIST uses `resae_3down` with 6 bottleneck channels; RGB datasets (CIFAR-10 and CelebA-HQ-64) use `resae_2down` with 20 bottleneck channels. Recovery starts from the corrupted observation.
- **Training configuration**: `train/ae.yaml` defines `original` and `auxiliary` training, both using MSE. Auxiliary models are selected based on PSNR on the complete clean validation split. The `direct` and FF-AE methods prepare only the original model; AEFP also prepares the auxiliary models.
- **Extensions**: Plain convolutional AEs, inpainting, block inpainting, deblurring, and the 256-resolution configuration are implementation extensions outside the scope of the paper.
- **Paper evaluation**: TODO: Define a one-command reproduction entry point once the remaining changes are complete.
