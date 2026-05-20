## Thesis Code

All training and dataset loading code for my thesis.

### Repository layout

- [train_resnet.py](train_resnet.py) - ResNet training entry point.
- [train_timesformer.py](train_timesformer.py) - TimeSformer training entry point.
- [sampling.py](sampling.py) - UCF101 temporal sampling class.
- [gpu_test.py](gpu_test.py) - Quick GPU sanity check.
- [analysis.ipynb](analysis.ipynb) - Analysis notebook.
- [requirements.txt](requirements.txt) - Python dependencies.
- [raw results/](raw%20results/) - Per-run logs and JSON outputs.
- [processed results/](processed%20results/) - Aggregated CSV summaries.

### Dataset

The UCF101 dataset and the train-test splits can be downloaded from [this website](https://www.crcv.ucf.edu/data/UCF101.php). The dataset was not included in the repository due to it's large size. All other materials needed to reproduce the results are here.

### Environment

The code was developed and tested with the following environment. Exact versions may vary, but matching them is recommended for reproducibility.

- Python 3.11.9
- CUDA 12.1 (for GPU training)
- PyTorch/torchvision versions from [requirements.txt](requirements.txt)

### Quick start

1. Create a Python environment.
2. Install dependencies from [requirements.txt](requirements.txt).
3. Download the dataset and train-test splits from [here](https://www.crcv.ucf.edu/data/UCF101.php).
4. Place the data in the expected layout (or pass custom paths via flags):

```
UCF101/
	ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi
	...
splits/
	trainlist01.txt
	testlist01.txt
```

5. Train a model (examples):

```bash
python train_resnet.py --video_root UCF101 --train_split splits/trainlist01.txt --test_split splits/testlist01.txt
python train_timesformer.py --video_root UCF101 --train_split splits/trainlist01.txt --test_split splits/testlist01.txt
```

6. (Optional) Run analysis and aggregate results in [analysis.ipynb](analysis.ipynb).

Outputs are written to the `outputs/` directory by default and include a model checkpoint plus a `metrics_summary_*.json` file.

### Notes

- Raw and processed results are included for reference and replication.

### Citations and attribution

If you reuse this code or results, please cite the following sources:

- UCF101 dataset: https://www.crcv.ucf.edu/data/UCF101.php
- R3D-18 architecture and pretrained weights from Torchvision: https://pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html
- TimeSformer model and pretrained weights from Hugging Face (Kinetics-400): https://huggingface.co/facebook/timesformer-base-finetuned-k400
- TimeSformer paper: https://arxiv.org/abs/2102.05095
- R3D-18 (3D ResNet) paper: https://arxiv.org/abs/1711.11248
