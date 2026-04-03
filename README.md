## Fatigue-Aware Learning to Defer via Constrained Optimisation (FALCON)

This repo is the official implementation of our paper [Fatigue-Aware Learning to Defer via Constrained Optimisation]
![image](https://github.com/zhengzhang37/FALCON/blob/main/workflow.jpg)

## Contributions

- L2D with workload-variant human performance: FALCON is the first L2D framework that explicitly models workload-dependent human performance and requires a sequential CMDP formulation because each deferral decision changes the future state via an workload accumulator, where allocations alter subsequent human’s cognitive state, unlike prior L2D works that assume static human accuracy and thus non-sequential gating.
- Psychologically Grounded Simulation Environment: We develop a human performance simulation environment grounded in psychological principles, offering a realistic testbed for evaluating L2D methods under {workload-variant} human performance conditions.
- Fatigue-Aware L2D (FA-L2D) Benchmark: We release the FA-L2D benchmark, based on [Cifar100](https://arxiv.org/abs/2110.12088), [Flickr](https://cdn.aaai.org/ojs/10485/10485-13-14013-1-2-20201228.pdf), [MiceBone](https://arxiv.org/abs/2106.16209), and [Chaoyang](https://github.com/bupt-ai-cz/HSA-NRL), which models controllable fatigue effects across varying time horizons, enabling scenarios from near-constant to highly variable human performance and replacing prior benchmarks that assumed static human performance.
  

## Environment Setup

Please refer to `jax.def` for the detailed setup of the Apptainer.

## Dataset Setup

The training and evaluation data is specified through json files. Each json file has a similar structure as follows:


```
[
    {
        "file": "test/538880-3-IMG009x012-3.JPG",
        "label": 3
    },
    {
        "file": "train/538880-3-IMG018x005-3.JPG",
        "label": 3
    },
    {
        "file": "train/539972_1-IMG009x011-3.JPG",
        "label": 3
    }
]
```
## Running

Please make changes in the `configs/conf.yaml` file for the dataset path. Note that some regularisations are required to avoid the overfitting of the classifier. Running the following command to train a classifier:

```
python3 "experiments/clf_training.py"
```

Please make changes in the `configs/falcon.yaml` and `configs/<dataset_name>.yaml` files for the dataset path, budget range and human fatigue parameters. For FALCON running please see the following command:

```
bash experiments/run.sh
```

Other human fatigue simulators can be replaced in `models/human_simulation_rl.py`.

## Citation

If you use this code/data for your research, please cite our paper [Fatigue-Aware Learning to Defer via Constrained Optimisation](https://arxiv.org/abs/2604.00904).

```bibtex
@article{zhang2026fatigue,
  title={Fatigue-Aware Learning to Defer via Constrained Optimisation},
  author={Zhang, Zheng and Nguyen, Cuong and Rosewarne, David and Wells, Kevin and Carneiro, Gustavo},
  journal={arXiv preprint arXiv:2604.00904},
  year={2026}
}
```

