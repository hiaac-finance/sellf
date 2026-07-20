# Long-Term Fairness with Selective Labels

Code for the paper "[Long-Term Fairness with Selective Labels](https://arxiv.org/abs/2605.22291)" presented at ICML 2026. Code contains the implementation of proposed algorithm, baselines and experiments.


## Requirements

conda was used for development with an environment with Python 3.8. To set up the environment, use the provided `environment.yml` file.

   ```bash
   conda env create -f environment.yml
   conda activate fairrl
   ```

## Repository Structure

- `fairrl/`: Directory containing the implementation of the proposed algorithm and baselines. The environment is implemented in `fairrl/environments/resampling.py` and all algorithms are implemented in `fairrl/agents/`. The `fairrl/configs/` directory contains the hyperparameter configurations for all algorithms.
- `notebooks/`: Directory containing Jupyter notebooks for data preprocessing and figure generation.
- `main.py`, `bound.py`, `ablation.py`: Scripts to run the main experiments, compute bounds and run ablation studies.

### Running the main experiment

To run the main experiment, use:

```bash
python main.py --env_name <env_name> --algorithm <algorithm> --mu_type <mu_type> --config_id
```

`<env_name>` is the environment selection and should be `fico`, `enem` or `compas`. `<algorithm>` is the algorithm to use and should be `ppo`, `pocar`, `pocar_full` or `sellf`. `<mu_type>` is the type of fairness metric and should be `tpr`, `accuracy` or `qualification`. --config_id is the id of the hyperparameter configuration to use.


### Running ablation studies

To run ablation experiments, use:

```bash
python ablation.py --env_name <env_name> --algorithm <algorithm> --mu_type <mu_type> --seed_id <seed_id> --ablation_type <ablation_type>
```

Where `<seed_id>` is the seed for the random number generator. Select `--ablation_type` to be `beta1` or `beta2` to run the corresponding ablation study.

### Running bound computation

To compute the bounds, use:

```bash
python bound.py --env_name <env_name> --algorithm <algorithm> --mu_type <mu_type> --seed_id <seed_id>
```

This script runs in similar fashion to the main experiment, but computes the bounds for the given algorithm and environment.

## Comments

This code was built on top of the code from [Policy Optimization with Constraint Advantage Regularization](https://github.com/ericyangyu/pocar). ELBERT was obtained from the [official repository](https://github.com/umd-huang-lab/ELBERT) and FOCOPS was also obtained from the [official repository](https://github.com/ymzhang01/focops).


Currently working in a a tiny implementation of the proposed algorithm! Contact me if you are interested.


## Contact

For any questions or issues, please reach me at: giovani.valdrighi at ic.unicamp.br


## Citation

If you use this code in your research, please cite the following paper:

```
@inproceedings{valdrighilong,
  title={Long-term Fairness with Selective Labels},
  author={Valdrighi, Giovani and Valera, Isabel and Raimundo, Marcos M},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
}
```


