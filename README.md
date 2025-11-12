# Long-Term Fairness with Selective Labels

This supplementary code contains the implementation for the experiments described in the paper "Long-Term Fairness with Selective Labels". The code is structured to facilitate running the main experiments as well as ablation studies.

## Requirements

We used conda for development with an environment with Python 3.8. To set up the environment, use the provided `environment.yml` file.

   ```bash
   conda env create -f environment.yml
   conda activate fairrl
   ```

## Code Organization

- `enem.ipynb`: Jupyter notebook for data preprocessing of ENEM dataset.
- `fico.ipynb`: Jupyter notebook for data preprocessing of FICO dataset.
- `figures.ipynb`: Jupyter notebook for generating figures from results.
- `main.py`: Script to run the main experiments.
- `ablation.py`: Script to run ablation studies.
- `config/`: Directory containing configuration of algorithms hyperparameters.

### Running the main experiment

To run the main experiment, use:

python main.py --env_name fico_equal --algorithm ppo --mu_type tpr --config_id $SLURM_ARRAY_TASK_ID

```bash
python main.py --env_name <env_name> --algorithm <algorithm> --mu_type <mu_type> --config <config_file>
```

Where `<env_name>` is the environment name (e.g., `fico_equal`, `enem`), `<algorithm>` is the algorithm to use (e.g., `ppo`, `pocar`, `pocar_full`, `sellf`), and `<mu_type>` is the type of fairness metric (e.g., `tpr`, `accuracy`, `qualification`). --config is the id of the hyperparameter configuration to use.

### Running ablation studies

To run ablation experiments, use:

```bash
python ablation.py --env_name fico_equal --algorithm sellf --mu_type accuracy --seed_id <seed_id>
```
Where `<seed_id>` is the seed for the random number generator.