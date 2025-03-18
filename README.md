## Repository for the ICLR paper "A Causal Lens for Learning Long-term Fair Policies."

The **environment.yml** file contains all the dependencies to run this code and can be installed using Anaconda. It has only been tested on Linux systems. The key requirements are Python, PyTorch, gym, stable-baselines3, and geomloss (geomloss is only for running F-PPO-L).

Once requirements are installed, you can run the code with the command

`$ python main.py --train --eval`.

In **config.py**, you can set which model to run with the booleans near the top. To reproduce results from the paper, set the DELAYED_IMPACT_CLUSTER_PROBS, DELAYED_IMPACT_SUCCESS_PROBS, and DRIFT_PROBS variables as follows for each setting: 

Setting 1
```
DELAYED_IMPACT_CLUSTER_PROBS = (
	(0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.0, 0.0),
	(0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.0, 0.0, 0.0, 0.0),   
)
DELAYED_IMPACT_SUCCESS_PROBS = (0.773, 0.804, 0.833, 0.857, 0.879, 0.898, 0.914, 0.928, 0.939, 0.949, 0.958, 0.965, 0.970, 0.975 )
DRIFT_PROBS = (
		(0.1, 0.8, 0.1),
		(0.1, 0.8, 0.1)
)
```
    
Setting 2
```
DELAYED_IMPACT_CLUSTER_PROBS = (
	(0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.0, 0.0),
	(0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.0, 0.0, 0.0, 0.0),   
)
DELAYED_IMPACT_SUCCESS_PROBS = (0.506, 0.594, 0.677, 0.750, 0.812, 0.861, 0.898, 0.927, 0.948, 0.963, 0.974, 0.982, 0.987, 0.991)
DRIFT_PROBS = (
		(0.1, 0.8, 0.1),
		(0.1, 0.8, 0.1)
)
```	

Setting 3
```
DELAYED_IMPACT_CLUSTER_PROBS = (
	(0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.0, 0.0),
	(0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.0, 0.0),   
)
DELAYED_IMPACT_SUCCESS_PROBS = (0.773, 0.804, 0.833, 0.857, 0.879, 0.898, 0.914, 0.928, 0.939, 0.949, 0.958, 0.965, 0.970, 0.975 )
DRIFT_PROBS = (
		(0.05, 0.85, 0.1),
		(0.25, 0.7, 0.05)
)
```