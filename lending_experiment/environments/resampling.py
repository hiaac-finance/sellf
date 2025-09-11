import attr
from copy import deepcopy
import numpy as np
import pickle as pkl

import gym
from gym import spaces
from time import time


class ResamplingEnv(gym.Env):
    """
    Resampling environment to facilitate calculating metrics along individuals.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        n_groups: int = 2,
        cost: float = 0.8,
        n_applicants: int = 10000,
        n_features: int = 10,
        utility_method: str = "accuracy",
        delta_method: str = "full",
        seed = None,
    ):
        assert utility_method in ["accuracy", "qualification", "tpr"]
        assert delta_method in ["full", "imputation", "imputation_hard", "accepted"]
        self.n_groups = n_groups
        self.cost = cost
        self.n_applicants = n_applicants
        self.n_features = n_features
        self.utility_method = utility_method
        self.delta_method = delta_method
        self.action_space = spaces.Discrete(2)

        resource_space = spaces.Box(
            low=0.0,
            high=100_000,
            dtype=np.float64,
            shape=(),
        )
        applicant_space = spaces.Box(
            low=0.0, high=1.0, dtype=np.float32, shape=(self.n_features,)
        )
        group_space = spaces.MultiBinary(n=self.n_groups)

        self.observable_state_vars = {
            "resource": resource_space,
            "applicant_features": applicant_space,
            "group": group_space,
        }
        self.observation_space = spaces.Dict(self.observable_state_vars)
        self.init_data = dict(
            [
                (col, np.zeros(self.n_applicants, dtype=np.float32))
                for col in [
                    "group",
                    "features",
                    "label",
                    "action",
                    "pred",
                    "utility",
                    "active",
                    "utility_obs",
                    "active_obs",
                ]
            ]
        )
        self.init_data["features"] = np.zeros((self.n_applicants, self.n_features), dtype=np.float32)
        self.init_data["group"] = self.init_data["group"].astype(np.int32)
        self.data = deepcopy(self.init_data)
        self.pool_accepted = [[] for _ in range(self.n_groups)]
        self.pool_rejected = [[] for _ in range(self.n_groups)]
        self.timestep = 0

        self.get_label_pred = lambda x, g : 0
        self.get_action = lambda x, g : 0
        self.get_action_prob = lambda x, g : 0
        self.get_action_prob_list = lambda x, g : 0
        self.get_action_prob_batch = lambda x, g : 0
        self.get_pred_batch = lambda x, g : 0
        self.delta_obs = 0
        self.error_bound = [0, 0]
        self.error_accepted = [0, 0]
        self.error_rejected = [0, 0]
        self.divergence = [0, 0]
        self.seed(seed)
        self._load_data()
        self._state_init()
        

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def _load_data(self):
        """This function should fill the entries in self.init_data"""
        return

    def _state_init(self):
        self.resource = 1_000
        self.sample_applicant()

    def sample_applicant(self):
        selected = np.random.choice(self.n_applicants, size=1)[0]
        self.idx = selected
        return

    def reset(self):
        """Resets the environment."""
        self.timestep = 0
        self.data = deepcopy(self.init_data)
        self._state_init()
        self.compute_disparity()
        return self._get_observable_state()

    def update_models(self):
        self.pool_rejected = [[] for _ in range(self.n_groups)]
        for idx in range(self.n_applicants):
            group = self.init_data["group"][idx]
            features = self.init_data["features"][idx]
            label = self.init_data["label"][idx]
            pred = self.get_label_pred(features, group)
            action = self.get_action(features, group)
            self.init_data["pred"][idx] = pred
            self.init_data["action"][idx] = action
            if action == 0:
                self.pool_rejected[group].append(
                    {
                        "features": features,
                        "label": label,
                        "one_minus_pi": None,
                    }
                )
            self.update_utility(idx, label, pred, action, init=True)
            

        if "imputation" in self.delta_method:
            # update accepted pool
            # Calculate "one_minus_pi" of new accepted individuals
            for i in range(self.n_groups):
                group = np.zeros(self.n_groups)
                group[i] = 1
                for j in range(len(self.pool_accepted[i])):
                    if self.pool_accepted[i][j]["one_minus_pi"] is None:
                        features = self.pool_accepted[i][j]["features"]
                        prob_list = self.get_action_prob_list(features, group)
                        self.pool_accepted[i][j]["one_minus_pi"] = np.prod(
                            [1 - x for x in prob_list]
                        )

            # Here I need to calculate the error_bound, that is the error of the predictor plus
            # the Chi Squared divergence term

            self.error_accepted = np.zeros(self.n_groups)
            self.error_rejected = np.zeros(self.n_groups)
            self.divergence = np.zeros(self.n_groups)

            if len(self.pool_accepted[0]) == 0 or len(self.pool_accepted[1]) == 0:
                return

            for i in range(self.n_groups):
                # Calculate first the error on the rejected population
                group = np.zeros(
                    (len(self.pool_rejected[i]), self.n_groups), dtype=np.float32
                )
                group[:, i] = 1
                features = np.array(
                    [applicant["features"] for applicant in self.pool_rejected[i]]
                )
                labels = np.array(
                    [applicant["label"] for applicant in self.pool_rejected[i]]
                )
                preds = self.get_pred_batch(features, group)
                self.error_rejected[i] = np.mean(preds - labels)

                # Create a large dataset of accepted applicants
                group = np.zeros(
                    (len(self.pool_accepted[i]), self.n_groups), dtype=np.float32
                )
                group[:, i] = 1
                features = np.array(
                    [applicant["features"] for applicant in self.pool_accepted[i]]
                )

                # Calculate the error of the updated classifier on the accepted population
                labels = np.array(
                    [applicant["label"] for applicant in self.pool_accepted[i]]
                )

                # Calculate the weights
                one_minus_pi = 1 - self.get_action_prob_batch(features, group)
                one_minus_pi_hist = np.array(
                    [applicant["one_minus_pi"] for applicant in self.pool_accepted[i]]
                )
                one_minus_pi_hist *= one_minus_pi
                # Save new one_minus_pi_hist
                for j in range(len(self.pool_accepted[i])):
                    self.pool_accepted[i][j]["one_minus_pi"] = one_minus_pi_hist[j]

                one_minus_pi_hist = 1 - one_minus_pi_hist

                weights = (one_minus_pi / one_minus_pi_hist)

                #print(f"Weights stats: {weights.mean():.2f} , {weights.max():.2f}")
                
                preds = self.get_pred_batch(features, group)
                self.error_accepted[i] = (weights * (preds - labels)).mean()

                renyi_div = np.mean((one_minus_pi / one_minus_pi_hist)**2)

                # calculate complexity term
                delta = 0.95
                m = len(one_minus_pi)
                p = self.n_features
                C = np.sqrt((p * np.log(2 * m * np.e / p) + np.log(4/ delta)) / m)
                C *= np.power(renyi_div, 3/8)
                C *= 2**(5/4)

                #print(f"Renyi div: {renyi_div:.2f}, C: {C:.4f}")
                self.divergence[i] = C

            self.error_bound = self.error_accepted + self.divergence
            self.error_bound = np.clip(self.error_bound, 0, 1)


    def compute_disparity(self):
        # calculate disparity using self.data
        self.data["utility"] *= self.data[
            "active"
        ]  # make utility of "inactive" be equal to 0

        self.group_counts = np.array(
            [
                np.sum(self.data["active"][self.data["group"] == i])
                for i in range(self.n_groups)
            ]
        )
        self.utility_sum = np.array(
            [
                np.sum(self.data["utility"][self.data["group"] == i])
                for i in range(self.n_groups)
            ]
        )

        self.utility_values = np.true_divide(
            self.utility_sum,
            self.group_counts,
            where=self.group_counts != 0,
            out=np.zeros_like(self.utility_sum),
        )
        old_delta = self.delta_obs
        self.delta = abs(self.utility_values[1] - self.utility_values[0])
        self.delta_obs = self.delta
        self.delta_pred = 0

        if "accepted" in self.delta_method or "imputation" in self.delta_method:
            self.data["utility_obs"] *= self.data["active_obs"]
            self.group_counts_obs = np.array(
                [
                    np.sum(self.data["active_obs"][self.data["group"] == i])
                    for i in range(self.n_groups)
                ]
            )
            self.utility_sum_obs = np.array(
                [
                    np.sum(self.data["utility_obs"][self.data["group"] == i])
                    for i in range(self.n_groups)
                ]
            )
            print(self.utility_sum_obs, self.group_counts_obs)
            self.utility_values_obs = np.true_divide(
                self.utility_sum_obs,
                self.group_counts_obs,
                where=self.group_counts_obs != 0,
                out=np.zeros_like(self.utility_sum_obs),
            )
            self.delta_obs = abs(
                self.utility_values_obs[1] - self.utility_values_obs[0]
            )

        if "imputation" in self.delta_method:
            # calculate other necessary delta values
            group_counts = np.array(
                [np.sum(self.data["group"] == i) for i in range(self.n_groups)]
            )
            accept_sum = np.array(
                [
                    np.sum(self.data["action"][self.data["group"] == i])
                    for i in range(self.n_groups)
                ]
            )
            accept_rate = np.true_divide(
                accept_sum,
                group_counts,
                where=group_counts != 0,
                out=np.zeros_like(accept_sum),
            )

            imputation = self.data["action"] * self.data["label"] + (1 - self.data["action"]) * self.data["pred"]
            pred_sum = np.array(
                [
                    np.sum(imputation[self.data["group"] == i])
                    for i in range(self.n_groups)
                ]
            )
            pred_rate = np.true_divide(
                pred_sum,
                group_counts,
                where=group_counts != 0,
                out=np.zeros_like(pred_sum),
            )

            # here, the error bound will already be calculated
            delta_pred = np.zeros(self.n_groups)
            if self.utility_method in ["qualification", "accuracy"]:
                delta_pred = self.error_bound * (1 - accept_rate)
            else:
                delta_pred = self.error_bound * (1 - accept_rate) / pred_rate

            if self.delta_method == "imputation_hard":
                self.delta_pred = max(delta_pred[1], delta_pred[0]) 
            else:
                self.delta_pred = abs(delta_pred[1] - delta_pred[0])

            delta_pred_real = np.zeros(self.n_groups)
            if self.utility_method in ["qualification", "accuracy"]:
                delta_pred_real = (self.error_rejected) * (1 - accept_rate)
            else:
                delta_pred_real = (self.error_rejected) * (1 - accept_rate) / pred_rate
            self.delta_pred_real = abs(delta_pred_real[1] - delta_pred_real[0])

        self.delta_delta = self.delta_obs - old_delta

    def _get_observable_state(self):
        group = np.zeros(self.n_groups, dtype = np.float32)
        group[int(self.data["group"][self.idx])] = 1
        return {
            "resource": np.array(self.resource),
            "applicant_features": np.array(self.data["features"][self.idx]),
            "group": group,
        }

    def _is_done(self):
        return self.resource <= 0 or self.timestep >= 10_000

    def step(self, action):
        self.timestep += 1
        old_resource = self.resource
        group = self.data["group"][self.idx]
        features = self.data["features"][self.idx]
        label = self.data["label"][self.idx]
        pred = self.data["pred"][self.idx]

        if action == 1 and len(self.pool_accepted[group]) < 50_000:
            self.pool_accepted[group].append(
                {
                    "features": features,
                    "label": label,
                    "one_minus_pi": None,
                }
            )
        
        self.update_resource(action, label)
        # Update utility based on this action for this applicant
        self.update_utility(self.idx, label, pred, action)
        # Update resource with action
        self.compute_disparity()
        # Update features of applicant based on action
        self.update_applicant(self.idx, action)
        self.sample_applicant()
        observation = self._get_observable_state()
        reward = self.resource - old_resource
        return observation, reward, self._is_done(), {}

    def update_resource(self, action, label):
        if action == 0:  # reject
            return
        self.resource += label - self.cost
        self.resource = max(0, self.resource)

    def update_utility(self, idx, label, pred, action, init=False):
        """Update the difference in utility for the current applicant. Also updates the utility in the pool."""
        # First, calculate real utility
        active = 1
        if self.utility_method == "accuracy":
            utility_value = 1 if label == action else 0
        elif self.utility_method == "qualification":
            utility_value = label
        elif self.utility_method == "tpr":
            utility_value = action
            active = 1 if label == 1 else 0

        utility_obs = utility_value
        active_obs = active
        if self.delta_method == "accepted":
            utility_obs = utility_value
            active_obs = active if action == 1 else 0

        if "imputation" in self.delta_method:
            active_obs = 1
            label_obs = label if action == 1 else pred
            if self.utility_method == "accuracy":
                utility_obs = 1 if label_obs == action else 0
            elif self.utility_method == "qualification":
                utility_obs = label_obs
            elif self.utility_method == "tpr":
                utility_obs = action
                active_obs = 1 if label_obs == 1 else 0

        data_to_change = self.init_data if init else self.data
        data_to_change["utility"][idx] = utility_value
        data_to_change["active"][idx] = active
        data_to_change["utility_obs"][idx] = utility_obs
        data_to_change["active_obs"][idx] = active_obs

    def update_applicant(self, idx, action):
        # Implement logic to update the applicant based on the action taken
        features = self.data["features"][idx]
        label = self.data["label"][idx]
        features = self.update_features(features, action, label)
        self.data["features"][idx] = features

        group = self.data["group"][idx]
        label = self.get_label(features, group)
        pred = self.get_label_pred(features, group)
        action = self.get_action(features, group)
        if action == 0:
            self.pool_rejected[group].append(
                {
                    "features": features,
                    "label": label,
                    "one_minus_pi": None,
                }
            )
        self.data["label"][idx] = label
        self.data["pred"][idx] = pred
        self.data["action"][idx] = action


    def update_features(self, features, action, label):
        return features


class LendingEnv(ResamplingEnv):
    """
    Environment for lending experiments.
    """

    def __init__(
        self,
        cost: float = 0.8,
        n_applicants: int = 10000,
        utility_method: str = "accuracy",
        delta_method: str = "full",
        group_ratios: str = "data",
        seed = None,
    ):
        assert group_ratios in ["data", "equal"]
        self.n_groups = 2
        self.cost = cost
        self.n_applicants = n_applicants
        self.n_features = 10
        self.utility_method = utility_method
        self.delta_method = delta_method
        self.group_ratios = group_ratios
        super().__init__(
            n_groups=self.n_groups,
            cost=self.cost,
            n_applicants=self.n_applicants,
            n_features=self.n_features,
            utility_method=self.utility_method,
            delta_method=self.delta_method,
            seed = seed,
        )

    def _load_data(self):
        with open("data/fico.pkl", "rb") as f:
            data = pkl.load(f)
        if self.group_ratios == "data":
            groups_probs = data["group_likelihoods"]
        else:
            groups_probs = [0.5, 0.5]
        cluster_probs = data["cluster_probabilities"]
        success_probs = data["success_probabilities"]

        def sample_label(x, g):
            # if x is not an scalar, get the argmax
            if not np.isscalar(x):
                x = np.argmax(x)
            return np.random.binomial(n=1, p=success_probs[g][x])

        self.get_label = sample_label

        num_groups = len(groups_probs)
        num_features = len(cluster_probs[0])

        for i in range(self.n_applicants):
            g = np.random.choice(num_groups, p=groups_probs)
            x = np.random.choice(num_features, p=cluster_probs[g])
            label = self.get_label(x, g)
            # one hot encode group and x
            features = np.zeros(num_features, dtype=np.float32)
            features[x] = 1
            group = np.zeros(num_groups, dtype=np.float32)
            group[g] = 1
            pred = self.get_label_pred(features, group)
            action = self.get_action(features, group)
            self.init_data["group"][i] = g
            self.init_data["features"][i] = features
            self.init_data["label"][i] = label
            self.init_data["pred"][i] = pred
            self.init_data["action"][i] = action
    

    def update_features(self, features, action, label):
        if action == 0:
            return features

        score = np.argmax(features)
        num_features = len(features)
        if label == 1:
            new_score = min(score + 1, num_features - 1)
        elif label == 0:
            new_score = max(score - 1, 0)
        features[score] = 0
        features[new_score] = 1
        return features

class EnemEnv(ResamplingEnv):
    """
    Environment for school admission experiments.
    """
    def __init__(
        self,
        cost: float = 0.5,
        n_applicants: int = 10000,
        utility_method: str = "accuracy",
        delta_method: str = "full",
        seed = None,
    ):
        super().__init__(
            n_groups=2,
            n_features=132,
            cost=cost,
            n_applicants=n_applicants,
            utility_method=utility_method,
            delta_method=delta_method,
            seed = seed,
        )

    def _load_data(self):
        with open("data/enem_pool.pkl", "rb") as f:
            pool = pkl.load(f)

        with open("data/enem_model.pkl", "rb") as f:
            self.model = pkl.load(f)

        def sample_label(x, g):
            if not np.isscalar(g):
                g = np.argmax(g)
            x = np.array([x[0], g, *x[1:]])
            p = self.model.predict_proba(x.reshape(1, -1))[0, 1]
            return 1 if np.random.rand() < p else 0

        self.get_label = sample_label

        for i in range(self.n_applicants):
            group = pool[i]["group"]
            features = pool[i]["features"]

            label = self.get_label(features, group)
            pred = self.get_label_pred(features, group)
            action = self.get_action(features, group)
            self.init_data["group"][i] = group
            self.init_data["features"][i] = features
            self.init_data["label"][i] = label
            self.init_data["pred"][i] = pred
            self.init_data["action"][i] = action

    def update_features(self, features, action, label):
        if action == 1 or features[0] == 1:
            # add marker that already passed
            features[0] = 1
            return features

        group = np.argmax(features[1:3])
        age_groups = 6
        age = np.argmax(features[(age_groups * group + 3) : (age_groups * (group + 1) + 3)])
        new_age = min(age + 1, age_groups - 1)

        new_features = features.copy()
        new_features[age_groups * group + 3 : age_groups * (group + 1) + 3] = 0
        new_features[age_groups * group + new_age + 3] = 1
        return new_features
