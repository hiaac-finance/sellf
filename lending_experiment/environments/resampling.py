import attr
import copy
import numpy as np
import pickle as pkl

from gym import spaces

from lending_experiment.environments import core


@attr.s
class Params(object):
    num_groups = attr.ib(default=2)
    gain = attr.ib(default=0.25)
    starting_resource = attr.ib(default=1_000.0)
    max_resource = attr.ib(default=100_000.0)
    num_applicants = attr.ib(default=10_000)
    num_features = attr.ib(default=10)


@attr.s(cmp=False)
class State(object):
    """
    State for the resampling environment.
    """

    # features of the population
    rng = attr.ib()
    params = attr.ib()
    resource = attr.ib()
    delta = attr.ib(default=0)
    delta_delta = attr.ib(default=0)
    delta_real = attr.ib(default=0)

    # features of the applicant
    applicant_features = attr.ib(default=None)
    group = attr.ib(default=None)
    label = attr.ib(default=None)
    pred = attr.ib(default=None)
    idx = attr.ib(default=None)


class ResamplingEnv(core.FairnessEnv):
    """
    Resampling environment to facilitate calculating metrics along individuals.
    """

    metadata = {"render.modes": ["human"]}
    group_membership_var = "group"

    def __init__(self, params, utility_method="accuracy", delta_method="full"):
        assert utility_method in ["accuracy", "qualification", "tpr"]
        assert delta_method in ["full", "imputation", "accepted"]
        self.utility_method = utility_method
        self.delta_method = delta_method
        self.action_space = spaces.Discrete(2)
        self.predict_fn = lambda x: 0

        resource_space = spaces.Box(
            low=0.0,
            high=params.max_resource,
            dtype=np.float64,
            shape=(),
        )
        applicant_space = spaces.Box(
            low=0.0, high=1.0, dtype=np.float32, shape=(params.num_features,)
        )
        group_space = spaces.MultiBinary(n=params.num_groups)

        self.observable_state_vars = {
            "resource": resource_space,
            "applicant_features": applicant_space,
            "group": group_space,
        }

        self.init_utility_matrix = np.zeros(
            (params.num_applicants, params.num_groups), dtype=np.float32
        )
        self.init_active_matrix = np.zeros(
            (params.num_applicants, params.num_groups), dtype=np.float32
        )
        self.utility_matrix = np.zeros(
            (params.num_applicants, params.num_groups), dtype=np.float32
        )
        self.active_matrix = np.zeros(
            (params.num_applicants, params.num_groups), dtype=np.float32
        )
        self.init_utility_real_matrix = np.zeros(
            (params.num_applicants, params.num_groups), dtype=np.float32
        )
        self.init_active_real_matrix = np.zeros(
            (params.num_applicants, params.num_groups), dtype=np.float32
        )
        self.utility_real_matrix = np.zeros(
            (params.num_applicants, params.num_groups), dtype=np.float32
        )
        self.active_real_matrix = np.zeros(
            (params.num_applicants, params.num_groups), dtype=np.float32
        )
        self.init_pool = []
        self.pool = []
        self.load_pool(params)

        super(ResamplingEnv, self).__init__(params)
        self._state_init()

    def _state_init(self, rng=None):
        self.state = State(
            params=copy.deepcopy(self.initial_params),
            rng=rng or np.random.RandomState(),
            resource=self.initial_params.starting_resource,
        )
        self.sample_applicant()

    def reset(self):
        """Resets the environment."""
        self.timestep = 0
        self.pool = copy.deepcopy(self.init_pool)
        self.utility_matrix = copy.deepcopy(self.init_utility_matrix)
        self.active_matrix = copy.deepcopy(self.init_active_matrix)
        self.utility_real_matrix = copy.deepcopy(self.init_utility_real_matrix)
        self.active_real_matrix = copy.deepcopy(self.init_active_real_matrix)
        self._state_init(self.state.rng)
        self.compute_disparity()
        return super(ResamplingEnv, self).reset()

    def _is_done(self):
        return self.state.resource <= 0 or self.timestep >= 10_000

    def _step_impl(self, state, action):
        self.timestep += 1
        self.update_resource(self.state, action)
        self.update_utility(
            self.state.idx,
            self.state.label,
            self.state.pred,
            np.argmax(self.state.group),
            action,
            init=False,
        )
        self.compute_disparity()
        self.update_applicant(self.state, action)
        self.sample_applicant()
        return self.state

    def load_pool(self, params):
        """Load the pool of applicants."""
        # Implement logic to load the pool of applicants based on params
        pass

    def update_resource(self, state, action):
        params = state.params
        if action == 0:  # reject
            return
        if state.label == 1:
            state.resource += params.gain
        else:
            state.resource -= 1

        state.resource = max(0, state.resource)

    def update_applicant(self, state, action):
        # Implement logic to update the applicant based on the action taken
        pass

    def update_utility(self, idx, label, pred, group_idx, action, init=False):
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

        if init:
            self.init_utility_real_matrix[idx, group_idx] = utility_value
            self.init_active_real_matrix[idx, group_idx] = active
        else:
            self.utility_real_matrix[idx, group_idx] = utility_value
            self.active_real_matrix[idx, group_idx] = active

        if self.delta_method == "full":
            pass  # Full delta is already calculated in the utility_real_matrix
        elif self.delta_method == "imputation":
            label = label if action == 1 else pred
            if self.utility_method == "accuracy":
                utility_value = 1 if label == action else 0
                active = 1
            elif self.utility_method == "qualification":
                utility_value = label
                active = 1
            elif self.utility_method == "tpr":
                utility_value = action
                active = 1 if label == 1 else 0
        elif self.delta_method == "accepted":
            label = label
            if self.utility_method == "accuracy":
                utility_value = 1 if label == action else 0
                active = 1 if action else 0
            elif self.utility_method == "qualification":
                utility_value = label
                active = 1 if action else 0
            elif self.utility_method == "tpr":
                utility_value = action
                active = 1 if action * label == 1 else 0

        if init:
            self.init_utility_matrix[idx, group_idx] = utility_value
            self.init_active_matrix[idx, group_idx] = active
        else:
            self.utility_matrix[idx, group_idx] = utility_value
            self.active_matrix[idx, group_idx] = active

    def compute_disparity(self):
        # Multiplity utility by active matrix
        self.utility_matrix *= self.active_matrix
        self.utility_real_matrix *= self.active_real_matrix

        # First, calculate real utility
        cur_util = np.sum(self.utility_real_matrix, axis=0)
        group_counts = np.sum(self.active_real_matrix, axis=0)
        cur_util = np.divide(
            cur_util,
            group_counts,
            out=np.zeros_like(cur_util),
            where=group_counts != 0,
        )
        self.state.delta_real = np.max(cur_util) - np.min(cur_util)

        # Then, calculate utility
        cur_util = np.sum(self.utility_matrix, axis=0)
        group_counts = np.sum(self.active_matrix, axis=0)
        cur_delta = self.state.delta
        cur_util = np.divide(
            cur_util,
            group_counts,
            out=np.zeros_like(cur_util),
            where=group_counts != 0,
        )
        self.state.delta = np.max(cur_util) - np.min(cur_util)
        self.state.delta_delta = self.state.delta - cur_delta

    def sample_applicant(self):
        selected = np.random.choice(len(self.pool), size=1)[0]
        applicant = self.pool[selected]
        self.state.applicant_features = applicant["features"]
        self.state.group = applicant["group"]
        self.state.label = applicant["label"]
        self.state.pred = applicant["pred"]
        self.state.idx = selected
        return

    def set_action_pred(self, list_action, list_pred):
        self.init_active_matrix *= 0
        self.init_utility_matrix *= 0
        self.init_active_real_matrix *= 0
        self.init_utility_real_matrix *= 0
        for idx, (action, pred) in enumerate(zip(list_action, list_pred)):
            label = self.pool[idx]["label"]
            group_idx = self.pool[idx]["group"].argmax()
            self.pool[idx]["pred"] = pred
            self.update_utility(idx, label, pred, group_idx, action, init=True)

        self.utility_matrix = copy.deepcopy(self.init_utility_matrix)
        self.active_matrix = copy.deepcopy(self.init_active_matrix)
        self.utility_real_matrix = copy.deepcopy(self.init_utility_real_matrix)
        self.active_real_matrix = copy.deepcopy(self.init_active_real_matrix)

        self.compute_disparity()


class LendingEnv(ResamplingEnv):
    """
    Environment for lending experiments.
    """

    def load_pool(self, params):
        with open("data/fico.pkl", "rb") as f:
            data = pkl.load(f)
        groups_probs = data["group_likelihoods"]
        cluster_probs = data["cluster_probabilities"]
        success_probs = data["success_probabilities"]

        def sample_label(g, x):
            return np.random.binomial(n=1, p=success_probs[g][x])

        self.label_fn = sample_label

        num_groups = len(groups_probs)
        num_features = len(cluster_probs[0])

        for i in range(params.num_applicants):
            g = np.random.choice(num_groups, p=groups_probs)
            x = np.random.choice(num_features, p=cluster_probs[g])
            y = self.label_fn(g, x)
            # one hot encode group and x
            features = np.zeros(num_features, dtype=np.float32)
            features[x] = 1
            group = np.zeros(num_groups, dtype=np.float32)
            group[g] = 1

            self.init_pool.append(
                {
                    "features": features,
                    "group": group,
                    "label": y,
                    "pred": None,
                }
            )
        
        self.pool = copy.deepcopy(self.init_pool)

    def update_applicant(self, state, action):
        if action == 0:
            return

        idx = state.idx
        score = np.argmax(state.applicant_features)
        num_features = len(state.applicant_features)
        if state.label == 1:
            new_score = min(score + 1, num_features - 1)
        elif state.label == 0:
            new_score = max(score - 1, 0)

        self.pool[idx]["features"][score] = 0
        self.pool[idx]["features"][new_score] = 1
        state.applicant_features = self.pool[idx]["features"]

        group = np.argmax(state.group)
        y = self.label_fn(group, new_score)
        self.pool[idx]["label"] = y
        state.label = y

        pred = self.predict_fn(self.pool[idx])
        self.pool[idx]["pred"] = pred
        state.pred = pred

class EnemEnv(ResamplingEnv):
    """
    Environment for school admission experiments.
    """

    def load_pool(self, params):
        with open("data/enem_pool.pkl", "rb") as f:
            self.init_pool = pkl.load(f)

        with open("data/enem_model.pkl", "rb") as f:
            self.model = pkl.load(f)

        def sample_label(x):
            p = self.model.predict_proba(x.reshape(1, -1))[0, 1]
            return 1 if np.random.rand() < p else 0

        self.label_fn = sample_label
        self.pool = copy.deepcopy(self.init_pool)

    def updated_applicant(self, state, action):
        if action == 1:
            return
        age_groups = 6
        idx = state.idx
        features = state.applicant_features
        group = np.argmax(state.group)
        age_features = features[(age_groups * group):(age_groups * (group + 1))]
        age = np.argmax(age_features)
        new_age = min(age + 1, age_groups - 1)

        new_features = features.copy()
        new_features[int(age_groups * group + age)] = 0
        new_features[int(age_groups * group + new_age)] = 1

        self.pool[idx]["features"] = new_features
        state.applicant_features = new_features

        label = self.label_fn(group, new_age)
        self.pool[idx]["label"] = label
        state.label = label

        pred = self.predict_fn(self.pool[idx])
        self.pool[idx]["pred"] = pred
        state.pred = pred