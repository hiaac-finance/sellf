import attr
import copy
import numpy as np

from gym import spaces

from environments import core

@attr.s
class UtilFunc():
    def __call__(self, label, action):
        """
        Calculate the utility based on the label and action.
        """
        pass

class QualiUtil(UtilFunc):
    def __call__(self, label, action):
        return label
    
class AccUtil(UtilFunc):
    def __call__(self, label, action):
        return 1 if label == action else 0

# TODO: implement true positive rate utility

@attr.s
class Params(core.Params):
    num_groups = attr.ib(default=2)
    cost = attr.ib(default=0.5)
    starting_resource = attr.ib(default=1_000.0)
    max_resource = attr.ib(default=100_000.0)
    num_applicants = attr.ib(default=10_000)
    num_features = attr.ib(default=7)


@attr.s(cmp=False)
class State(core.State):
    """
    State for the resampling environment.
    """

    # features of the population
    rng = attr.ib()
    params = attr.ib()
    resource = attr.ib()
    utility = attr.ib(default=None)
    utility_obs = attr.ib(default=None)

    # features of the applicant
    applicant_features = attr.ib(default=None)
    group = attr.ib(default=None)
    label = attr.ib(default=None)
    prediction = attr.ib(default = None)
    idx = attr.ib(default=None)


class ResamplingEnv(core.FairnessEnv):
    """
    Resampling environment to facilitate calculating metrics along individuals.
    """

    metadata = {"render.modes": ["human"]}
    group_membership_var = "group"

    def __init__(self, params):
        self.action_space = spaces.Discrete(2)

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
        self._state_init(self.state.rng)
        return super(ResamplingEnv, self).reset()

    def _is_done(self):
        return self.state.resource <= 0

    def _step_impl(self, state, action):
        self.update_resource(self.state, action)
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
        state.resource += state.label - params.cost

    def update_applicant(self, state, action):
        pass

    def sample_applicant(self):
        selected = np.random.choice(len(self.pool), size=1)[0]
        applicant = self.pool[selected]
        self.state.applicant_features = applicant["features"]
        self.state.group = applicant["group"]
        self.state.label = applicant["label"]
        self.state.idx = selected
        return
    
    def apply_models(self, policy, predictor):
        """
        Apply the policy to get the action and predicted label.
        """
        for i, applicant in enumerate(self.pool):
            action = ...
            prediction = ...

            self.state.utility += self.util_fn(self.state.label, action)
            self.state.utility_obs += self.util_fn(prediction, action)

            # TODO: finish this function

class LendingEnv(ResamplingEnv):
    """
    Environment for lending experiments.
    """

    def __init__(self, params):
        super(LendingEnv, self).__init__(params)

    def load_pool(self, params):
        groups_probs = [0.5, 0.5]
        cluster_probs = [
            [0.0, 0.1, 0.1, 0.2, 0.3, 0.3, 0.0],
            [0.1, 0.1, 0.2, 0.3, 0.3, 0.0, 0.0],
        ]
        success_probs = [
            [0.1, 0.2, 0.45, 0.6, 0.65, 0.7, 0.7],
            [0.1, 0.2, 0.45, 0.6, 0.65, 0.7, 0.7],
        ]
        self.pool = []

        num_groups = len(groups_probs)
        num_features = len(cluster_probs[0])

        for i in range(params.num_applicants):
            g = np.random.choice(num_groups, p=groups_probs)
            x = np.random.choice(num_features, p=cluster_probs[g])
            y = np.random.binomial(n=1, p=success_probs[g][x])
            y = int(y)
            # one hot encode group and x
            features = np.zeros(num_features, dtype=np.float32)
            features[x] = 1
            group = np.zeros(num_groups, dtype=np.float32)
            group[g] = 1

            self.pool.append(
                {
                    "features": features,
                    "group": group,
                    "label": y,
                }
            )

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
