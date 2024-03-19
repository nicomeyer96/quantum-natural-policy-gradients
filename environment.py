import numpy as np


class ContextualBandits:
    """
    Contextual Bandits environment class
    """

    def __init__(self, n_arms: int = 2, n_bandits: int = 1024):
        """
        Set up contextual bandits environment

        :param n_arms: number of arm, i.e. actions, for each bandit
        :param n_bandits: number of bandits, i.e. states, in the environment
        """

        if not 2 == n_arms:
            raise NotImplementedError('Environment is currently only implemented for two actions!')

        self.n_arms = n_arms
        self.n_bandits = n_bandits

        # expression returns 0, iff state/string has even parity, and +1 otherwise
        self.env = lambda x: bin(x)[2:].count('1') % 2

        self.action_space = np.arange(n_arms)
        self.state = None

    def _get_reward(self, action: int) -> int:
        """
        Get (ground-truth) reward for executed action and current state, uncertainty is applied later

        :param action: action to execute
        :return: reward (ground-truth)
        """

        opt_action = self.env(self.state)

        if opt_action == action:
            return +1
        else:
            return -1

    def _get_expected_reward(self, policy: np.ndarray) -> float:
        """
        Get (ground-truth) expected reward for current state

        :param policy: current policy
        :return: expected reward
        """

        opt_action = self.env(self.state)

        return 2 * policy[opt_action] - 1

    def step(self, action: int, policy: np.ndarray = None) -> (None, float, bool, float):
        """
        Execute one step in the environment

        :param action: action to execute
        :param policy: complete policy for computing expected rewards (optional)
        :return: Next state (None in this environment), reward, done (always True after one step),
        expected reward (just for tracking performance!)
        """

        if action not in self.action_space:
            raise RuntimeError('Action {} does not exist in the environment!'.format(action))

        # sample reward and apply uncertainty with N(mu=0,sigma=1)
        reward = self._get_reward(action) + np.random.randn()

        # if full policy is provided, determine expected reward
        expected_reward = None
        if policy is not None:
            expected_reward = self._get_expected_reward(policy)

        return None, reward, True, expected_reward

    def reset(self) -> int:
        """
        Call to initialize environment first time and after done=True (i.e. after one step in this instance)

        :return: stated sampled uniform at random
        """

        self.state = np.random.randint(self.n_bandits)
        return self.state

    def set_state(self, state: int):
        """
        Can be used as a workaround for allowing more efficient batch-processing
        (important for efficiency when using quantum backend)

        :param state: state to set the environment to (in decimal representation)
        """

        self.state = state


def state_prep(raw_state: int, n_qubits: int) -> list[int]:
    """
    Convert decimal state to binary representation of correct length (2**n_qubits), which can be encoded into the VQC

    :param raw_state: state in decimal representation
    :param n_qubits: number of qubits in system (required for determining number of leading zeros)
    :return: binary representation of state
    """

    bin_state = bin(raw_state)[2:].zfill(n_qubits)
    state = []
    for b in bin_state:
        state.append(int(b))
    return state
