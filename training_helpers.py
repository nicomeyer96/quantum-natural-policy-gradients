import numpy as np


class Parameters:
    """
    Class for handling parameter initialization and updates
    """

    def __init__(self,
                 num_params: int,
                 seed: int | None = None,
                 lr: float = 0.1,
                 initializer: float = 1.1):
        """
        Set up variational parameter handler

        :param num_params: should correspond to number of variational parameters in circuit
        :param seed: fix seed for parameter initialization
        :param lr: learning rate for updating parameters (via gradient ascent)
        :param initializer: helps to speed up convergence by initializing in neighborhood of optimal parameters
        """

        # set random seed
        np.random.seed(seed)

        if initializer > 1:
            # parameters are initialized uniform at random in entire parameter space
            self.params = np.random.uniform(0, 2 * np.pi, size=num_params)
        else:
            # Initialize in N(mu=0, sigma=initializer)-neighborhood of optimal parameters
            # (obviously only works for the pre-defined VQCs and environment)
            layer_12 = np.tile([np.pi/2, -np.pi/2], num_params // 6)
            layer_3 = np.tile([-np.pi/2, 0], num_params // 6)
            self.params = np.concatenate((layer_12, layer_12, layer_3)) + initializer * np.random.randn(num_params)

        # unset random seed
        np.random.seed(None)

        self.lr = lr

    def get_params(self) -> np.ndarray:
        """
        :return: current parameters
        """

        return self.params

    def update_params(self, update: np.ndarray):
        """
        :param update: value to update with
        :return: gradient-ascent update
        """

        self.params += self.lr * update

    def set_params(self, params: np.ndarray):
        """
        :param params: hand-set parameters
        """
        self.params = params

    def set_lr(self, lr: float):
        """
        :param lr: re-set learning rate
        """
        self.lr = lr


class ProgressTracker:
    """
    Class for tracking training progress (rewards and expected rewards)
    """

    def __init__(self):
        self.rewards = []
        self.expected_rewards = []
        self.batch_rewards = []
        self.batch_expected_rewards = []
        self.validation_rewards = []
        self.validation_expected_rewards = []

    def add_reward(self, reward: float):
        self.rewards.append(reward)

    def add_expected_reward(self, expected_reward: float):
        self.expected_rewards.append(expected_reward)

    def add_batch_reward(self, batch_reward):
        self.batch_rewards.extend(batch_reward)

    def add_batch_expected_reward(self, batch_expected_reward):
        self.batch_expected_rewards.extend(batch_expected_reward)

    def add_validation_reward(self, validation_reward):
        self.validation_rewards.append(validation_reward)

    def add_validation_expected_reward(self, validation_expected_reward):
        self.validation_expected_rewards.append(validation_expected_reward)

    def convert_to_dict(self) -> dict:
        """
        Convert training trajectory to dictionary

        :return: collected training results
        """

        history = {
            'history_reward': np.array(self.rewards),
            'history_expected_reward': np.array(self.expected_rewards),
            'history_batch_reward': np.array(self.batch_rewards),
            'history_batch_expected_reward': np.array(self.batch_expected_rewards),
            'history_validation_reward': np.array(self.validation_rewards),
            'history_validation_expected_reward': np.array(self.validation_expected_rewards)
        }

        return history
