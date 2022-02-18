import ray.rllib
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel

tf1, tf, tfv = ray.rllib.utils.framework.try_import_tf()


class Connect4Model(DistributionalQTFModel):
    """
    The input network for the DQN, adapted to the Connect4 game
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        """Initialize variables of this model"""
        super(Connect4Model, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        self.base_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=obs_space.shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(84, activation='relu'),
            tf.keras.layers.Dense(84, activation='relu'),
            tf.keras.layers.Dense(num_outputs, activation='relu')
        ])

    def forward(self, input_dict, state, seq_lens):
        """Call the model with the given input tensors and state."""
        model_out = self.base_model(input_dict["obs"])
        return model_out, state
