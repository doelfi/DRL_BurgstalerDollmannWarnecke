import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import gymnasium as gym
import matplotlib as plt


class ExperienceReplayBuffer():
    def __init__(self, max_size : int, environment_name : str, parallel_game_unrolls : int, observation_preprocessing_function : callable, unroll_steps : int):
        self.max_size = max_size
        self.environment_name = environment_name
        # Number of parallel env the agents plays
        self.parallel_game_unrolls = parallel_game_unrolls
        self.observation_preprocesssing_function = observation_preprocessing_function
        self.unroll_steps = unroll_steps
        self.envs = gym.vector.make(environment_name, self.parallel_game_unrolls)
        self.num_possible_actions = self.envs.single_action_space.n
        self.current_states, _ = self.envs.reset()
        self.data = []

    def fill_with_samples(self, dqn_network, epsilon : float):
        # Adds new samples into the ERP
        states_list = []
        actions_list = []
        rewards_list = []
        subsequent_states_list = []
        terminateds_list = []
        
        for i in range(self.unroll_steps):
            actions = self.sample_epsilon_greedy(dqn_network, epsilon)
            next_states, rewards, terminateds, _, _ = self.envs.step(actions)
            # Put observation, action, reward, next observation into ERP
            
            states_list.append(self.current_states)
            actions_list.append(actions)
            rewards_list.append(rewards)
            subsequent_states_list.append(next_states)
            terminateds_list.append(terminateds)

            self.current_states = next_states

        def data_generator():
            for states_batch, actions_batch, rewards_batch, subsequent_states_batch, terminateds_batch in zip(states_list, actions_list, rewards_list,subsequent_states_list, terminateds_list):
                for game_idx in range(self.parallel_game_unrolls):
                    state = states_batch[game_idx, :, :, :]
                    action = actions_batch[game_idx]
                    reward = rewards_batch[game_idx]
                    subsequent_state = subsequent_states_batch[game_idx, :, :, :]
                    terminated = terminateds_batch[game_idx]
                    yield (state, action, reward, subsequent_state, terminated)

        dataset_tensor_specs = (tf.TensorSpec(shape=(210,160,3), dtype=tf.uint8), 
                                tf.TensorSpec(shape=(), dtype=tf.int32), 
                                tf.TensorSpec(shape=(), dtype=tf.float32), 
                                tf.TensorSpec(shape=(210,160,3), dtype=tf.uint8), 
                                tf.TensorSpec(shape=(), dtype=tf.bool))
        new_samples_dataset = tf.data.Dataset.from_generator(data_generator, output_signature=dataset_tensor_specs) #TODO

        new_samples_dataset = new_samples_dataset.map(lambda state, action, reward, subsequent_state, terminated : 
                                                      (self.observation_preprocesssing_function(state), action, reward, 
                                                       self.observation_preprocesssing_function(subsequent_state), terminated))
        new_samples_dataset = new_samples_dataset.cache().shuffle(buffer_size=self.unroll_steps*self.parallel_game_unrolls, reshuffle_each_iteration=True).batch(1)
        # Make sure cache is applied
        for elem in new_samples_dataset:
            continue

        self.data.append(new_samples_dataset)
        datapoints_in_data = len(self.data)*self.unroll_steps*self.parallel_game_unrolls
        if datapoints_in_data > self.max_size:
            self.data.pop(0)


    def create_dataset(self):
        # Creates a tf.data.Dataset from the ERP 
        erp_dataset = tf.data.Dataset.sample_from_datasets(self.data, weights=[1/float(len(self.data)) for _ in self.data], stop_on_empty_dataset=False)
        return erp_dataset

    def sample_epsilon_greedy(self, dqn_network, epsilon : float):
        # Samples actions from DQN
        observations = self.observation_preprocesssing_function(self.current_states)
        q_values = dqn_network(observations) # Tensor of type tf.float32, shape(parallel_game_unrolls, num_actions)
        greedy_actions = tf.argmax(q_values, axis=1) # Tensor of type tf.int64, shape(parallel_game_unrolls,)
        random_actions = tf.random.uniform(shape=(self.parallel_game_unrolls,), 
                                           minval=0, 
                                           maxval=self.num_possible_actions, 
                                           dtype=tf.int64) # Tensor of type tf.int64, shape(parrallel,game_unrolls,)
        epsilon_sampling =tf.random.uniform(shape=(self.parallel_game_unrolls,),
                                            minval=0,
                                            maxval=1,
                                            dtype=tf.float32) > epsilon
        actions = tf.where(epsilon_sampling, greedy_actions, random_actions).numpy()
        return actions
    
def observation_preprocessing_function(observation):
    # Converts an observation to shape 84x84
    observation = tf.image.resize(observation, size=(84,84))
    observation = tf.cast(observation, dtype=tf.float32) / 128.0 - 1.0
    return observation

def create_dqn_network(num_actions : int):
    # Create input for functional tf model api
    input_layer = tf.keras.Input(shape=(84, 84, 3), dtype=tf.float32)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same')(input_layer)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same')(x) + x     # Residual connection
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same')(x) + x     # Residual connection
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(x) + x     # Residual connection
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(x) + x     # Residual connection
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(x) + x     # Residual connection
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(x) + x     # Residual connection
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(x) + x     # Residual connection
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(x) + x     # Residual connection
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(x) + x     # Residual connection
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dense(units=64, activation='relu')(x) + x
    x = tf.keras.layers.Dense(units=num_actions, activation='linear')(x)
    model = tf.keras.Model(inputs=input_layer, outputs=x)
    return model

def train_dqn(train_dqn_network, target_network, dataset, optimizer, gamma : float, num_training_steps : int, use_double=False):
    def training_step(q_target, observations, actions):
        with tf.GradientTape() as tape:
            q_predictions_all_actions = train_dqn_network(observations) # shape = (batch_size, num_actions)
            q_predictions = tf.gather(q_predictions_all_actions, actions, batch_dims=1)
            loss = tf.reduce_mean(tf.square(q_predictions - q_target))
        gradients = tape.gradient(loss, train_dqn_network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, train_dqn_network.trainable_variables))
        return loss.numpy()
    
    losses = []
    q_values = []
    for i, state_transition in enumerate(dataset):
        # Train on data
        state, action, reward, subsequent_state, terminated = state_transition
        # Calculate q-targets
        # if use_double is false, a normal DQN approach with delayed target is used
        # if true, a Double DQN approach is used
        if use_double:
            q_val_source = train_dqn_network(subsequent_state)
            q_val_target = target_network(subsequent_state)
            # get the indices of the highest value from the source network
            max_q_value_indices = tf.argmax(q_val_source, axis=1)
            # get the value from the target_network using the indices from the source network
            max_q_values = tf.gather(q_val_target,max_q_value_indices,axis=1) # untested, I hope the function does what I think it does :)
        else:
            q_val = target_network(subsequent_state)
            q_values.append(q_val)
            max_q_values = tf.reduce_max(q_val, axis=1)
        use_subsequent_state = tf.where(terminated, tf.zeros_like(max_q_values, dtype=tf.float32), tf.ones_like(max_q_values, dtype=tf.float32))
        q_target = reward + (gamma*max_q_values*use_subsequent_state)
        loss = training_step(q_target, state, action)
        losses.append(loss)
        if i >= num_training_steps:
            break
    return np.mean(losses), np.mean(q_values)

def test_q_network(test_dqn_network, environment_name : str, num_parallel_tests : int, gamma : float):
    envs = gym.vector.make(environment_name, num_parallel_tests)
    states, _ = envs.reset()
    states = observation_preprocessing_function(states)
    returns = np.zeros(num_parallel_tests)
    episodes_finished = np.zeros(num_parallel_tests, dtype=bool)

    timestep = 0
    done = False
    while not done: 
        q_values = test_dqn_network(states)
        actions = tf.argmax(q_values, axis=1)
        states, rewards, terminateds, _, _ = envs.step(actions)
        episodes_finished = np.logical_or(episodes_finished, terminateds)
        returns += ((gamma**timestep)*rewards)*(np.logical_not(episodes_finished).astype(np.float32))
        timestep += 1
        done = np.all(episodes_finished)
    return np.mean(returns)

def polyak_averaging_weights(source_network, target_network, polyak_averaging_factor: float):
    source_network_weights = source_network.get_weights()
    target_network_weights = target_network.get_weights()
    averaged_weights = []
    for source_network_weight, target_network_weight in zip(source_network_weights, target_network_weights):
        fraction_kept_weights = polyak_averaging_factor * target_network_weight
        fraction_updated_weights = (1 - polyak_averaging_factor) * source_network_weight
        average_weight = fraction_kept_weights + fraction_updated_weights
        averaged_weights.append(average_weight)
    target_network.set_weights(averaged_weights)

def visualize_results(results_df, step):
    # create three subplots
    fig, axis = plt.subplots(3, 1)
    # include the row idxs explicitly in the results_df
    results_df['step'] = results_df.index
    # plot the average return
    sns.lineplot(x='step', y='average_return', data=results_df, ax=axis[0])
    # plot the average loss
    sns.lineplot(x='step', y='average_loss', data=results_df, ax=axis[1])
    # plot the average q values
    sns.lineplot(x='step', y='average_q_values', data=results_df, ax=axis[2])
    # create a timestring from the timestamp
    timestring = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # save the figure
    plt.savefig(f'./results/{timestring}_results:step{step}.png')
    # close the figure
    plt.close(fig)


def dqn():
    ENVIRONMENT_NAME = 'ALE/Breakout-v5'
    NUM_ACTIONS = gym.make(ENVIRONMENT_NAME).action_space.n
    ERP_SIZE = 100_000
    PARALLEL_GAME_UNROLS = 64
    UNROLL_STEPS = 4
    EPSILON = 0.2
    GAMMA = 0.98
    NUM_TRAINING_STEPS = 4
    NUM_TRAINING_ITER = 5000
    TEST_EVERY_N_STEPS = 1000
    TEST_NUM_PARALLEL_ENVS = 32
    PREFILL_STEPS = int(40_000 / (PARALLEL_GAME_UNROLS * UNROLL_STEPS)) # so that we can change the values and still get enough prefill
    POLYAK_AVERAGING_FACTOR = 0.99

    erp = ExperienceReplayBuffer(max_size=ERP_SIZE, 
                                 environment_name=ENVIRONMENT_NAME, 
                                 parallel_game_unrolls=PARALLEL_GAME_UNROLS, 
                                 observation_preprocessing_function=observation_preprocessing_function, 
                                 unroll_steps=UNROLL_STEPS)
    # This is the DQN we train
    dqn_agent = create_dqn_network(num_actions=NUM_ACTIONS)
    dqn_optimizer = tf.keras.optimizers.Adam()
    # Test the agent
    dqn_agent(tf.random.uniform(shape=(1,84,84,3)))
    # This is the DQN we use to calculate the q_estimation targets
    target_network = create_dqn_network(num_actions=NUM_ACTIONS)
    # copy over the weights from the dqn_agent to the target_network via polyak averaging with factor 0.0
    polyak_averaging_weights(source_network=dqn_agent, target_network=target_network, polyak_averaging_factor=0.0)

    return_tracker = []
    dqn_prediction_error = []
    average_q_values = []

    # prefill the replay buffer
    prefill_exploration_epsilon = 1.0
    for prefill_step in range(PREFILL_STEPS):
        erp.fill_with_samples(dqn_agent, prefill_exploration_epsilon)
    
    for step in range(NUM_TRAINING_ITER):
        # Step 1: Put s, a, r, s', t transitions into replay buffer
        erp.fill_with_samples(dqn_agent, EPSILON)
        dataset = erp.create_dataset()
        # Step 2: Train some samples from the replay buffer
        average_loss, average_q_vals = train_dqn(train_dqn_network=dqn_agent,
                                                 target_network=target_network,
                                                 dataset=dataset, 
                                                 optimizer=dqn_optimizer, 
                                                 gamma=GAMMA, 
                                                 num_training_steps=NUM_TRAINING_STEPS,
                                                 use_double=True)

        # update the target network via polyak averaging
        polyak_averaging_weights(source_network=dqn_agent, target_network=target_network, polyak_averaging_factor=POLYAK_AVERAGING_FACTOR)

        # Test the agent
        if step % TEST_EVERY_N_STEPS == 0:
            average_return = test_q_network(dqn_agent, ENVIRONMENT_NAME, TEST_NUM_PARALLEL_ENVS, GAMMA)
            return_tracker.append(average_return)
            dqn_prediction_error.append(average_loss)
            average_q_values.append(average_q_vals)
            print(f'TESTING: Average return: {average_return}, average loss: {average_loss}, average q-value-estimation: {average_q_values}')
            # put all the result lists into a dataframe by transforming them into a dict first
            results_dict = {'average_return': return_tracker, 'average_loss': dqn_prediction_error, 'average_q_values': average_q_values}
            results_df = pd.DataFrame(results_dict)
            # visualize the results with sns
            visualize_results(results_df, step)
            print(results_df)

    print(results_dict)
    # Visualize
    #sns.lineplot(data=results_df, x=results_df.index, y='average_return')

if __name__ == "__main__":
    print(1)
    dqn()
    
# Sachen, die ich zu den Videos geändert habe: 
# 1. padding='same' hinzugefügt, weil dim-error
# 2. .batch(1) in fill_with_samples, weil dim-error
# 3. loss = training_step(q_target, state, action), weil Argument gefehlt hat
# 4. states = observation_preprocessing_function(states) in test_q_network(), weil dim-error
# Changed hyperparameter:NUM_TRAINING_ITER, TEST_NUM_PARALLEL_ENVS
#%%

# Sachen die ich im Vergleich zu den Videos geändert habe:
# 1. PREFILL_STEPS wird so berechnet, dass wir immer etwa 40_000 values im ERP haben, selbst wenn wir die Anzahl der erps ändern

# Sachen die ich hinzugefügt habe
# 1. Double DQN approach. Ich hab das über einen boolean gelöst, d.h. wenn man den in der Trainingsmethode auf false setzt,
# dann behält man die Version aus den Videos bei.
# Ich hoffe, dass tf.gather so funktioniert wie ich mir das vorstelle
