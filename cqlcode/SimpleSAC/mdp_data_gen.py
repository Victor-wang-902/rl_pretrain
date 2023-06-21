import numpy.random
import torch
from torch import Tensor
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
import joblib

def softmax_with_torch(x, temperature):
    return F.softmax(Tensor(x/temperature), dim=0).numpy()

def gen_mdp_data(n_traj, max_length, n_state, n_action, policy_temperature, transition_temperature):
    n_data = n_traj * max_length
    states = np.zeros(n_data, dtype=int)
    actions = np.zeros(n_data, dtype=int)
    next_states = np.zeros(n_data, dtype=int)
    i = 0

    for i_traj in tqdm(range(n_traj)):
        np.random.seed(n_traj)
        state = np.random.randint(n_state)
        for t in range(max_length):
            states[i] = state
            # for each step, an action is taken, and a next state is decided.
            # if we assume a fixed policy is generating the data, then the action probability only depends on the state
            np.random.seed(state)
            action_probs = softmax_with_torch(np.random.rand(n_action), policy_temperature)

            np.random.seed(42 + i_traj * 1000 + t * 333)
            action = np.random.choice(n_action, p=action_probs)

            np.random.seed(state * 888 + action * 777)
            next_state_probs = softmax_with_torch(np.random.rand(n_state), transition_temperature)

            np.random.seed(666 + i_traj * 1000 + t * 333)
            next_state = np.random.choice(n_state, p=next_state_probs)
            next_states[i] = next_state
            actions[i]=action
            state = next_state
            i += 1
    return {'observations': states,
            'actions': actions,
            'next_observations': next_states}

# generate 1M data, each trajectory has 1000 steps
n_traj, max_length, n_state, n_action = 1000, 1000, 1000, 1000
policy_temperature = 1 # higher temperature -> uniform random actions, close to 0 temperature -> deterministic action
transition_temperature = 1 # higher -> uniform random transition, close to 0 -> deterministic transition


data = gen_mdp_data(n_traj, max_length, n_state, n_action, policy_temperature, transition_temperature)
joblib.dump(data, '../../mdpdata/random.pkl')

# the variations:
    # a. whether action is random, or from a fixed policy
    # b. whether next state is deterministically decided by s-a, or come from a distribution that is decided by s-a.
    # that gives following possibilities:
    # and then we can also use temperature to control the prob distribution of action and next_state.
    # 1) rand_action - deterministic_next_state (transition)
    # 2) rand_action - stochastic_next_state
    # 3) action depends on state - deterministic next state
    # 4) action depends on state - stochastic next state

    # so I guess we can have an action temperature, and a transition temperature. At an extreme we have totally random.
    # at the other extreme we have deterministic action (depends on state), and deterministic transition (depends on s-a)
    # we can also control the size of the state and action sets.

# and then we have to decide, how do we convert these integers to float values:
    # 1) map each into a learnable embedding
    # 2) map each into a fixed embedding
    #    here we can try, use a random vector, or use a random network to map into a vector
    #    potentially we can also only allow the state to change a bit, so they cannot massively change for each transition...










