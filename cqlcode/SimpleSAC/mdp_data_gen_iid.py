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

    if policy_temperature == transition_temperature == 'inf2':
        np.random.seed(0)
        origin = np.random.randint(n_state)

        np.random.seed(13)
        states = np.random.randint(n_state, size=n_data, dtype=int)
        next_states = states.copy()
        next_states = np.concatenate((next_states[1:], np.random.randint(n_state, size=1)), dtype=int)
        states[[i*1000 for i in range(n_traj)]] = origin

        np.random.seed(47)
        actions = np.random.randint(n_action, size=n_data)
    elif policy_temperature == transition_temperature == 'fix_sprime':
        np.random.seed(0)
        origin = np.random.randint(n_state)

        np.random.seed(13)
        states = np.random.randint(n_state, size=n_data, dtype=int)
        states[[i*1000 for i in range(n_traj)]] = origin

        np.random.seed(28)
        next_states = np.full(n_data, np.random.randint(n_state), dtype=int)

        np.random.seed(47)
        actions = np.random.randint(n_action, size=n_data)
    elif policy_temperature == transition_temperature == 'mean_sprime':
        np.random.seed(0)
        origin = np.random.randint(n_state)

        np.random.seed(13)
        states = np.random.randint(n_state, size=n_data, dtype=int)
        next_states = states.copy()
        next_states = np.concatenate((next_states[1:], np.random.randint(n_state, size=1)), dtype=int)
        states[[i*1000 for i in range(n_traj)]] = origin

        np.random.seed(47)
        actions = np.random.randint(n_action, size=n_data)
    elif str(policy_temperature).startswith('sigma') and str(policy_temperature)[-1] == 'S':
        np.random.seed(0)
        origin = np.random.randint(n_state)

        np.random.seed(13)
        states = np.random.randint(n_state, size=n_data, dtype=int)
        next_states = states.copy()
        next_states = np.concatenate((next_states[1:], np.random.randint(n_state, size=1)), dtype=int)
        states[[i * 1000 for i in range(n_traj)]] = origin

        np.random.seed(47)
        actions = np.random.randint(n_action, size=n_data)
    elif str(policy_temperature).startswith('sigma') and str(policy_temperature)[-1] == 'N':
        np.random.seed(13)
        states = np.concatenate(
            [np.random.randint(n_state//5*i, n_state//5*(i+1), n_data//5) for i in range(5)], axis=None)
        next_states = np.concatenate((states[1:], np.random.randint(n_state, size=1)), dtype=int)
        for i in range(5):
            next_states[(i+1) * n_data//5 - 1] = np.random.randint(n_state//5*i, n_state//5*(i+1), size=1)

        np.random.seed(47)
        actions = np.concatenate(
            [np.random.randint(n_action // 5 * i, n_state // 5 * (i + 1), n_data // 5) for i in range(5)], axis=None)
    else:
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
                if policy_temperature == 'inf':
                    action_probs = None
                else:
                    np.random.seed(state)
                    action_probs = softmax_with_torch(np.random.rand(n_action), policy_temperature)

                np.random.seed(42 + i_traj * 1000 + t * 333)
                action = np.random.choice(n_action, p=action_probs)

                if transition_temperature == 'inf':
                    next_state_probs = None
                else:
                    np.random.seed(state * 888 + action * 777)
                    next_state_probs = softmax_with_torch(np.random.rand(n_state), transition_temperature)

                np.random.seed(666 + i_traj * 1000 + t * 333)
                next_state = np.random.choice(n_state, p=next_state_probs)
                next_states[i] = next_state
                actions[i]=action
                state = next_state
                i += 1
    data_dict = {'observations': states,
            'actions': actions,
            'next_observations': next_states}
    data_name = 'mdp_traj%d_ns%d_na%d_pt%s_tt%s.pkl' % (n_traj, n_state, n_action,
                                                        str(policy_temperature), str(transition_temperature))
    save_name = '../mdpdata/%s' % data_name
    joblib.dump(data_dict, save_name)
    print("Data saved to:", save_name)

# generate 1M data, each trajectory has 1000 steps
n_traj, max_length = 1000, 1000
# policy_temperature = 1 # higher temperature -> uniform random actions, close to 0 temperature -> deterministic action
# transition_temperature = 1 # higher -> uniform random transition, close to 0 -> deterministic transition

# 1. different state-action space size
# for n_state_action in [1, 10, 100, 1000, 10000, 50257]:
#     gen_mdp_data(n_traj, max_length, n_state_action, n_state_action, policy_temperature, transition_temperature)

# 2. different temperatures
# for temperature in [0.01, 0.1, 1, 10, 100]:
#     gen_mdp_data(n_traj, max_length, 1000, 1000, temperature, temperature)

# n_state = 100
# n_action = 100
# # new ones with extreme temperature values
# for temperature in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
#     gen_mdp_data(n_traj, max_length, n_state, n_action, temperature, temperature)

for n_state in [1, 10, 1000, 10000, 100000]:
    temperature = 1
    n_action = n_state
    gen_mdp_data(n_traj, max_length, n_state, n_action, temperature, temperature)


# for n_state in [100]:
#     for temperature in [10]:
#         n_action = n_state
#         gen_mdp_data(n_traj, max_length, n_state, n_action, temperature, temperature)

# for n_state in [100]:
#     for temperature in ['sigma0.01S', 'sigma0.1S', 'sigma1S', 'sigma2S', 'sigma0.01N', 'sigma0.1N', 'sigma1N', 'sigma2N']:
#         n_action = n_state
#         gen_mdp_data(n_traj, max_length, n_state, n_action, temperature, temperature)

# for n_state in [42]:
#     for temperature in [1]:
#         n_action = n_state
#         gen_mdp_data(n_traj, max_length, n_state, n_action, temperature, temperature)


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










