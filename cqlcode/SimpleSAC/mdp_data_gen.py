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
    if policy_temperature >= 9999999 and transition_temperature >= 9999999: # if use this number or larger, will hard switch to iid generation
        iid = True
    else:
        iid = False

    n_data = n_traj * max_length
    states = np.zeros(n_data, dtype=int)
    actions = np.zeros(n_data, dtype=int)
    next_states = np.zeros(n_data, dtype=int)
    i = 0

    if n_state in [12345678, int(2e4) * 1000 * 2] and temperature >= 9999999:
        # 1m data, infinite state space, iid
        print("generate finite data, inf state space, iid.")
        np.random.seed(0)
        states_next_states = np.random.choice(n_state, size=n_data*2, replace=False)
        states = states_next_states[:n_data]
        next_states = states_next_states[n_data:]
        actions = np.random.choice(n_state, size=n_data, replace=False)
        print(states.shape, actions.shape, next_states.shape)
    else:
        for i_traj in tqdm(range(n_traj)):
            np.random.seed(i_traj)
            state = np.random.randint(n_state)
            for t in range(max_length):
                states[i] = state
                # for each step, an action is taken, and a next state is decided.
                # if we assume a fixed policy is generating the data, then the action probability only depends on the state
                if not iid:
                    np.random.seed(state)
                    action_probs = softmax_with_torch(np.random.rand(n_action), policy_temperature)

                    np.random.seed(42 + i_traj * 1000 + t * 333)
                    action = np.random.choice(n_action, p=action_probs)

                    np.random.seed(state * 888 + action * 777)
                    next_state_probs = softmax_with_torch(np.random.rand(n_state), transition_temperature)

                    np.random.seed(666 + i_traj * 1000 + t * 333)
                    next_state = np.random.choice(n_state, p=next_state_probs)
                else:
                    action = np.random.randint(n_action)
                    next_state = np.random.randint(n_state)

                next_states[i] = next_state
                actions[i]=action
                state = next_state
                i += 1

    data_dict = {'observations': states,
            'actions': actions,
            'next_observations': next_states}
    data_name = 'mdp_traj%d_ns%d_na%d_pt%s_tt%s.pkl' % (n_traj, n_state, n_action,
                                                        str(policy_temperature), str(transition_temperature))
    save_name = '/cqlcode/mdpdata/%s' % data_name
    joblib.dump(data_dict, save_name)
    print("Data saved to:", save_name)

# generate 1M data, each trajectory has 1000 steps
n_traj, max_length = 1000, 1000

# for n_state in [100, 100000, 10000000]:
#     for temperature in [9999999]:
#         n_action = n_state
#         gen_mdp_data(n_traj, max_length, n_state, n_action, temperature, temperature)
#
# for n_state in [42]: # in the newest cql main code, when see 42 they will do state space inf IID transition pretrain
#     for temperature in [42]:
#         n_action = n_state
#         gen_mdp_data(n_traj, max_length, n_state, n_action, temperature, temperature)
#
# for n_state in [100000]:
#     for temperature in [1000]:
#         n_action = n_state
#         gen_mdp_data(n_traj, max_length, n_state, n_action, temperature, temperature)

for n_traj in [int(1e4), int(4e4)]:
    n_state = int(4e4) * 1000 * 2
    temperature = 9999999
    n_action = n_state
    gen_mdp_data(n_traj, max_length, n_state, n_action, temperature, temperature)



