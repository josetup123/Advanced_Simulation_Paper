"""
IE-608 DQN example.
"""

# import
import logging
import numpy as np
from DRL import Agent, DRL_Env
from matplotlib import pyplot as plt


def main():
    """
    mian
    """
    # --------- parameters -----------

    # number of states
    num_states = 4

    # number of plans
    num_plans = 2

    # horizon
    T = 20

    # success probability
    success_pr = [0.0, 0.5, 0.9]

    # degrade pr, implies four states
    degrade_pr = np.mat([
        [0.3, 0.5, 0.2, 0.0],
        [0.0, 0.4, 0.4, 0.2],
        [0.0, 0.0, 0.4, 0.6],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # revenue
    C_r = 10

    # operating cost
    C_o = [
        i for i in range(num_states)
    ]

    # maintenance cost
    C_m = [
        i for i in range(num_plans)
    ]

    # ------------ MDP --------------

    # initial state: t, condition
    initial_state = (0, 0)

    # number of actions
    actions = range(num_plans)

    # transition function
    def trans_func(t, s, a):
        """
        transition
        """
        # terminal
        if t >= T:
            return "Delta"
        # calculate pr
        pr = []
        for i in range(num_states):
            # smaller state
            if i < s[1]:
                pr.append(
                    success_pr[a] * degrade_pr[0, i]
                )
            else:
                pr.append(np.sum([
                    success_pr[a] * degrade_pr[0, i],
                    (1 - success_pr[a]) * degrade_pr[s[1], i]
                ]))
        # sample new state according to pr
        return (t + 1, np.random.choice(
            range(num_states), size=1, replace=False,
            p=pr
        )[0])
    
    # reward function
    def reward_func(s, a, s_new):
        """
        reward
        """
        return C_r - C_o[s[1]] - C_m[a]

    # agent
    manager = Agent(
        name="manager",
        actions=actions,
        input_size=2,
        hidden_layers=[10, 10],
        output_size=len(actions),
        learning_rate=1e-5
    )

    # define problem
    problem = DRL_Env(
        name="maintenance",
        agent=manager,
        initial_state=initial_state,
        trans_func=trans_func,
        reward_func=reward_func,
        memory_size=5000,
        sample_episodes=500
    )

    # logging
    logging.basicConfig(
        filename='maintenance.log', filemode='w+',
        format='%(levelname)s - %(message)s', level=logging.INFO
    )

    # DQN
    G = problem.deep_Q_Network(
        episodes=5000,
        discount_factor=1.0,
        learn_step=20,
        eps_init=1.0,
        eps_end=0.0,
        write_log=True
    )

    # plot
    manager.plot_loss(dir='figs/')

    manager.plot_G(dir='figs/')

    return


if __name__ == "__main__":
    main()
