"""
Deep Reinforcement learning.
"""

# import
# import pickle
import logging
import numpy as np
import torch
import torch.optim as optim
from ActorCritic import Actor, Critic
import matplotlib.pyplot as plt


def action_filter(state):
    """
    output valid actions for the state
    """
    actions = []
    for i in range(1, len(state)):
        if state[i] == 0:
            actions.append(i)
    if len(actions) == 0:
        actions.append(0)
    return actions


class Agent:
    """
    Agent use in the DRL class.\n
    `name`: str, name of the agent;\n
    `actions`: a list of all actions;\n
    `input_size`: int, size of the input, len(state);\n
    `actor_hidden_layers`: list, the number of neurons for the hidden layer;\n
    `critic_hidden_layers`: list, the number of neurons for the hidden layer;\n
    `output_zise`: int, size of the output, len(all actions);\n
    Keyword Arguments:\n
    `action_filter`: Boolean, True to use the action filter, False to not use;\n
    """

    def __init__(
        self, name, actions, input_size, actor_hidden_layers,
        critic_hidden_layers, output_size, actor_lr, critic_lr, **kwargs
    ):
        super().__init__()
        # ------------------- name & actions ----------------------
        self.name, self.actions = name, actions
        # ------------------- GPU ----------------
        if torch.cuda.is_available():
            self.dev = "cuda:0"
        else:
            self.dev = "cpu"
        # ------------- whether to use action_filter ----------------
        if 'action_filter' in kwargs:
            self.filter_action = False
            if kwargs['action_filter']:
                self.filter_action = True
                self.action_filter = action_filter
        # ----------------- construct network --------------------
        # actor
        self.actor = Actor(
            hidden_layers=actor_hidden_layers,
            input_size=input_size, output_size=output_size
        )
        self.actor.to(self.dev)
        # critic
        self.critic = Critic(
            hidden_layers=critic_hidden_layers, input_size=input_size
        )
        self.critic.to(self.dev)
        # ------------------------ optimizer ----------------------------
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )
        # self.actor_optimizer = optim.Adadelta(self.actor.parameters())
        # self.critic_optimizer = optim.Adadelta(self.critic.parameters())
        # ------------------------ training step ------------------------
        self.train_step = 1
        self.actor_loss = []
        self.critic_loss = []
        self.reward_memory = []
        self.log_prob = []
        self.V_pred = []
        self.G_memory = []
        self.entropy = 0

    def load_parameter(self, actor_path, critic_path):
        """
        Load network parameter from path
        """
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        return

    def take_action(self, state):
        """
        take action (make prediction), based on the input state.
        """
        # make state to tensor
        input_seq = torch.FloatTensor(state).to(self.dev)
        # make a prediction, with grad
        policy_dist = self.actor(input_seq)
        value = self.critic(input_seq)
        # value, distribution
        self.V_pred.append(value[0])
        dist = policy_dist.to('cpu').detach().numpy()
        # valid_actions
        if self.filter_action:
            valid_actions = self.action_filter(state)
        else:
            valid_actions = self.actions
        # choose action
        action_pr = np.array([
            np.exp(dist[i]) if self.actions[i] in valid_actions
            else 0 for i in range(len(self.actions))
        ])
        if np.sum(action_pr) == 0:
            action_pr = np.array([
                1 if self.actions[i] in valid_actions
                else 0 for i in range(len(self.actions))
            ])
        action_pr = action_pr / np.sum(action_pr)
        action_ind = np.random.choice(
            range(len(self.actions)), 1, False, action_pr
        )[0]
        # log prob
        self.log_prob.append(policy_dist.squeeze(0)[action_ind])
        # entropy
        self.entropy += -1 * np.sum(np.mean(np.exp(dist)) * dist)
        # return
        return self.actions[action_ind]

    def simulate_action(self, state):
        """
        take action (for simulation), based on the input state.
        """
        # make state to tensor
        input_seq = torch.FloatTensor(list(state.values())).to(self.dev)
        # make a prediction, without grad
        self.actor.eval()
        with torch.no_grad():
            dist = self.actor(input_seq).to('cpu').numpy()
        self.actor.train()
        # valid_actions
        if self.filter_action:
            valid_actions = self.action_filter(state)
        else:
            valid_actions = self.actions
        # choose action
        action_pr = np.array([
            np.exp(dist[i]) if self.actions[i] in valid_actions
            else 0 for i in range(len(self.actions))
        ])
        if np.sum(action_pr) == 0:
            action_pr = np.array([
                1 if self.actions[i] in valid_actions
                else 0 for i in range(len(self.actions))
            ])
        action_pr = action_pr / np.sum(action_pr)
        action_ind = np.argmax(action_pr)
        return self.actions[action_ind]

    def learn(self, discount_factor):
        """
        train the network
        """
        # set zero grad
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        # memories
        # reward_memory = rewards
        # target values
        V_targ = np.zeros(len(self.V_pred))
        V_future = 0
        for t in range(len(self.reward_memory) - 1, -1, -1):
            V_future = self.reward_memory[t] + discount_factor * V_future
            V_targ[t] = V_future
        # make tensors
        V_pred = torch.stack(self.V_pred)
        V_targ = torch.FloatTensor(V_targ).to(self.dev)
        log_prob = torch.stack(self.log_prob)
        # loss
        advantage = V_targ - V_pred
        actor_loss = (-1 * log_prob * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        # back propagate
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        # save loss
        self.actor_loss.append(actor_loss.to('cpu').detach().numpy())
        self.critic_loss.append(critic_loss.to('cpu').detach().numpy())
        # clear memory
        self.log_prob = []
        self.V_pred = []
        self.reward_memory = []
        return
    
    def plot_G(self, window=20, start_ind=0, sample=1):
        """
        plot return using time window
        """
        G_plot = {}
        ind = window
        while ind <= len(self.G_memory):
            # sample
            if ind % sample == 0:
                G_plot[ind - 1] = np.mean([
                    self.G_memory[i] for i in range(ind - window, ind, 1)
                ])
            ind += 1
        # plot
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        ax.plot(
            list(G_plot.keys())[start_ind:],
            list(G_plot.values())[start_ind:],
            'b-'
        )
        ax.set_xlabel('Epochs')
        ax.set_ylabel('G')
        fig.tight_layout()
        fig.savefig('figs/{}_G.png'.format(self.name), dpi=600)
        plt.close()
        return
    
    def plot_loss(self, window=20, start_ind=0, sample=1):
        """
        plot train loss for agent
        """
        for i in range(2):
            loss_memory = self.actor_loss if i == 0 else self.critic_loss
            loss_plot = {}
            ind = window
            while ind <= len(loss_memory):
                # sample
                if ind % sample == 0:
                    loss_plot[ind - 1] = np.mean([
                        loss_memory[i] for i in range(ind - window, ind, 1)
                    ])
                ind += 1
            # plot
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
            ax.plot(
                list(loss_plot.keys())[start_ind:],
                list(loss_plot.values())[start_ind:],
                'b-'
            )
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            fig.tight_layout()
            fig.savefig('figs/{}_{}_loss.png'.format(
                self.name, "actor" if i == 0 else "critic"
            ), dpi=600)
        return


class MDRL_Env:
    """
    Multi-agents Deep Reinforcement learning class.
    - initial state: dict, or functions that returns a dict,
        denoting the initial state;
        Terminal state = 'Delta'!!!!!!!!!!!!!!!!!;
    - reward: input state and action, output a number;
        Output should be a dict, {agent_name: agent_action};
    - transition: input state and action, action should be a dict,
        {agent_name: agent_action};
        If additional environment is needed, input should be
        (state, action, envs), otherwise, (state, action);
        Output a new state;
    Keyword arguments:
    - every kwargs is considered to be an agent.
    """

    def __init__(
        self, name, initial_state, trans_func, reward_func, agent
    ):
        # input parameters
        self.name = name
        self.initial_state = initial_state
        self.trans_func = trans_func
        self.reward_func = reward_func
        self.learn_step = 50
        # agent
        self.agent = agent

    def __plot_loss(self, name):
        """
        plot train loss for agent
        """
        if name == 'actor':
            loss = self.agent.actor_loss
        elif name == 'critic':
            loss = self.agent.critic_loss
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        ax.plot(
            list(range(len(loss)))[0:], loss[0:]
        )
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        fig.tight_layout()
        fig.savefig('figs/loss/{}_{}.png'.format(self.name, name))
        return

    def __A2C_update(self, discount_factor, write_log):
        """
        one episode of A2C
        """
        # =================== initialize ===================
        # initialize
        if callable(self.initial_state):
            state = self.initial_state()
        else:
            state = self.initial_state
        # return
        G = 0
        # env
        if self.agent.use_env:
            envs = self.agent.environment
        # memory
        reward_memory = []
        # ======================= LOOP ========================
        epoch, V_future = 0, 0
        while state != 'Delta':
            # ----------------- take action ------------------
            action = self.agent.take_action(state)
            # --------------- transit and reward ---------------
            if self.agent.use_env:
                # transition
                new_state = self.trans_func(state, action, envs)
                # rewards
                R = self.reward_func(state, action, new_state, envs)
            else:
                # transition
                new_state = self.trans_func(state, action)
                # rewards
                R = self.reward_func(state, action, new_state)
            # --------------------- record ---------------------
            reward_memory.append(R)
            # ------------------- update G --------------------
            G += R
            # --------------- output experience ----------------
            if write_log:
                # logging state, action and reward
                logging.info('    epoch: {}'.format(epoch))
                logging.info('    state: {}'.format(state))
                logging.info('    action: {}'.format(action))
                logging.info('    reward: {}'.format(R))
            # ---------- on step, transit to new state -------------
            state = new_state
            epoch += 1
            if epoch >= self.learn_step:
                V_future = self.agent.critic(list(state.values))
                V_future = V_future.to('cpu').detach().numpy()[0]
                break
        # ---------------- Learning --------------------
        self.agent.learn(
            epoch=epoch, V_future=V_future,
            rewards=reward_memory, discount_factor=discount_factor
        )
        return G

    def advantage_actor_acritic(
        self, episodes, discount_factor, write_log, plot_loss
    ):
        """
        Deep Q learning.
        - episodes: how many iterations to simulation in total.
        - alpha: learning rate;
        - discount factor: perspective of future.
        """
        G = []
        # ---------------------- Learning ----------------------
        if write_log:
            logging.info("Learning...")
        for iter in range(episodes):
            if iter % 10000 == 0:
                print("Iteration {}".format(iter))
            # if True:
            if write_log:
                logging.info("Iteration {}".format(iter))
            # run one update
            # try:
            step_G = self.__A2C_update(discount_factor, write_log)
            # ---------------- step control --------------------
            # record return
            G.append(step_G)
            # log return
            if write_log:
                logging.info("    return: {}".format(step_G))
                logging.info("    -----------------------")
            # except ValueError:
            #     break
        # ------------------- plot loss -----------------------
        if plot_loss:
            self.__plot_loss('actor')
            self.__plot_loss('critic')
        return G

    def simulate(self, write_to_file, run_time):
        """
        simulate the delivery process.
        """
        if write_to_file:
            output_file = open('results/{}.txt'.format(self.name), 'w+')
            output_file.write("=============== SIMULATION ==============\n")
            output_file.write("Run Time = {}\n".format(run_time))
        # =================== initialize ===================
        # initialize
        if callable(self.initial_state):
            state = self.initial_state()
        else:
            state = self.initial_state
        # return
        G = 0
        # env
        if self.agent.use_env:
            envs = self.agent.environment
        # ======================= LOOP ========================
        epoch = 0
        while state != 'Delta':
            # ------------------ take action ------------------
            action = self.agent.simulate_action(state)
            # --------------- transit and reward ---------------
            if self.agent.use_env:
                # transition
                new_state = self.trans_func(state, action, envs)
                # rewards
                R = self.reward_func(state, action, new_state, envs)
            else:
                # transition
                new_state = self.trans_func(state, action)
                # rewards
                R = self.reward_func(state, action, new_state)
            if write_to_file:
                # logging state, action and reward
                output_file.write('    epoch: {}\n'.format(epoch))
                output_file.write('    state: {}\n'.format(state))
                output_file.write('    action: {}\n'.format(action))
                output_file.write('    reward: {}\n'.format(R))
            # ------------ exit if too many epochs --------------
            if epoch > 100:
                new_state = 'Delta'
            # -------------------- update G ----------------------
            G += R
            # ---------- on step, transit to new state -----------
            state = new_state
            epoch += 1
        if write_to_file:
            output_file.write('Return: {}\n'.format(G))
        return G
