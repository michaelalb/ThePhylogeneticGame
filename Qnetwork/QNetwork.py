import os

import joblib
import torch
import torch.optim as optim

import SharedConsts as SC
from Qnetwork.Qnet_architecture import Net
from Qnetwork.ReplayBuffer import ReplayBuffer
from SharedConsts import GAMMA, LOSS_FUNC, SOFT_UPDATE_RATE, UPDATE_TARGET_NET, \
    TARGET_NET_UPDATE_POLICY, USE_TARGET, LEARNING_RATE, BUFFER_SIZE, BATCH_SIZE


class QNetwork:
    """
    Interacts with and learns from environment.
    """

    def __init__(self, state_action_size, experiment_unique_dir_name,
                 local_weight_path=SC.Q_NETWORK_LOCAL_WEIGHTS_FILE_NAME,
                 target_weight_path=SC.Q_NETWORK_TARGET_WEIGHTS_FILE_NAME):
        """
        :param state_action_size: dimension of each state_action
        :param local_weight_path:
        :param target_weight_path:
        """
        # Q- Network
        self.device = torch.device("cuda" if SC.USE_CUDA else "cpu")
        self.q_network = Net(in_features=state_action_size).to(self.device)
        self.q_network_target = Net(in_features=state_action_size).to(self.device) if USE_TARGET else self.q_network

        self.experiment_unique_dir_name = experiment_unique_dir_name
        self.local_weight_path = SC.EXPERIMENTS_RESDIR / self.experiment_unique_dir_name / local_weight_path
        self.target_weight_path = SC.EXPERIMENTS_RESDIR / self.experiment_unique_dir_name / target_weight_path
        self.load_model(local_weights_path=self.local_weight_path, target_weights_path=self.target_weight_path)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)

        # Replay memory
        memory_path = SC.EXPERIMENTS_RESDIR / self.experiment_unique_dir_name / SC.REPLAY_BUFFER_FILE_NAME
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, memory_path=memory_path)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.to_train_mode()

    def prepare_to_exit(self, episode):
        self.save_all(episode)

    def save_all(self, episode):
        self.save_memory(episode)
        self.save_weights(episode)

    def save_memory(self, episode):
        self.memory.save_buffer()
        episode_file = 'Replay_Buffer_memory' + f"_{episode}.pkl"
        episode_path = SC.EXPERIMENTS_RESDIR / self.experiment_unique_dir_name / 'Replay_Buffers'
        episode_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.memory.memory, str(episode_path / episode_file))

    def save_weights(self, episode):
        # keep regular save
        torch.save(self.q_network.state_dict(), self.local_weight_path)
        if USE_TARGET:
            torch.save(self.q_network_target.state_dict(), self.target_weight_path)

        # add episode save
        file_name = self.local_weight_path.stem
        suffix = self.local_weight_path.suffix
        episode_file = file_name + f"_{episode}" + suffix
        episode_path = SC.EXPERIMENTS_RESDIR / self.experiment_unique_dir_name / 'weights_folder'
        episode_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_network.state_dict(), episode_path / episode_file)

    def load_weights(self, weights_path, model):
        model.load_state_dict(torch.load(weights_path, map_location=self.device))

    def load_model(self, local_weights_path, target_weights_path):

        if not os.path.isfile(local_weights_path):
            print("No weights found at all.  created Net from scratch according to QNet param consts")
            return

        print('Load QNet parameters ...')
        self.load_weights(local_weights_path, self.q_network)

        if USE_TARGET and os.path.isfile(target_weights_path):
            print('Load Target-QNet parameters ...')
            self.load_weights(target_weights_path, self.q_network_target)
        else:
            self.q_network_target = self.q_network

    def add_memory(self, old_state_action, reward, done, new_state_action):
        # Save experience in replay memory
        self.memory.add(old_state_action, reward, done, new_state_action)

    def predict(self, state_action_feature_matrix):
        """
        this function receives state_action vectors from the agent and predicts their Q values
        :param state_action_feature_matrix:
        :return: predictions
        """
        with torch.no_grad():
            return self.q_network(state_action_feature_matrix)

    def learn(self, gamma=GAMMA):
        """
        Update both network parameters.
        Params
        =======
        """
        state_actions, rewards, dones, next_state_actions = self.memory.sample()
        loss = self.learn_on_batch(state_actions, rewards, dones, next_state_actions, gamma)

        return loss

    def learn_on_batch(self, state_actions, rewards, dones, next_state_actions, gamma=GAMMA):

        criterion = LOSS_FUNC
        predictions = self.q_network(state_actions)

        with torch.no_grad():
            # .detach() ->  Returns a new Tensor, detached from the current graph.
            target_next_predictions = self.q_network_target(next_state_actions).detach()

        # we choose to ignore dones for inf-horizon Q-learning
        labels = rewards + (gamma * target_next_predictions)

        loss = criterion(predictions, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if USE_TARGET:
            self.update_target()

        return loss.item()

    def update_target(self):
        self.t_step += 1
        if TARGET_NET_UPDATE_POLICY == 'hard':
            if self.t_step % UPDATE_TARGET_NET == 0:
                self.q_network_target.load_state_dict(self.q_network.state_dict())
                self.t_step = 0
        else:
            # soft - update target network
            self.soft_update()

    def soft_update(self, tau=SOFT_UPDATE_RATE):
        """
        Soft update ll_model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local ll_model (PyTorch ll_model): weights will be copied from
            target ll_model (PyTorch ll_model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(self.q_network_target.parameters(),
                                             self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def to_eval_mode(self):
        self.q_network.eval()
        self.q_network_target.eval()

    def to_train_mode(self):
        self.q_network.train()
        if USE_TARGET:
            self.q_network_target.eval()
