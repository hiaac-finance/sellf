import numpy as np
import argparse
import gym
import torch
import torch.nn as nn
import time
from .data_generator import DataGenerator
from .models import GaussianPolicy, Value, CategoricalPolicy
from .utils import RunningStats, graph_detach, to_dytype_device
from collections import deque

def categorical_kl(new_logits, old_logits):
    """
    KL( new || old ) for Categorical distributions given logits.

    new_logits, old_logits: [B, act_dim]
    Returns: [B, 1]
    """
    # log softmax
    log_p = torch.log_softmax(new_logits, dim=-1)   # log p
    log_q = torch.log_softmax(old_logits, dim=-1)   # log q

    p = torch.exp(log_p)                            # p

    kl = torch.sum(p * (log_p - log_q), dim=-1, keepdim=True)
    return kl

class FOCOPS:
    """
    Implement FOCOPS algorithm
    """
    def __init__(self,
        env,
        learning_rate = 1e-5,
        max_iter_num = 500,
        n_epochs = 10,
        batch_size = 64,
        gamma = 0.99,
        c_gamma = 0.99,
        gae_lambda = 0.95,
        c_gae_lambda = 0.95,
        l2_reg = 1e-3,
        lam = 1.5,
        delta = 0.02,
        eta = 0.02,
        nu = 0,
        nu_lr = 0.01,
        nu_max =2.0,
        cost_lim = 0.1,
        n_steps = 2048,
        max_eps_len = 2_000,
        clip_range = 0.2,
        normalize_advantage = True,
        ent_coef = 0.,
        vf_coef = 0.5,
        max_grad_norm = 0.5,
    ):


        self.env = env
        self.learning_rate = learning_rate
        self.max_iter_num = max_iter_num
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.c_gamma = c_gamma
        self.gae_lambda = gae_lambda
        self.c_gae_lambda = c_gae_lambda
        self.l2_reg = l2_reg
        self.lam = lam
        self.delta = delta
        self.eta = eta
        self.nu = nu
        self.nu_lr = nu_lr
        self.nu_max = nu_max
        self.cost_lim = cost_lim
        self.n_steps = n_steps
        self.max_eps_len = max_eps_len

        self.obs_dim = env.observation_space.shape[0]
    

        self.pi_loss = None
        self.vf_loss = None
        self.cvf_loss = None
        self.device = torch.device("cpu")
        self._setup_model()

    
    def _setup_model(self):
        # Initialize neural nets
        self.policy = CategoricalPolicy(self.obs_dim, 2)
        self.value_net = Value(self.obs_dim)
        self.cvalue_net = Value(self.obs_dim)
        self.policy.to(self.device)
        self.value_net.to(self.device)
        self.cvalue_net.to(self.device)

        # Initialize optimizer
        self.pi_optimizer = torch.optim.Adam(self.policy.parameters(), self.learning_rate)
        self.vf_optimizer = torch.optim.Adam(self.value_net.parameters(), self.learning_rate)
        self.cvf_optimizer = torch.optim.Adam(self.cvalue_net.parameters(), self.learning_rate)

        # Initialize learning rate scheduler
        #lr_lambda = lambda it: max(1.0 - it / self.max_iter_num, 0)
        #self.pi_scheduler = torch.optim.lr_scheduler.LambdaLR(self.pi_optimizer, lr_lambda=lr_lambda)
        #self.vf_scheduler = torch.optim.lr_scheduler.LambdaLR(self.vf_optimizer, lr_lambda=lr_lambda)
        #self.cvf_scheduler = torch.optim.lr_scheduler.LambdaLR(self.cvf_optimizer, lr_lambda=lr_lambda)

        # Initialize RunningStat for state normalization, score queue, logger
        self.running_stat = None #RunningStats(clip=5)
        self.score_queue = deque(maxlen=100)
        self.cscore_queue = deque(maxlen=100)

    @property
    def logger(self):
        return self._logger
    
    def set_logger(self, logger):
        self._logger = logger
        self._custom_logger = True

    def update_params(self, rollout, dtype, device):

        # Convert data to tensor
        obs = torch.Tensor(rollout['states']).to(dtype).to(device)
        act = torch.Tensor(rollout['actions']).to(torch.long).to(device)
        vtarg = torch.Tensor(rollout['v_targets']).to(dtype).to(device).detach()
        adv = torch.Tensor(rollout['advantages']).to(dtype).to(device).detach()
        cvtarg = torch.Tensor(rollout['cv_targets']).to(dtype).to(device).detach()
        cadv = torch.Tensor(rollout['c_advantages']).to(dtype).to(device).detach()

        # Get log likelihood, mean, and std of current policy
        old_logprob, old_logits = self.policy.logprob(obs, act)
        old_logprob, old_logits = to_dytype_device(dtype, device, old_logprob, old_logits)
        old_logprob, old_logits = graph_detach(old_logprob, old_logits)


        # Store in TensorDataset for minibatch updates
        dataset = torch.utils.data.TensorDataset(obs, act, vtarg, adv, cvtarg, cadv,
                                                 old_logprob, old_logits)
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        avg_cost = rollout['avg_cost']


        # Update nu
        self.nu += self.nu_lr * (avg_cost - self.cost_lim)
        if self.nu < 0:
            self.nu = 0
        elif self.nu > self.nu_max:
            self.nu = self.nu_max

        pi_loss_list = []
        vf_loss_list = []
        cvf_loss_list = []

        for epoch in range(self.n_epochs):

            for _, (obs_b, act_b, vtarg_b, adv_b, cvtarg_b, cadv_b,
                    old_logprob_b, old_logits_b) in enumerate(loader):


                # Update reward critic
                mse_loss = nn.MSELoss()
                vf_pred = self.value_net(obs_b)
                self.vf_loss = mse_loss(vf_pred, vtarg_b)
                # weight decay
                for param in self.value_net.parameters():
                    self.vf_loss += param.pow(2).sum() * self.l2_reg
                vf_loss_list.append(self.vf_loss.item())
                self.vf_optimizer.zero_grad()
                self.vf_loss.backward()
                self.vf_optimizer.step()

                # Update cost critic
                cvf_pred = self.cvalue_net(obs_b)
                self.cvf_loss = mse_loss(cvf_pred, cvtarg_b)
                # weight decay
                for param in self.cvalue_net.parameters():
                    self.cvf_loss += param.pow(2).sum() * self.l2_reg
                cvf_loss_list.append(self.cvf_loss.item())
                self.cvf_optimizer.zero_grad()
                self.cvf_loss.backward()
                self.cvf_optimizer.step()


                # Update policy
                logprob, logits = self.policy.logprob(obs_b, act_b)
                kl_new_old = categorical_kl(logits, old_logits_b)
                ratio = torch.exp(logprob - old_logprob_b)
                self.pi_loss = (kl_new_old - (1 / self.lam) * ratio * (adv_b - self.nu * cadv_b)) \
                          * (kl_new_old.detach() <= self.eta).type(dtype)
                self.pi_loss = self.pi_loss.mean()
                pi_loss_list.append(self.pi_loss.item())
                self.pi_optimizer.zero_grad()
                self.pi_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
                self.pi_optimizer.step()


            # Early stopping
            logprob, logits = self.policy.logprob(obs, act)
            kl_val = categorical_kl(logits, old_logits).mean().item()
            if kl_val > self.delta:
            #    println('Break at epoch {} because KL value {:.4f} larger than {:.4f}'.format(epoch + 1, kl_val, self.delta))
                break

            
        self.logger.record("accept_rate", np.mean(rollout["actions"]))
        self.logger.record("reward", np.mean(self.score_queue))
        self.logger.record("cost", avg_cost)
        self.logger.record("pi_loss", np.mean(pi_loss_list))
        self.logger.record("vf_loss", np.mean(vf_loss_list))
        self.logger.record("cvf_loss", np.mean(cvf_loss_list))

        #print("Reward: ", np.mean(self.score_queue), " Cost: ", avg_cost, " Nu: ", self.nu)
        #print("Pi loss: ", self.pi_loss.item(), " Vf loss: ", self.vf_loss.item(), " CVf loss: ", self.cvf_loss.item())



        # Store everything in log
        #self.logger.record('MinR', np.min(self.score_queue))
        #self.logger.record('MaxR', np.max(self.score_queue))
        #self.logger.record('AvgR', np.mean(self.score_queue))
        #self.logger.record('MinC', np.min(self.cscore_queue))
        #self.logger.record('MaxC', np.max(self.cscore_queue))
        #self.logger.record('AvgC', np.mean(self.cscore_queue))
        self.logger.record('nu', self.nu)


        # Save models
        # self.logger.save_model('policy_params', self.policy.state_dict())
        # self.logger.save_model('value_params', self.value_net.state_dict())
        # self.logger.save_model('cvalue_params', self.cvalue_net.state_dict())
        # self.logger.save_model('pi_optimizer', self.pi_optimizer.state_dict())
        # self.logger.save_model('vf_optimizer', self.vf_optimizer.state_dict())
        # self.logger.save_model('cvf_optimizer', self.cvf_optimizer.state_dict())
        # self.logger.save_model('pi_loss', self.pi_loss)
        # self.logger.save_model('vf_loss', self.vf_loss)
        # self.logger.save_model('cvf_loss', self.cvf_loss)

    def learn(self, total_timesteps):
        n_episodes = total_timesteps // 2_000

        for i in range(n_episodes):
            data_generator = DataGenerator(self.obs_dim, 1, self.n_steps, self.max_eps_len)
            rollout = data_generator.run_traj(
                self.env, 
                self.policy, 
                self.value_net, 
                self.cvalue_net,
                self.running_stat, 
                self.score_queue, 
                self.cscore_queue,
                self.gamma, 
                self.c_gamma, 
                self.gae_lambda, 
                self.c_gae_lambda,
                torch.float32, 
                self.device, 
            )

            self.update_params(rollout, torch.float32, self.device)
            #self.pi_scheduler.step()
            #self.vf_scheduler.step()
            #self.cvf_scheduler.step()

            self.logger.dump(step = i)


    def set_random_seed(self, seed):
        if seed is None:
            return
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

    def save(self, path: str) -> None:
        """
        Save the model to a file.
        """
        torch.save(self.policy.state_dict(), path + ".pth")

    def load(self, path: str) -> None:
        """
        Load the model from a file.
        """
        self.policy.load_state_dict(torch.load(path + ".pth", map_location=self.device))
        self.policy.eval()

    
    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation
        :param observation: the input observation
        :return: the action to take
        """
        return self.policy.get_action(observation)

    def get_label(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation
        :param observation: the input observation
        :return: the action to take
        """
        return self.policy.get_label(observation)