import torch
import numpy as np
import torch.distributions as D
import torch.nn.functional as F
import torch.nn as nn
from .common import *

import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd

class VariationalLinear(nn.Module):
    def __init__(self, num_inputs, num_outputs, reparam_noise=1e-4):
        super(VariationalLinear, self).__init__()
        self.mu = nn.Linear(num_inputs, num_outputs)
        self.sigma = nn.Linear(num_inputs, num_outputs)
        self.reparam_noise = reparam_noise

    def forward(self, x):
        mu = self.mu(x)
        sigma = self.sigma(x)

        sigma = F.softplus(sigma) + self.reparam_noise
        
        return mu, sigma

class TransReprModule(nn.Module):
    def __init__(self, obs_size, act_size, state_size, hidden_size):
        super(TransReprModule, self).__init__()


        self.trans_det = nn.Sequential(
            nn.Linear(obs_size + act_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.trans_stoc = VariationalLinear(hidden_size, state_size)

        self.repr_model = nn.Sequential(
            nn.Linear(hidden_size + obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            VariationalLinear(hidden_size, state_size),
        )

    def forward(self, obs, act, next_obs):
        trans_det_states = self.trans_det(torch.cat([obs, act], dim=1))

        trans_stoch_mu, trans_stoch_sigma = self.trans_stoc(trans_det_states)
        trans_stoch_distr = D.independent.Independent(D.Normal(trans_stoch_mu, trans_stoch_sigma), 1)

        repr_stoch_mu, repr_stoch_sigma = self.repr_model(torch.cat([trans_det_states, next_obs], dim=1))
        repr_stoch_distr = D.independent.Independent(D.Normal(repr_stoch_mu, repr_stoch_sigma), 1)

        return trans_stoch_distr, repr_stoch_distr, trans_det_states

class ConvTransReprModule(nn.Module):
    def __init__(self, obs_size, act_size, state_size, hidden_size, use_bn, use_ln):
        super(ConvTransReprModule, self).__init__()

        self.trans_cnn = get_conv_net(state_size, use_bn=use_bn, use_ln=use_ln)

        self.trans_det = nn.Sequential(
            nn.Linear(state_size + act_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )

        self.trans_stoc = VariationalLinear(hidden_size, state_size)

        self.repr_cnn = get_conv_net(state_size, use_bn=use_bn, use_ln=use_ln)

        self.repr_model = nn.Sequential(
            nn.Linear(hidden_size + state_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            VariationalLinear(hidden_size, state_size),
        )

    def forward(self, obs, act, next_obs):
        trans_cnn_features = self.trans_cnn(obs)
        trans_det_states = self.trans_det(torch.cat([trans_cnn_features, act], dim=1))

        trans_stoch_mu, trans_stoch_sigma = self.trans_stoc(trans_det_states)
        trans_stoch_distr = D.independent.Independent(D.Normal(trans_stoch_mu, trans_stoch_sigma), 1)

        repr_obs_state = self.repr_cnn(next_obs)

        repr_stoch_mu, repr_stoch_sigma = self.repr_model(torch.cat([trans_det_states, repr_obs_state], dim=1))
        repr_stoch_distr = D.independent.Independent(D.Normal(repr_stoch_mu, repr_stoch_sigma), 1)

        return trans_stoch_distr, repr_stoch_distr, trans_det_states

class LBS(nn.Module):
    def __init__(self, obs_size, act_size, state_size, hidden_size, 
                        device='cuda' if torch.cuda.is_available() else 'cpu', 
                        out_type='distribution', 
                        use_bn=False,
                        use_ln=False,
                        cur_acc=False,
                        beta=1.):
        super(LBS, self).__init__()

        self.use_bn = use_bn
        self.use_ln = use_ln

        self.obs_size = obs_size
        self.act_size = act_size
        self.hidden_size = hidden_size
        self.beta = beta
        print("Beta value is ", self.beta)
        
        self.out_type = out_type # either 'distribution' or 'mean'
        print(f"The LBS model will output {self.out_type}s")
        self.cur_acc = cur_acc
        print(f"Curiosity will use posterior accuracy: {self.cur_acc}")

        if len(obs_size) == 1:
            obs_size = obs_size[0]
            state_size = obs_size
            self.target_model = nn.Identity()

            self.lbs = TransReprModule(obs_size, act_size, state_size, hidden_size)
        elif len(obs_size) == 3:
            self.target_model = get_random_cnn(state_size, use_bn=self.use_bn, use_ln=self.use_ln)

            self.lbs = ConvTransReprModule(obs_size, act_size, state_size, hidden_size, use_bn=self.use_bn, use_ln=self.use_ln) 
            
        self.state_size = state_size

        self.feat_mean = 0.
        self.feat_std = 1.

        if not self.use_bn and not self.use_ln:
            print("Collecting features mean and std at the beginning ...")

        self.free_nats = int(0) 
        print(f"Constraining KL to {self.free_nats:d} free nats")

        self.head = nn.Linear(self.state_size, self.state_size) 

        self.device = device
        self.to(device)
        
    def get_feature_moments(self, obs):
        states = []
        if len(obs) > 512:
            for i in range(0, len(obs), 512):
                states.append(self.target_model(obs[i:i+512]))
        states = torch.cat(states)
        self.feat_mean = torch.mean(states, dim=0) 
        self.feat_std = torch.std(states, dim=0)

    def normalize(self, state):
        state = (state - self.feat_mean) / (self.feat_std)
        return state

    def forward(self, obs, act, next_obs):
        target_next_states = self.target_model(next_obs)

        if not self.use_bn and not self.use_ln:
            target_next_states = self.normalize(target_next_states)
        
        trans_stoch_distr, repr_stoch_distr, _ = self.lbs(obs, act, next_obs)

        return target_next_states, trans_stoch_distr, repr_stoch_distr

    def loss(self, obs, act, next_obs, *args, **kwargs):

        # batch_size = act.shape[0]
    
        target_next_states, trans_pred_distr, repr_pred_distr = self.forward(obs, act, next_obs)
        repr_samples = repr_pred_distr.rsample()

        target_distr = D.independent.Independent(D.Normal(target_next_states, torch.ones_like(target_next_states)), 1)
        repr_projections = self.head(repr_samples)
        logprob_target = target_distr.log_prob(repr_projections)

        kl_div_post_prior = D.kl.kl_divergence(repr_pred_distr, trans_pred_distr) 

        if self.out_type == 'distribution':
            loss = self.beta * kl_div_post_prior - logprob_target

            mse_post_prior = (repr_pred_distr.mean - trans_pred_distr.mean).pow(2).mean(dim=1)
            mse_state_posterior = (repr_projections - target_next_states).pow(2).mean(dim=1)
        else:
            repr_samples = repr_pred_distr.mean
            trans_samples = trans_pred_distr.mean
            repr_projections = self.head(repr_samples)

            mse_post_prior = (repr_samples - trans_samples).pow(2).mean(dim=1)
            mse_state_posterior = (repr_projections - target_next_states).pow(2).mean(dim=1)

            loss = mse_post_prior + mse_state_posterior

        
        self.last_kl = kl_div_post_prior.detach().mean().cpu().item()
        self.last_acc = logprob_target.detach().mean().cpu().item()

        self.last_prior_mse = mse_post_prior.detach().mean().cpu().item()
        self.last_prior_std = trans_pred_distr.stddev.mean().cpu().item() 

        self.last_post_mse = mse_state_posterior.detach().mean().cpu().item()
        self.last_post_std = repr_pred_distr.stddev.mean().cpu().item()
        
        return loss 

    def curiosity(self, obs, act, next_obs, *args, **kwargs):
        target_next_states, trans_pred_distr, repr_pred_distr = self.forward(obs, act, next_obs)
        repr_samples = repr_pred_distr.sample()

        target_distr = D.independent.Independent(D.Normal(target_next_states, torch.ones_like(target_next_states)), 1)
        repr_projections = self.head(repr_samples)
        logprob_target = target_distr.log_prob(repr_projections)

        kl_div_post_prior = D.kl.kl_divergence(repr_pred_distr, trans_pred_distr) # .mean(-1)

        if self.out_type == 'distribution':
            curiosity = kl_div_post_prior
            if self.cur_acc:
                curiosity -= logprob_target

            mse_post_prior = (repr_pred_distr.mean - trans_pred_distr.mean).pow(2).mean(dim=1)
            mse_state_posterior = (repr_projections - target_next_states).pow(2).mean(dim=1)
        else:
            repr_samples = repr_pred_distr.mean
            trans_samples = trans_pred_distr.mean
            repr_projections = self.head(repr_samples)

            mse_post_prior = (repr_samples - trans_samples).pow(2).mean(dim=1)
            mse_state_posterior = (repr_projections - target_next_states).pow(2).mean(dim=1)

            curiosity = mse_post_prior 
            if self.cur_acc:
                curiosity += mse_state_posterior
        
        return curiosity