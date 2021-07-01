'''
Licensed under the MIT license ( http://valums.com/mit-license/ )
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixer(nn.Module):

    INPUT_KEYS = ['heroes', 'nonheroes', 'towers']
    N_HEROES = 5 * 2
    N_NONHEROES = 16 * 2
    N_TOWERS = 3 * 2
    N_UNIT_BASIC_STATS = 80
    N_UNIT_BASIC_FILTER = 32
    N_EMBED_FILTER = 128

    def __init__(self, num_agents):
        super().__init__()
        self.num_agents = num_agents

        self.affine_unit_basic_stats = nn.Linear(self.N_UNIT_BASIC_STATS, self.N_UNIT_BASIC_FILTER)

        self.affine_heros = nn.Linear(self.N_UNIT_BASIC_FILTER * self.N_HEROES, self.N_EMBED_FILTER)
        self.affine_nonheros = nn.Linear(self.N_UNIT_BASIC_FILTER * self.N_NONHEROES,
                                         self.N_EMBED_FILTER)
        self.affine_towers = nn.Linear(self.N_UNIT_BASIC_FILTER * self.N_TOWERS,
                                       self.N_EMBED_FILTER)

        self.w1 = nn.Linear(self.N_EMBED_FILTER * len(self.INPUT_KEYS),
                            self.N_EMBED_FILTER * self.num_agents)
        self.b1 = nn.Linear(self.N_EMBED_FILTER * len(self.INPUT_KEYS), self.N_EMBED_FILTER)
        self.w2 = nn.Linear(self.N_EMBED_FILTER * len(self.INPUT_KEYS), self.N_EMBED_FILTER)
        self.b2 = nn.Sequential(
            nn.Linear(self.N_EMBED_FILTER * len(self.INPUT_KEYS), self.N_EMBED_FILTER), nn.ReLU(),
            nn.Linear(self.N_EMBED_FILTER, 1))

    def forward(self, heroes, nonheroes, towers, qs):
        # Heroes
        h_basic = F.relu(self.affine_unit_basic_stats(heroes))
        h_embedding = self.affine_heros(h_basic.view(*h_basic.shape[:2], -1))  # (b, s, n)

        # Non heros
        nh_basic = F.relu(self.affine_unit_basic_stats(nonheroes))
        nh_embedding = self.affine_nonheros(nh_basic.view(*nh_basic.shape[:2], -1))  # (b, s, n)

        # Towers
        t_basic = F.relu(self.affine_unit_basic_stats(towers))
        t_embedding = self.affine_towers(t_basic.view(*t_basic.shape[:2], -1))  # (b, s, n)

        states = torch.cat([h_embedding, nh_embedding, t_embedding], dim=-1)

        # 1st layer
        w1 = torch.abs(self.w1(states))
        w1 = w1.view(*w1.shape[:2], self.num_agents, -1)
        b1 = self.b1(states)
        b1 = b1.view(*b1.shape[:2], 1, -1)
        middle = F.elu(qs.view(*qs.shape[:2], 1, self.num_agents) @ w1 + b1)

        # 2nd layer
        w2 = torch.abs(self.w2(states))
        w2 = w2.view(*w2.shape[:2], -1, 1)
        b2 = self.b2(states)
        b2 = b2.view(*b2.shape[:2], 1, 1)
        q_tot = middle @ w2 + b2

        return q_tot
