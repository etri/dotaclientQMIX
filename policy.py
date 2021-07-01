import logging
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

eps = np.finfo(np.float32).eps.item()

TICKS_PER_OBSERVATION = 15  # HACK!
# N_DELAY_ENUMS = 5  # HACK!

REWARD_KEYS = ['enemy', 'win', 'xp', 'hp', 'kills', 'death', 'lh', 'denies', 'tower_hp', 'mana']


class MaskedCategorical():
    def __init__(self, log_probs, mask):
        self.log_probs = log_probs
        self.mask = mask
        self.masked_probs = torch.exp(log_probs).clone()
        self.masked_probs[~mask] = 0.
        # print('self.masked_probs=', self.masked_probs)

    def sample(self):
        return torch.multinomial(self.masked_probs[-1], num_samples=1)


class EpsilonGreedy():
    def __init__(self, logits, mask, epsilon=.1):
        _logits = logits.detach().clone()
        _logits[mask != 1] = -np.Inf
        self.max_actions = _logits.max(dim=-1)[1]
        self.avail_actions = Categorical(mask.float())
        self.epsilon = epsilon

    def sample(self, test=False):
        return self.max_actions if test or np.random.uniform(
        ) > self.epsilon else self.avail_actions.sample().long()


class Policy(nn.Module):

    TICKS_PER_SECOND = 30
    MAX_MOVE_SPEED = 550
    MAX_MOVE_IN_OBS = (MAX_MOVE_SPEED / TICKS_PER_SECOND) * TICKS_PER_OBSERVATION
    N_MOVE_ENUMS = 9
    MOVE_ENUMS = np.arange(N_MOVE_ENUMS, dtype=np.float32) - int(N_MOVE_ENUMS / 2)
    MOVE_ENUMS *= MAX_MOVE_IN_OBS / (N_MOVE_ENUMS - 1) * 2
    OBSERVATIONS_PER_SECOND = TICKS_PER_SECOND / TICKS_PER_OBSERVATION
    MAX_TARGET_UNITS = (5 + 16 + 3) * 2 - 1
    INPUT_KEYS = [
        'env', 'this_hero', 'allied_heroes', 'enemy_heroes', 'allied_nonheroes', 'enemy_nonheroes',
        'allied_towers', 'enemy_towers'
    ]

    _ACTION_CLASS = ['nothing', 'move', 'attack', 'ability']
    _ACTION_CLASS_SIZE = [1, N_MOVE_ENUMS**2, MAX_TARGET_UNITS, 3]
    NUM_ACTIONS = sum(_ACTION_CLASS_SIZE)
    _ACTION_CLASS_START_INDEX = reduce(lambda acc, cur: acc + [acc[-1] + cur], \
        _ACTION_CLASS_SIZE, [0])[:-1]
    ACTION_CLASS_RANGE = {
        cls: range(i, i + sz)
        for cls, i, sz in zip(_ACTION_CLASS, _ACTION_CLASS_START_INDEX, _ACTION_CLASS_SIZE)
    }

    def __init__(self):
        super().__init__()

        self.affine_env = nn.Linear(3, 128)

        self.affine_unit_basic_stats = nn.Linear(12, 128)

        self.affine_unit_ah = nn.Linear(128, 128)
        self.affine_unit_eh = nn.Linear(128, 128)
        self.affine_unit_anh = nn.Linear(128, 128)
        self.affine_unit_enh = nn.Linear(128, 128)
        self.affine_unit_ath = nn.Linear(128, 128)
        self.affine_unit_eth = nn.Linear(128, 128)

        self.affine_pre_rnn = nn.Linear(1024, 256)
        self.rnn = nn.GRU(input_size=256, hidden_size=256, num_layers=1, batch_first=True)

        # Heads
        self.affine_nothing = nn.Linear(256, 1)
        self.affine_move = nn.Linear(256, self.N_MOVE_ENUMS**2)
        self.affine_unit_attention = nn.Linear(256, 128)
        self.affine_ability = nn.Linear(256, 3)

    def init_hidden(self):
        return torch.zeros([1, 1, 256], dtype=torch.float32)

    def single(self, hidden, **kwargs):
        """Inputs a single element of a sequence."""
        for k in kwargs:
            kwargs[k] = kwargs[k].unsqueeze(0).unsqueeze(0)
        return self.__call__(**kwargs, hidden=hidden)

    def sequence(self, hidden, **kwargs):
        """Inputs a single sequence."""
        for k in kwargs:
            kwargs[k] = kwargs[k].unsqueeze(0)
        return self.__call__(**kwargs, hidden=hidden)

    def forward(self, env, this_hero, allied_heroes, enemy_heroes, allied_nonheroes,
                enemy_nonheroes, allied_towers, enemy_towers, hidden):
        """Input as batch."""

        # Environment.
        env = F.relu(self.affine_env(env))  # (b, s, n)

        # This Hero.
        h_basic = F.relu(self.affine_unit_basic_stats(this_hero))
        h_embedding = self.affine_unit_ah(h_basic)  # (b, s, 1, n)

        # Allied Heroes.
        ah_basic = F.relu(self.affine_unit_basic_stats(allied_heroes))
        ah_embedding = self.affine_unit_ah(ah_basic)  # (b, s, units, n)
        ah_embedding_max, _ = torch.max(ah_embedding, dim=2)  # (b, s, n)

        # Enemy Heroes.
        eh_basic = F.relu(self.affine_unit_basic_stats(enemy_heroes))
        eh_embedding = self.affine_unit_eh(eh_basic)  # (b, s, units, n)
        eh_embedding_max, _ = torch.max(eh_embedding, dim=2)  # (b, s, n)

        # Allied Non-Heroes.
        anh_basic = F.relu(self.affine_unit_basic_stats(allied_nonheroes))
        anh_embedding = self.affine_unit_anh(anh_basic)  # (b, s, units, n)
        anh_embedding_max, _ = torch.max(anh_embedding, dim=2)  # (b, s, n)

        # Enemy Non-Heroes.
        enh_basic = F.relu(self.affine_unit_basic_stats(enemy_nonheroes))
        enh_embedding = self.affine_unit_enh(enh_basic)  # (b, s, units, n)
        enh_embedding_max, _ = torch.max(enh_embedding, dim=2)  # (b, s, n)

        # Allied Towers.
        ath_basic = F.relu(self.affine_unit_basic_stats(allied_towers))
        ath_embedding = self.affine_unit_ath(ath_basic)  # (b, s, units, n)
        ath_embedding_max, _ = torch.max(ath_embedding, dim=2)  # (b, s, n)

        # Enemy Towers.
        eth_basic = F.relu(self.affine_unit_basic_stats(enemy_towers))
        eth_embedding = self.affine_unit_eth(eth_basic)  # (b, s, units, n)
        eth_embedding_max, _ = torch.max(enh_embedding, dim=2)  # (b, s, n)

        # Create the full unit embedding
        unit_embedding = torch.cat((ah_embedding, eh_embedding, anh_embedding, enh_embedding,
                                    ath_embedding, eth_embedding),
                                   dim=2)  # (b, s, units, n)
        unit_embedding = torch.transpose(unit_embedding, dim0=3,
                                         dim1=2)  # (b, s, units, n) -> (b, s, n, units)

        # Combine for RNN.
        x = torch.cat((env, h_embedding.squeeze(2), ah_embedding_max, eh_embedding_max,
                       anh_embedding_max, enh_embedding_max, ath_embedding_max, eth_embedding_max),
                      dim=2)  # (b, s, n)

        x = F.relu(self.affine_pre_rnn(x))  # (b, s, n)

        # RNN
        x, hidden = self.rnn(x, hidden)  # (b, s, n)

        # Unit attention.
        unit_attention = self.affine_unit_attention(x)  # (b, s, n)
        unit_attention = unit_attention.unsqueeze(2)  # (b, s, n) ->  (b, s, 1, n)

        # Output
        action_score_nothing = self.affine_nothing(x)
        action_scores_move = self.affine_move(x)
        action_target_unit = torch.matmul(
            unit_attention, unit_embedding)  # (b, s, 1, n) * (b, s, n, units) = (b, s, 1, units)
        action_target_unit = action_target_unit.squeeze(2)  # (b, s, 1, units) -> (b, s, units)
        action_ability = self.affine_ability(x)
        logits = torch.cat(
            (action_score_nothing, action_scores_move, action_target_unit, action_ability), dim=-1)

        # Return
        return logits, hidden

    @classmethod
    def masked_softmax(cls, logits, mask, dim=2):
        """Returns log-probabilities."""
        exp = torch.exp(logits)
        masked_exp = exp.clone()
        masked_exp[~mask] = 0.
        masked_sumexp = masked_exp.sum(dim, keepdim=True)
        logsumexp = torch.log(masked_sumexp)
        log_probs = logits - logsumexp
        return log_probs

    @classmethod
    def flatten_selections(cls, action_dict):
        t = torch.zeros(cls.NUM_ACTIONS, dtype=torch.uint8)
        t[action_dict['action']] = 1
        return t

    @classmethod
    def sample_action(cls, logits, mask):
        return EpsilonGreedy(logits=logits, mask=mask).sample()

    @classmethod
    def select_action(cls, logits, mask):
        act = cls.sample_action(logits, mask)
        action_dict = {'action': act}

        if act in cls.ACTION_CLASS_RANGE['nothing']:
            action_dict['enum'] = 0  # Nothing
        elif act in cls.ACTION_CLASS_RANGE['move']:
            action_dict['enum'] = 1  # Move
            s = cls.ACTION_CLASS_RANGE['move'][0]
            action_dict['x'] = (act - s) // cls.N_MOVE_ENUMS
            action_dict['y'] = (act - s) % cls.N_MOVE_ENUMS
        elif act in cls.ACTION_CLASS_RANGE['attack']:
            action_dict['enum'] = 2  # Attack
            action_dict['target_unit'] = act - cls.ACTION_CLASS_RANGE['attack'][0]
        elif act in cls.ACTION_CLASS_RANGE['ability']:
            action_dict['enum'] = 3  # Ability
            action_dict['ability'] = act - cls.ACTION_CLASS_RANGE['ability'][0]
        else:
            ValueError("Invalid Action Selection.")

        return action_dict

    @staticmethod
    def ability_available(ability):
        return ability.is_activated and ability.level > 0 and ability.cooldown_remaining == 0 \
                and ability.is_fully_castable

    @classmethod
    def action_mask(cls, player_unit, unit_handles):
        """Mask the head with possible actions."""
        if not player_unit.is_alive:
            # Dead player means it can only do the NoOp.
            mask = torch.zeros(1, 1, cls.NUM_ACTIONS).byte()
            mask[0] = 1
            return mask

        mask = torch.ones(1, 1, cls.NUM_ACTIONS).byte()
        for ability in player_unit.abilities:
            if ability.slot >= 3:
                continue
            # Note: `is_fully_castable` implies there is mana for it.
            # Note: `is_in_ability_phase` means it is currently doing an ability.
            if not cls.ability_available(ability):
                # Can't use ability
                mask_idx = cls.ACTION_CLASS_RANGE['ability'][0] + ability.slot
                mask[0, 0, mask_idx] = 0

        attack_idx = cls.ACTION_CLASS_RANGE['attack']
        mask[0, 0, attack_idx[0]:attack_idx[-1] + 1] = unit_handles != -1

        return mask
