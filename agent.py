from collections import Counter
from collections import deque
from datetime import datetime
from functools import reduce
from itertools import chain
from pprint import pformat
import argparse
import asyncio
import io
import logging
import math
import pickle
import random
import time
import traceback

from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from dotaservice.protos.dota_shared_enums_pb2 import DOTA_GAMEMODE_1V1MID
from dotaservice.protos.DotaService_grpc import DotaServiceStub
from dotaservice.protos.DotaService_pb2 import Actions
from dotaservice.protos.DotaService_pb2 import Empty
from dotaservice.protos.DotaService_pb2 import GameConfig
from dotaservice.protos.DotaService_pb2 import HostMode
from dotaservice.protos.DotaService_pb2 import ObserveConfig
from dotaservice.protos.DotaService_pb2 import Status
from dotaservice.protos.DotaService_pb2 import TEAM_DIRE, TEAM_RADIANT, HeroPick
from dotaservice.protos.DotaService_pb2 import HERO_CONTROL_MODE_IDLE, HERO_CONTROL_MODE_DEFAULT, HERO_CONTROL_MODE_CONTROLLED
from dotaservice.protos.DotaService_pb2 import NPC_DOTA_HERO_NEVERMORE, NPC_DOTA_HERO_SNIPER

from grpclib.client import Channel
from tensorboardX import SummaryWriter
import aioamqp
import numpy as np
import png
import torch

import pika  # TODO(tzaman): remove in favour of aioamqp

from policy import Policy
from policy import REWARD_KEYS

torch.set_grad_enabled(False)

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

# Static variables
OPPOSITE_TEAM = {TEAM_DIRE: TEAM_RADIANT, TEAM_RADIANT: TEAM_DIRE}

TICKS_PER_OBSERVATION = 15
N_DELAY_ENUMS = 5
HOST_TIMESCALE = 10
N_GAMES = 10000000
MAX_AGE_WEIGHTSTORE = 64
MAP_HALF_WIDTH = 7000.  # Approximate size of the half of the map.

HOST_MODE = HostMode.Value('HOST_MODE_DEDICATED')

DOTASERVICE_HOST = '127.0.0.1'

# RMQ
EXPERIENCE_QUEUE_NAME = 'experience'
MODEL_EXCHANGE_NAME = 'model'

# Derivates.
DELAY_ENUM_TO_STEP = math.floor(TICKS_PER_OBSERVATION / N_DELAY_ENUMS)

xp_to_reach_level = {
    1: 0,
    2: 230,
    3: 600,
    4: 1080,
    5: 1680,
    6: 2300,
    7: 2940,
    8: 3600,
    9: 4280,
    10: 5080,
    11: 5900,
    12: 6740,
    13: 7640,
    14: 8865,
    15: 10115,
    16: 11390,
    17: 12690,
    18: 14015,
    19: 15415,
    20: 16905,
    21: 18405,
    22: 20155,
    23: 22155,
    24: 24405,
    25: 26905
}

writer = None

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_total_xp(level, xp_needed_to_level):
    if level == 25:
        return xp_to_reach_level[level]
    xp_required_for_next_level = xp_to_reach_level[level + 1] - xp_to_reach_level[level]
    missing_xp_for_next_level = (xp_required_for_next_level - xp_needed_to_level)
    return xp_to_reach_level[level] + missing_xp_for_next_level


def get_reward(prev_obs, obs, player_id):
    """Get the reward."""
    unit_init = get_unit(prev_obs, player_id=player_id)
    unit = get_unit(obs, player_id=player_id)
    player_init = get_player(prev_obs, player_id=player_id)
    player = get_player(obs, player_id=player_id)

    # TODO(tzaman): make a nice reward container?
    reward = {key: 0. for key in REWARD_KEYS}

    # XP Reward
    xp_init = get_total_xp(level=unit_init.level, xp_needed_to_level=unit_init.xp_needed_to_level)
    xp = get_total_xp(level=unit.level, xp_needed_to_level=unit.xp_needed_to_level)
    reward['xp'] = (xp - xp_init) * 0.001  # One creep is around 40 xp; 40*0.001=0.04

    # HP and death reward
    if unit_init.is_alive and unit.is_alive:
        hp_rel = unit.health / unit.health_max
        # rel=0 -> 3; rel=0 -> 2; rel=0.5->1.25; rel=1 -> 1.
        low_hp_factor = 1. + (1 - hp_rel)**2
        hp_rel_init = unit_init.health / unit_init.health_max
        reward['hp'] = (hp_rel - hp_rel_init) * low_hp_factor * 0.3
        # NOTE: Fully depleting hp costs: (0 - 1) * (1+(1-0)^2) * 0.2 = - 0.4

        mana_rel = unit.mana / unit.mana_max
        # rel=0 -> 3; rel=0 -> 2; rel=0.5->1.25; rel=1 -> 1.
        low_mana_factor = 1. + (1 - mana_rel)**2
        mana_rel_init = unit_init.mana / unit_init.mana_max
        reward['mana'] = (mana_rel - mana_rel_init) * low_mana_factor * 0.3
        # NOTE: Fully depleting mana costs: (0 - 1) * (1+(1-0)^2) * 0.1 = - 0.2

    # Kill and death rewards
    reward['kills'] = (player.kills - player_init.kills) * 0.4
    reward['death'] = (player.deaths - player_init.deaths) * -0.4

    # Last-hit reward
    lh = unit.last_hits - unit_init.last_hits
    reward['lh'] = lh * 0.1

    # Deny reward
    denies = unit.denies - unit_init.denies
    reward['denies'] = denies * 0.05

    # Tower hp reward
    reward['tower_hp'] = 0.
    for i in range(1, 4):
        mid_tower_init = get_mid_tower(prev_obs, player.team_id, i)
        mid_tower = get_mid_tower(obs, player.team_id, i)
        reward['tower_hp'] += (mid_tower.health - mid_tower_init.health) / mid_tower.health_max

    return reward


class WeightStore:
    def __init__(self, maxlen):
        self.ready = None  # HACK: Will be set to an event
        self.weights = deque(maxlen=maxlen)

        # The latest policy is used as a pointer to the latest and greatest policy. It is updated
        # even while the agents are playing.
        self.latest_policy = Policy()
        self.latest_policy.eval()

    def add(self, version, state_dict):
        # TODO(tzaman): delete old ones
        self.weights.append((version, state_dict))

        # Add to latest policy immediatelly
        self.latest_policy.load_state_dict(state_dict, strict=True)
        self.latest_policy.weight_version = version

    def oldest_weights(self):
        return self.weights[0]

    def latest_weights(self):
        return self.weights[-1]


weight_store = WeightStore(maxlen=MAX_AGE_WEIGHTSTORE)


async def model_callback(channel, body, envelope, properties):
    # TODO(tzaman): add a future so we can wait for first weights
    version = properties.headers['version']
    logger.info("Received new model: version={}, size={}b".format(version, len(body)))
    state_dict = torch.load(io.BytesIO(body), map_location=torch.device(device))
    weight_store.add(version=version, state_dict=state_dict)
    weight_store.ready.set()


async def rmq_connection_error_cb(exception):
    logger.error('rmq_connection_error_cb(exception={})'.format(exception))
    exit(1)


async def setup_model_cb(host, port):
    # TODO(tzaman): setup proper reconnection, see https://github.com/Polyconseil/aioamqp/issues/65#issuecomment-301737344
    logger.info('setup_model_cb(host={}, port={})'.format(host, port))
    transport, protocol = await aioamqp.connect(host=host,
                                                port=port,
                                                on_error=rmq_connection_error_cb,
                                                heartbeat=300)
    channel = await protocol.channel()
    await channel.exchange(exchange_name=MODEL_EXCHANGE_NAME,
                           type_name='x-recent-history',
                           arguments={'x-recent-history-length': 1})
    result = await channel.queue(queue_name='', exclusive=True)
    queue_name = result['queue']
    await channel.queue_bind(exchange_name=MODEL_EXCHANGE_NAME,
                             queue_name=queue_name,
                             routing_key='')
    await channel.basic_consume(model_callback, queue_name=queue_name, no_ack=True)


def get_player(state, player_id):
    for player in state.players:
        if player.player_id == player_id:
            return player
    raise ValueError("hero {} not found in state:\n{}".format(player_id, state))


def get_unit(state, player_id):
    for unit in state.units:
        if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO') \
            and unit.player_id == player_id and not unit.is_illusion:
            return unit
    raise ValueError("unit {} not found in state:\n{}".format(player_id, state))


def get_mid_tower(state, team_id, tower_no):
    for unit in state.units:
        if unit.unit_type == CMsgBotWorldState.UnitType.Value('TOWER') \
            and unit.team_id == team_id \
            and f'tower{tower_no}_mid' in unit.name:
            return unit
    raise ValueError("tower{}_mid not found in state:\n{}".format(tower_no, state))


def is_unit_attacking_unit(unit_attacker, unit_target):
    # Check for a direct attack.
    if unit_attacker.attack_target_handle == unit_target.handle:
        return 1.
    # Go over the incoming projectiles from this unit.
    for projectile in unit_target.incoming_tracking_projectiles:
        if projectile.caster_handle == unit_attacker.handle and projectile.is_attack:
            return 1.
    # Otherwise, the unit is not attacking the target, and there are no incoming projectiles.
    return 0.


def is_invulnerable(unit):
    for mod in unit.modifiers:
        if mod.name == "modifier_invulnerable":
            return True
    return False


class Player:

    MAX_HEROES = 5
    MAX_NONHEROES = 16
    MAX_TOWERS = 3
    MAX_UNITS = MAX_HEROES + MAX_NONHEROES + MAX_TOWERS

    END_STATUS_TO_TEAM = {
        Status.Value('RADIANT_WIN'): TEAM_RADIANT,
        Status.Value('DIRE_WIN'): TEAM_DIRE,
    }

    def __init__(self, game_id, player_id, team_id, hero, use_latest_weights, drawing, validation):
        self.game_id = game_id
        self.player_id = player_id
        self.team_id = team_id
        self.hero = hero
        self.use_latest_weights = use_latest_weights

        self.policy_inputs = []
        self.global_states = []
        self.actions = []
        self.action_masks = []
        self.rewards = []
        self.drawing = drawing
        self.validation = validation

        self.creeps_had_spawned = False
        self.prev_level = 0

        use_synced_weights = use_latest_weights and not self.validation

        if use_synced_weights:
            # This will actually use the latest policy, that is even updated while the agent is playing.
            self.policy = weight_store.latest_policy
        else:  # Use non-synchronized weights
            if self.validation or use_latest_weights:
                # Use the latest weights for validation
                version, state_dict = weight_store.latest_weights()
            else:
                # Use the oldest weights.
                version, state_dict = weight_store.oldest_weights()
            self.policy = Policy()
            self.policy.load_state_dict(state_dict, strict=True)
            self.policy.weight_version = version
            self.policy.eval()  # Set to evaluation mode.

        self.policy.to(device)
        self.hidden = self.policy.init_hidden().to(device)

        logger.info('Player {} using weights version {}'.format(self.player_id,
                                                                self.policy.weight_version))

    def summed_subrewards(self):
        reward_counter = Counter()
        for r in self.rewards:
            reward_counter.update(r)
        return dict(reward_counter)

    def print_reward_summary(self):
        subrewards = self.summed_subrewards()
        reward_sum = sum(subrewards.values())
        logger.info('Player {} reward sum: {:.2f} subrewards:\n{}'.format(
            self.player_id, reward_sum, pformat(subrewards)))

    def process_endstate(self, end_state):
        # The end-state adds rewards to the last reward.
        if not self.rewards:
            return
        if end_state in self.END_STATUS_TO_TEAM.keys():
            if self.team_id == self.END_STATUS_TO_TEAM[end_state]:
                self.rewards[-1]['win'] = 1
            else:
                self.rewards[-1]['win'] = -1
            return

        # Add a negative win reward, because we did not have a clear winner.
        self.rewards[-1]['win'] = -0.25

    @staticmethod
    def pack_list_of_dicts(inputs):
        """Convert the list-of-dicts into a dict with a single tensor per input for the sequence."""
        d = {key: [] for key in inputs[0]}
        for inp in inputs:  # go over steps: (list of dicts)
            for k, v in inp.items():  # go over each input in the step (dict)
                d[k].append(v)

        # Pack it up
        for k, v in d.items():
            # Concatenate together all inputs into a single tensor.
            # We formerly padded this instead of stacking, but that presented issues keeping track
            # of the chosen action ids related to units.
            d[k] = torch.stack(v)
        return d

    @staticmethod
    def pack_rewards(inputs):
        """Pack a list or reward dicts into a dense 2D tensor"""
        t = np.zeros([len(inputs), len(REWARD_KEYS)], dtype=np.float32)
        for i, reward in enumerate(inputs):
            for ir, key in enumerate(REWARD_KEYS):
                t[i, ir] = reward[key]
        return t

    @staticmethod
    def pack_actions(inputs):
        data = [Policy.flatten_selections(inp) for inp in inputs]
        return torch.stack(data)

    @staticmethod
    def pack_masks(inputs):
        # Concatenate over sequence axis and remove batch axis
        return torch.cat(inputs, dim=1).squeeze(0)

    def pack_experience(self):

        # Pack all the policy inputs into dense tensors
        observations = self.pack_list_of_dicts(self.policy_inputs)
        global_states = self.pack_list_of_dicts(self.global_states)
        masks = self.pack_masks(self.action_masks)
        actions = self.pack_actions(self.actions)
        rewards = self.pack_rewards(inputs=self.rewards)

        data = {
            'game_id': self.game_id,
            'team_id': self.team_id,
            'player_id': self.player_id,
            'weight_version': self.policy.weight_version,
            'canvas': self.drawing.canvas,
            'observations': observations,
            'global_states': global_states,
            'masks': masks,
            'actions': actions,
            'rewards': rewards,
        }

        return data

    @property
    def steps_queued(self):
        return len(self.rewards)

    def write_validation(self):
        it = self.policy.weight_version
        writer.add_image('game/canvas', self.drawing.canvas, it, dataformats='HWC')
        writer.add_scalar('game/steps', self.steps_queued, it)
        subrewards = self.summed_subrewards()
        reward_sum = sum(subrewards.values())
        writer.add_scalar('game/rewards_sum', reward_sum, it)
        for key, reward in subrewards.items():
            writer.add_scalar('game/rewards_{}'.format(key), reward, it)
        writer.file_writer.flush()

    def rollout(self):
        logger.info('Player {} rollout, len={}'.format(self.player_id, self.steps_queued))

        if not self.rewards:
            logger.info('nothing to roll out.')
            return

        self.print_reward_summary()

        experience = None
        if self.use_latest_weights:
            experience = self.pack_experience()
        else:
            logger.info('Not using latest weights: not rolling out.')

        # Reset states.
        self.policy_inputs = []
        self.global_states = []
        self.rewards = []
        self.actions = []
        self.action_masks = []

        return experience

    @staticmethod
    def unit_separation(state, team_id):
        # Break apart the full unit-list into specific categories for allied and
        # enemy unit groups of various types so we don't have to repeatedly iterate
        # the full unit-list again.
        allied_heroes = []
        enemy_heroes = []
        allied_nonheroes = []
        enemy_nonheroes = []
        allied_creep = []
        enemy_creep = []
        allied_towers = []
        enemy_towers = []
        for unit in state.units:
            # check if allied or enemy unit
            if unit.team_id == team_id:
                if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO'):
                    # If illusion, treat as creep hero
                    if unit.is_illusion:
                        allied_nonheroes.append(unit)
                    else:
                        allied_heroes.append(unit)
                elif unit.unit_type == CMsgBotWorldState.UnitType.Value('CREEP_HERO'):
                    allied_nonheroes.append(unit)
                elif unit.unit_type == CMsgBotWorldState.UnitType.Value('LANE_CREEP'):
                    allied_creep.append(unit)
                elif unit.unit_type == CMsgBotWorldState.UnitType.Value('TOWER'):
                    if unit.name[-4:] == '_mid':  # Only consider the mid towers for now.
                        allied_towers.append(unit)
            else:
                if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO'):
                    # enemy.is_illusion is always false
                    enemy_heroes.append(unit)
                elif unit.unit_type == CMsgBotWorldState.UnitType.Value('CREEP_HERO'):
                    enemy_nonheroes.append(unit)
                elif unit.unit_type == CMsgBotWorldState.UnitType.Value('LANE_CREEP'):
                    enemy_creep.append(unit)
                elif unit.unit_type == CMsgBotWorldState.UnitType.Value('TOWER'):
                    if unit.name[-4:] == '_mid':  # Only consider the mid towers for now.
                        enemy_towers.append(unit)

        return allied_heroes, enemy_heroes, allied_nonheroes, enemy_nonheroes, \
               allied_creep, enemy_creep, allied_towers, enemy_towers

    @staticmethod
    def unit_matrix(unit_list, hero_unit, include_this_hero=False, max_units=16):
        # We are always inserting an 'zero' unit to make sure the policy doesn't barf
        # We can't just pad this, because we will otherwise lose track of corresponding chosen
        # actions relating to output indices. Even if we would, batching multiple sequences together
        # would then be another error prone nightmare.
        handles = torch.full([max_units], -1, dtype=torch.get_default_dtype())
        m = torch.zeros(max_units, 12)
        i = 0
        for unit in unit_list:
            if unit.is_alive:
                if include_this_hero and unit != hero_unit:
                    continue
                elif not include_this_hero and unit == hero_unit:
                    continue

                if i >= max_units:
                    break
                rel_hp = 1.0 - (unit.health / unit.health_max)
                rel_mana = 0.0
                if unit.mana_max > 0:
                    rel_mana = 1.0 - (unit.mana / unit.mana_max)
                loc_x = unit.location.x / MAP_HALF_WIDTH
                loc_y = unit.location.y / MAP_HALF_WIDTH
                loc_z = (unit.location.z / 512.) - 0.5
                distance_x = (hero_unit.location.x - unit.location.x)
                distance_y = (hero_unit.location.y - unit.location.y)
                distance = math.sqrt(distance_x**2 + distance_y**2)
                norm_distance = (distance / MAP_HALF_WIDTH) - 0.5

                # Get the direction where the unit is facing.
                facing_sin = math.sin(unit.facing * (2 * math.pi) / 360)
                facing_cos = math.cos(unit.facing * (2 * math.pi) / 360)

                if hero_unit.is_alive:
                    # Calculates normalized boolean value [-0.5 or 0.5] of if unit is within
                    # attack range of hero.
                    in_attack_range = float(distance <= hero_unit.attack_range) - 0.5

                    # Calculates normalized boolean value [-0.5 or 0.5] of if that unit
                    # is currently targeting me with right-click attacks.
                    is_attacking_me = float(is_unit_attacking_unit(unit, hero_unit)) - 0.5
                    me_attacking_unit = float(is_unit_attacking_unit(hero_unit, unit)) - 0.5
                else:
                    in_attack_range = -0.5
                    is_attacking_me = -0.5
                    me_attacking_unit = -0.5

                in_ability_phase = -0.5
                for a in unit.abilities:
                    if a.is_in_ability_phase or a.is_channeling:
                        in_ability_phase = 0.5
                        break

                m[i] = (torch.tensor([
                    rel_hp, loc_x, loc_y, loc_z, norm_distance, facing_sin, facing_cos,
                    in_attack_range, is_attacking_me, me_attacking_unit, rel_mana, in_ability_phase
                ]))

                # Because we are currently only attacking, check if these units are valid
                # HACK: Make a nice interface for this, per enum used?
                if unit.is_invulnerable or unit.is_attack_immune or not hero_unit.is_alive:
                    handles[i] = -1
                elif unit.team_id == OPPOSITE_TEAM[
                        hero_unit.team_id] and unit.unit_type == CMsgBotWorldState.UnitType.Value(
                            'TOWER') and unit.anim_activity == 1500:
                    # Enemy tower. Due to a dota bug, the bot API can only attack towers (and move to it)
                    # when they are attacking (activity 1503; stationary is activity 1500)
                    handles[i] = -1
                elif unit.team_id == hero_unit.team_id and unit.unit_type == CMsgBotWorldState.UnitType.Value(
                        'TOWER'):
                    # Its own tower:
                    handles[i] = -1
                elif unit.team_id == hero_unit.team_id and (unit.health / unit.health_max) > 0.5:
                    # Not denyable
                    handles[i] = -1
                else:
                    handles[i] = unit.handle

                i += 1
        return m.to(device), handles.to(device)

    @staticmethod
    def global_unit_key(unit):
        return unit.unit_type, unit.name, unit.location.x, unit.location.y, unit.location.z

    @staticmethod
    def global_unit_matrix(unit_list, enemy_units, enemy_indices, max_units=16):
        units = []
        for unit in unit_list:
            rel_hp = 1.0 - (unit.health / unit.health_max)
            rel_mana = 1.0 - (unit.mana / unit.mana_max) if unit.mana_max > 0 else 0.0
            loc_x = unit.location.x / MAP_HALF_WIDTH
            loc_y = unit.location.y / MAP_HALF_WIDTH
            loc_z = (unit.location.z / 512.) - 0.5
            facing_sin = math.sin(unit.facing * (2 * math.pi) / 360)
            facing_cos = math.cos(unit.facing * (2 * math.pi) / 360)

            in_attack_range_list = [-0.5] * Player.MAX_UNITS
            is_attacking_me_list = [-0.5] * Player.MAX_UNITS
            me_attacking_unit_list = [-0.5] * Player.MAX_UNITS
            for enemy in enemy_units:
                enemy_idx = enemy_indices[Player.global_unit_key(enemy)]

                distance_x = (unit.location.x - enemy.location.x)
                distance_y = (unit.location.y - enemy.location.y)
                distance = math.sqrt(distance_x**2 + distance_y**2)
                if distance <= unit.attack_range:
                    in_attack_range_list[enemy_idx] = 0.5

                if is_unit_attacking_unit(enemy, unit):
                    is_attacking_me_list[enemy_idx] = 0.5
                if is_unit_attacking_unit(unit, enemy):
                    me_attacking_unit_list[enemy_idx] = 0.5

            in_ability_phase = -0.5
            for a in unit.abilities:
                if a.is_in_ability_phase or a.is_channeling:
                    in_ability_phase = 0.5
                    break

            units.append([
                rel_hp, loc_x, loc_y, loc_z, facing_sin, facing_cos, *in_attack_range_list,
                *is_attacking_me_list, *me_attacking_unit_list, rel_mana, in_ability_phase
            ])

        assert len(units) <= max_units, len(units)
        unit_tensor = torch.tensor(units).detach() if units \
                      else torch.zeros(max_units, 8 + 3 * Player.MAX_UNITS).detach()
        if len(unit_tensor) < max_units:
            unit_tensor = torch.nn.functional.pad(unit_tensor,
                                                  (0, 0, 0, max_units - len(unit_tensor)))

        return unit_tensor

    def select_action(self, world_state, hero_unit):
        dota_time_norm = world_state.dota_time / 1200.  # Normalize by 20 minutes
        creepwave_sin = math.sin(world_state.dota_time * (2. * math.pi) / 60)
        team_float = -.2 if self.team_id == TEAM_DIRE else .2

        env_state = torch.Tensor([dota_time_norm, creepwave_sin, team_float]).to(device)

        # Separate units into unit-type groups for both teams
        # The goal is to iterate only once through the entire unit list
        # in the provided world-state protobuf and for further filtering
        # only iterate across the unit-type specific list of interest.
        ah, eh, anh, enh, ac, ec, at, et = self.unit_separation(world_state, hero_unit.team_id)

        # Process units into Tensors & Handles
        this_hero, _ = self.unit_matrix(
            unit_list=ah,
            hero_unit=hero_unit,
            include_this_hero=True,
            max_units=1,
        )

        allied_heroes, allied_hero_handles = self.unit_matrix(
            unit_list=ah,
            hero_unit=hero_unit,
            include_this_hero=False,
            max_units=self.MAX_HEROES - 1,
        )

        enemy_heroes, enemy_hero_handles = self.unit_matrix(
            unit_list=eh,
            hero_unit=hero_unit,
            max_units=self.MAX_HEROES,
        )

        allied_nonheroes, allied_nonhero_handles = self.unit_matrix(
            unit_list=[*anh, *ac],
            hero_unit=hero_unit,
            max_units=self.MAX_NONHEROES,
        )

        enemy_nonheroes, enemy_nonhero_handles = self.unit_matrix(
            unit_list=[*enh, *ec],
            hero_unit=hero_unit,
            max_units=self.MAX_NONHEROES,
        )

        allied_towers, allied_tower_handles = self.unit_matrix(
            unit_list=at,
            hero_unit=hero_unit,
            max_units=self.MAX_TOWERS,
        )

        enemy_towers, enemy_tower_handles = self.unit_matrix(
            unit_list=et,
            hero_unit=hero_unit,
            max_units=self.MAX_TOWERS,
        )

        unit_handles = torch.cat([
            allied_hero_handles, enemy_hero_handles, allied_nonhero_handles, enemy_nonhero_handles,
            allied_tower_handles, enemy_tower_handles
        ])

        if not self.creeps_had_spawned and world_state.dota_time > 0.:
            # Check that creeps have spawned. See dotaclient/issues/15.
            # TODO(tzaman): this should be handled by DotaService.
            # self.creeps_had_spawned = bool((allied_nonhero_handles != -1).any())
            self.creeps_had_spawned = len(ac) > 0
            if not self.creeps_had_spawned:
                raise ValueError('Creeps have not spawned at timestep {}'.format(
                    world_state.dota_time))

        policy_input = {
            'env': env_state,
            'this_hero': this_hero,
            'allied_heroes': allied_heroes,
            'enemy_heroes': enemy_heroes,
            'allied_nonheroes': allied_nonheroes,
            'enemy_nonheroes': enemy_nonheroes,
            'allied_towers': allied_towers,
            'enemy_towers': enemy_towers,
        }

        logger.debug('policy_input:\n' + pformat(policy_input))

        logits, self.hidden = self.policy.single(**policy_input, hidden=self.hidden)

        logger.debug('logits:\n' + pformat(logits))

        # Get valid actions. This mask contains all viable actions.
        action_mask = Policy.action_mask(player_unit=hero_unit, unit_handles=unit_handles)
        logger.debug('action_mask:\n' + pformat(action_mask))

        # From the logits and the mask, select the actions.
        action_dict = Policy.select_action(logits=logits, mask=action_mask)
        logger.debug('action_dict:\n' + pformat(action_dict))

        return policy_input, action_dict, action_mask, unit_handles

    def action_to_pb(self, action_dict, state, unit_handles):
        # TODO(tzaman): Recrease the scope of this function. Make it a converter only.
        hero_unit = get_unit(state, player_id=self.player_id)
        action_pb = CMsgBotWorldState.Action()
        action_pb.actionDelay = 0  # action_dict['delay'] * DELAY_ENUM_TO_STEP
        action_enum = action_dict['enum']

        if action_enum == 0:
            action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')
        elif action_enum == 1:
            action_pb.actionType = CMsgBotWorldState.Action.Type.Value(
                'DOTA_UNIT_ORDER_MOVE_DIRECTLY')
            m = CMsgBotWorldState.Action.MoveToLocation()
            hero_location = hero_unit.location
            m.location.x = hero_location.x + Policy.MOVE_ENUMS[action_dict['x']]
            m.location.y = hero_location.y + Policy.MOVE_ENUMS[action_dict['y']]
            m.location.z = 0
            action_pb.moveDirectly.CopyFrom(m)
        elif action_enum == 2:
            action_pb.actionType = CMsgBotWorldState.Action.Type.Value(
                'DOTA_UNIT_ORDER_ATTACK_TARGET')
            m = CMsgBotWorldState.Action.AttackTarget()
            if 'target_unit' in action_dict:
                m.target = unit_handles[action_dict['target_unit']]
            else:
                m.target = -1
            m.once = True
            action_pb.attackTarget.CopyFrom(m)
        elif action_enum == 3:
            action_pb = CMsgBotWorldState.Action()
            action_pb.actionType = CMsgBotWorldState.Action.Type.Value(
                'DOTA_UNIT_ORDER_CAST_NO_TARGET')
            action_pb.cast.abilitySlot = action_dict['ability']
        else:
            raise ValueError("unknown action {}".format(action_enum))
        action_pb.player = self.player_id
        return action_pb

    def train_ability(self, hero_unit):
        # Check if we leveled up
        leveled_up = hero_unit.level > self.prev_level
        if leveled_up:
            self.prev_level = hero_unit.level
            # Just try to level up the first ability.
            action_pb = CMsgBotWorldState.Action()
            action_pb.actionType = CMsgBotWorldState.Action.Type.Value(
                'DOTA_UNIT_ORDER_TRAIN_ABILITY')
            action_pb.player = self.player_id
            action_pb.trainAbility.ability = "nevermore_shadowraze1"
            return action_pb
        return None

    def obs_to_actions(self, obs):
        actions = []
        hero_unit = get_unit(state=obs, player_id=self.player_id)

        policy_input, action_dict, action_mask, unit_handles = self.select_action(
            world_state=obs,
            hero_unit=hero_unit,
        )

        self.policy_inputs.append(policy_input)
        self.actions.append(action_dict)
        self.action_masks.append(action_mask)
        logger.debug('action:\n' + pformat(action_dict))

        action_pb = self.action_to_pb(action_dict=action_dict, state=obs, unit_handles=unit_handles)
        actions.append(action_pb)

        level_pb = self.train_ability(hero_unit)
        if level_pb is not None:
            actions.append(level_pb)

        return actions

    def compute_reward(self, prev_obs, obs):
        # Draw.
        self.drawing.step(state=obs, team_id=self.team_id, player_id=self.player_id)

        reward = get_reward(prev_obs=prev_obs, obs=obs, player_id=self.player_id)
        self.rewards.append(reward)


class Drawing:

    TEAM_COLORS = {
        TEAM_DIRE: [[255, 0, 0], [229, 0, 0], [255, 77, 77], [255, 128, 0], [255, 0, 128]],
        TEAM_RADIANT: [[0, 212, 89], [0, 182, 67], [0, 159, 59], [0, 131, 48], [0, 111, 2]]
    }

    def __init__(self, size=256):
        # Notice the shape is in (H, W, C)
        self.size = size
        self.sizeh = self.size / 2.
        self.canvas = np.ones((self.size, self.size, 3), dtype=np.uint8) * 255
        self.ratio = self.sizeh / (8000.)

    def normalize_location(self, l):
        return int((l.x * self.ratio) + self.sizeh), int(self.size - (l.y * self.ratio) -
                                                         self.sizeh)

    def step(self, state, team_id, player_id):
        for unit in state.units:
            if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO') \
                and unit.player_id == player_id:
                x, y = self.normalize_location(l=unit.location)
                color = player_id - 5 if player_id >= 5 else player_id
                self.canvas[y, x] = self.TEAM_COLORS[team_id][color]

    def save(self, stem):
        png.from_array(self.canvas, 'RGB').save('logs/{}.png'.format(stem))


class Game:

    _UNIT_CLASS = ['heroes', 'nonheroes', 'towers']
    _UNIT_CLASS_SIZE = [Player.MAX_HEROES, Player.MAX_NONHEROES, Player.MAX_TOWERS]
    _UNIT_CLASS_START_IDX = reduce(lambda acc, cur: acc + [acc[-1] + cur], \
        _UNIT_CLASS_SIZE, [0])[:-1]
    UNIT_CLASS_IDX_RANGE = {
        cls: range(i, i + sz)
        for cls, i, sz in zip(_UNIT_CLASS, _UNIT_CLASS_START_IDX, _UNIT_CLASS_SIZE)
    }

    ENV_RETRY_DELAY = 15

    def __init__(self, dota_service, experience_channel, max_rollout_size, max_dota_time,
                 latest_weights_prob, validation):
        self.dota_service = dota_service
        self.experience_channel = experience_channel
        self.max_rollout_size = max_rollout_size
        self.max_dota_time = max_dota_time
        self.latest_weights_prob = latest_weights_prob
        self.validation = validation

    async def play(self, config, game_id):
        logger.info('Starting game.')

        # Use the latest weights by default.
        use_latest_weights = {TEAM_RADIANT: True, TEAM_DIRE: True}
        if random.random() > self.latest_weights_prob:
            # Randomly pick the ream that will use the old weights.
            old_model_team = random.choice([TEAM_RADIANT, TEAM_DIRE])
            use_latest_weights[old_model_team] = False

        drawing = Drawing(
        )  # TODO(tzaman): drawing should include include what's visible to the player

        # Reset and obtain the initial observation. This dictates who we are controlling,
        # this is done before the player definition, because there might be humand playing
        # that take up bot positions.
        response = await asyncio.wait_for(self.dota_service.reset(config), timeout=120)

        player_request = config.hero_picks
        players_response = response.players  # Lists all human and bot players.
        players = {TEAM_RADIANT: [], TEAM_DIRE: []}
        for p_req, p_res in zip(player_request, players_response):
            assert p_req.team_id == p_res.team_id  # TODO(tzaman): more tests?
            if p_res.is_bot and p_req.control_mode == HERO_CONTROL_MODE_CONTROLLED:
                player = Player(
                    game_id=game_id,
                    player_id=p_res.id,
                    team_id=p_res.team_id,
                    hero=p_res.hero,
                    use_latest_weights=use_latest_weights[p_res.team_id],
                    drawing=drawing,
                    validation=self.validation,
                )
                players[p_res.team_id].append(player)

        prev_obs = {
            TEAM_RADIANT: response.world_state_radiant,
            TEAM_DIRE: response.world_state_dire,
        }
        done = False
        step = 0
        dota_time = -float('Inf')
        end_state = None
        while dota_time < self.max_dota_time:
            reward_sum_step = {TEAM_RADIANT: 0, TEAM_DIRE: 0}
            for team_id in [TEAM_RADIANT, TEAM_DIRE]:
                logger.debug('\ndota_time={:.2f}, team={}'.format(dota_time, team_id))

                response = await self.dota_service.observe(ObserveConfig(team_id=team_id))
                if response.status != Status.Value('OK'):
                    end_state = response.status
                    done = True
                    break
                obs = response.world_state
                dota_time = obs.dota_time

                # We not loop over each player in this team and get each players action.
                actions = []
                for player in players[team_id]:
                    player.compute_reward(prev_obs=prev_obs[team_id], obs=obs)
                    reward_sum_step[team_id] += sum(player.rewards[-1].values())
                    with torch.no_grad():
                        actions_player = player.obs_to_actions(obs=obs)
                    actions.extend(actions_player)

                actions_pb = CMsgBotWorldState.Actions(actions=actions)
                actions_pb.dota_time = obs.dota_time

                _ = await self.dota_service.act(Actions(actions=actions_pb, team_id=team_id))

                prev_obs[team_id] = obs

            if done:
                if team_id == TEAM_DIRE:
                    # This game finished abnormally
                    logger.info(f'This game finished abnormally: {end_state}')
                    for player in players[TEAM_RADIANT]:
                        del player.policy_inputs[-1]
                        del player.action_masks[-1]
                        del player.actions[-1]
                        del player.rewards[-1]
                break

            if not self.validation:
                global_state = self.generate_global_state(prev_obs)

                to_rollout = {TEAM_RADIANT: [], TEAM_DIRE: []}
                for team_id in [TEAM_RADIANT, TEAM_DIRE]:
                    for player in players[team_id]:
                        # Subtract each other's rewards for zero-sum games
                        player.rewards[-1]['enemy'] = -reward_sum_step[OPPOSITE_TEAM[team_id]]

                        player.global_states.append({
                            key: torch.cat([
                                global_state[team_id][key],
                                global_state[OPPOSITE_TEAM[team_id]][key]
                            ])
                            for key in global_state[team_id]
                        })

                        if player.steps_queued > 0 and player.steps_queued % self.max_rollout_size == 0:
                            to_rollout[team_id].append(player)
                await self.rollout(to_rollout)

        drawing.save(stem=game_id)  # HACK

        # Finish (e.g. final rollout or send validation metrics).
        await self.finish(players, end_state)

        # TODO(tzaman): the worldstate ends when game is over. the worldstate doesn't have info
        # about who won the game: so we need to get info from that somehow

        logger.info('Game finished.')

    async def finish(self, players, end_state):
        for player in [*players[TEAM_RADIANT], *players[TEAM_DIRE]]:
            player.process_endstate(end_state)
            if self.validation:
                player.write_validation()
        if not self.validation:
            await self.rollout(players)

    async def rollout(self, players):
        for team_id in [TEAM_RADIANT, TEAM_DIRE]:
            team_experience = []
            for player in players[team_id]:
                experience = player.rollout()
                if experience:
                    team_experience.append(experience)
            if team_experience:
                self._send_experience_rmq(team_experience)

    def _send_experience_rmq(self, team_experience):
        logger.debug('_send_experience_rmq')

        team_data = pickle.dumps(team_experience)
        self.experience_channel.basic_publish(exchange='',
                                              routing_key=EXPERIENCE_QUEUE_NAME,
                                              body=team_data)

    @staticmethod
    def generate_global_state(observations):
        global_state, ally_units, enemy_units, enemy_indices = {}, {}, {}, {}
        for team_id, world_state in observations.items():
            ah, eh, anh, enh, ac, ec, at, et = Player.unit_separation(world_state, team_id)
            ally_units[team_id] = {
                'heroes': [u for u in ah if u.is_alive],
                'nonheroes': [u for u in [*anh, *ac] if u.is_alive],
                'towers': [u for u in at if u.is_alive]
            }
            enemy_units[team_id] = [u for u in [*eh, *enh, *ec, *et] if u.is_alive]

            unit_idx = {}
            for key, unit_list in ally_units[team_id].items():
                start_idx = Game.UNIT_CLASS_IDX_RANGE[key][0]
                for i, unit in enumerate(unit_list):
                    unit_idx[Player.global_unit_key(unit)] = start_idx + i
            enemy_indices[OPPOSITE_TEAM[team_id]] = unit_idx

        for team_id, unit_map in ally_units.items():
            global_state[team_id] = {}
            for key, unit_list in unit_map.items():
                global_state[team_id][key] = \
                    Player.global_unit_matrix(unit_list,
                                              enemy_units[team_id],
                                              enemy_indices[team_id],
                                              max_units=len(Game.UNIT_CLASS_IDX_RANGE[key]))

        return global_state


async def main(rmq_host, rmq_port, dota_port, max_rollout_size, max_dota_time, latest_weights_prob,
               validation, log_dir, num_agents, self_play):
    logger.info('main(rmq_host={}, rmq_port={})'.format(rmq_host, rmq_port))

    # RMQ
    rmq_connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=rmq_host, port=rmq_port, heartbeat=300))
    experience_channel = rmq_connection.channel()
    experience_channel.queue_declare(queue=EXPERIENCE_QUEUE_NAME)

    weight_store.ready = asyncio.Event(loop=asyncio.get_event_loop())

    global writer
    writer = SummaryWriter(log_dir=log_dir)

    # Set up the model callback.
    await setup_model_cb(host=rmq_host, port=rmq_port)

    # Wait for the first model weight to come in.
    await weight_store.ready.wait()

    # Connect to dota
    channel_dota = Channel(DOTASERVICE_HOST, dota_port, loop=asyncio.get_event_loop())
    dota_service = DotaServiceStub(channel_dota)

    game = Game(dota_service=dota_service,
                experience_channel=experience_channel,
                max_rollout_size=max_rollout_size,
                max_dota_time=max_dota_time,
                latest_weights_prob=latest_weights_prob,
                validation=validation)

    for i in range(0, N_GAMES):
        logger.info('=== Starting Game {}.'.format(i))
        game_id = str(datetime.now().strftime('%b%d_%H-%M-%S'))

        if validation:
            config = get_1v1_bot_vs_default_config(validation_team=validation)
        else:
            config = get_config(num_agents, self_play)

        try:
            await game.play(config=config, game_id=game_id)
        except:
            traceback.print_exc()
            return

    channel_dota.close()


def get_config(num_agents, self_play):
    modes = {TEAM_RADIANT: HERO_CONTROL_MODE_CONTROLLED, TEAM_DIRE: HERO_CONTROL_MODE_CONTROLLED}
    if not self_play:
        # Choose a scripted bot team
        modes[np.random.choice([TEAM_RADIANT, TEAM_DIRE])] = HERO_CONTROL_MODE_DEFAULT
    hero_picks = chain.from_iterable([
        chain.from_iterable([[
            HeroPick(team_id=team_id,
                     hero_id=NPC_DOTA_HERO_NEVERMORE +
                     (i if modes[team_id] == HERO_CONTROL_MODE_DEFAULT else 0),
                     control_mode=modes[team_id]) for i in range(num_agents)
        ],
                             [
                                 HeroPick(team_id=team_id,
                                          hero_id=NPC_DOTA_HERO_SNIPER,
                                          control_mode=HERO_CONTROL_MODE_IDLE)
                                 for _ in range(5 - num_agents)
                             ]]) for team_id in [TEAM_RADIANT, TEAM_DIRE]
    ])

    return GameConfig(
        ticks_per_observation=TICKS_PER_OBSERVATION,
        host_timescale=HOST_TIMESCALE,
        host_mode=HOST_MODE,
        game_mode=DOTA_GAMEMODE_1V1MID,
        hero_picks=hero_picks,
    )


def get_1v1_bot_vs_default_config(validation_team):
    # Randomize the mode between dire and radiant players.
    if validation_team == 'RADIANT':
        modes = [HERO_CONTROL_MODE_CONTROLLED, HERO_CONTROL_MODE_DEFAULT]
    else:
        modes = [HERO_CONTROL_MODE_DEFAULT, HERO_CONTROL_MODE_CONTROLLED]
    hero_picks = [
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_NEVERMORE, control_mode=modes[0]),
        HeroPick(team_id=TEAM_RADIANT,
                 hero_id=NPC_DOTA_HERO_SNIPER,
                 control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_RADIANT,
                 hero_id=NPC_DOTA_HERO_SNIPER,
                 control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_RADIANT,
                 hero_id=NPC_DOTA_HERO_SNIPER,
                 control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_RADIANT,
                 hero_id=NPC_DOTA_HERO_SNIPER,
                 control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_NEVERMORE, control_mode=modes[1]),
        HeroPick(team_id=TEAM_DIRE,
                 hero_id=NPC_DOTA_HERO_SNIPER,
                 control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE,
                 hero_id=NPC_DOTA_HERO_SNIPER,
                 control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE,
                 hero_id=NPC_DOTA_HERO_SNIPER,
                 control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE,
                 hero_id=NPC_DOTA_HERO_SNIPER,
                 control_mode=HERO_CONTROL_MODE_IDLE),
    ]
    return GameConfig(
        ticks_per_observation=TICKS_PER_OBSERVATION,
        host_timescale=HOST_TIMESCALE,
        host_mode=HOST_MODE,
        game_mode=DOTA_GAMEMODE_1V1MID,
        hero_picks=hero_picks,
    )


def get_1v1_selfplay_config():
    hero_picks = [
        HeroPick(team_id=TEAM_RADIANT,
                 hero_id=NPC_DOTA_HERO_NEVERMORE,
                 control_mode=HERO_CONTROL_MODE_CONTROLLED),
        HeroPick(team_id=TEAM_RADIANT,
                 hero_id=NPC_DOTA_HERO_SNIPER,
                 control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_RADIANT,
                 hero_id=NPC_DOTA_HERO_SNIPER,
                 control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_RADIANT,
                 hero_id=NPC_DOTA_HERO_SNIPER,
                 control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_RADIANT,
                 hero_id=NPC_DOTA_HERO_SNIPER,
                 control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE,
                 hero_id=NPC_DOTA_HERO_NEVERMORE,
                 control_mode=HERO_CONTROL_MODE_CONTROLLED),
        HeroPick(team_id=TEAM_DIRE,
                 hero_id=NPC_DOTA_HERO_SNIPER,
                 control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE,
                 hero_id=NPC_DOTA_HERO_SNIPER,
                 control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE,
                 hero_id=NPC_DOTA_HERO_SNIPER,
                 control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE,
                 hero_id=NPC_DOTA_HERO_SNIPER,
                 control_mode=HERO_CONTROL_MODE_IDLE),
    ]
    return GameConfig(
        ticks_per_observation=TICKS_PER_OBSERVATION,
        host_timescale=HOST_TIMESCALE,
        host_mode=HOST_MODE,
        game_mode=DOTA_GAMEMODE_1V1MID,
        hero_picks=hero_picks,
    )


# Test configuration for 2v2
def get_2v2_selfplay_config():
    hero_picks = [
        HeroPick(team_id=TEAM_RADIANT,
                 hero_id=NPC_DOTA_HERO_NEVERMORE,
                 control_mode=HERO_CONTROL_MODE_CONTROLLED),
        HeroPick(team_id=TEAM_RADIANT,
                 hero_id=NPC_DOTA_HERO_NEVERMORE,
                 control_mode=HERO_CONTROL_MODE_CONTROLLED),
        HeroPick(team_id=TEAM_RADIANT,
                 hero_id=NPC_DOTA_HERO_SNIPER,
                 control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_RADIANT,
                 hero_id=NPC_DOTA_HERO_SNIPER,
                 control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_RADIANT,
                 hero_id=NPC_DOTA_HERO_SNIPER,
                 control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE,
                 hero_id=NPC_DOTA_HERO_NEVERMORE,
                 control_mode=HERO_CONTROL_MODE_CONTROLLED),
        HeroPick(team_id=TEAM_DIRE,
                 hero_id=NPC_DOTA_HERO_NEVERMORE,
                 control_mode=HERO_CONTROL_MODE_CONTROLLED),
        HeroPick(team_id=TEAM_DIRE,
                 hero_id=NPC_DOTA_HERO_SNIPER,
                 control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE,
                 hero_id=NPC_DOTA_HERO_SNIPER,
                 control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE,
                 hero_id=NPC_DOTA_HERO_SNIPER,
                 control_mode=HERO_CONTROL_MODE_IDLE),
    ]
    return GameConfig(
        ticks_per_observation=TICKS_PER_OBSERVATION,
        host_timescale=HOST_TIMESCALE,
        host_mode=HOST_MODE,
        game_mode=DOTA_GAMEMODE_1V1MID,
        hero_picks=hero_picks,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ip", type=str, help="mq ip", default='127.0.0.1')
    parser.add_argument("--port", type=int, help="mq port", default=5672)
    parser.add_argument("--dota-port", type=int, help="DotaService port", default=13337)
    parser.add_argument("--max-rollout-size",
                        type=int,
                        help="Maximum size of each rollout (steps)",
                        default=1000000)
    parser.add_argument("--max-dota-time",
                        type=int,
                        help="Maximum in-game (dota) time of a game before restarting",
                        default=600)
    parser.add_argument("--num-agents", type=int, help="Number of players in a team", default=2)
    parser.add_argument("--self-play",
                        action="store_true",
                        help="If set, AI vs AI, or AI vs scripted bot.")
    parser.add_argument("--use-latest-weights-prob",
                        type=float,
                        help="Probability of using the latest weights. "
                        "Otherwise some old one is chosen if available.",
                        default=1.0)
    parser.add_argument("--validation",
                        help="Function as validation runner. "
                        "If empty, training will proceed.",
                        choices=['', 'DIRE', 'RADIANT'],
                        default='')
    parser.add_argument("--log-dir", type=str, help="Logging directory", default='')
    parser.add_argument("-l",
                        "--log",
                        dest="log_level",
                        help="Set the logging level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO')
    args = parser.parse_args()

    logger.setLevel(args.log_level)

    loop = asyncio.get_event_loop()
    coro = main(rmq_host=args.ip,
                rmq_port=args.port,
                dota_port=args.dota_port,
                max_rollout_size=args.max_rollout_size,
                max_dota_time=args.max_dota_time,
                latest_weights_prob=args.use_latest_weights_prob,
                validation=args.validation,
                log_dir=args.log_dir,
                num_agents=args.num_agents,
                self_play=args.self_play)

    loop.run_until_complete(coro)
