from datetime import datetime
from itertools import chain
import argparse
import copy
import io
import logging
import os
import pickle
import re
import socket
import time
import random

from tensorboardX import SummaryWriter
import numpy as np
import pika
import torch
from scipy.signal import lfilter

from dotaservice.protos.DotaService_pb2 import TEAM_DIRE, TEAM_RADIANT
from policy import Policy
from policy import REWARD_KEYS
from qmix import QMixer

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# torch.set_printoptions(profile="full")

torch.manual_seed(7)
random.seed(7)
np.random.seed(7)

eps = np.finfo(np.float32).eps.item()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def discount(x, gamma):
    return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1].astype(np.float32)


def advantage_returns(rewards, values, gamma, lam):
    """Compute the advantage and returns from rewards and values."""
    # GAE-Lambda advantage calculation.
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
    advantages = discount(deltas, gamma * lam)
    # Compute rewards-to-go (targets for the value function).
    returns = discount(rewards, gamma)[:-1]
    return advantages, returns


class MessageQueue:
    EXPERIENCE_QUEUE_NAME = 'experience'
    MODEL_EXCHANGE_NAME = 'model'
    MAX_RETRIES = 10

    def __init__(self, host, port, prefetch_count, use_model_exchange):
        """
        Args:
            prefetch_count (int): Amount of messages to prefetch. Settings this variable too
                high can result in blocked pipes that time out.
        """
        self._params = pika.ConnectionParameters(
            host=host,
            port=port,
            heartbeat=300,
        )
        self.prefetch_count = prefetch_count
        self.use_model_exchange = use_model_exchange

        self._conn = None
        self._xp_channel = None
        self._model_exchange = None

    def process_events(self):
        try:
            self._conn.process_data_events()
        except:
            pass

    def connect(self):
        if not self._conn or self._conn.is_closed:
            # RMQ.
            for i in range(10):
                try:
                    self._conn = pika.BlockingConnection(self._params)
                except pika.exceptions.ConnectionClosed:
                    logger.error('Connection to RMQ failed. retring. ({}/{})'.format(
                        i, self.MAX_RETRIES))
                    time.sleep(5)
                    continue
                else:
                    logger.info('Connected to RMQ')
                    break

            # Experience channel.
            self._xp_channel = self._conn.channel()
            self._xp_channel.basic_qos(prefetch_count=self.prefetch_count)
            self._xp_channel.queue_declare(queue=self.EXPERIENCE_QUEUE_NAME)

            # Model Exchange.
            if self.use_model_exchange:
                self._model_exchange = self._conn.channel()
                self._model_exchange.exchange_declare(
                    exchange=self.MODEL_EXCHANGE_NAME,
                    exchange_type='x-recent-history',
                    arguments={'x-recent-history-length': 1},
                )

    @property
    def xp_queue_size(self):
        try:
            res = self._xp_channel.queue_declare(queue=self.EXPERIENCE_QUEUE_NAME, passive=True)
            return res.method.message_count
        except:
            return None

    def process_data_events(self):
        # Sends heartbeat, might keep conn healthier.
        try:
            self._conn.process_data_events()
        except:  # Gotta catch em' all!
            pass

    def _publish_model(self, msg, hdr):
        self._model_exchange.basic_publish(
            exchange=self.MODEL_EXCHANGE_NAME,
            routing_key='',
            body=msg,
            properties=pika.BasicProperties(headers=hdr),
        )

    def publish_model(self, *args, **kwargs):
        try:
            self._publish_model(*args, **kwargs)
        except (pika.exceptions.ConnectionClosed, pika.exceptions.ChannelClosed):
            logger.error('reconnecting to queue')
            self.connect()
            self._publish_model(*args, **kwargs)

    def _consume_xp(self):
        method, properties, body = next(
            self._xp_channel.consume(
                queue=self.EXPERIENCE_QUEUE_NAME,
                auto_ack=False,
            ))
        self._xp_channel.basic_ack(delivery_tag=method.delivery_tag)
        return method, properties, body

    def consume_xp(self):
        try:
            return self._consume_xp()
        except (pika.exceptions.ConnectionClosed, pika.exceptions.ChannelClosed):
            logger.error('reconnecting to queue')
            self.connect()
            return self._consume_xp()

    def close(self):
        if self._conn and self._conn.is_open:
            logger.info('closing queue connection')
            self._conn.close()


class Sequence:
    def __init__(self, game_id, weight_version, team_id, observations, global_states, actions,
                 masks, rewards, hidden):
        self.game_id = game_id
        self.weight_version = weight_version
        self.team_id = team_id
        self.observations = observations
        self.global_states = global_states
        self.actions = actions
        self.masks = masks
        self.rewards = rewards
        self.hidden = hidden


class DotaOptimizer:

    MODEL_FILENAME_FMT = "model_%09d.pt"
    BUCKET_NAME = 'dotaservice'
    MODEL_HISTOGRAM_FREQ = 128
    MAX_GRAD_NORM = 0.5
    SPEED_KEY = 'steps per s'

    def __init__(self, rmq_host, rmq_port, num_agents, batch_size, seq_len, buffer_size,
                 target_update_freq, learning_rate, gamma, td_lambda, qmix, checkpoint,
                 pretrained_model, mq_prefetch_count, log_dir):
        super().__init__()
        self.rmq_host = rmq_host
        self.rmq_port = rmq_port
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.td_lambda = td_lambda
        self.checkpoint = checkpoint
        self.mq_prefetch_count = mq_prefetch_count
        self.iteration_start = 1
        self.policy = Policy()
        self.qmixer = QMixer(num_agents=num_agents) if qmix else None
        self.log_dir = log_dir

        self.iterations = 100000
        self.target_update_freq = target_update_freq
        self.eventfile_refresh_freq = 100
        self.e_clip = 0.1

        if self.checkpoint:
            self.writer = None
            logger.info('Checkpointing to: {}'.format(self.log_dir))
            try:
                os.mkdir(self.log_dir)
                if self.qmixer is not None:
                    os.mkdir(os.path.join(self.log_dir, 'qmixer'))
            except:
                pass

            # First, check if logdir exists.
            latest_model = self.get_latest_model(prefix=self.log_dir)
            # If there's a model in here, we resume from there
            if latest_model is not None:
                logger.info('Found a latest model in pretrained dir: {}'.format(latest_model))
                if pretrained_model is not None:
                    logger.warning('Overriding pretrained model by latest model.')
                pretrained_model = latest_model

            if pretrained_model is not None:
                self.iteration_start = self.iteration_from_model_filename(
                    filename=pretrained_model) + 1

        if pretrained_model is not None:
            self.policy.load_state_dict(torch.load(pretrained_model,
                                                   map_location=torch.device(device)),
                                        strict=False)
            if self.qmixer is not None:
                self.mixer.load_state_dict(torch.load(os.path.join(
                    os.path.dirname(pretrained_model), os.path.basename(pretrained_model)),
                                                      map_location=torch.device(device)),
                                           strict=False)

        self.target_policy = copy.deepcopy(self.policy).eval()
        self.policy.to(device)
        self.target_policy.to(device)

        if self.qmixer is not None:
            self.target_qmixer = copy.deepcopy(self.qmixer).eval()
            self.qmixer.to(device)
            self.target_qmixer.to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.time_last_it = time.time()

        self.mq = MessageQueue(host=self.rmq_host,
                               port=self.rmq_port,
                               prefetch_count=mq_prefetch_count,
                               use_model_exchange=self.checkpoint)
        self.mq.connect()

        # Upload initial model before any step is taken, and only if we're not resuming.
        self.upload_model(version=self.iteration_start)

    @staticmethod
    def iteration_from_model_filename(filename):
        x = re.search('(\d+)(?=.pt)', filename)
        return int(x.group(0))

    def get_latest_model(self, prefix):
        blobs = [f for f in os.listdir(prefix) if os.path.isfile(f)]

        if not blobs:
            # Directory does not exist, or no files in directory.
            return None
        else:
            fns = [x.name for x in blobs if x.name[-3:] == '.pt']
            if not fns:
                # No relevant files in directory.
                return None
            fns.sort()
            latest_model = fns[-1]
            return latest_model

    def get_rollout(self):
        # TODO(tzaman): make a rollout object
        method, properties, body = self.mq.consume_xp()
        team_data = pickle.loads(body)
        rollout_len = team_data[-1]['rewards'].shape[0]
        version = team_data[-1]['weight_version']
        canvas = team_data[-1]['canvas']

        # Compute rewards per topic, reduce-sum down the sequences.
        subrewards = np.mean([data['rewards'].sum(axis=0) for data in team_data], axis=0)

        return team_data, subrewards, rollout_len, version, canvas

    def experiences_from_rollout(self, data):
        # TODO(tzaman): The rollout can consist out of multiple viable sequences.
        # These should be padded and then sliced into separate experiences.
        observations = data['observations']
        global_states = data['global_states']
        masks = data['masks']
        actions = data['actions']
        rewards = data['rewards']

        rollout_len = data['rewards'].shape[0]
        logger.debug('rollout_len={}'.format(rollout_len))

        sequences = []
        hidden = self.policy.init_hidden().to(device)
        rewards_sum = []
        slice_indices = range(0, rollout_len, self.seq_len)
        for i1 in slice_indices:
            pad = 0
            # Check if this slice requires padding.
            if rollout_len - i1 < self.seq_len:
                pad = self.seq_len - (rollout_len - i1)
            i2 = i1 + self.seq_len - pad
            logger.debug('Slice[{}:{}], pad={}'.format(i1, i2, pad))

            # Slice out the relevant parts.
            s_observations = {}
            for key, val in observations.items():
                s_observations[key] = val[i1:i2, :].to(device)

            s_global_states = {}
            for key, val in global_states.items():
                s_global_states[key] = val[i1:i2, :].to(device)

            s_masks = masks[i1:i2, :].to(device)
            s_actions = actions[i1:i2, :].to(device)
            s_rewards = rewards[i1:i2]

            if pad:
                dim_pad = {
                    1: (0, pad),
                    2: (0, 0, 0, pad),
                    3: (0, 0, 0, 0, 0, pad),
                }
                for key, val in s_observations.items():
                    s_observations[key] = torch.nn.functional.pad(val,
                                                                  dim_pad[val.dim()],
                                                                  mode='constant',
                                                                  value=0).detach()

                for key, val in s_global_states.items():
                    s_global_states[key] = torch.nn.functional.pad(val,
                                                                   pad=dim_pad[val.dim()],
                                                                   mode='constant',
                                                                   value=0).detach()
                s_masks = torch.nn.functional.pad(s_masks,
                                                  pad=dim_pad[s_masks.dim()],
                                                  mode='constant',
                                                  value=0).detach()
                s_actions = torch.nn.functional.pad(s_actions,
                                                    pad=dim_pad[s_actions.dim()],
                                                    mode='constant',
                                                    value=0).detach()
                s_rewards = np.pad(s_rewards, ((0, pad), (0, 0)), mode='constant')

            input_hidden = hidden
            _, hidden = self.policy.sequence(**s_observations, hidden=input_hidden)

            # The values and rewards are gathered here over all sequences, because the values are
            # cumulative, and therefore need the first step from each next sequence. To optimize this,
            # we gather them here, and process these after the loop, and add them to the Sequence
            # object later.
            rewards_sum.append(np.sum(s_rewards, axis=1).ravel())

            sequence = Sequence(
                game_id=data['game_id'],
                weight_version=data['weight_version'],
                team_id=data['team_id'],
                observations={k: v.cpu()
                              for k, v in s_observations.items()},
                global_states={k: v.cpu()
                               for k, v in s_global_states.items()},
                actions=s_actions.cpu(),
                masks=s_masks.cpu(),
                rewards=s_rewards,
                hidden=input_hidden.detach().cpu(),
            )
            sequences.append(sequence)

        return sequences

    @staticmethod
    def list_of_dicts_to_dict_of_lists(x):
        return {k: torch.stack([d[k] for d in x]) for k in x[0]}

    def run(self):
        replay_buffer = []
        for it in range(self.iteration_start, self.iterations):
            logger.info('iteration {}/{}'.format(it, self.iterations))

            # First grab a bunch of experiences
            experiences = []
            subrewards = []
            rollout_lens = []
            weight_ages = []
            canvas = None  # Just save only the last canvas.
            start_xp = time.time()
            xp_waits = 0
            while len(experiences) < self.batch_size:
                logger.debug(' adding experience @{}/{}'.format(len(experiences), self.batch_size))

                # Get new experiences from a new rollout.
                with torch.no_grad():
                    start_xp_wait = time.time()
                    team_rollout, rollout_subrewards, rollout_len, weight_version, canvas = self.get_rollout(
                    )
                    xp_waits += time.time() - start_xp_wait
                    rollout_experiences = [
                        self.experiences_from_rollout(data=rollout) for rollout in team_rollout
                    ]

                # Decouple experiences when qmix is not used
                experiences.extend(
                    list(
                        zip(*rollout_experiences) if self.qmixer is not None else chain.
                        from_iterable(rollout_experiences)))

                subrewards.append(rollout_subrewards)
                rollout_lens.append(rollout_len)
                weight_ages.append(it - weight_version)
            time_xp = time.time() - start_xp

            replay_buffer.extend(experiences)
            if len(replay_buffer) > self.buffer_size:
                replay_buffer = replay_buffer[-self.buffer_size:]

            losses = []
            grad_norms = []
            start_optimizing = time.time()
            self.mq.process_data_events()
            sampled_experiences = [
                replay_buffer[i]
                for i in np.random.choice(len(replay_buffer), self.batch_size, replace=False)
            ]
            loss_d, grad_norm_d = self.train(experiences=sampled_experiences)
            losses.append(loss_d)
            grad_norms.append(grad_norm_d)
            time_optimizing = time.time() - start_optimizing

            losses = self.list_of_dicts_to_dict_of_lists(losses)
            loss = losses['loss'].mean()

            grad_norms = self.list_of_dicts_to_dict_of_lists(grad_norms)

            n_steps = len(experiences) * self.seq_len

            subrewards_per_sec = np.stack(subrewards) / n_steps * Policy.OBSERVATIONS_PER_SECOND
            rollout_rewards = subrewards_per_sec.sum(axis=1)
            reward_dict = dict(zip(REWARD_KEYS, subrewards_per_sec.sum(axis=0)))
            reward_per_sec = rollout_rewards.sum()

            rollout_lens = torch.tensor(rollout_lens, dtype=torch.float32)
            avg_rollout_len = rollout_lens.mean()

            weight_ages = torch.tensor(weight_ages, dtype=torch.float32)
            avg_weight_age = weight_ages.mean()

            time_it = time.time() - self.time_last_it
            steps_per_s = n_steps / (time_it)
            self.time_last_it = time.time()

            metrics = {
                self.SPEED_KEY: steps_per_s,
                'reward_per_sec/sum': reward_per_sec,
                'loss/sum': loss,
                'avg_rollout_len': avg_rollout_len,
                'avg_weight_age': avg_weight_age,
            }

            metrics['timing/it'] = time_it  # Full total time since last step
            metrics['timing/xp_total'] = time_xp
            metrics['timing/xp_mq_wait'] = xp_waits
            metrics['timing/optimizer'] = time_optimizing

            for k, v in grad_norms.items():
                metrics['grad_norm/{}'.format(k)] = v.mean()

            for k, v in reward_dict.items():
                metrics['reward_per_sec/{}'.format(k)] = v

            logger.info(
                ' steps_per_s={:.2f}, avg_weight_age={:.1f}, reward_per_sec={:.4f}, loss={:.4f}'.
                format(steps_per_s, float(avg_weight_age), reward_per_sec, float(loss)))

            if self.checkpoint:
                start_checkpoint = time.time()
                # TODO(tzaman): re-introduce distributed metrics. See commits from december 2017.

                if self.writer is None or it % self.eventfile_refresh_freq == 0:
                    logger.info('(Re-)Creating TensorBoard eventsfile (#iteration={})'.format(it))
                    self.writer = SummaryWriter(log_dir=self.log_dir)

                # Write metrics to events file.
                for name, metric in metrics.items():
                    self.writer.add_scalar(name, metric, it)

                # TODO(tzaman): How to add the time spent on writing events file and model?

                # Add per-iteration histograms
                self.writer.add_histogram('losses', losses['loss'], it)
                # self.writer.add_histogram('entropies', entropies, it)
                self.writer.add_histogram('rollout_lens', rollout_lens, it)
                self.writer.add_histogram('weight_age', weight_ages, it)

                # Rewards histogram
                self.writer.add_histogram('rewards_per_sec_per_rollout', rollout_rewards, it)

                # Model
                if it % self.MODEL_HISTOGRAM_FREQ == 1:
                    for name, param in self.policy.named_parameters():
                        self.writer.add_histogram('param/' + name,
                                                  param.clone().cpu().data.numpy(), it)
                        self.writer.add_image('canvas', canvas, it, dataformats='HWC')

                # RMQ Queue size.
                queue_size = self.mq.xp_queue_size
                if queue_size is not None:
                    self.writer.add_scalar('mq_size', queue_size, it)

                self.writer.file_writer.flush()
                time_checkpoint = time.time() - start_checkpoint

                start_model_upload = time.time()
                if it % self.target_update_freq == 0:
                    self.upload_model(version=it)
                    self.target_policy.load_state_dict(self.policy.state_dict())
                    self.target_qmixer.load_state_dict(self.qmixer.state_dict())
                time_model_upload = time.time() - start_model_upload

                logger.info(
                    'Timings: it={:.2f}, xp={:.2f}, xp_wait={:.2f} opt={:.2f}, upload_tb={:.2f}, upload_model={:.2f}'
                    .format(time_it, time_xp, xp_waits, time_optimizing, time_checkpoint,
                            time_model_upload))

    def train(self, experiences):
        # Train on one epoch of data.
        # Experiences is a list of (padded) experience chunks.
        logger.debug('train(experiences=#{})'.format(len(experiences)))

        # Stack together all experiences.
        if self.qmixer is not None:
            # Global states for group
            global_states = {key: [] for key in QMixer.INPUT_KEYS}
            for e in experiences:
                # All e has same global states
                for key, val in e[0].global_states.items():
                    global_states[key].append(val)
            for key, val in global_states.items():
                global_states[key] = torch.stack(val).to(device)

            # Expand batch
            experiences = list(chain.from_iterable(experiences))

        hidden = torch.cat([e.hidden for e in experiences], dim=1).detach().to(device)

        actions = torch.stack([e.actions for e in experiences]).to(device)
        masks = torch.stack([e.masks for e in experiences]).to(device)
        rewards_sum = np.array([e.rewards.sum(axis=-1) for e in experiences])
        terminated = np.array([np.abs(e.rewards[:, 8]) == 1 for e in experiences])

        observations = {key: [] for key in Policy.INPUT_KEYS}
        for e in experiences:
            for key, val in e.observations.items():
                observations[key].append(val)
        for key, val in observations.items():
            observations[key] = torch.stack(val).to(device)

        # Notice there is no notion of loss masking here, this is unnessecary as we only work
        # use selected probabilties. E.g. when things were padded, nothing was selected, so no data.
        logits, _ = self.policy(**observations, hidden=hidden)

        # actions_step contains if we took an action of this key during the respective step
        actions_step = actions.sum(dim=-1) != 0
        if actions_step.sum() == 0:
            loss = torch.zeros([])
        else:
            cur_actions = actions[:, :-1].max(dim=-1, keepdim=True)[1]
            chosen_action_qvals = logits[:, :-1].gather(dim=-1, index=cur_actions).squeeze(-1)

            target_logits, _ = self.target_policy(**observations, hidden=hidden)
            # double q
            logits_detach = logits.clone().detach()
            logits_detach[masks != 1] = -np.Inf
            cur_max_actions = logits_detach[:, 1:].max(dim=-1, keepdim=True)[1]
            target_max_qvals = target_logits[:, 1:].gather(dim=-1,
                                                           index=cur_max_actions).squeeze(-1)

            if self.qmixer is not None:
                chosen_action_qvals = self.qmixer(**{
                    k: v[:, :-1]
                    for k, v in global_states.items()
                },
                                                  qs=chosen_action_qvals.view(
                                                      self.batch_size, -1,
                                                      self.seq_len - 1).permute(0, 2,
                                                                                1)).squeeze(-1)
                target_max_qvals = self.target_qmixer(**{
                    k: v[:, 1:]
                    for k, v in global_states.items()
                },
                                                      qs=target_max_qvals.view(
                                                          self.batch_size, -1,
                                                          self.seq_len - 1).permute(0, 2,
                                                                                    1)).squeeze(-1)
                masks = masks.view(self.batch_size, -1, *masks.shape[1:]).permute(0, 2, 1, 3)
                rewards_sum = rewards_sum.reshape(self.batch_size, -1,
                                                  self.seq_len).transpose(0, 2, 1)
                terminated = terminated.reshape(self.batch_size, -1,
                                                self.seq_len).transpose(0, 2, 1)

            deltas = rewards_sum[:, :-1] + self.gamma * (1 - self.td_lambda) * (
                1 - terminated[:, :-1]) * target_max_qvals.detach().cpu().numpy()
            targets = lfilter([1], [1, -self.gamma * self.td_lambda], deltas[:, ::-1],
                              axis=1)[:, ::-1]
            targets = torch.from_numpy(targets.copy()).to(device)

            td_error = chosen_action_qvals - targets
            td_error_masked = torch.masked_select(input=td_error,
                                                  mask=(masks[:, :-1] == 1).any(dim=-1))

            loss = td_error_masked**2

        loss = loss.mean()

        if torch.isnan(loss):
            raise ValueError(f'loss={loss}')

        self.optimizer.zero_grad()
        loss.backward()

        grad_norm = self.mean_gradient_norm()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.MAX_GRAD_NORM)
        grad_norm_clipped = self.mean_gradient_norm()

        if torch.isnan(grad_norm):
            raise ValueError('grad_norm={}'.format(grad_norm))

        self.optimizer.step()
        losses = {'loss': loss.detach()}

        return losses, {'unclipped': grad_norm.detach(), 'clipped': grad_norm_clipped.detach()}

    def mean_gradient_norm(self):
        gs = []
        for p in list(filter(lambda p: p.grad is not None, self.policy.parameters())):
            gs.append(p.grad.data.norm(2))
        return torch.stack(gs).mean()

    def upload_model(self, version):
        filename = self.MODEL_FILENAME_FMT % version
        rel_path = os.path.join(self.log_dir, filename)

        # Serialize the model.
        buffer = io.BytesIO()
        state_dict = self.policy.state_dict()
        torch.save(obj=state_dict, f=buffer)
        state_dict_b = buffer.getvalue()

        # Write model to file.
        with open(rel_path, 'wb') as f:
            f.write(state_dict_b)

        # Send to exchange.
        self.mq.publish_model(msg=state_dict_b, hdr={'version': version})

        if self.qmixer is not None:
            rel_path = os.path.join(self.log_dir, 'qmixer', filename)

            # Serialize the model.
            buffer = io.BytesIO()
            state_dict = self.qmixer.state_dict()
            torch.save(obj=state_dict, f=buffer)
            state_dict_b = buffer.getvalue()

            # Write model to file.
            with open(rel_path, 'wb') as f:
                f.write(state_dict_b)


def main(rmq_host, rmq_port, num_agents, batch_size, seq_len, buffer_size, target_update_freq,
         learning_rate, gamma, td_lambda, qmix, pretrained_model, mq_prefetch_count, log_dir):
    logger.info(
        f'main(rmq_host={rmq_host}, rmq_port={rmq_port}, num_agents={num_agents}, batch_size={batch_size},'
        f' seq_len={seq_len}, buffer_size={buffer_size}, learning_rate={learning_rate}, qmix={qmix},'
        f' pretrained_model={pretrained_model}, mq_prefetch_count={mq_prefetch_count})')

    dota_optimizer = DotaOptimizer(
        rmq_host=rmq_host,
        rmq_port=rmq_port,
        num_agents=num_agents,
        batch_size=batch_size,
        seq_len=seq_len,
        buffer_size=buffer_size,
        target_update_freq=target_update_freq,
        learning_rate=learning_rate,
        gamma=gamma,
        td_lambda=td_lambda,
        qmix=qmix,
        checkpoint=True,
        pretrained_model=pretrained_model,
        mq_prefetch_count=mq_prefetch_count,
        log_dir=log_dir,
    )

    dota_optimizer.run()


def default_log_dir():
    return 'logs/{}_{}'.format(datetime.now().strftime('%b%d_%H-%M-%S'), socket.gethostname())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ip", type=str, help="mq ip", default='127.0.0.1')
    parser.add_argument("--port", type=int, help="mq port", default=5672)
    parser.add_argument("--seq-len",
                        type=int,
                        help="Sequence length (as one sample in a minibatch). "
                        "This is also the length that will be (truncated) backpropped into.",
                        default=16)
    parser.add_argument("--batch-size",
                        type=int,
                        help="Amount of sequences per iteration.",
                        default=400)
    parser.add_argument("--buffer-size", type=int, help="Replay buffer size", default=100000)
    parser.add_argument("--target-update-freq",
                        type=int,
                        help="Target policy update frequency",
                        default=25)
    parser.add_argument("--learning-rate", type=float, help="Learning rate", default=5e-5)
    parser.add_argument("--gamma", type=float, help="Discount factor", default=0.98)
    parser.add_argument("--td-lambda", type=float, help="Trace decay parameter", default=0.97)
    parser.add_argument("--qmix", action="store_true", help="Use QMIX")
    parser.add_argument("--num-agents",
                        type=int,
                        help="Number of players in a team. Ignored if qmix is unset.",
                        default=2)
    parser.add_argument("--pretrained-model", type=str, help="Pretrained model file", default=None)
    parser.add_argument("--mq-prefetch-count",
                        type=int,
                        help="Amount of experience messages to prefetch from mq",
                        default=1)
    parser.add_argument("--log-dir",
                        type=str,
                        help="Log and job dir name",
                        default=default_log_dir())
    parser.add_argument("-l",
                        "--log",
                        dest="log_level",
                        help="Set the logging level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO')

    args = parser.parse_args()

    logger.setLevel(args.log_level)

    try:
        main(
            rmq_host=args.ip,
            rmq_port=args.port,
            num_agents=args.num_agents,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            buffer_size=args.buffer_size,
            target_update_freq=args.target_update_freq,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            td_lambda=args.td_lambda,
            qmix=args.qmix,
            pretrained_model=args.pretrained_model,
            mq_prefetch_count=args.mq_prefetch_count,
            log_dir=args.log_dir,
        )
    except KeyboardInterrupt:
        pass
