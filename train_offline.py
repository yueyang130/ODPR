import os
from typing import Tuple
from pathlib import Path
import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
import wandb

import wrappers
from dataset_utils import D4RLDataset, split_into_trajectories
from evaluation import evaluate
from learner import Learner
from utils import get_user_flags

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './result/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
# network architecture
flags.DEFINE_boolean('encoder', False, 'an encoder for actor and critic input')
flags.DEFINE_enum('rep_module', 'backbone', ['backbone', 'encoder'], 'The network for representation learning')
# pretrain
flags.DEFINE_integer('pretrain_steps', int(1e6), '')
flags.DEFINE_enum('pretrain_sample', 'uniform', ['uniform', 'return-balance', 'inverse-return-balance'], '')
# offline learning
flags.DEFINE_integer('offline_steps', int(1e6), 'Number of total training steps.')
flags.DEFINE_enum('sample', 'return-balance', ['uniform', 'return-balance', 'inverse-return-balance'], '')
flags.DEFINE_enum('finetune', 'freeze', ['freeze', 'reduced-lr', 'none'], 'representation finutune schemes') 
flags.DEFINE_enum('retrain', 'repr', ['repr', 'pred', 'all'], 'retrain which part of network') 
flags.DEFINE_boolean('reinitialize', False, 'reinitialize the output layer')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_string('tag', '', 'tag of the run.')
config_flags.DEFINE_config_file(
    'config',
    'default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def make_env_and_dataset(env_name: str,
                         seed: int) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    pretrain_dataset = D4RLDataset(env, FLAGS.batch_size, FLAGS.pretrain_sample, FLAGS.config.base_prob)
    dataset = D4RLDataset(env, FLAGS.batch_size, FLAGS.sample, FLAGS.config.base_prob)

    if 'antmaze' in FLAGS.env_name:
        pretrain_dataset.rewards -= 1.0
        dataset.rewards -= 1.0
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
          or 'hopper' in FLAGS.env_name):
        normalize(pretrain_dataset)
        normalize(dataset)

    return env, pretrain_dataset, dataset


def main(_):
    kwargs = dict(FLAGS.config)
    # set up wandb
    wandb.init(project="IQL", config={"env": FLAGS.env_name, "seed": FLAGS.seed,
            "encoder": FLAGS.encoder, "rep_module": FLAGS.rep_module,
            "pretrain_sample": FLAGS.pretrain_sample, "pretrain_steps": FLAGS.pretrain_steps, 
            "offline_steps": FLAGS.offline_steps, "sampler": FLAGS.sample,
            "retrain": FLAGS.retrain, "finetune": FLAGS.finetune, "reinitialize": FLAGS.reinitialize, 
            "encoder_hidden_dims": FLAGS.config.encoder_hidden_dims, "dynamic_hidden_dims": FLAGS.config.dynamic_hidden_dims,
            "embedding_dim": FLAGS.config.embedding_dim, "hidden_dims": FLAGS.config.hidden_dims,
            "base_prob": FLAGS.config.base_prob, "tag": FLAGS.tag})

    FLAGS.save_dir = Path(os.path.join(FLAGS.save_dir, FLAGS.tag, FLAGS.env_name, str(FLAGS.seed)))
    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, 'tb'),
                                #    write_to_disk=True)
                                   write_to_disk=False)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    env, pretrain_dataset, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)

    # pretrain
    if FLAGS.pretrain_steps > 0:

        eval_returns = []
        rep_agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.pretrain_steps,
                    finetune=None,
                    encoder = FLAGS.encoder,
                    **kwargs)
        for i in tqdm.tqdm(range(1, FLAGS.pretrain_steps + 1),
                        smoothing=0.1,
                        disable=not FLAGS.tqdm):
            batch = pretrain_dataset.sample()

            update_info = rep_agent.update(batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    if v.ndim == 0:
                        summary_writer.add_scalar(f'pretrain/training/{k}', v, i)
                        wandb.log({f"pretrain/training_{k}": v}, step=i)
                    else:
                        summary_writer.add_histogram(f'pretrain/training/{k}', v, i)
                summary_writer.flush()

            if i % FLAGS.eval_interval == 0:
                eval_episode = max(100, FLAGS.eval_episodes) if i ==  FLAGS.pretrain_steps else FLAGS.eval_episodes
                eval_stats = evaluate(rep_agent, env, eval_episode)

                for k, v in eval_stats.items():
                    summary_writer.add_scalar(f'pretrain/evaluation/average_{k}s', v, i)
                    wandb.log({f'pretrain/evaluation/{k}': v}, step=i)

                summary_writer.flush()

                # eval_returns.append((i, eval_stats['return']))
                # np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                #            eval_returns,
                #            fmt=['%d', '%.1f'])

        # save and load
        rep_agent.save(FLAGS.save_dir / 'ckpt')

    # offline learning
    if FLAGS.offline_steps > 0:
        agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.offline_steps,
                    finetune=FLAGS.finetune,
                    encoder = FLAGS.encoder,
                    rep_module=FLAGS.rep_module,
                    retrain=FLAGS.retrain,
                    **kwargs)
        if FLAGS.pretrain_steps > 0:
            agent.load(FLAGS.save_dir / 'ckpt')
            if FLAGS.reinitialize:
                agent.reinitialize_output_layer()

        eval_episode = max(100, FLAGS.eval_episodes)
        eval_stats = evaluate(agent, env, FLAGS.eval_episodes)
        for k, v in eval_stats.items():
            summary_writer.add_scalar(f'offline/evaluation/average_{k}s', v, FLAGS.pretrain_steps)
            wandb.log({f'offline/evaluation/{k}': v}, step=FLAGS.pretrain_steps)

        eval_returns = []
        for i in tqdm.tqdm(range(FLAGS.pretrain_steps + 1, FLAGS.pretrain_steps + FLAGS.offline_steps + 1),
                        smoothing=0.1,
                        disable=not FLAGS.tqdm):
            batch = dataset.sample()

            update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    if v.ndim == 0:
                        summary_writer.add_scalar(f'offline/training/{k}', v, i)
                        wandb.log({f"offline/training/{k}": v}, step=i)
                    else:
                        summary_writer.add_histogram(f'offline/training/{k}', v, i)
                summary_writer.flush()

            if i % FLAGS.eval_interval == 0:
                eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

                for k, v in eval_stats.items():
                    summary_writer.add_scalar(f'offline/evaluation/average_{k}s', v, i)
                    wandb.log({f'offline/evaluation/{k}': v}, step=i)

                summary_writer.flush()

                # eval_returns.append((i, eval_stats['return']))
                # np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                #            eval_returns,
                #            fmt=['%d', '%.1f'])


if __name__ == '__main__':
    app.run(main)
