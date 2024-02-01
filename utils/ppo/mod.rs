/// This program implements a bot that through the use of reinforcement learning
/// learns how to efficiently play a game of Space Invaders.
///
/// To run this project make sure that you:
///    - run a following command: pip install 'autorom[accept-rom-license]'
///    - run a following command: pip install 'gym[atari]'
///    - install [Rust](https://www.rust-lang.org/tools/install)
///
/// Project created by:
///    Kajetan Welc
///    Daniel Wirzba
mod clipped_loss;
mod collector;
mod estimator;
mod sampler;

use std::{fmt, rc::Rc, sync::Arc};

use derive_setters::Setters;

use clipped_loss::ClippedLoss;
use collector::Collector;
use sampler::{ActorSample, CriticSample, Sampler};
use tch::nn::{self, OptimizerConfig, VarStore};

use crate::env::Env as EnvTrait;
use crate::model;

/// Following code is organized into several modules (clipped loss, collector,
/// estimator, sampler) that encapsulate different components of the PPO
/// algorithm since it also defines structs that represent key elements of the
/// algorithm, such as PPO for the main algorithm, Payload for holding data
/// during training, and Builder for configuring the PPO instance
#[must_use]
pub struct PPO<Env>
where
    Env: EnvTrait,
{
    actor_vs: Arc<VarStore>,
    critic_vs: VarStore,
    actor: Rc<model::Actor>,
    ga_estimator: estimator::GeneralAdvantage,
    return_estimator: estimator::Return,
    clipped_loss: ClippedLoss,
    collector: Collector,
    actor_optimizer: nn::Optimizer,
    critic_optimizer: nn::Optimizer,
    sampler: Sampler,
    eval_env: Env,
    actor_train_num_batches: usize,
    critic_train_num_batches: usize,
}

impl<Env> PPO<Env>
where
    Env: EnvTrait,
{
    pub fn save<T>(&self, path: T)
    where
        T: fmt::Display,
    {
        self.actor_vs.save(format!("{path}-actor.ot")).unwrap();
        self.critic_vs.save(format!("{path}-critic.ot")).unwrap();
    }

    pub fn train(&mut self, num_epochs: usize) {
        for _ in 0..num_epochs {
            let collector::Payload {
                observations,
                next_observations,
                episodes_not_terminated,
                actions,
                action_log_probs,
                rewards,
            } = self.collector.collect();

            let advantages = self.ga_estimator.compute_advantages(
                &observations,
                &next_observations,
                &episodes_not_terminated,
                &rewards,
            );

            let returns = self
                .return_estimator
                .compute_returns(&episodes_not_terminated, &rewards);

            self.sampler
                .fill_observations(observations)
                .fill_actions(actions)
                .fill_action_log_probs(action_log_probs)
                .fill_advantages(advantages)
                .fill_returns(returns);

            for _ in 0..self.actor_train_num_batches {
                let ActorSample {
                    actions,
                    action_log_probs,
                    observations,
                    advantages,
                } = self.sampler.actor_sample();

                let actor_loss = self.clipped_loss.compute_actor_loss(
                    &actions,
                    &action_log_probs,
                    &observations,
                    &advantages,
                );

                self.actor_optimizer.zero_grad();
                actor_loss.backward();
                self.actor_optimizer.step();
            }

            self.collector.sync_collecting_actors();

            for _ in 0..self.critic_train_num_batches {
                let CriticSample {
                    observations,
                    returns,
                } = self.sampler.critic_sample();

                let critic_loss = self
                    .clipped_loss
                    .compute_critic_loss(&observations, returns);

                self.critic_optimizer.zero_grad();
                critic_loss.backward();
                self.critic_optimizer.step();
            }
        }
    }

    pub fn evaluate_avg_return(&mut self) -> f64 {
        self.actor.evaluate_avg_return(&mut self.eval_env)
    }

    pub fn builder() -> Builder<Env> {
        Builder {
            gamma: None,
            actor_learning_rate: None,
            critic_learning_rate: None,
            env: None,
            actor_vs: None,
            critic_vs: None,
            num_steps_per_epoch: None,
            batch_size: None,
            lmbda: None,
            clip_epsilon: None,
            actor_train_num_batches: None,
            critic_train_num_batches: None,
            num_threads: None,
        }
    }
}

/// The builder allows flexible configuration of parameters such as variable
/// stores for actor and critic models, hyperparameters, learning rates,
/// environment settings, and training parameters
#[must_use]
#[derive(Setters, Debug)]
#[setters(strip_option)]
pub struct Builder<Env>
where
    Env: EnvTrait,
{
    actor_vs: Option<nn::VarStore>,
    critic_vs: Option<nn::VarStore>,
    gamma: Option<f64>,
    lmbda: Option<f64>,
    clip_epsilon: Option<f64>,
    actor_learning_rate: Option<f64>,
    critic_learning_rate: Option<f64>,
    env: Option<Env>,
    num_steps_per_epoch: Option<usize>,
    num_threads: Option<usize>,
    batch_size: Option<usize>,
    actor_train_num_batches: Option<usize>,
    critic_train_num_batches: Option<usize>,
}

impl<Env> Builder<Env>
where
    Env: EnvTrait,
{
    pub fn build(self) -> PPO<Env> {
        let actor_learning_rate = self.actor_learning_rate.unwrap();
        let critic_learning_rate = self.critic_learning_rate.unwrap();
        let num_steps_per_epoch = self.num_steps_per_epoch.unwrap();
        let mut actor_vs = self.actor_vs.unwrap();
        let mut critic_vs = self.critic_vs.unwrap();
        let mut env = self.env.unwrap();
        env.reset();
        let action_space = env.action_space();

        let actor: Rc<model::Actor> = Rc::new(model::Actor::new(&actor_vs.root(), action_space));
        let critic = Rc::new(model::Critic::new(&critic_vs.root()));

        let actor_optimizer = nn::Adam::default()
            .build(&actor_vs, actor_learning_rate)
            .unwrap();
        let critic_optimizer = nn::Adam::default()
            .build(&critic_vs, critic_learning_rate)
            .unwrap();

        actor_vs.double();
        critic_vs.double();

        let device = actor_vs.device();
        assert_eq!(device, critic_vs.device());

        let actor_vs = Arc::new(actor_vs);

        let collector = Collector::new(collector::Options {
            global_actor_vs: &actor_vs,
            device,
            num_steps: num_steps_per_epoch,
            num_threads: self.num_threads.unwrap(),
            env: &env,
        });

        let batch_size = self.batch_size.unwrap();

        let sampler = Sampler::new(num_steps_per_epoch, batch_size, device);

        let gamma = self.gamma.unwrap();
        let lmbda = self.lmbda.unwrap();

        let ga_estimator = estimator::GeneralAdvantage::builder()
            .critic(Rc::clone(&critic))
            .num_steps(num_steps_per_epoch)
            .gamma(gamma)
            .lmbda(lmbda)
            .device(device)
            .build();

        let return_estimator = estimator::Return::builder()
            .num_steps(num_steps_per_epoch)
            .gamma(gamma)
            .device(device)
            .build();

        let clipped_loss = ClippedLoss::builder()
            .actor(Rc::clone(&actor))
            .critic(critic)
            .clip_epsilon(self.clip_epsilon.unwrap())
            .batch_size(batch_size)
            .device(device)
            .build();

        PPO {
            actor_vs,
            critic_vs,
            actor,
            actor_optimizer,
            critic_optimizer,
            collector,
            sampler,
            ga_estimator,
            return_estimator,
            clipped_loss,
            eval_env: env,
            actor_train_num_batches: self.actor_train_num_batches.unwrap(),
            critic_train_num_batches: self.critic_train_num_batches.unwrap(),
        }
    }
}
