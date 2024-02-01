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

/// Below are the implementations for computing advantages and returns, which
/// are essential components in the RL PPO algorithm for estimating the quality
/// of actions and sequences of actions
use std::rc::Rc;

use tch::{nn::Module, Device, Kind, Tensor};
use typed_builder::TypedBuilder;

use crate::model;

// From Pytorch a class wrapper around the Generalized Advantage Estimate
// functional (GAE) is used to provide an interface for processing advantage,
// target value entries and not only, for the models.
#[derive(TypedBuilder)]
pub struct GeneralAdvantage {
    critic: Rc<model::Critic>,
    gamma: f64,
    lmbda: f64,

    #[builder(setter(transform = |num_steps: usize| num_steps as i64))]
    num_steps: i64,

    device: Device,
}

impl GeneralAdvantage {
    pub fn compute_advantages(
        &self,
        observations: &Tensor,
        next_observations: &Tensor,
        episodes_not_terminated: &Tensor,
        rewards: &Tensor,
    ) -> Tensor {
        let values = tch::no_grad(|| self.critic.forward(observations).squeeze());
        let next_values = tch::no_grad(|| self.critic.forward(next_observations).squeeze());

        let advantages = Tensor::zeros([self.num_steps], (Kind::Double, self.device));
        let mut prev_advantage = 0.0;

        let gamma_not_terminated = episodes_not_terminated * self.gamma;

        let delta = rewards + (&gamma_not_terminated * next_values) - &values;

        let discount = &gamma_not_terminated * self.lmbda;

        for i in (0..self.num_steps).rev() {
            let advantage = delta.get(i) + discount.get(i) * prev_advantage;
            advantages.get(i).copy_(&advantage);
            prev_advantage = advantage.double_value(&[]);
        }

        (&advantages - advantages.mean(Kind::Double)) / advantages.std(false)
    }
}

#[derive(TypedBuilder)]
pub struct Return {
    gamma: f64,

    #[builder(setter(transform = |num_steps: usize| num_steps as i64))]
    num_steps: i64,

    device: Device,
}

impl Return {
    pub fn compute_returns(&self, episodes_not_terminated: &Tensor, rewards: &Tensor) -> Tensor {
        let gamma_not_terminated = episodes_not_terminated * self.gamma;
        let returns = Tensor::zeros([self.num_steps], (Kind::Double, self.device));

        let mut prev_return = 0.0;

        for i in (0..self.num_steps).rev() {
            let return_ = rewards.get(i) + gamma_not_terminated.get(i) * prev_return;
            returns.get(i).copy_(&return_);
            prev_return = return_.double_value(&[]);
        }

        returns
    }
}
