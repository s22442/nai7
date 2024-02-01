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
use std::rc::Rc;

use tch::{nn::Module, Device, Kind, Tensor};
use typed_builder::TypedBuilder;

use crate::model;

/// Clipped Loss in RL is a technique used in policy gradient methods to address
/// the issue of high variance in gradient estimates during training. Below
/// implementation computes losses for the PPO algorithm (optimizing a policy to
/// maximize expected rewards)
#[derive(TypedBuilder)]
pub struct ClippedLoss {
    actor: Rc<model::Actor>,
    critic: Rc<model::Critic>,
    clip_epsilon: f64,

    #[builder(setter(transform = |batch_size: usize| batch_size as i64))]
    batch_size: i64,

    device: Device,
}

impl ClippedLoss {
    pub fn compute_actor_loss(
        &self,
        actions: &Tensor,
        action_log_probs: &Tensor,
        observations: &Tensor,
        advantages: &Tensor,
    ) -> Tensor {
        let logits = self.actor.forward(observations);
        let new_action_log_probs = logits
            .log_softmax(-1, Kind::Double)
            .gather(1, actions, false)
            .squeeze_dim(1);

        let ratio = (&new_action_log_probs - action_log_probs).exp();

        let min_advantages = Tensor::zeros([self.batch_size], (Kind::Double, self.device));

        // In the loop we make sure that the algorithm would not learn too
        // quick which can potentially lead to a model collapse
        for i in 0..self.batch_size {
            let advantage = advantages.get(i);
            let multiplier = if advantage.double_value(&[]) > 0.0 {
                1.0 + self.clip_epsilon
            } else {
                1.0 - self.clip_epsilon
            };
            min_advantages.get(i).copy_(&(advantage * multiplier));
        }

        -(&ratio * advantages)
            .min_other(&min_advantages)
            .mean(Kind::Double)
    }

    pub fn compute_critic_loss(&self, observations: &Tensor, returns: Tensor) -> Tensor {
        let values = self.critic.forward(observations).squeeze();
        (returns - values).pow_tensor_scalar(2).mean(Kind::Double)
    }
}
