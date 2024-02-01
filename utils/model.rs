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
// This code defines neural network modules for an actor and a critic in reinforcement learning
// using the PPO algorithm.
use tch::{
    nn::{self, ConvConfig, LinearConfig, Module},
    Kind, Tensor,
};

use crate::env::Env as EnvTrait;

fn build_seq(vs_path: &nn::Path, out_dim: usize) -> nn::Sequential {
    let out_dim = out_dim as i64;

    let stride = |stride| ConvConfig {
        stride,
        ..ConvConfig::default()
    };

    let seq = nn::seq()
        .add(nn::conv2d(vs_path / "c1", 3, 32, 8, stride(4)))
        .add_fn(Tensor::relu)
        .add(nn::conv2d(vs_path / "c2", 32, 64, 4, stride(2)))
        .add_fn(Tensor::relu)
        .add(nn::conv2d(vs_path / "c3", 64, 64, 3, stride(1)))
        .add_fn(|xs| xs.relu().flat_view())
        .add(nn::linear(
            vs_path / "l1",
            22528,
            512,
            LinearConfig::default(),
        ))
        .add_fn(Tensor::relu);

    seq.add(nn::linear(
        vs_path / "out",
        512,
        out_dim,
        LinearConfig::default(),
    ))
}

/// The Actor struct and its later implementation constructs a neural network
/// with convolution and linear layers to process observations and output
/// actions
#[must_use]
#[derive(Debug)]
pub struct Actor {
    seq: nn::Sequential,
}

impl Module for Actor {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.seq)
    }
}

impl Actor {
    pub fn new(vs_path: &nn::Path, action_space: usize) -> Self {
        Self {
            seq: build_seq(vs_path, action_space),
        }
    }

    #[must_use]
    pub fn chose_action(&self, observation: &Tensor) -> u32 {
        let logits = tch::no_grad(|| self.forward(&observation.unsqueeze(0)));
        let probs = logits.softmax(-1, Kind::Double);
        let action = probs.argmax(-1, false);
        action.int64_value(&[]) as u32
    }

    #[must_use]
    pub fn evaluate_avg_return<Env>(&self, env: &mut Env) -> f64
    where
        Env: EnvTrait,
    {
        const EVAL_EPISODE_COUNT: u8 = 10;

        let mut total_return = 0.0;
        for _ in 0..EVAL_EPISODE_COUNT {
            env.reset();

            while !env.episode_ended() {
                let observation = env.observation();
                let action = self.chose_action(&observation);
                let reward = env.step(action);
                total_return += reward;
            }
        }

        total_return / EVAL_EPISODE_COUNT as f64
    }
}

/// The Critic struct and its later implementation due to the value-based
/// methods (like the one we are using - PPO) is especially essential in RL
/// algorithms. Its main goal is to estimate the value function, which
/// represents the expected return from a given state under a certain policy
#[must_use]
#[derive(Debug)]
pub struct Critic {
    seq: nn::Sequential,
}

impl Module for Critic {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.seq)
    }
}

impl Critic {
    pub fn new(vs_path: &nn::Path) -> Self {
        Self {
            seq: build_seq(vs_path, 1),
        }
    }
}
