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
use derive_setters::Setters;
use tch::{Device, Kind, Tensor};

/// following 3 structs are defined:
/// - Sampler - responsible for generating samples with multiple parameters
///   defining its work
/// - `ActorSample` - represents a sample generated for an actor in RL, later
///   initialized in the Sampler implementation
/// - Critic Sample - represents a sample generated for a critic in R, later
///   initialized in the Sampler implementation
/// Beside that `ActorSample` and `CriticSample` are used to calculate the loss
/// between the values produced by the agent and the optimal values with the use
/// of a PPO algorithm and also to train the models themselves.
#[must_use]
#[derive(Setters, Debug)]
#[setters(prefix = "fill_", borrow_self)]
pub struct Sampler {
    #[setters(skip)]
    num_steps: i64,

    #[setters(skip)]
    batch_size: i64,

    #[setters(skip)]
    device: Device,

    actions: Tensor,
    action_log_probs: Tensor,
    observations: Tensor,
    advantages: Tensor,
    returns: Tensor,
}

pub struct ActorSample {
    pub actions: Tensor,
    pub action_log_probs: Tensor,
    pub observations: Tensor,
    pub advantages: Tensor,
}

pub struct CriticSample {
    pub observations: Tensor,
    pub returns: Tensor,
}

impl Sampler {
    pub fn new(num_steps: usize, batch_size: usize, device: Device) -> Self {
        let num_steps = num_steps as i64;
        let batch_size = batch_size as i64;

        let empty_tensor = || Tensor::zeros([0], (Kind::Double, device));

        Self {
            num_steps,
            batch_size,
            actions: Tensor::zeros([0], (Kind::Int64, device)),
            action_log_probs: empty_tensor(),
            observations: empty_tensor(),
            advantages: empty_tensor(),
            returns: empty_tensor(),
            device,
        }
    }

    fn generate_sample_indexes(&self) -> Tensor {
        Tensor::randint(
            self.num_steps,
            [self.batch_size],
            (Kind::Int64, self.device),
        )
    }

    pub fn actor_sample(&self) -> ActorSample {
        let sample_indexes = self.generate_sample_indexes();

        let actions = self.actions.index_select(0, &sample_indexes);
        let action_log_probs = self.action_log_probs.index_select(0, &sample_indexes);
        let observations = self.observations.index_select(0, &sample_indexes);
        let advantages = self.advantages.index_select(0, &sample_indexes);

        ActorSample {
            actions,
            action_log_probs,
            observations,
            advantages,
        }
    }

    pub fn critic_sample(&self) -> CriticSample {
        let sample_indexes = self.generate_sample_indexes();

        let observations = self.observations.index_select(0, &sample_indexes);
        let returns = self.returns.index_select(0, &sample_indexes);

        CriticSample {
            observations,
            returns,
        }
    }
}
