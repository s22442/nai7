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
use tch::{nn::VarStore, Device};
use utils::{env, panic_hook, ppo::PPO};

const NUM_EPOCHS: usize = 5000;

const EVAL_INTERVAL: usize = 50;

const SAVE_INTERVAL: usize = 1000;

/// This function focuses on initializing and then building the actor and critic
/// while also later on teaching the model itself.
fn main() {
    panic_hook::init();

    let env = env::GymWrapper::new("SpaceInvaders-v4", None).unwrap();
    let device = Device::Cpu;

    let mut ppo = PPO::builder()
        .actor_vs(VarStore::new(device))
        .critic_vs(VarStore::new(device))
        .env(env)
        .gamma(0.99)
        .lmbda(0.95)
        .clip_epsilon(0.2)
        .actor_learning_rate(1e-4)
        .critic_learning_rate(1e-4)
        .num_steps_per_epoch(250)
        .batch_size(50)
        .actor_train_num_batches(5)
        .critic_train_num_batches(5)
        .num_threads(5)
        .build();

    for i in 1..=(NUM_EPOCHS / EVAL_INTERVAL) {
        ppo.train(EVAL_INTERVAL);
        let avg_return = ppo.evaluate_avg_return();
        println!(
            "Epoch: {}, Average return: {}",
            i * EVAL_INTERVAL,
            avg_return
        );

        if i % SAVE_INTERVAL == 0 {
            ppo.save(format!("ppo-{}", i * EVAL_INTERVAL));
        }
    }
}
