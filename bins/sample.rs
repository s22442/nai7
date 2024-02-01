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
use utils::{
    env::{self, Env},
    model, panic_hook,
};

// This function launches the trained models while also generating appropriate
// images
fn main() {
    panic_hook::init();

    let args = std::env::args().collect::<Vec<String>>();
    let [_, vs_path] = args.as_slice() else {
        panic!("invalid args")
    };

    let mut env = env::GymWrapper::new("SpaceInvaders-v4", Some("dist")).unwrap();
    let mut vs = VarStore::new(Device::Cpu);
    let policy = model::Actor::new(&vs.root(), env.action_space());
    vs.load(vs_path).unwrap();
    vs.double();

    let avg_return = policy.evaluate_avg_return(&mut env);

    println!("Average return: {avg_return}");
}
