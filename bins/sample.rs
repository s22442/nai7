use tch::{nn::VarStore, Device};
use utils::{
    env::{self, Env},
    model, panic_hook,
};

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
