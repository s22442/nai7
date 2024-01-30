use tch::Tensor;

mod gym;

pub trait Env: Clone + Send + 'static {
    fn observation_space(&self) -> Vec<usize>;
    fn action_space(&self) -> usize;
    fn observation(&self) -> Tensor;
    fn episode_ended(&self) -> bool;
    fn reset(&mut self);
    fn step(&mut self, action: u32) -> f64;
}

pub use gym::Wrapper as GymWrapper;
