use pyo3::{types::PyTuple, IntoPy, PyObject, PyResult, Python};
use tch::{Kind, Tensor};

use super::Env;

fn parse_py_observation(py: Python<'_>, obs: &PyObject, observation_space: &[usize]) -> Tensor {
    let observation_space = observation_space
        .iter()
        .map(|&x| x as i64)
        .collect::<Vec<_>>();

    let obs = obs.call_method(py, "flatten", (), None).unwrap();

    Tensor::from_slice(&obs.extract::<Vec<f32>>(py).unwrap())
        .view_(observation_space)
        .permute([2, 0, 1])
}

fn reset_py_env(py: Python<'_>, env: &PyObject, observation_space: &[usize]) -> Tensor {
    parse_py_observation(
        py,
        &env.call_method(py, "reset", (), None).unwrap(),
        observation_space,
    )
}

#[derive(Debug)]
pub struct Wrapper {
    name: String,
    img_dir: Option<String>,
    env: PyObject,
    action_space: usize,
    observation_space: Vec<usize>,
    episode_ended: bool,
    observation: Tensor,
}

impl Clone for Wrapper {
    fn clone(&self) -> Self {
        Self::new(&self.name, self.img_dir.as_deref()).unwrap()
    }
}

impl Wrapper {
    pub fn new(name: &str, img_dir: Option<&str>) -> PyResult<Self> {
        Python::with_gil(|py| {
            let env = if let Some(img_dir) = img_dir {
                let sys = py.import("sys").unwrap();
                let path = sys.getattr("path").unwrap();
                let _ = path.call_method("append", ("utils/env",), None).unwrap();
                let gym = py.import("img_gym_env")?;
                gym.call_method("make", (name, img_dir), None)?
            } else {
                let gym = py.import("gym")?;
                gym.call_method("make", (name,), None)?
            };
            let action_space = env.getattr("action_space")?;
            let action_space = action_space.getattr("n")?.extract()?;

            let observation_space = env.getattr("observation_space")?;
            let observation_space: Vec<usize> = observation_space.getattr("shape")?.extract()?;

            let env = env.into();

            let observation = reset_py_env(py, &env, &observation_space);

            Ok(Self {
                name: name.to_owned(),
                img_dir: img_dir.map(ToOwned::to_owned),
                env,
                action_space,
                observation_space,
                episode_ended: false,
                observation,
            })
        })
    }
}

impl Env for Wrapper {
    fn reset(&mut self) {
        Python::with_gil(|py| {
            let observation = reset_py_env(py, &self.env, &self.observation_space);
            self.observation = observation;
            self.episode_ended = false;
        });
    }

    fn step(&mut self, action: u32) -> f64 {
        Python::with_gil(|py| {
            let step = self.env.call_method(py, "step", (action,), None).unwrap();
            let step: &PyTuple = step.extract(py).unwrap();

            let obs = parse_py_observation(
                py,
                &step.get_item(0).unwrap().into_py(py),
                &self.observation_space,
            );
            let reward = step.get_item(1).unwrap().extract().unwrap();
            let episode_ended = step.get_item(2).unwrap().extract().unwrap();

            self.observation = obs;
            self.episode_ended = episode_ended;

            reward
        })
    }

    fn action_space(&self) -> usize {
        self.action_space
    }

    fn observation_space(&self) -> Vec<usize> {
        let mut observation_space = self.observation_space.clone();

        observation_space.swap(0, 1);
        observation_space.swap(0, 2);

        observation_space
    }

    fn observation(&self) -> Tensor {
        self.observation.to_kind(Kind::Double)
    }

    fn episode_ended(&self) -> bool {
        self.episode_ended
    }
}
