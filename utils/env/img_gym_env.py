import gym
from PIL import Image


class ImageEnvWrapper(gym.Wrapper):
    def __init__(self, env, img_path):
        gym.Wrapper.__init__(self, env)
        self._cnt = 0
        self._img_path = img_path

    def step(self, action):
        step_result = self.env.step(action)
        obs, _, _, _ = step_result
        img = Image.fromarray(obs, 'RGB')
        img.save(f"{self._img_path}/env-{self._cnt:05d}.png")
        self._cnt += 1
        return step_result


def make(env_name, img_dir):
    env = gym.make(env_name)
    return ImageEnvWrapper(env, img_dir)
