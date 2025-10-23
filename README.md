# light_mappo

Lightweight version of MAPPO to help you quickly migrate to your local environment.

- [Video (in Chinese)](https://www.bilibili.com/video/BV1bd4y1L73N)  
This is a translated English version. Please click [here](README_CN.md) for the orginal Chinese readme.

This code has been used in the following paper:

```bash
@inproceedings{he2024intelligent,
  title={Intelligent Decentralized Multiple Access via Multi-Agent Deep Reinforcement Learning},
  author={He, Yuxuan and Gang, Xinyuan and Gao, Yayu},
  booktitle={2024 IEEE Wireless Communications and Networking Conference (WCNC)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
@article{qiu2024enhancing,
  title={Enhancing UAV Communications in Disasters: Integrating ESFM and MAPPO for Superior Performance},
  author={Qiu, Wen and Shao, Xun and Loke, Seng W and He, Zhiqiang and Alqahtani, Fayez and Masui, Hiroshi},
  journal={Journal of Circuits, Systems and Computers},
  year={2024},
  publisher={World Scientific}
}
@article{qiu2024optimizing,
  title={Optimizing Drone Energy Use for Emergency Communications in Disasters via Deep Reinforcement Learning},
  author={Qiu, Wen and Shao, Xun and Masui, Hiroshi and Liu, William},
  journal={Future Internet},
  volume={16},
  number={7},
  pages={245},
  year={2024},
  publisher={MDPI}
}
@inproceedings{yu2024path,
  title={Path Planning for Multi-AGV Systems Based on Globally Guided Reinforcement Learning Approach},
  author={Yu, Lanlin and Wang, Yusheng and Sheng, Zixiang and Xu, Pengfei and He, Zhiqiang and Du, Haibo},
  booktitle={2024 IEEE International Conference on Unmanned Systems (ICUS)},
  pages={819--825},
  year={2024},
  organization={IEEE}
}
```

## Table of Contents

- [Background](#Background)
- [Installation](#Installation)
- [Usage](#Usage)

## Background

The original MAPPO code was too complex in terms of environment encapsulation, so this project directly extracts and encapsulates the environment. This makes it easier to transfer the MAPPO code to your own project.

## Installation

Simply download the code, create a Conda environment, and then run the code, adding packages as needed. Specific packages will be added later.

## Usage

- The environment part is an empty implementation, and the implementation of the environment part in the light_mappo/envs/env_core.py file is: [Code] (https://github.com/tinyzqh/light_mappo/blob/main/envs/env_core.py)

```python
import numpy as np
class EnvCore(object):
    """
    # Environment Agent
    """
    def __init__(self):
        self.agent_num = 2 # set the number of agents(aircrafts), here set to two
        self.obs_dim = 14 # set the observation dimension of agents
        self.action_dim = 5 # set the action dimension of agents, here set to a five-dimensional

    def reset(self):
        """
        # When self.agent_num is set to 2 agents, the return value is a list, and each list contains observation data of shape = (self.obs_dim,)
        """
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = np.random.random(size=(14, ))
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions):
        """
        # When self.agent_num is set to 2 agents, the input of actions is a two-dimensional list, and each list contains action data of shape = (self.action_dim,).
        # By default, the input is a list containing two elements, because the action dimension is 5, so each element has a shape of (5,)
        """
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(self.agent_num):
            sub_agent_obs.append(np.random.random(size=(14,)))
            sub_agent_reward.append([np.random.rand()])
            sub_agent_done.append(False)
            sub_agent_info.append({})

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
```


Just write this part of the code, and you can seamlessly connect with MAPPO. After env_core.py, two files, env_discrete.py and env_continuous.py, were separately extracted to encapsulate the action space and discrete action space. In elif self.continuous_action: in algorithms/utils/act.py, this judgment logic is also used to handle continuous action spaces. The # TODO here in runner/shared/env_runner.py is also used to handle continuous action spaces.

In the train.py file, choose to comment out continuous environment or discrete environment to switch the demo environment.

## Cite this work

If you use `light_mappo`, please cite:

```bibtex
@software{light_mappo,
  author  = {Zhiqiang He},
  title   = {light\_mappo: Lightweight MAPPO implementation},
  year    = {2025},
  url     = {https://github.com/tinyzqh/light_mappo},
  note    = {Version v0.1.0}
}
```

## Related Efforts

- [on-policy](https://github.com/marlbenchmark/on-policy) - 💌 Learn the author implementation of MAPPO.

## Maintainers

[@tinyzqh](https://github.com/tinyzqh).

## Translator
[@tianyu-z](https://github.com/tianyu-z)

## License

[MIT](LICENSE) © tinyzqh

