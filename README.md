# light_mappo

Lightweight version of MAPPO to help you quickly migrate to your local environment.

轻量版MAPPO，帮助你快速移植到本地环境。


## Table of Contents

- [背景](#背景)
- [安装](#安装)
- [用法](#用法)


## 背景

MAPPO原版代码对于环境的封装过于复杂，本项目直接将环境封装抽取出来。更加方便将MAPPO代码移植到自己的项目上。

## 安装

直接将代码下载下来，创建一个Conda环境，然后运行代码，缺啥补啥包。具体什么包以后再添加。

## 用法

- 环境部分是一个空的的实现，文件`light_mappo/envs/env_core.py`里面环境部分的实现：[Code](https://github.com/tinyzqh/light_mappo/blob/main/envs/env_core.py)

```python
import numpy as np
class EnvCore(object):
    """
    # 环境中的智能体
    """
    def __init__(self):
        self.agent_num = 2  # 设置智能体(小飞机)的个数，这里设置为两个
        self.obs_dim = 14  # 设置智能体的观测纬度
        self.action_dim = 5  # 设置智能体的动作纬度，这里假定为一个五个纬度的

    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        """
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = np.random.random(size=(14, ))
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作纬度为5，所里每个元素shape = (5, )
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


只需要编写这一部分的代码，就可以无缝衔接MAPPO。在env_core.py之后，单独提出来了两个文件env_discrete.py和env_continuous.py这两个文件用于封装处理动作空间和离散动作空间。在algorithms/utils/act.py中elif self.continuous_action:这个判断逻辑也是用来处理连续动作空间的。和runner/shared/env_runner.py部分的# TODO 这里改造成自己环境需要的形式即可都是用来处理连续动作空间的。

在train.py文件里面，选择注释连续环境，或者离散环境进行demo环境的切换。

## Related Efforts

- [on-policy](https://github.com/marlbenchmark/on-policy) - 💌 Learn the author implementation of MAPPO.

## Maintainers

[@tinyzqh](https://github.com/tinyzqh).

## License

[MIT](LICENSE) © tinyzqh

