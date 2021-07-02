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

- 环境部分是一个空的的实现，文件`light_mappo/envs/env_wrappers.py`里面环境部分的实现：[Code](https://github.com/tinyzqh/light_mappo/blob/main/envs/env_wrappers.py)

```python
class Env(object):
    """
    # 环境中的智能体
    """
    def __init__(self, i):
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


只需要编写这一部分的代码，就可以无缝衔接MAPPO。初始版本，后期这一部分会单独提出来。


## Related Efforts

- [on-policy](https://github.com/marlbenchmark/on-policy) - 💌 Learn the author implementation of MAPPO.

## Maintainers

[@tinyzqh](https://github.com/tinyzqh).

## License

[MIT](LICENSE) © tinyzqh

