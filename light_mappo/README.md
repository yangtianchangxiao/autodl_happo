# light_mappo

如何运行：
在 \train\train.py中，有'main'

我们的环境在\envs\EnvDrone中，包含两个init文件和一个EnvDrones.py文件，主要修改在于使得每个agent都拥有独立的reward和state与action，并且没有采用传统mappo中，将多个智能体的运行视作各个智能体依次运行的做法，而是在同一时间计算多个智能体的reward和state, 将时间复杂度从o（kn）降到了o（n）,其中k是智能体的数量。为此，我们主要对目标点检测与删除，障碍物检测，智能体之间互相感知彼此的距离等机制进行了改动，使得在同一时间内，每个智能体的上述三种工作不受其他智能体的干扰。

注：在传统mappo中，由于把多个智能体的执行分解成了各个智能体依次执行，上一个智能体检测到目标点后会把目标点直接删除，导致下一个智能体如果也做出了检测该目标点的action，那么他并不会得到对应的reward, 我认为这一点在IPPO中是不合理的，相当于智能体的决策被其他智能体干扰了。但是此处也有许多的疑问，比如，这么设计是否会导致智能体们集体冲向相同的目标点。 一个暂时的回答是不会，因为在这种机制中，只有在同一时间内，多个智能体同时检测到同一个目标点，才会使得多个智能体同时收到对应的reward, 这个条件比较苛刻。

注2：如果该机制确实有问题，那么改回按照各个智能体按照顺序执行也很方便

注3：即便是按照顺序执行，也应当使用“只执行一次env.step”的做法，而不是分别使用env.step()来执行多个动作，从而尽量避免重复运行程序.

以下是原文的ReadMe, 原文链接：https://github.com/tinyzqh/light_mappo

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

