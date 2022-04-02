# PiMBRL
This repo provides code for our paper [Physics-informed Dyna-style model-based deep reinforcement learning for dynamic control](https://royalsocietypublishing.org/doi/pdf/10.1098/rspa.2021.0618)  ([arXiv version](https://arxiv.org/abs/2108.00128)), implemented in Pytorch.
* Authors: Xin-Yang Liu \[ [Google Scholar](https://scholar.google.com/citations?user=DI9KTLoAAAAJ&hl=en) \], Jian-Xun Wang \[ [Google Scholar](https://scholar.google.com/citations?user=1cXHUD4AAAAJ&hl=en) | [Homepage](http://sites.nd.edu/jianxun-wang/) \]


<p align="center">
<img align="center" width="600" src="/docs/uncontrolled.png">
</p>
<p align="center" > An uncontrolled KS environment. </p>

<p align="center">
<img align="center" width="600" src="/docs/controlled.png">
</p>
<p align="center"> A RL controlled KS environment. </p>

<p align="center">
  <img width="400" src="/docs/performance.png">         
</p>
<p align="center" > PiMBRL performance vs. Model-free RL baseline.</p>
<p align="center"> (Vanilla Model-based RL failed to converge, thus not shown in this figure). </p>




## Abstract
Model-based reinforcement learning (MBRL) is believed to have much higher sample efficiency compared with model-free algorithms by learning a predictive model of the environment. However, the performance of MBRL highly relies on the quality of the learned model, which is usually built in a blackbox manner and may have poor predictive accuracy outside of the data distribution. The deficiencies of the learned model may prevent the policy from being fully optimized. Although some uncertainty analysisbased remedies have been proposed to alleviate this issue, model bias still poses a great challenge for MBRL. In this work, we propose to leverage the prior knowledge of underlying physics of the environment, where the governing laws are (partially) known. In particular, we developed a physics-informed MBRL framework, where governing equations and physical constraints are used to inform the model learning and policy search. By incorporating the prior information of the environment, the quality of the learned model can be notably improved, while the required interactions with the environment are significantly reduced, leading to better sample efficiency and learning performance. The effectiveness and merit have been emonstrated over a handful of classic control problems, where the environments are governed by canonical ordinary/partial differential equations.

## code structure:
* `src/` contains the source code of the framework.
  * `NN/` includes deep nerual network related code.
    * `RL/` includes policy & Q function networks
    * `model.py` contains surrogate models
  * `RLago/` contains RL algorithms including `DDPG`, `TD3`, `SAC`.
  * `ModelBased/` 
    * `dyna.py` & `dynav2.py` contains the code for the dyna-style algorithms.
  * `utils/` useful tools.
  * `envs.py` contains self-defined environments, mimic the gym environments.
  
* `examples/` contains a set of examples of using the framework.
  * `ks.py` one-dimensional environment governed by KS equation.
  * `cartpole.py` the classic cartpole problem
  * `pendu.py` modified pendulum problem
  * `burgers.py` one-dimensional environment governed by Burgers' equation.

* `.env`

## Usage:
Please refer to `examples/ks.py` for usage.

## Requirements:
```
    python>=3.8.8
    pytorch==1.8.1
    gym==0.19.0
    numpy>=1.19.2
```

## Citation
If you find this repo useful in your research, please consider citing our paper: [Physics-informed Dyna-style model-based deep reinforcement learning for dynamic control](https://royalsocietypublishing.org/doi/pdf/10.1098/rspa.2021.0618).

``` 
@article{liu2021physics,
  title={Physics-informed Dyna-style model-based deep reinforcement learning for dynamic control},
  author={Liu, Xin-Yang and Wang, Jian-Xun},
  journal={Proceedings of the Royal Society A},
  volume={477},
  number={2255},
  pages={20210618},
  year={2021},
  publisher={The Royal Society}
}
```

## Problems
If you find any bugs in the code or have trouble in running PiMBRL in your machine, you are very welcome to [open an issue](https://github.com/Jianxun-Wang/PIMBRL/issues) in this repository.


## Acknoledgements
The code in `src/RLalgo` is inspired by OpenAI's [spinningup](https://spinningup.openai.com/en/latest/).
