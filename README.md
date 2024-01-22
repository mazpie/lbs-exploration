# LBS - Curiosity-Driven Exploration via Latent Bayesian Surprise

[[website](https://lbsexploration.github.io/)] [[paper](https://arxiv.org/abs/2104.07495)]

This repository is the official implementation of Curiosity-Driven Exploration via Latent Bayesian Surprise.

<p align="center">
    <img src='_cover.gif' width=90%>
</p>


If you find the code useful, please refer to our work using:

```
@article{Mazzaglia2022LBS, 
    title={Curiosity-Driven Exploration via Latent Bayesian Surprise}, 
    volume={36}, 
    url={https://ojs.aaai.org/index.php/AAAI/article/view/20743}, 
    DOI={10.1609/aaai.v36i7.20743},  
    number={7}, 
    journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
    author={Mazzaglia, Pietro and Catal, Ozan and Verbelen, Tim and Dhoedt, Bart}, 
    year={2022}, 
    month={Jun.}, 
    pages={7752-7760} 
}
```

## Installation

### [RECOMMENDED] Conda env

 Create and activate a conda environment running:

```[bash]
conda create -n lbs python=3.8`
conda activate lbs
```

### Dependencies

To install dependencies, run:

`pip install -r requirements.txt`

## Training Code

In order to run experiments you can use the following:

### Mountain Car

```
python main.py --env-name "MountainCarSparse-v0" --algo ppo-lbs --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0.01 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 102400 --beta 0.1 --no-cuda --log-dir ./logs/mountaincarsparse/lbs-0 --seed 0
```

### Ant Maze

Make sure you have correctly installed and configured `mujoco-py`.

```
python main.py --env-name "MagellanAnt-v2" --algo ppo-lbs --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0.01 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 251904 --beta 0.1 --no-cuda --log-dir ./logs/antmaze/lbs-0 --seed 0 
```

### Half Cheetah

Make sure you have correctly installed and configured `mujoco-py`.

```
python main.py --env-name "HalfCheetahSparse-v3" --algo ppo-lbs --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0.01 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 501760 --beta 0.1 --no-cuda --log-dir ./logs/halfcheetahsparse/lbs-0 --seed 0
```

### Atari 

Make sure you have correctly installed and configured `atari-py` (you may need to import the Atari ROMs).

```
python main.py --env-name SpaceInvadersNoFrameskip-v4 --algo ppo-lbs --use-gae --lr 1e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 128 --num-steps 128 --num-mini-batch 8 --ppo-epoch 3 --log-interval 1 --entropy-coef 0.001 --num-env-steps 100000000 --log-dir ./logs/SpaceInvaders/lbs-5 --seed 1 --beta 0.01
```

### Super Mario Bros. 

Make sure to have correctly configured `gym-retro` (you may need to import Mario's ROM).

```
python main.py --env-name MarioBrosNoFrameskip-v4 --algo ppo-lbs --use-gae --lr 1e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 128 --num-steps 128 --num-mini-batch 8 --ppo-epoch 3 --log-interval 1 --entropy-coef 0.001 --num-env-steps 100000000 --log-dir ./logs/MarioBros/lbs-5 --seed 1 --beta 0.01
```

### Mountain Car Stochasic Frozen/Evolving

```
python main.py --env-name "MountainCarStochastic-Frozen" --algo ppo-lbs --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0.01 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 102400 --beta 2. --no-cuda --log-dir ./logs/mountaincarstoch-frozen/lbs-0 --seed 0
```

#### Acknowledgments

We would like to thank the authors of the following repositories for their open source code:

[PPO implementation](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) [PPO training code]

[Model-based active exploration](https://github.com/nnaisense/MAX) [Ant Maze environment]

[Large-Scale Study of Curiosity-Driven Learning](https://github.com/openai/large-scale-curiosity) [Arcade games experiments]