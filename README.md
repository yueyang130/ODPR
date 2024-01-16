# ODPR: Decoupled Prioritized Resampling for Offline RL

ODPR is a plug-and-play component for offline RL algorithms, where comprises two stage: (1) calculate static priority weights. (2) Train offline RL algorithms on the prioritized dataset by resampling or reweighting. 

As case studies, we evaluate ODPR on five different algorithms, including BC, TD3+BC, Onestep RL, CQL, and IQL on D4RL. Extensive experiments demonstrate that both ODPR-A and ODPR-R significantly improve the performance for all baseline methods.
We make public the priority weights of D4RL benchmark [here](https://drive.google.com/drive/folders/1QybIPy90nSrIIQbZrWBuPfoXMeIs1jCS?usp=sharing). 


## Usage

### ODPR-A
To calculate (unnormalized) ODPR-A weights, run the code in `main` branch by:
```
python main.py --env hopper-medium-expert-v2 --seed 1 --n_step 1  --iter 5  --first_eval_steps 1000000 --bc_eval_steps 1000000 
```
Except for kitchen and pen:
```
python main.py --env kitchen-complete-v0 --seed 1 --n_step 5  --iter 5  --first_eval_steps 500000 --bc_eval_steps 500000 
```
In the paper, we run 3 seeds and get the average to reduce variance of ODPR-A weights. 

### ODPR-R
The code of getting (unnormalized) ODPR-R weights is [here](https://github.com/yueyang130/TD3_BC/blob/9285f1c0ce95cc5e2b8c4eb52fccccb6c7b523bd/utils.py#L174), which is also included in the case study codes. No need for extra running to get ODPR-R weights.

### Visualization 
Visualization of ODPR-A and ODPR-R weights can be found in `load_weights.ipynb`. 

### Case Studies
We provide the codes for cases studies (bandit, BC, TD3+BC, IQL, CQL, OnestepRL), which can be found in the corresponding branches. The code for BC is at the `OnestepRL` branch. Detailed usages is at `README.md` in these branches. 
We provide both re-sampling and re-weighting implementations for ODPR in case studies, both of which achieves similar performance in the majority of environments. However, re-sampling is more stable than re-weighting in several games (e.g., kitchen). Therefore, we recommond  re-sampling implementations.
Note that for re-sampling and re-weighting, the priority weights from the first stage should be normalized. The normalization has been included in the codes of the second stage, no need for extra processing.

We give some explanation about the config variables in the case study codes:
- reweight: bool representing whether to reweight by ODPR-A/ODPR-R priority weights
- resample: bool representing whether to resampler by ODPR-A/ODPR-R priority weights
- reweight_eval: bool representing whether to reweight the policy evaluation term. Be valid only when reweight is true. If it's set to false, only reweight the policy constraint and improvement terms.
- two_sampler: bool representing whether to use an extra uniform sampler for the policy evaluation term. If two_sampler is set to true and resample is set to true, only resample for the policy constraint and improvement terms.
- bc_eval: bool representing whether to use ODPR-A. If it's set to false, use ODPR-R weights for resampling/reweighting.
- weight_num: how many ODPR-R weights to compute the average. Default is 3.
- weight_path: A string representing where the ODPR-A priority weights is saved. An example is `./weights/hopper-medium-v2_%s.npy`, where `%s` is a placeholder for seed. Then the script would automatically load `./weights/hopper-medium-v2_1.npy`, `./weights/hopper-medium-v2_2.npy`, and `./weights/hopper-medium-v2_3.npy`.
- iter: The iteration which the ODPR-A priroity weights comes from.
- std: The standard deviation the ODPR-A priroity weight is sacled to.


## *Note
The code in this repository has been reorganized. Errors that may arise during the organizing process could lead to code malfunctions or discrepancies from the original research results. If you encounter any problems, please raise issues. I will go and fix these bugs.



## Credits
The code of case studies are built on [TD3+BC](https://github.com/sfujim/TD3_BC), [CQL](https://github.com/young-geng/JaxCQL), [IQL](https://github.com/ikostrikov/implicit_q_learning), and [OnetepRL](https://github.com/davidbrandfonbrener/onestep-rl).


