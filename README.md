# The Artificial Experience

It's time to give artificial intelligence a taste of reality! The `artificial-experience` is a library to facilitate training and evaluating models, optimziers, pipelines, and training paradigms across dozens of tasks, domains, dataset loaders, environments, and hubs simultaneously, lifelong, and in-context. This library also provides a highly complex, multi-task, open-world learning environment the `ArtificialExperience` which can be used quickly run AGI experiments:
```python
import artificial_experience as ae

env = ae.ArtificialExperience()

# `env` is a `dm_env.DmEnv` instance.
timestep = env.reset()
while True:
    action = agent.forward(timestep.observation)
    timestep = env.step(action)
    # timestep is a `TimeStep` namedtuple  with fields (step_type, reward, discount, observation)
```

## Key Principles

**All data streams are environments**.  Datasets are wrapped into `DatasetEnv`s. 1 minibatch = 1 environment step. You can compose environments into pipelines You can specify a loss function for the `DatasetEnv` or make it instinsic to the agent. We provide utilities to convert supervised learning datasets into Markovian environments (e.g.: observe X, agent's action is a prediction, then observe Y and recieve reward).

**Inputs and outputs are structured into modalities**. Each observation and action is a Python `dict` object with nested `Tensor` or `None` values. Environments also have a dictionary of `Modality` objects which define a combination of `structure` (flat, set, sequence, grid, or graph), `representation` (binary, categorical, integer, real), and `context` ("natural", "computer", or other natural language tag) that can be used to determine network architecture.

**Environments compose lifelong learning pipelines**. We provide the following pipeline components:

- `Interleave` is a high level version of `SynEnvironment` that interleaves interactions from a list of environments with an arbitrary interleave pattern. For example, the interleave patten `[EnvA, EnvB, EnvC, EnvB, EnvC]` takes the first interaction from `EnvA`, the second from `EnvB`, the third from `EnvC`, the fourth from `EnvB`, and the fifth from `EnvC`. The environment is done when either the first or all sub-environments are done.

- `Multitasking` makes an agent interact in multiple environments simultaneously. The environment is done when either the first or all sub-environments are done.

- `Teacher` occasionally reverts the wrapped environment's state to a previous state where performance maximally increased. It can buffer with a rolling history, top k, or arbitrary `should_store_state` function. This wrapper is useful for implementing Go-Explore-type algorithms.

- `Augment` is a base class for wrapper environments that augment specific inputs and outputs.

- `Dropout` is an `Augment` environment wrapper that occasionally replaces an input or output value with 0 or another specified value.

- `Noisy` is an `Augment` environment wrapper that adds noise to an input or output value.

- `Repeat` is an `Augment` environment wrapper that occasionally repeats an input or output value for multiple interaction steps.

- `DropKeyValue` is an `Augment` environment wrapper that occasionally drops an input or output value *and key* from the dictionary.

- `ObserveReward` directly includes wrapped environment's reward in the observation space.

- `Advantage` directly includes the wrapped environment's Nth-order reward advantage in the observation space.

- `ObserveActions` feeds the agent's actions into the next interaction step observation. 

- `PredictInputs` expects and rewards agents for predicting the next input values.

- `Imaginary` uses a prediction agents to generate imaginary environments.

- `StaticTimescaled` allows developers to control the ratio of agent steps to environment steps (ratio = agent timesteps / environment timesteps). This ratio can be any nonnegative floating point value (0 <= ratio <= inf). For example, the ratio is 1, the agent and environment are synchronized. If the ratio is 0, the agent is never notified of an environment update. If the ratio is 0.5, the agent is notified of an environment update half the time. If the ratio is 4, the agent gets to observe the same observation 4 times before its action is sent to the environment. When the environment get more steps than the agent, the agent's last action can be repeated or ignored. When the agent gets more steps than the environment, only the agent's last action is sent to the environment. Developers can also specify custom `pool_observation` and `pool_action` functions. 

- `DynamicTimescaled` is like `StaticTimescaled` but it gives the agent the ability to observe and modify its external environment interaction timescale instead by observing and acting on the `ratio` modality.

- `PenalizeCompute` decreases reward proportional to the amount of compute used since the last interaction. This penalizes the agent (such as in a `DynamicTimescaled`) for using too much compute.

- `ReplayBuffer`: a wrapper that stores and replays observation-action-reward trajectories. Can save/load from disk. Extendable for real-time monitoring.

- `Lambda` allows arbitrary code to modify the observations and actions as they are passed along the pipeline.


2. **Datasets are wrapped into `DatasetEnv`s** 1 minibatch = 1 environment step. For supervised datasets, agents observe both labels and targets simultaneously. NOTE: the environment doesn't provide an external reward by default; your agent should train itself given only x and y. `DatasetEnv` tries to automatically guess modalities, but you can override the observation and action structure. Many supervised and self-supervised learning problems can be structured using this environment and a prediction-based learning objective.

TODO: make this into the description for dataset env

3. **Environments are wrapped into `SynEnv`s** which presents a single batched interaction sequence spanning multiple environments staggered along the batch and time axis. `SynEnv` maintains a separate running environment on each of its batch indices. Agents can select when and which environment to transition to by observing a set of `all_environments` and a sequence of `current_environments` and producing a sequence of `next_environments` on each timestep. Whenever an element of `next_environment` is not `None`, the `SynEnv` replaces the environment on that respective batch axis. Environments are not presented to the agent whenever they are done. Whenever an environment changes, the `SynEnv` calls a `transition_fn` which developers can supply to add input and output modalities to the policy. The `SynEnv` can optionally present a few interaction steps where the agent observes a natural language instruction (e.g. `predict the class of images on imagenet`) on the input key `'text:instruction'` associated with the new environment. Finally, interaction begins in the new environment.

4. **Environments compose lifelong learning pipelines**. Each environment is a `dm_env` and can be wrapped into pipelines and networks.
    - `Interleave` is a high level version of `SynEnvironment` that interleaves interactions from a list of environments with an arbitrary interleave pattern. For example, the interleave patten `[EnvA, EnvB, EnvC, EnvB, EnvC]` takes the first interaction from `EnvA`, the second from `EnvB`, the third from `EnvC`, the fourth from `EnvB`, and the fifth from `EnvC`. The environment is done when either the first or all sub-environments are done. 
    - `Multitasking` makes an agent interact in multiple environments simultaneously. The environment is done when either the first or all sub-environments are done.
    - `Teacher` occasionally reverts the wrapped environment's state to a previous state where performance maximally increased. It can buffer with a rolling history, top k, or arbitrary `should_store_state` function. This wrapper is useful for implementing Go-Explore-type algorithms.
    - `Augment` is a base class for wrapper environments that augment specific inputs and outputs.
    - `Dropout` is an `Augment` environment wrapper that occasionally replaces an input or output value with 0 or another specified value.
    - `Noisy` is an `Augment` environment wrapper that adds noise to an input or output value.
    - `Repeat` is an `Augment` environment wrapper that occasionally repeats an input or output value for multiple interaction steps.
    - `DropKeyValue` is an `Augment` environment wrapper that occasionally drops an input or output value *and key* from the dictionary.
    - `ObserveReward` directly includes wrapped environment's reward in the observation space.
    - `Advantage` directly includes the wrapped environment's Nth-order reward advantage in the observation space.
    - `ObserveActions` feeds the agent's actions into the next interaction step observation. 
    - `PredictInputs` expects and rewards agents for predicting the next input values.
    - `Imaginary` uses a prediction agents to generate imaginary environments.
    - `StaticTimescaled` allows tuning the amount of 'ponder' steps that a model gets between external environment interactions. Inputs can be dropped or repeated. Outputs can be averaged, max pooled, min pooled, randomly selected, or last index selected along the time dimension.
    - `DynamicTimescaled` is like `StaticTimescaled` but it gives the agent the ability to observe and modify its external environment interaction timescale.
    - `PenalizeCompute` decreases reward proportional to the amount of compute used since the last interaction. This penalizes the agent (such as in a `DynamicTimescaled`) for using too much compute.
    - `ReplayBuffer`: a wrapper that stores and replays observation-action-reward trajectories. Can save/load from disk. Extendable for real-time monitoring.
    - `Lambda` allows arbitrary code to modify the observations and actions as they are passed along the pipeline.

5. **The `ArtificialExperience` provides a ready-made pipeline of environments and datasets to train on**.

## Suggestions for developing general agents
- Get a good training pipeline established including qualitative sanity checks with personal qualitative observation.
- Make sure your agent can master limited domains before introducing it to the `ArtificialExperience`.
- Maintain reccurent states across transitions to learn in-context meta-learning, and architect your model so that the recurrent state has strong expressive potential over the activation landscape.
- Make sure your model can dynamically add new encoders and decoders.
- May train on input prediction error. Try to put the 1st-order optimizer inside your model.

## Getting Started

**NOT CURRENTLY PUBLISHED**

First, install the `artificial-experience`
```base
$ pip install artificial-experience
```

You can optionally install extras:
```
$ pip install artificial-experience[heavy]  # includes environments that take up a lot of disk space
$ pip install artificial-experience[baselines]  # with baselines
$ pip install artificial-experience[all]  # all extras
```

## Examples

TODO

```python



```


```python
env = AEEnv(envs=[
    DatasetEnv(tfds.load('coco')), # multimodal information
    DatasetEnv(hub.load('hub://activeloop/mnist-train'))  # cloud-native data
    DatasetEnv(tfds.load('anli'), epochs=4, batch_size=1024), # quick customization
    gym.make('CartPole-v0'), # continous observation, discrete control
    gym.make('Pong-v0'), # rgb image, discrete actions
    gym.make('HalfCheetah-v2'), # continuous observation, continuous control
    gym_starcraft.envs.starcraft_base_env(), # starcraft env
    pettingzoo.atari.mario_bros_v2.env() # multiagent atari env
])
```

`AEEnv` also makes it easy to train on prespecified problem domains with datasets and environments minimally specified by some overlapping hierarchial tag-based system. Not all environments have the `.tag` attribute, so those will be ignored. However, the inbuilt list of envionrments should all support this schema. These filters can be changed at any moment between `AEEnv` steps. See Appendix A for a list of what I want to support.
```python
env = AEEnv(
    include=[
        domains.text_commonsense, 
        'domains.image', 
        'domains.multiagent'],
    exclude=[
        lambda x: False if isinstance(x, Env) and x.version<2 else True,
        domains.multiagent.atari],
) # train on text-commonsense (specific), image datasets (broad), and multiagent RL environments (broad) but don't train on the multiagent/atari environment or multiagent environments that don't have a environment specified reward.

env = AEEnv() # train on all inbuilt datasets and environments
```

## Features

TODO. List every public function, class, and method.

- `ArtificialExperience` presents an intersection of [API-compatible](https://jacobfv.github.io/blog/the-api/) environments and datasets. 


## Datasets and environments

The categories overlap. For instance, image captioning might be in the `image` category, but also in the `text` category. The high-level hierarchy might be:
- images
- text
- video
- audio
TODO

### NLP
from Google's [FLAN blog post](https://ai.googleblog.com/2021/10/introducing-flan-more-generalizable.html): 
- Natural language inference: ANLI, RTE, CB, SNLI, MNLI, QNLI, WNLI, QNLI, 
- Commonsense: CoPA, HeliaSwag, PiQA, StoryCloze
- Sentiment: IMDB, Sent140, SST-2, Yelp
- Paraphrase: MRPC, QQP, PAWS, STS-B
- Closed book QA: ARC (easy/chal), NQ, TQA
- Struct to Text: CommonGen, DART, E2ENLG, WEBNLG
- Reading Comp: 
- Reading Comp w/o commonsensne:
- Conference:
- Misc.:
- Summarization:
- Translation:

### Images

### Video

### 
CLEVERER
BEHAVIOR
Anymal universe in IsaacGym

VNCEnv
