class BaseEnv(dataclass):

    state: any
    observation_spec: dict[str, Modality]
    action_spec: dict[str, Modality]
    last_observation: dict[str, NestedTensor]
    last_action: dict[str, NestedTensor]

    def step(self, obs, *args, **kwargs):
        raise NotImplementedError('subclasses should implement this method')