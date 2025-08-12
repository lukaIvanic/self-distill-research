


def get_project_config():

    # print(f"USING TRAIN SETTINGS FOR PROJECT CONFIG. TRAIN!!!!")
    from main_space.train_management import get_global_train_manager
    return get_global_train_manager().projectConfig


    # TODO: this is horrible, fix this
    # print(f"USING VALIDATION SETTINGS FOR PROJECT CONFIG. VALIDATION!!!!")
    # from main_space.validation_management import get_global_validation_manager
    # return get_global_validation_manager().projectConfig

def get_training_config():
    return get_project_config().trainingConfig

def get_wandb_config():
    return get_project_config().wandbConfig

def get_logging_config():
    return get_wandb_config().loggingConfig

def get_profiler_config():
    return get_wandb_config().profilerConfig

def get_dataset_config():
    return get_project_config().datasetConfig

def get_hyperparameter_config():
    return get_training_config().hyperParamConfig

def get_distill_config():
    return get_training_config().distillConfig

def get_checkpoint_config():
    return get_project_config().checkpointConfig