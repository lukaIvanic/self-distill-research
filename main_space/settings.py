import os

import torch
from torch.amp import GradScaler

class WandbConfig:

    class LoggingConfig:
        def __init__(self):
            self.strict_error = True

    class ProfilerConfig:
        def __init__(self):
            self.profiler_wait_steps = 100
            self.profiler_warmup_steps = 10
            self.profiler_active_steps = 2
            self.profiler_repeat_cycles = 1  # Total active steps = profiler_active_steps * profiler_repeat_cycles
            self.profiler_record_shapes = True
            self.profiler_profile_memory = True
            self.profiler_with_ops = False
            self.profiler_with_stack = False  # Set to False by default, as it can be costly

    def __init__(self):
        # --- Overall Logging Control ---
        self.is_wandb_enabled = True  # Master switch for all W&B interactions
        self.is_profiler_enabled = False  # Master switch for torch.profiler
        self.is_trace_handler_custom = False  # Choose dir where pytorch.profiler will save measured metrics

        # --- W&B Configuration ---
        self.wandb_project_name = "self-distill-research"
        self.wandb_entity = "luka_newbie"
        self.wandb_watch_level = "all"  # Options: "all", "gradients", "parameters", "none"
        self.wandb_log_freq_model_watch = 1000  # Frequency for wandb.watch
        self.wandb_log_freq_metrics = 200  # Frequency for wandb.log() for loss, lr, etc.
        self.does_wandb_log_graph = False  # Enable to get 'model' tab in wandb
        self.log_validation = True

        self.profilerConfig = self.ProfilerConfig()
        self.loggingConfig = self.LoggingConfig()


class DatasetConfig:
    def __init__(self):

        self.tokenizer_path  = os.path.join(os.getcwd(), "dataset_creation/tokenizer/1_raw_wikitext103_bpe_vocab_5000.json")
        self.cache_dir = os.path.join(os.getcwd(), "/dataset_creation/temp_files/cache_hf_datasets")
        self.dataset_name = "wikitext"
        self.dataset_config = "wikitext-103-v1"
        self.num_workers_dataloader = 8
        print(f"NUMBER OF WORKERS FOR DATALOADER IS SET TO: {self.num_workers_dataloader}!!!!!!!!!")
        self.pin_memory_dataloader = True
        print(f"PIN MEMORY DATALOADER IS SET TO: {self.pin_memory_dataloader}!!!!!!!!!")
        self.vocab_size_from_tokenizer = True


class TrainingConfig:


    class HyperparameterConfig:

        def __init__(self):
            self.run_name = "2M_only_first_half_for_12k_steps"
            self.vocab_size = 5000  # Dummy vocab size
            self.d_model = 128  # Embedding dimension / model dimension
            self.num_heads = 8  # Number of attention heads
            self.num_layers = 5  # Number of Transformer blocks
            self.ctx_len = 512  # Max sequence length for dummy data and positional embeddings
            self.dropout_rate = 0.01
            self.attach_aux_heads = True

    class DistillConfig:

        def __init__(self):

            self.distill_mode = 'logits_outputs' # Can be "logits_outputs", "hidd_distill", "hidd_and_logits_distill", "attn_distill"


    def __init__(self):

        self.warmup_steps = int(2000)
        self.train_steps = int(12000)
        self.peak_lr = 1e-3
        self.batch_size = 32
        self.scheduler_type = "cosine"  # Options: "cosine", "inverse_sqrt", "linear", "custom", "adaptable"
        self.min_lr = 1e-4
        self.max_hidd_loss = 8  # Only applies for "adaptable" scheduler type
        self.training_precision = "bfloat16"  # Options: "bfloat16", "float16" (uses GradScaler), "float32"
        self.gradient_clip_norm = 1.0
        self.seed = 42

        self.beta1 = 0.9  # default 0.9
        self.beta2 = 0.999  # default 0.999, others 0.95
        self.weight_decay = 0.01


        self.experimental_steps = int(150)
        self.experimental_stop = False


        self.precision_dtype = None
        self.scaler = None
        self.initialize_precision()
        self.doesClipGradients = True
        self.distill_enabled = False
        self.distill_stop_step = 1000 # After this many steps, loss will be calculated classically
        self.distill_stop_cooldown_steps = 1 # After self.distill_stop_step num of steps, perform linear cooldown, after which loss will be purely from hard targets
        self.teacher_alias_tag = "30M_classic:run_lc5nraox_step_11999"

        self.half_by_half_training = False # If enabled, will first train the first half_index+1 trans blocks for self.separation_step steps, and then the rest of the transformer blocks for self.train_steps - self.separation_step steps.
        self.separation_step = 4000 # Num of steps to train the first part of the transformer block
        self.continue_full_step = 6000 # The step at which the first part of the model will unfreeze, and the whole model will continue to train.
        self.half_index = 5 # If half_by_half training is enabled, this index represent the last transformer block in the "first half" of the training. After the First half has been trained, the rest of the transformer blocks will be attached newly initialized.



        self.hyperParamConfig = self.HyperparameterConfig()
        self.distillConfig = self.DistillConfig()


    def initialize_precision(self):
        if self.training_precision == "float16":
            self.precision_dtype = torch.float16
            self.scaler = GradScaler(device='cuda', enabled=True)
            print(f"Using Automatic Mixed Precision with dtype: {self.precision_dtype} and GradScaler.")
        elif self.training_precision == "bfloat16":
            self.precision_dtype = torch.bfloat16
            print(f"Using Automatic Mixed Precision with dtype: {self.precision_dtype}.")
        elif self.training_precision == "float32":
            self.precision_dtype = torch.float32
            print("Using float32 precision.")


class CheckpointConfig:
    def __init__(self):
        # TODO: set option to save or not to save, be careful it doesn't
        #       contradict with other wandb logging options,
        #       in the validate_config function

        self.artifact_base_names = ["50_mil_classic",
                                    "sub_1M_distill_dummies",
                                    "distilled_model_dummies",
                                    "sub_1M_distill_dummies_smaller",
                                    "ctx_len_1M_exp",
                                    "100M_teacher",
                                    "1M_classic",
                                    "2M_classic",
                                    "5M_classic",
                                    "10M_classic",
                                    "30M_classic",
                                    "70M_classic",
                                    "150M_classic",
                                    "5M_classic_GPU",
                                    "500M_classic",
                                    "VECI_M_classic_GPU",
                                    "30M_distill_from_30M",
                                    "30M_distil_from_30M_soft_focused",
                                    "30M_distill_from_30M_hidd",
                                    "5M_distill_from_30_logits",
                                    "5M_distill_from_30_hidd",
                                    "5M_distill_from_10M_logits",
                                    "800M_classic",
                                    "1B_classic",
                                    "4M_distilled_5M",
                                    "1M_classic_2x_train",
                                    "1M_classic_2x_train_incr_lr",
                                    "1M_classic_2x_train_incr_lr_1e-3",
                                    "1M_distill_2M_outs_alpha_0.9",
                                    "1M_distill_2M_outs_alpha_0.1",
                                    "1M_distill_2M_outs_alpha_0.5",
                                    "1M_distill_2M_outs_norm_alpha_0.5",
                                    "1M_distill_2M_hidd_norm_alpha_0.5",
                                    "30M_self_distill_first_checkpoint",
                                    "10k_classic",
                                    "30M_testing_script",
                                    "30M_classic_w_heads",
                                    "30M_distill_30M_hidd_val",
                                    "12M_distill_30M_hidd_val",
                                    "12M_classic",
                                    "12M_distill_30M_hidd",
                                    "12M_distill_12M_hidd",
                                    "12M_distill_30M_hidd_2_5_7",
                                    "25M_classic",
                                    "25M_distill_30M_hidd_2_5_7",
                                    "25M_distill_25M_hidd_2_3_4",
                                    "25M_classic_aux_test",
                                    "Generic",
                                    "30M_into_30M_reproduce_result",
                                    "70M_distill_into_30M_logits",
                                    "30M_6k_step_chk_pnt_into_30M_logits",
                                    "30M_distill_into_30M_hidd_reproduce_results",
                                    "30M_stop_distill_after_3k_steps_into_30M_hidd",
                                    "30M_distill_tot_6k_steps_into_30M_hidd",
                                    "30M_distill_from_30M_hiddn",
                                    "30M_stop_distill_after_3k_steps_into_30M_hidd_reproduce_result",
                                    "30M_smooth_stop_distill_after_2k_steps_into_30M_hidd",
                                    "30M_smooth_stop_distill_after_1k_steps_into_30M_hidd",
                                    "30M_stop_distill_after_2k_steps_into_30M_hidd_higher_lr",
                                    "30M_stop_distill_after_2k_steps_into_30M_hidd_custom_lr",
                                    "30M_stop_distill_after_3k_steps_into_30M_hidd_custom_lr",
                                    "30M_distill_into_30M_hiddn_and_logits",
                                    "30M_distill_into_30M__logits",
                                    "30M_distill_into_30M_logits_6k",
                                    "30M_distill_into_30M_logits_6k_adamW",
                                    "30M_distill_into_30M_hidd_6k_adamW",
                                    "30M_distill_into_30M_hidd_6k_customizations",
                                    "30M_distill_into_30M_logits_6k_customizations",
                                    "70M_distill_into_30M_logits_6k_customizations",
                                    "30M_classic_optimizer",
                                    "30M_classic_dropout",
                                    "30M_classic_optimizer_and_dropout",
                                    "nodrop_30M_distill_into_30M_hidd_6k_customizations",
                                    "baseline_30M_distill_into_30M_hidd_6k_customizations",
                                    "30M_distill_into_30M_hidd_6k_non_regularized_loss",
                                    "30M_distill_into_30M_hidd_6k_adaptable_lr",
                                    "30M_distill_into_30M_hidd_6k_adaptable_lr_sqrt",
                                    "30M_distill_into_30M_hidd_6k_adaptable_lr_hard_independent",
                                    "30M_distill_into_30M_hidd_6k_full_adaptable_lr",
                                    "30M_distill_into_30M_logits_6k_full_adaptable_lr",
                                    "30M_classic_first_half",
                                    "30M_classic_aux_heads",
                                    "30M_classic_train_4th_aux_head",
                                    "30M_classic_train_6k",
                                    "30M_half_half_train",
                                    "2M_classic",
                                    "2M_half_half",
                                    "2M_only_first_half_for_12k_steps"]

        self.artifact_base_name = "2M_only_first_half_for_12k_steps"  # TODO: fix for consistency
        self.alias_to_load = None
        if self.artifact_base_name not in self.artifact_base_names:
            raise ValueError("CheckpointConfig.__init__() error, chose invalid artifact base.")

        self.checkpoint_frequency = 12000

        self.attempt_load_checkpoint_if_exists= False
        self.strict_state_dict_loading = True



class Config:

    def __init__(self):
        self.wandbConfig = WandbConfig()
        self.datasetConfig = DatasetConfig()
        self.trainingConfig = TrainingConfig()
        self.checkpointConfig = CheckpointConfig()
