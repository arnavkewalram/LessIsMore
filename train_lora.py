import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed
import os
import hydra
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import mask_trainers as trainers
import wandb
import json
import socket
from typing import Optional, Set
from huggingface_hub import login
from peft import LoraConfig, PeftModel, get_peft_model

dist.set_debug_level(dist.DebugLevel.OFF)

OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    if 'FSDP' in config.trainer:
        init_distributed(rank, world_size, port=config.fsdp_port)

    if rank == 0 and config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.local_dirs)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )

    TrainerClass = getattr(trainers, config.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, reference_model=reference_model, rank=rank, world_size=world_size)

    trainer.train()

    # ---------------- SAVE ARTIFACTS (rank 0 only) ----------------
    if rank == 0:
        out_dir = config.local_run_dir
        os.makedirs(out_dir, exist_ok=True)
        # Save tokenizer (recreate here to keep signature minimal)
        tok = transformers.AutoTokenizer.from_pretrained(config.model.name_or_path)
        try:
            tok.save_pretrained(out_dir)
        except Exception:
            pass

        # 1) Save LoRA adapter (small)
        try:
            if isinstance(policy, PeftModel):
                policy.save_pretrained(out_dir)
                print(f"💾 Saved LoRA adapter to: {out_dir}")
        except Exception as e:
            print(f"⚠️ Skipped saving adapter: {e}")

        # 2) Save merged full model (fp16) to <run_dir>/merged
        merged_dir = os.path.join(out_dir, "merged")
        try:
            if isinstance(policy, PeftModel):
                try:
                    merged = policy.merge_and_unload()     # works if base isn't quantized
                except Exception:
                    # fallback for quantized base: reload fp16 base and the just-saved adapter, then merge
                    base_fp16 = transformers.AutoModelForCausalLM.from_pretrained(
                        config.model.name_or_path, torch_dtype=torch.float16, device_map="auto"
                    )
                    peft_loaded = PeftModel.from_pretrained(base_fp16, out_dir)
                    merged = peft_loaded.merge_and_unload()
            else:
                merged = policy

            try:
                merged = merged.to(dtype=torch.float16)
            except Exception:
                pass

            os.makedirs(merged_dir, exist_ok=True)
            merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="2GB")
            try:
                tok.save_pretrained(merged_dir)
            except Exception:
                pass
            print(f"✅ Saved merged full model to: {merged_dir}")
        except Exception as e:
            print(f"❌ Failed to save merged full model: {e}")
    # ----------------------------------------------------------------


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 140)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 140)

    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)

    model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)

    load_path = config.model.name_or_path
    print('building policy from path', load_path)

    # ---- minimal quantization toggle (training-time) ----
    # works with either `quantization=...` (root) or `model.quantization=...`
    quant = str(getattr(getattr(config, "model", {}), "quantization", getattr(config, "quantization", "none"))).lower()
    if quant in ("4", "4bit", "qlora"):
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=policy_dtype,
        )
        policy = transformers.AutoModelForCausalLM.from_pretrained(
            load_path, low_cpu_mem_usage=True, use_cache=False, quantization_config=bnb_config, device_map="auto"
        )
    elif quant in ("8", "8bit"):
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        policy = transformers.AutoModelForCausalLM.from_pretrained(
            load_path, low_cpu_mem_usage=True, use_cache=False, quantization_config=bnb_config, device_map="auto"
        )
    else:
        policy = transformers.AutoModelForCausalLM.from_pretrained(
            load_path, low_cpu_mem_usage=True, use_cache=False, torch_dtype=policy_dtype, **model_kwargs
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(load_path)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})
        policy.config.pad_token_id = tokenizer.pad_token_id
        policy.resize_token_embeddings(len(tokenizer))

    if config.model.archive is None:
        peft_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
        )
        policy = get_peft_model(policy, peft_config)
        policy.print_trainable_parameters()
    else:
        policy = PeftModel.from_pretrained(policy, config.model.archive)
        print('loading from archive', config.model.archive)
        for name, param in policy.named_parameters():
            if 'lora_B' in name or 'lora_A' in name:
                param.requires_grad = True
        # Print the trainable parameters
        policy.print_trainable_parameters()

    disable_dropout(policy)

    if config.loss.name in ['dpo', 'soft_sft']:
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            load_path, use_cache=False, low_cpu_mem_usage=True, torch_dtype=reference_model_dtype, **model_kwargs
        )
        disable_dropout(reference_model)
    else:
        reference_model = None

    if 'FSDP' in config.trainer:
        world_size = torch.cuda.device_count()
        print('starting', world_size, 'processes for FSDP training')
        mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, policy, reference_model), join=True)
    else:
        print('starting single-process worker')
        worker_main(0, 1, config, policy, reference_model)


if __name__ == '__main__':
    main()
