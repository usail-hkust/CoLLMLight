import torch
import torch.nn as nn
import json
import time
import os
import shutil
from tqdm import tqdm
import random
from types import SimpleNamespace
from copy import deepcopy
import yaml

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from utils.prompts import get_ra_user_prompt_wsys, get_atr_user_prompt_wsys, SYSTEM_PROMPT
from utils.my_utils import extract_json_reactive_action_results, count_double_hash_pattern

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# torch.cuda.set_device(0)

class VLLMLocalGenerator:
    """Local VLLM generator, supports dynamic reloading of LoRA weights."""
    def __init__(self, base_model_path, lora_path=None, max_model_len=8192):
        self.base_model_path = base_model_path
        self.current_lora_path = lora_path
        self.max_model_len = max_model_len
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize VLLM model
        self.llm = None
        self._load_vllm_model()
        
        # Set sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_k=50,
            top_p=0.95,
            max_tokens=2048,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )
    
    def _load_vllm_model(self):
        """Load or reload the VLLM model."""
        num_gpus = torch.cuda.device_count()
        print(f"Auto-detected {num_gpus} GPUs for VLLM.")
        
        llm_args = {
            "model": self.base_model_path,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": 0.9,
            "dtype": "bfloat16",
            "tensor_parallel_size": num_gpus
        }
        
        if self.current_lora_path and os.path.exists(self.current_lora_path):
            print(f"Loading VLLM with LoRA enabled for path: {self.current_lora_path}")
            llm_args["enable_lora"] = True
            llm_args["max_lora_rank"] = 8
        
        self.llm = LLM(**llm_args)
    
    def reload_with_lora(self, lora_path):
        """Reload the model and apply new LoRA weights."""
        if lora_path != self.current_lora_path:
            print(f"Reloading VLLM model with LoRA: {lora_path}")
            self.current_lora_path = lora_path
            # Delete the old model to free up VRAM
            if self.llm is not None:
                del self.llm
            torch.cuda.empty_cache()
            # Reload the model
            self._load_vllm_model()
    
    def unload_model(self):
        """Unload the VLLM model to free up VRAM."""
        if self.llm is not None:
            del self.llm
            self.llm = None
        torch.cuda.empty_cache()
        print("VLLM model unloaded, GPU memory cleared")
    
    def _format_prompt(self, user_prompt, system_prompt=SYSTEM_PROMPT):
        """Format the prompt."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def generate_batch(self, prompts, system_prompt=SYSTEM_PROMPT):
        """Generate text in batches."""
        if self.llm is None:
            raise RuntimeError("VLLM model is not loaded. Call _load_vllm_model() first.")
            
        formatted_prompts = [self._format_prompt(prompt, system_prompt) for prompt in prompts]
        
        if self.current_lora_path and os.path.exists(self.current_lora_path):
            # Generate using LoRA
            outputs = self.llm.generate(
                formatted_prompts,
                self.sampling_params,
                lora_request=LoRARequest("default", 1, self.current_lora_path)
            )
        else:
            # Generate using the base model
            outputs = self.llm.generate(formatted_prompts, self.sampling_params)
        
        responses = [output.outputs[0].text for output in outputs]
        return responses
    
    def generate_single(self, prompt, system_prompt=SYSTEM_PROMPT):
        """Generate a single text."""
        return self.generate_batch([prompt], system_prompt)[0]

def load_models_for_ppo(model_name, max_new_tokens=2048):
    """Load models required for PPO training."""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure quantization
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant_config
    )
    base_model.gradient_checkpointing_enable() 

    # Create LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0,
    )

    # Create the training model with a value head
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        peft_config=peft_config,
        device_map="auto",
        load_in_4bit=True
    )

    # Add generation config
    if not hasattr(ppo_model, 'generation_config'):
        ppo_model.generation_config = GenerationConfig(
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
        )

    return ppo_model, tokenizer

def generate_offline_data(vllm_generator, dataset):
    """Generate offline data using VLLM - batched version."""
    offline_data = []
    
    print("Generating offline data with VLLM...")
    
    # Prepare all ATR prompts
    prompts_atr = [get_atr_user_prompt_wsys(data) for data in dataset]
    
    # Batch generate all ATR responses
    print("Generating ATR responses...")
    responses_atr = vllm_generator.generate_batch(prompts_atr)
    
    # Prepare all RA prompts
    prompts_ra = [get_ra_user_prompt_wsys(dataset[i], responses_atr[i]) for i in range(len(dataset))]
    
    # Batch generate all RA responses
    print("Generating RA responses...")
    responses_ra = vllm_generator.generate_batch(prompts_ra)
    
    # Calculate rewards and save data
    print("Calculating rewards...")
    for i, data in enumerate(dataset):
        reward_ra, used_atr = calculate_reactive_action_reward(responses_ra[i], data['Best_Signal'])
        reward_atr = calculate_advance_reasoning_reward(responses_atr[i], reward_ra, used_atr)
        
        offline_data.append({
            'data': data,
            'response_atr': responses_atr[i],
            'response_ra': responses_ra[i],
            'reward_atr': reward_atr,
            'reward_ra': reward_ra,
            'prompt_atr': prompts_atr[i],
            'prompt_ra': prompts_ra[i]
        })
    
    return offline_data

def prepare_ppo_data_from_offline(offline_data, tokenizer, device):
    """Prepare PPO training data from offline data."""
    queries = []
    responses = []
    rewards = []
    
    for item in offline_data:
        # ATR data
        query_atr = tokenizer.encode(item['prompt_atr'], return_tensors="pt", truncation=True, max_length=8192).squeeze().to(device)
        response_atr = tokenizer.encode(item['response_atr'], return_tensors="pt", truncation=True, max_length=2048).squeeze().to(device)
        
        # RA data
        query_ra = tokenizer.encode(item['prompt_ra'], return_tensors="pt", truncation=True, max_length=8192).squeeze().to(device)
        response_ra = tokenizer.encode(item['response_ra'], return_tensors="pt", truncation=True, max_length=2048).squeeze().to(device)
        
        queries.extend([query_atr, query_ra])
        responses.extend([response_atr, response_ra])
        rewards.extend([torch.tensor(item['reward_atr'], device=device), torch.tensor(item['reward_ra'], device=device)])
    
    return queries, responses, rewards

@torch.no_grad()
def evaluate_model(vllm_generator, eval_dataset):
    """Evaluate model performance on the validation set - batched version."""
    # Prepare all ATR prompts
    prompts_atr = [get_atr_user_prompt_wsys(data) for data in eval_dataset]
    
    # Batch generate all ATR responses
    responses_atr = vllm_generator.generate_batch(prompts_atr)
    
    # Prepare all RA prompts
    prompts_ra = [get_ra_user_prompt_wsys(eval_dataset[i], responses_atr[i]) for i in range(len(eval_dataset))]
    
    # Batch generate all RA responses
    responses_ra = vllm_generator.generate_batch(prompts_ra)
    
    # Calculate all rewards
    total_reward_ra = 0
    total_reward_atr = 0
    
    for i, data in enumerate(eval_dataset):
        # Calculate reward
        reward_ra, used_atr = calculate_reactive_action_reward(responses_ra[i], data['Best_Signal'])
        reward_atr = calculate_advance_reasoning_reward(responses_ra[i], reward_ra, used_atr)
        
        total_reward_ra += reward_ra
        total_reward_atr += reward_atr
    
    avg_ra_reward = total_reward_ra / len(eval_dataset) if eval_dataset else 0
    avg_atr_reward = total_reward_atr / len(eval_dataset) if eval_dataset else 0
    
    return avg_ra_reward, avg_atr_reward

def main():
    # Load configuration from YAML file
    config_path = "config/ppo_config.yaml"
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert the loaded dictionary into objects for easy access
    model_name = config['model_name']
    max_seq_length = config['max_seq_length']
    dataset_path = config['dataset_path']
    eval_split_ratio = config['eval_split_ratio']
    random_seed = config['random_seed']
    training_args = SimpleNamespace(**config['training_args'])
    ppo_config = PPOConfig(**config['ppo_config'])
    print("Configuration loaded successfully.")
    
    # Create output directories
    best_model_dir = training_args.best_model_dir
    os.makedirs(training_args.output_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    
    # Load and split the dataset
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'r') as file:
        traffic_dataset = json.load(file)
    
    # Set random seed and split the dataset
    random.seed(random_seed)
    random.shuffle(traffic_dataset)
    
    split_index = int(len(traffic_dataset) * (1 - eval_split_ratio))
    
    train_dataset = traffic_dataset[:split_index]
    eval_dataset = traffic_dataset[split_index:]
    
    print(f"Dataset loaded and split successfully.")
    print(f"Total samples: {len(traffic_dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Training loop
    best_avg_ra_reward = -float('inf')
    
    print("Starting PPO Training...")
    
    for epoch in range(int(training_args.num_train_epochs)):
        print(f"\n=== Epoch {epoch+1}/{int(training_args.num_train_epochs)} ===")
        
        # Phase 1: Generate offline data using VLLM
        print("Phase 1: Generating offline data with VLLM")
        
        # Determine the LoRA path for the current epoch
        if epoch == 0:
            current_lora_path = None  # Use the base model for the first epoch
        else:
            current_lora_path = f"{training_args.output_dir}/epoch_{epoch}"
        
        # Initialize the VLLM generator
        vllm_generator = VLLMLocalGenerator(
            model_name, 
            lora_path=current_lora_path, 
            max_model_len=max_seq_length
        )
        
        # Generate offline data
        offline_data = generate_offline_data(
            vllm_generator, 
            train_dataset, 
        )
        
        # Calculate epoch reward statistics
        epoch_total_reward_ra = sum(item['reward_ra'] for item in offline_data)
        epoch_total_reward_atr = sum(item['reward_atr'] for item in offline_data)
        avg_ra_reward = epoch_total_reward_ra / len(offline_data) if offline_data else 0
        avg_atr_reward = epoch_total_reward_atr / len(offline_data) if offline_data else 0
        
        print(f"Offline data generated. Avg RA Reward: {avg_ra_reward:.4f}, Avg ATR Reward: {avg_atr_reward:.4f}")
        
        # Validation phase - evaluate using the current model
        print("Running evaluation on validation set...")
        eval_avg_ra_reward, eval_avg_atr_reward = evaluate_model(vllm_generator, eval_dataset)
        print(f"Validation -> Avg RA Reward: {eval_avg_ra_reward:.4f}, Avg ATR Reward: {eval_avg_atr_reward:.4f}")
        
        # Unload the VLLM model to free up VRAM
        vllm_generator.unload_model()
        del vllm_generator
        torch.cuda.empty_cache()
        
        # Phase 2: Load PPO models and start training
        print("Phase 2: Loading PPO models and training")
        
        # Load the PPO training model
        model, tokenizer = load_models_for_ppo(model_name, max_seq_length)
        
        # If not the first epoch, load the previous checkpoint
        if epoch > 0:
            checkpoint_path = f"{training_args.output_dir}/epoch_{epoch}"
            if os.path.exists(checkpoint_path):
                print(f"Loading checkpoint from {checkpoint_path}")
                model = PeftModel.from_pretrained(model, checkpoint_path)
        
        # Create PPO trainer
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            tokenizer=tokenizer,
            model=model,
            ref_model=None,
        )
        
        # Train with offline data
        print("Training with offline data...")
        
        batch_size = ppo_config.batch_size
        num_complete_batches = len(offline_data) // batch_size
        total_samples_to_use = num_complete_batches * batch_size
        
        print(f"Total offline data: {len(offline_data)}, Using: {total_samples_to_use} samples in {num_complete_batches} batches")
        
        # PPO training in batches with offline data
        for start_idx in tqdm(range(0, total_samples_to_use, batch_size), desc="PPO Training"):
            end_idx = start_idx + batch_size
            batch_offline_data = offline_data[start_idx:end_idx]
            
            queries, responses, rewards = prepare_ppo_data_from_offline(
                batch_offline_data, tokenizer, ppo_trainer.accelerator.device
            )
            
            if queries and responses and rewards:
                stats = ppo_trainer.step(queries, responses, rewards)
        
        # Save the model for the current epoch
        epoch_checkpoint = f"{training_args.output_dir}/epoch_{epoch+1}"
        ppo_trainer.save_pretrained(epoch_checkpoint)
        print(f"Epoch {epoch+1} model saved to {epoch_checkpoint}")
        
        # Save the best model
        if eval_avg_ra_reward > best_avg_ra_reward:
            best_avg_ra_reward = eval_avg_ra_reward
            print(f"🎉 New best model found at epoch {epoch+1} with Avg RA Reward: {best_avg_ra_reward:.4f}!")
            
            if os.path.exists(best_model_dir):
                shutil.rmtree(best_model_dir)
            
            shutil.copytree(epoch_checkpoint, best_model_dir)
            print(f"Best model saved to {best_model_dir}")
        else:
            print(f"Did not beat best validation Avg RA Reward of {best_avg_ra_reward:.4f}.")
        
        # Clean up PPO models to free VRAM
        del model, tokenizer, ppo_trainer
        torch.cuda.empty_cache()
        print("PPO models unloaded, GPU memory cleared")
    
    print(f"\n--- Training Finished ---")
    print(f"The best model was saved to '{best_model_dir}' with an average RA reward of {best_avg_ra_reward:.4f}")


def calculate_reactive_action_reward(reactive_action_response: str, best_signal: str) -> tuple[float, int]:
    """Calculate the reward for the reactive action."""
    try:
        ra_results_json = extract_json_reactive_action_results(reactive_action_response)
        signal = ra_results_json['signal']
        used_atr = ra_results_json.get('used_atr', 0)
        
        reward = 1.0 if signal == best_signal else -1.0
    except Exception:
        reward = -1.0
        used_atr = 0
    
    return reward, int(used_atr)


def calculate_advance_reasoning_reward(
    advance_reasoning_response: str,
    reactive_action_reward: float,
    used_atr: int
) -> float:
    """Calculate the reward for the advance reasoning."""
    L_max = 8.0
    L = min(count_double_hash_pattern(advance_reasoning_response), L_max)
    beta = 0.5
    length_reward = beta * (1 - L / L_max)
    effectiveness_reward = (1 - beta) * float(used_atr)
    reward = reactive_action_reward * (length_reward + effectiveness_reward)
    return reward

if __name__ == "__main__":
    main()