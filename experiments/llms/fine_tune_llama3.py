from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import TrainingArguments
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
import os

dataset =  dataset = load_dataset('csv',data_files='data/dataset_llama2_220224.csv', split='train')
dataset = dataset.train_test_split(test_size=0.2, seed=42)
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)

dataset_train = dataset['train']
dataset_valid = dataset['test']

def format_chat_template(row):
    row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
    row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
    return row

dataset = dataset.map(
    format_chat_template,
    num_proc= os.cpu_count(),
)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
tokenizer = AutoTokenizer.from_pretrained(model_id)

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
# https://github.com/Lightning-AI/litgpt/issues/327

outputs = model.generate(
    input_ids,
    max_new_tokens=128,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

response = outputs[0][input_ids.shape[-1]:]

print(tokenizer.decode(response, skip_special_tokens=True))

# !pip install -U transformers[torch] datasets
# !pip install -q bitsandbytes trl peft accelerate
# !pip install flash-attn --no-build-isolation

from transformers import BitsAndBytesConfig

# For 8 bit quantization
#quantization_config = BitsAndBytesConfig(load_in_8bit=True,
#                                        llm_int8_threshold=200.0)

## For 4 bit quantization
quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,)

model = AutoModelForCausalLM.from_pretrained(model_id, 
                                             quantization_config=quantization_config, 
                                             device_map='auto')


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
trained_model_id = "Llama-3-8B-sft-lora-ultrachat"
output_dir = 'kaggle/working/' + trained_model_id

# based on config
training_args = TrainingArguments(
    fp16=False, # specify bf16=True instead when training on GPUs that support bf16 else fp16
    bf16=False,
    do_eval=True,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2.0e-05,
    log_level="info",
    logging_steps=5,
    logging_strategy="steps",
    lr_scheduler_type="cosine",
    max_steps=-1,
    num_train_epochs=1,
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_eval_batch_size=1, # originally set to 8
    per_device_train_batch_size=1, # originally set to 8
    push_to_hub=True,
    hub_model_id=trained_model_id,
    # hub_strategy="every_save",
    # report_to="tensorboard",
    report_to="none",  # for skipping wandb logging
    save_strategy="no",
    save_total_limit=None,
    seed=42,
)

# based on config
peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model_kwargs = dict(
    torch_dtype="auto",
    use_cache=False, # set to False as we're going to use gradient checkpointing
    device_map='auto',
    quantization_config=quantization_config,
)

trainer = SFTTrainer(
        model=model_id,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=True,
        peft_config=peft_config,
        max_seq_length=tokenizer.model_max_length,
    )

# To clear out cache for unsuccessful run
torch.cuda.empty_cache()

train_result = trainer.train()