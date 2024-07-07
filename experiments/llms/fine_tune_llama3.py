from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import TrainingArguments
from transformers import BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from experiments.llms.oyxoy_datasets import MetaphorPromptHandler, InferencePromptHandler

inference_handler = InferencePromptHandler('zero_shot_nli_label', None)
dev_chats = inference_handler.get_chats()

# dataset_train =  Dataset.from_dict(train_chats)
dataset_dev =  Dataset.from_dict(dev_chats)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.model_max_length = 1024

def format_chat_template(row):
    row["chat"] = tokenizer.apply_chat_template(row["chat"], tokenize=False)
    return row

# dataset_train = dataset_train.map(
#     format_chat_template,
#     num_proc= os.cpu_count(),
# )
dataset_dev = dataset_dev.map(
    format_chat_template,
    num_proc= os.cpu_count(),
)


# For 8 bit quantization
#quantization_config = BitsAndBytesConfig(load_in_8bit=True,
#                                        llm_int8_threshold=200.0)

## For 4 bit quantization
quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,)

# model = AutoModelForCausalLM.from_pretrained(model_id, 
#                                              quantization_config=quantization_config, 
#                                              device_map='auto')


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
trained_model_id = "Llama-3-8B-inference"
output_dir = 'fine_tuned_models/' + trained_model_id

# based on config
training_args = TrainingArguments(
    fp16=False, # specify bf16=True instead when training on GPUs that support bf16 else fp16
    bf16=True,
    do_eval=True,
    evaluation_strategy="no",
    eval_steps=None,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2.0e-05,
    log_level="info",
    logging_steps=5,
    logging_strategy="steps",
    lr_scheduler_type="cosine",
    max_steps=-1,
    num_train_epochs=2,
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_eval_batch_size=2, # originally set to 8
    per_device_train_batch_size=2, # originally set to 8
    # push_to_hub=True,
    hub_model_id=trained_model_id,
    # hub_strategy="every_save",
    # report_to="tensorboard",
    report_to="none",  # for skipping wandb logging
    save_strategy="steps",
    save_steps=200,
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
        train_dataset=dataset_dev,
        eval_dataset=None,
        dataset_text_field="chat",
        tokenizer=tokenizer,
        # packing=True,
        peft_config=peft_config,
        max_seq_length=tokenizer.model_max_length,
    )

# To clear out cache for unsuccessful run
torch.cuda.empty_cache()

train_result = trainer.train()
trainer.save_model(output_dir+'/checkpoint-end')

