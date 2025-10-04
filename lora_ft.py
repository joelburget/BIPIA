from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel

from transformers import AutoTokenizer, AutoModelForCausalLM
from tokenizers import AddedToken
import torch
import datasets
import os

model_id = "google/gemma-2-9b-it"  # valid Gemma 2 repo
dataset_name = "joelb/bipia-train"
auth_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

# Detect NVIDIA/CUDA capabilities
bf16_available = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if bf16_available else torch.float16
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=auth_token)
new_tokens = {
    "additional_special_tokens": [
        AddedToken("<|begin_embed|>", normalized=False, lstrip=False, rstrip=False),
        AddedToken("<|end_embed|>", normalized=False, lstrip=False, rstrip=False),
    ]
}
num_added = tok.add_special_tokens(new_tokens)
tok.pad_token = tok.eos_token


raw_ds = datasets.load_dataset(
    dataset_name,
    # data_files={"train": "all.jsonl"}
    data_files={
        "train": [
            "code_code.jsonl",
            "code_text.jsonl",
            "email_code.jsonl",
            "email_text.jsonl",
            "qa_code.jsonl",
            "qa_text.jsonl",
            "abstract_code.jsonl",
            "abstract_text.jsonl",
            "table_code.jsonl",
            "table_text.jsonl",
        ]
    },
    streaming=False,  # materialize to regular Dataset for TRL compatibility
    token=auth_token,
)["train"]


def to_text(example):
    # Prefer chat-format if present
    if "msgs" in example and example["msgs"] is not None:
        # Gemma does not support a 'system' role.
        if "gemma" in model_id:
            msgs = example["msgs"]
            # Some chat templates (e.g., Gemma) do not support a 'system' role.
            # Merge any system messages into the first user message content.
            system_parts = []
            filtered_msgs = []
            for message in msgs:
                role = message.get("role")
                content = message.get("content", "")
                if role == "system":
                    if content:
                        system_parts.append(content)
                    continue
                filtered_msgs.append({"role": role, "content": content})

            if system_parts:
                merged_system = "\n\n".join(system_parts)
                # Prepend system text to the first user message, or create one if absent
                for m in filtered_msgs:
                    if m.get("role") == "user":
                        m["content"] = (
                            merged_system + "\n\n" + (m.get("content") or "")
                        ).strip()
                        break
                else:
                    filtered_msgs.insert(0, {"role": "user", "content": merged_system})
        else:
            filtered_msgs = example["msgs"]

        text = tok.apply_chat_template(
            filtered_msgs, tokenize=False, add_generation_prompt=False
        )
        if not text.endswith(tok.eos_token):
            text += tok.eos_token
        return {"text": text}

    raise ValueError("Example does not contain 'msgs' field")


ds = raw_ds.map(
    to_text,
    # remove_columns=[c for c in raw_ds.column_names if c != "text"],
)

model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=dtype, use_cache=False, token=auth_token
)

if num_added > 0:
    model.resize_token_embeddings(len(tok))

lora = LoraConfig(
    r=8,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj",
    ],
    task_type="CAUSAL_LM",
    modules_to_save=["embed_tokens", "lm_head"],
)  # adjust targets per arch

args = SFTConfig(
    output_dir="ft-lora",
    per_device_train_batch_size=4,  # tune per GPU memory
    num_train_epochs=1,  # tune
    max_steps=10,
    logging_steps=10,
    dataloader_drop_last=False,
    push_to_hub=True,
    hub_model_id="joelb/gemma-ft-lora",
    hub_token=auth_token,
    gradient_checkpointing=True,
    bf16=bf16_available,
    fp16=not bf16_available,
    tf32=True if torch.cuda.is_available() else False,
    dataset_text_field="text",
    packing=False,  # disable to avoid Flash Attention requirement
    max_length=2048,  # tune
)

trainer = SFTTrainer(
    model=model,
    processing_class=tok,
    train_dataset=ds,
    args=args,
    peft_config=lora,
)
trainer.train()

# Save/push just the adapter (and optionally merge for a standalone model)
trainer.model.save_pretrained("adapter")

merged = PeftModel.from_pretrained(model, "adapter").merge_and_unload()
merged.push_to_hub("joelb/gemma-ft-lora", safe_serialization=True, token=auth_token)
tok.push_to_hub("joelb/gemma-ft-lora", token=auth_token)
