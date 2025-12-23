import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from my_llm_training.data.utils import distinguish, train_data, valid_data
from peft import LoraConfig, TaskType, get_peft_model


# === 🔥 重要：数据必须在这里转换成 prompt + completion 格式 ===
data_train = distinguish(train_data)
eval_data  = distinguish(valid_data)


def train(args):
    # ---------- 训练参数 ----------
    training_args = SFTConfig(
        output_dir=args.checkpoint_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,

        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.epochs,

        fp16=True,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
    )

    # ---------- LoRA ----------
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        inference_mode=False,
        lora_alpha=32,
        r=8,
        lora_dropout=0.1
    )

    # ---------- 模型 ----------
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=args.cache_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        cache_dir=args.cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = get_peft_model(model, config)

    # ---------- 核心：SFTTrainer ----------
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=eval_data,
        # ❗❗ 不要 formatting_func（我们不再用 messages）
    )

    trainer.train()
