import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from my_llm_training.data.utils import distinguish,train_data,valid_data
from peft import LoraConfig, TaskType, get_peft_model

# data=distinguish(train_data)[0]
# print(f'{data}')
#{'sentence': '现头昏口苦', 'labels': [['口苦', '临床表现']], 'prompt': [{'content': '你是一个智能助手。给定一句话，你需要抽取其中的医学实体，并判断其类别，输出格式为[[实体，类别]...]', 'role': 'system'}, {'content': '现头昏口苦', 'role': 'user'}, {'content': '[["\\u53e3\\u82e6", "\\u4e34\\u5e8a\\u8868\\u73b0"]]', 'role': 'assistant'}]}

data_train=distinguish(train_data)
eval_data=distinguish(valid_data)

def train(args):
    training_args = SFTConfig(
        output_dir=args.checkpoint_dir,          # 模型输出路径
        overwrite_output_dir=True,               # 是否覆盖已有输出
        do_train=True,
        do_eval=True,                            # 开启评估
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
        fp16=True,                               # 半精度训练
        save_steps=args.save_steps,
        save_total_limit=3,                       # 最多保留 3 个 checkpoint
        report_to="none",                        # 不用 wandb/logging
    )

    config=LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        inference_mode=False,
        lora_alpha=32,
        r=8,
        lora_dropout=0.1
    )


    model=AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float16,
        device_map=None,
        cache_dir=args.cache_dir
    ).to("cuda")

    tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    model=get_peft_model(model,config)
    #model.print_trainable_parameters()----trainable params: 20,185,088 || all params: 7,635,801,600 || trainable%: 0.2643

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=eval_data,
        dataset_text_field="prompt",
    )
    trainer.train()