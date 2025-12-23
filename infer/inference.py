# from transformers import AutoTokenizer, AutoModelForCausalLM
# from srctwo.data.utils import SYSTEM_prompt
# from peft import PeftModel

# def infer(args):
#     base_model=AutoModelForCausalLM.from_pretrained(
#         "D:\\vscode--llm\\my-llm-training\\srctwo\\models\\Qwen2.5-7B",
#         torch_dtype="auto",
#         device_map="auto"
#     )
#     model=PeftModel(base_model,"D:\\vscode--llm\\my-llm-training\\srctwo\\checkpoints\\bestmodel")

#     tokernizer=AutoTokenizer.from_pretrained(
#         "D:\\vscode--llm\\my-llm-training\\srctwo\\models\\Qwen2.5-7B"
#     )

#     while True:
#         print("请输入你的问题")
#         prompt=input()
#         if prompt in ("bye","exit"):
#             break
        
#         message=[
#             {'role':'system','content':SYSTEM_prompt},
#             {'role':'user','content':prompt},
#         ]
#         text=tokenizer.from_chat_template(
#             message,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#         models_input=tokenizer([text],return_tensors="pt").to(model.device)
#         model_output=model.generate(
#             models_input,
#             max_new_tokens=args.max_completion_length
#         )
#         generate_ids=[b[len(a):] for a,b in zip(models_input.input_ids,model_output)]

#         response=tokenizer.batch_decode(generate_ids,skip_special_tokens=True)


