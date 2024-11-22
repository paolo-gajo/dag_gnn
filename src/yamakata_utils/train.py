from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = 'meta-llama/Meta-Llama-3-70B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

system_prompt = "You are a text-to-graph model tasked to convert recipes into flow graphs."
recipe_text = open('./data/english_recipe_flow_graph_corpus/r-200/recipe-00000-05793.txt').read()
user_message = f"""Convert the following into a flow graph: {recipe_text}"""

messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
        # {"role": "assistant", "content": user_message},
    ]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
