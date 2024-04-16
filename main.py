from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
import torch

tokenizer=AutoTokenizer.from_pretrained("google/flan-t5-large")

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

from accelerate import Accelerator

accelerator = Accelerator()

model_File_path="C:/Users/anshh/Capstone/Sharding/model"
accelerator.save_model(model=model, save_directory=model_File_path, max_shard_size='1GB')

from accelerate import load_checkpoint_and_dispatch
device_map={'':'cpu'}
model = load_checkpoint_and_dispatch(model, checkpoint=model_File_path, device_map=device_map, no_split_module_classes=['Block'])

raw_input = input("Enter the input text: ")
while raw_input!="exit":
    
    inputs=tokenizer(raw_input,padding=True, truncation= True, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100, return_dict_in_generate=True, output_scores=True)

    generated_token_ids=outputs.sequences
    generated_text = tokenizer.decode(generated_token_ids[0], skip_special_tokens=True)

    print(generated_text)

    raw_input = input("Enter the input text: ")
