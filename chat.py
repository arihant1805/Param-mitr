import warnings
warnings.filterwarnings('ignore')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
from peft import PeftModel, PeftConfig


model_name = 'google/flan-t5-small'
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype = torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

peft_model_path = './peft_for_param_mitr'
config = PeftConfig.from_pretrained(peft_model_path)
model = PeftModel.from_pretrained(
            base_model, 
            peft_model_path,
            is_trainable = False,
            config = config
            )
print('\n\n\n - - - WELCOME TO PARAM-MITR - - - ')
print('\nTell me you issue now!! I am here to help')
print('==='.join(['==' for _ in range(10)]))
while True:
    inp = input('\nYou : ')
    if inp == '0':
        break

    print('Param Mitr :', tokenizer.decode(
                model.generate(
                        **tokenizer(
                            inp, 
                            return_token_type_ids = False, 
                            return_tensors = 'pt'
                            ), 
                        generation_config = GenerationConfig(max_new_tkoens=150, num_beams = 1) 
                        )[0], skip_special_tokens = True
                )
            )
   





