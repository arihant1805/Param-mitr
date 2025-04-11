# Param-mitr


Param-mitr is a Seq2Seq conversational AI model designed to provide mental health support and help you overcome personal challenges. The model is fine-tuned using LoRA (Low-Rank Adaptation) on a mental health dialogue dataset, enabling it to generate empathetic and contextually aware responses.

---

## Overview

- **Purpose:**  
  Provide mental health guidance by generating thoughtful conversational responses.

- **Technology:**  
  - Uses a fine-tuned FLAN-T5-Small model.
  - Enhanced with LoRA via the PEFT library to optimize training efficiency.

- **Key Features:**  
  - Empathetic conversational responses.
  - Context-aware, addressing issues such as anxiety, depression, and stress.
  - Supports both GPU and CPU inference.

---

## Project Structure

```
├── peft_for_param_mitr/       # Directory containing the saved LoRA adapter
├── notebook.ipynb             # Jupyter Notebook with the complete project code
└── README.md                  # This file
```

---

## Requirements

- Python 3.10+
- [PyTorch](https://pytorch.org/) (with CUDA for GPU-based training/inference)
- [Transformers](https://github.com/huggingface/transformers)
- [Datasets](https://github.com/huggingface/datasets)
- [PEFT](https://github.com/huggingface/peft)
- [Evaluate](https://huggingface.co/spaces/evaluate-metric/rouge)
- [tqdm](https://github.com/tqdm/tqdm)

Install the necessary dependencies with:

```bash
pip install torch transformers datasets peft evaluate tqdm
```

---

## Usage

### Model Initialization & Tokenization

- **Loading the Base Model and Tokenizer:**
  ```python
  from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  model_name = 'google/flan-t5-small'
  model = AutoModelForSeq2SeqLM.from_pretrained(
      model_name,
      device_map="auto",
      torch_dtype=torch.bfloat16
  )
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  ```

- **Loading the PEFT Adapter:**
  ```python
  from peft import PeftModel
  peft_model = PeftModel.from_pretrained(
                              model, 
                              './peft_for_param_mitr', 
                              is_trainable=False
                              )
  ```


### Inference & Evaluation

- **Inference Function:**  
  Generate responses by passing formatted user inputs:
```python
def inference(input_data, model_):
"""
print the sentences in input_data and output of the model in conversational form.
input_data : list of the input sentences.
model: model you want to use for inference
"""
      	# Ensure the dataset variable is defined or imported
      	instruct = "Provide a supportive response to the following:\n"
      	inp = [instruct + question + "\nAnswer:" for question in input_data]

      	output = model_.generate(
          	**tokenizer(
			inp, 
			return_token_type_ids=False, 
			return_tensors='pt', 
			padding=True, truncation=True).to('cuda')
      		)

	decoded = [tokenizer.decode(out, skip_special_tokens=True) for out in output]
      
	for i, j in zip(input_data, decoded):
        	print(f"Me: {i}\nParam-mitr: {j}\n")
  ```




- **Evaluation:**  
  Compare generated responses against a human baseline using metrics like ROUGE:
  ```python
  import evaluate
  rouge = evaluate.load('rouge')
  rouge_score_new = rouge.compute(
			predictions=predicted_new, 
			references=human_base_line
			)
  ```

---

## License

This project is licensed under the Apache License.
