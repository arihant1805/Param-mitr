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

### Data Loading & Preprocessing

- **Loading the Dataset:**  
  Load your pre-saved mental health dialogue dataset from disk:
  ```python
  from datasets import load_dataset
  dataset = load_dataset('samhog/psychology-10k', split = ['train'])[0]
  Dataset = dataset.train_test_split(test_size=0.2)
  ```

- **Data Exploration:**  
  Use helper functions (like `print_conv`) to inspect sample conversations.

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

- **Tokenization:**  
  Prepare inputs by concatenating the `instruction`, user query, and a task prompt (e.g., `"Answer :"`).

### Fine-Tuning with LoRA

- **LoRA Configuration:**
  ```python
  from peft import LoraConfig, TaskType, get_peft_model
  config = LoraConfig(
      task_type=TaskType.SEQ_2_SEQ_LM,
      r=32,
      target_modules="all-linear",
      lora_alpha=32,
      lora_dropout=0.05
  )
  Model = get_peft_model(model, config)
  Model.print_trainable_parameters()
  ```

- **Training:**  
  Use Hugging Face’s `Trainer` along with custom tokenization to fine-tune the model on your dataset.

### Inference & Evaluation

- **Inference Function:**  
  Generate responses by passing formatted user inputs:
  ```python
  def inference(input_data, model_=Model):
      intruct = dataset['instruction'][0]
      task = "Answer :"
      inp = [intruct + "\n" + sent + "\n" + task for sent in input_data]
      output = model_.generate(
          **tokenizer(inp, return_token_type_ids=False, return_tensors='pt', padding=True, truncation=True).to('cuda')
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
  rouge_score_new = rouge.compute(predictions=predicted_new, references=human_base_line)
  ```

### Saving and Loading the Adapter

- **Saving the Adapter:**
  ```python
  trainer.model.save_pretrained('./peft_for_param_mitr')
  ```

- **Loading for CPU Inference:**
  ```python
  from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  from peft import PeftModel
  tokenizer = AutoTokenizer.from_pretrained('/kaggle/working/Model', device_map="auto")
  model = AutoModelForSeq2SeqLM.from_pretrained('/kaggle/working/Model', device_map="auto", torch_dtype=torch.bfloat16)
  Model = PeftModel.from_pretrained(model, './peft_for_param_mitr', is_trainable=False)
  ```

---

## License

This project is licensed under the MIT License.
