import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline
import gc


def pred_verbalized(model_prompt:AutoModelForCausalLM, tokenizer_prompt:AutoTokenizer, prompt:str, data:str):
  inputs = tokenizer_prompt.encode(prompt + data + 'label=', return_tensors="pt").to("cuda")
  outputs = model_prompt.generate(inputs, max_new_tokens=inputs.shape[1] + 5)
  pred = tokenizer_prompt.decode(outputs[0], skip_special_tokens=True)
  torch.cuda.empty_cache()
  return pred, pred.split('label=')[-1]

def zero_shot_prediction(model_prompt:AutoModelForCausalLM, tokenizer_prompt:AutoTokenizer, prompt:str, datalist:list[str]):
  preds = []
  for sample in tqdm(datalist):
    _, label = pred_verbalized(model_prompt, tokenizer_prompt, prompt, sample)
    gc.collect()
    preds.append(label)
  return preds

def pred_with_pipeline(prediction_pipeline:Pipeline, data:list[str], additional_prompt:str='') -> list:
  pred = prediction_pipeline([additional_prompt + d for d in data])
  if prediction_pipeline.task == 'text2text-generation':
      pred_key = 'generated_text'
  else:
      pred_key = 'label'
  return [p[pred_key] for p in pred]

class LanguagePredictions:

    def __init__(self, data_path:str, prediction_tasks:dict=None, prompts: dict=None):
        self.data_path = data_path
        self.prediction_tasks = prediction_tasks
        self.prompts = prompts
        if prompts:
            print('*Loading bloom')
            checkpoint = "bigscience/bloomz-560m"
            self.tokenizer_prompt = AutoTokenizer.from_pretrained(checkpoint)
            self.model_prompt = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

    def predict_and_store(self, column_name:str, identifier: str, filename: str):
        outname = filename if '_pred' in filename else filename.replace('.', '_pred.')
        df = pd.read_csv(self.data_path + filename)
        print('*', column_name)
        samples = list(df[column_name].dropna().unique())
        for task_name, (task_pipe, additional) in self.prediction_tasks.items():
            print('**', task_name)
            pipe_pred = pred_with_pipeline(task_pipe, samples, additional)
            task_preds = {s: p for s, p in zip(df[column_name], pipe_pred)}
            df[task_name + identifier] = df[column_name].apply(lambda x: task_preds.get(x))
            df.to_csv(self.data_path + outname, index=False)

        for prompt_name, prompt in self.prompts.items():
            print('**', prompt_name)
            prompt_pred = zero_shot_prediction(self.model_prompt, self.tokenizer_prompt, prompt, samples)
            task_preds = {s: p for s, p in zip(df[column_name], prompt_pred)}
            df[prompt_name + identifier] = df[column_name].apply(lambda x: task_preds.get(x))
            df.to_csv(self.data_path + outname, index=False)