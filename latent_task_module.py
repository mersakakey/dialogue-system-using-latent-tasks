import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

from name_to_prompt import name_to_prompt
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Datasetの定義
class NewsDataset(Dataset):
  def __init__(self, X, y, tokenizer, max_len):
    self.X = X
    self.y = y
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):  # len(Dataset)で返す値を指定
    return len(self.y)

  def __getitem__(self, index):  # Dataset[index]で返す値を指定
    text = self.X[index]
    inputs = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      truncation=True,
      padding='max_length'
    )
    ids = inputs['input_ids']
    mask = inputs['attention_mask']

    return {
      'ids': torch.LongTensor(ids),
      'mask': torch.LongTensor(mask),
      'labels': torch.Tensor(self.y[index])
    }

def input_and_predict_with_softmax(model, loader, device, criterion=None):
  """ 入力データから推論結果を出力 """
  model.eval()
  with torch.no_grad():
    for data in loader:
      # デバイスの指定
      ids = data['ids'].to(device)
      mask = data['mask'].to(device)
      labels = data['labels'].to(device)

      # 順伝播
      outputs = model(ids, mask)

      pred = torch.nn.functional.softmax(outputs.logits, dim=-1)

  return pred

def load_model(hf_token):
    model = AutoModelForSequenceClassification.from_pretrained("mersakakey/autotrain-latent_tasks_classification-73664139317", token = hf_token)

    tokenizer = AutoTokenizer.from_pretrained("mersakakey/autotrain-latent_tasks_classification-73664139317", token = hf_token)

    return model, tokenizer

def extract_latent_task(text, model, tokenizer):
    columns = ['data','telling_about_oneself', 'knowing_the_conversation_partner', 'gaining_empathy', 'Empathizing_with_the_conversation_partner', 'discussion', 'ending_the_conversation', 'have_a_clear_task', "generic_utterance"]
    categories = ['telling_about_oneself', 'knowing_the_conversation_partner', 'gaining_empathy', 'Empathizing_with_the_conversation_partner', 'discussion', 'ending_the_conversation', 'have_a_clear_task', "generic_utterance"]


    MAX_LEN = 64

    input_utter = pd.DataFrame([[text,0,0,0,0,0,0,0,0]],columns = columns)

    dataset_eval = NewsDataset(input_utter['data'], input_utter[categories].values, tokenizer, MAX_LEN)

    dataloader_eval = DataLoader(dataset_eval, batch_size=1, shuffle=False)

    p = input_and_predict_with_softmax(model, dataloader_eval, "cpu")[0]

    print(f'推論結果：{p}')

    return p

def p_to_prompt(p):
    categories = ['telling_about_oneself', 'knowing_the_conversation_partner', 'gaining_empathy', 'Empathizing_with_the_conversation_partner', 'discussion', 'ending_the_conversation', 'have_a_clear_task', "generic_utterance"]
    
    category_line = ""

    for i, category in enumerate(categories):
        if p[i] > 0.3:
            if category != "ending_the_conversation":
                category_line += category + " "
            elif p[i] > 0.5:
                category_line += category + " "

    prompt = name_to_prompt(category_line)
    return category_line, prompt

if __name__ == "__main__":

    model, tokenizer = load_model()
    text = "こんにちは．僕は榊原といいます．" 
    extract_latent_task(text, model, tokenizer)