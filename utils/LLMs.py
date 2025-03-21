import requests
import json
from transformers import AutoTokenizer
import time

    
class GPT_model:
    def __init__(self, llm_params = {}, model = "gpt-3.5-turbo", key = None) -> None:
        self.url = None ## you need to prepare your api key
        self.key = key
        if self.url is None or key is None:
            raise ValueError("Please provide a valid API key and URL.")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": key
            
        }
        self.model = model
        self.total_consume = 0
        self.llm_params = {
            "model": model,  # or "gpt-4-32k", "gpt-4", "gpt-3.5-turbo"
            "temperature": 0.1
        }
        self.llm_params.update(llm_params)
        self.max_retries = 5

    def ask(self, user_prompt, system_prompt = 'You are a helpful AI assistant.'):
        
        data = {
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}]}
        data.update(self.llm_params)
        retries = 0
        while retries < self.max_retries:
            try:
                response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
                response.raise_for_status()  # Ensure we raise an error for bad status codes
                response_text = response.json()['choices'][0]['message']['content']
                # print(response_text)
                self.total_consume += response.json()['consume']
                return response_text
            except requests.exceptions.RequestException as e:
                # print(f"Request failed: {e}")
                retries += 1
                if retries < self.max_retries:
                    continue
                return "Request failed after multiple retries"
            except json.JSONDecodeError as e:
                # print(f"JSON decode error: {e}")
                retries += 1
                if retries < self.max_retries:
                    continue
                return "Error decoding JSON response after multiple retries"
            except KeyError as e:
                # print(f"Key error: {e}")
                retries += 1
                if retries < self.max_retries:
                    continue
                return "Key error after multiple retries"
    
    def batch_ask(self, user_prompts, system_prompt = 'You are a helpful AI assistant.'):
        messages_list = []
        responses = []
        for user_prompt in user_prompts:
            response = self.ask(user_prompt, system_prompt)
            responses.append(response)
        return responses
            

class LLAMA_model:
    def __init__(self, model, llm_params = {}, port = 23333) -> None:
        self.url = "http://localhost:{}/v1/completions".format(port)
        self.headers = {
            "Content-Type": "application/json",
        }
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model)
        self.llm_params = {
            "model": model,  
            "temperature": 0.1,
            # "prompt": prompt,
            "max_tokens": 8192,
            "stop_token_ids": [128001, 128009],
            "top_k": 50,
            "top_p": 0.95,
        }
        self.llm_params.update(llm_params)

    def ask(self, user_prompt, system_prompt = 'You are a helpful AI assistant.'):
        messages=[  {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

        data = {
        "prompt": prompt,
        }
        data.update(self.llm_params)   
        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
            response_text = response.json()['choices'][0]['text']
            return response_text
        except requests.exceptions.RequestException as e:
            return "Request failed"
        except json.JSONDecodeError as e:
            return "Error decoding JSON response"
        except KeyError as e:
            return "Key error"

    def batch_ask(self, user_prompts, system_prompt = 'You are a helpful AI assistant.'):
        messages_list = []
        batch_num = len(user_prompts)
        for user_prompt in user_prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            messages_list.append(prompt)

        data = {
            "prompt": messages_list,  
        }
        data.update(self.llm_params)   
        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
            response = response.json()['choices']
            response_text = [c['text'] for c in response]
            # print(response_text)
            return response_text
        except requests.exceptions.RequestException as e:
            # print(f"Request failed: {e}")
            return ["Request failed"] * batch_num
        except json.JSONDecodeError as e:
            # print(f"JSON decode error: {e}")
            return ["Error decoding JSON response"] * batch_num
        except KeyError as e:
            # print(f"Key error: {e}")
            return ["Key error"] * batch_num 