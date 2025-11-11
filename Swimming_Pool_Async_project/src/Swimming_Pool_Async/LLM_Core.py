

import copy
import json
import re

from openai import AsyncOpenAI, OpenAI

class LLM_Core:
    def __init__(self, tokenizer=None, use_async=True, api_model="",base_url="http://0.0.0.0:6001/v1", api_key='EMPTY', task="chat"):
        self.use_async = use_async
        self.api_key = api_key
        self.base_url = base_url
        self.api_model = api_model
        self.tokenizer = tokenizer
        self.task = task
        self.timeout=360
        self.extra_body = {
            'repetition_penalty': 1.05,
                    }
        if use_async==True:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
    def process_chunk(self,chunk):
        return json.dumps(chunk, default=lambda o: o.__dict__, ensure_ascii=False).replace('\\n', '\n')
        
    async def async_model(self, data):
        data_copy = copy.deepcopy(data)
        completion = await self.client.chat.completions.create(
            model=data["model"],
            messages=data["messages"],
            temperature=data.get("temperature", 0.95),
            top_p=data.get("top_p", 0.35),
            extra_body=data.get("extra_body", self.extra_body),
            stream=data.get("stream", False),
            #logprobs=data.get("logprobs", False),
            timeout=self.timeout
        )
        count = 0
        if data_copy.get("stream", False):
            async for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    count += 1
                    chunk_dict = {
                    "id": chunk.id,
                    "chunk": chunk,
                    "choices": [{
                        "delta": {
                            "content": chunk.choices[0].delta.content,
                            "function_call": chunk.choices[0].delta.function_call,
                            "role": chunk.choices[0].delta.role,
                            "tool_calls": chunk.choices[0].delta.tool_calls
                        },
                        "finish_reason": chunk.choices[0].finish_reason,
                        "index": chunk.choices[0].index
                    }],
                    "created": chunk.created,
                    "model": chunk.model,
                    "object": chunk.object
                        }
                    chunk_json = json.dumps(chunk_dict, default=str, ensure_ascii=False)

                    # Use regex to clean up multiple backslashes
                    chunk_json = re.sub(r'\\+', r'\\', chunk_json)

                    # Yield the cleaned-up JSON string
                    yield f"data: {chunk_json}\n\n"
            yield "data: [DONE]\n\n"
        else:
            chunk = completion
            try:
                reasoning_content = chunk.choices[0].message.reasoning_content
            except:
                reasoning_content = ""
            chunk_dict = {
                    "id": chunk.id,
                    "chunk": chunk,
                    "choices": [{
                        "message": {
                            "content": chunk.choices[0].message.content,
                            "reasoning_content": reasoning_content,
                            "function_call": chunk.choices[0].message.function_call,
                            "role": chunk.choices[0].message.role,
                            "tool_calls": chunk.choices[0].message.tool_calls
                        },
                        "finish_reason": chunk.choices[0].finish_reason,
                        "index": chunk.choices[0].index
                    }],
                    "created": chunk.created,
                    "model": chunk.model,
                    "object": chunk.object
                        }
            
            yield f"{json.dumps(chunk_dict, default=str,ensure_ascii=False)}\n\n"
            
    async def receive_data_RW(self, data):
        """Asynchronously receive model data and return output along with rewards."""
        data_copy = copy.deepcopy(data)
        output = ""
        rewards = None
       # try:
        async for chunk in self.async_logprob(data=data_copy):
            response_data = json.loads(chunk.replace("data: ", "").strip())
            if 'response' in response_data:
                output = response_data['response']
                rewards = response_data['rewards']
            elif 'choices' in response_data:
                # Handle streaming data if needed
                output += response_data['choices'][0]['delta']['content']
        # except Exception as e:
        #     print(data_copy["messages"])
        #     print(f"接收数据时出错: {e}")
        
        return {"response": output, "rewards": rewards}

    
    async def receive_data(self, data):
        data_copy = copy.deepcopy(data)
        #("1",data_copy["messages"],"2")
        output = ""
        #print(self.llm)
        async for chunk in self.async_model(data=data_copy):
            if data_copy['stream']==True:
                json_string = chunk[len('data: '):]
            else:
                json_string = chunk
            
            if json_string.strip() == '[DONE]':
                break
            try:
                chunk_data = json.loads(json_string)
            except json.JSONDecodeError as e:
                print("error!!!!!!!!",json_string)
                continue
            if data_copy['stream']==True:
                output += chunk_data['choices'][0]['delta']['content']
            else:
                output += chunk_data['choices'][0]['message']['content']
        csj_ = output
        #print("1",csj_,"2")
        return csj_
