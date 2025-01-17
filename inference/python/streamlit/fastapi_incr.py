# Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Running Instructions:
- To run this FastAPI application, make sure you have FastAPI and Uvicorn installed.
- Save this script as 'fastapi_incr.py'.
- Run the application using the command: `uvicorn fastapi_incr:app --reload --port PORT_NUMBER`
- The server will start on `http://localhost:PORT_NUMBER`. Use this base URL to make API requests.
- Go to `http://localhost:PORT_NUMBER/docs` for API documentation.
"""


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import flexflow.serve as ff
import uvicorn
import json, os, argparse
from types import SimpleNamespace
from typing import Optional, List
import time


# Initialize FastAPI application
app = FastAPI()

# Define the request model
class PromptRequest(BaseModel):
    prompt: str

# data models
class Message(BaseModel):
    role: str
    content: str


# class ChatCompletionRequest(BaseModel):
#     model: Optional[str] = "mock-gpt-model"
#     messages: List[Message]
#     max_tokens: Optional[int] = 512
#     temperature: Optional[float] = 0.1
#     stream: Optional[bool] = False

class ChatCompletionRequest(BaseModel):
    max_new_tokens: Optional[int] = 1024
    messages: List[Message]

# Global variable to store the LLM model
llm = None


def get_configs():
    
    # Fetch configuration file path from environment variable
    config_file = os.getenv("CONFIG_FILE", "")

    # Load configs from JSON file (if specified)
    if config_file:
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Config file {config_file} not found.")
        try:
            with open(config_file) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print("JSON format error:")
            print(e)
    else:
        # Define sample configs
        ff_init_configs = {
            # required parameters
            "num_gpus": 8,
            "memory_per_gpu": 20000,
            "zero_copy_memory_per_node": 40000,
            # optional parameters
            "num_cpus": 4,
            "legion_utility_processors": 8,
            "data_parallelism_degree": 1,
            "tensor_parallelism_degree": 4,
            "pipeline_parallelism_degree": 1,
            "offload": False,
            "offload_reserve_space_size": 8 * 1024, # 8GB
            "use_4bit_quantization": False,
            "use_8bit_quantization": False,
            "enable_peft": False,
            "peft_activation_reserve_space_size": 1024, # 1GB
            "profiling": False,
            "benchmarking": False,
            "inference_debugging": False,
            "fusion": True,
        }
        llm_configs = {
            # required parameters
            "llm_model": "meta-llama/Llama-3.1-8B-Instruct",
            # optional parameters
            "cache_path": os.environ.get("FF_CACHE_PATH", ""),
            "refresh_cache": False,
            "full_precision": False,
            "prompt": "",
            "output_file": "",
        }
        # Merge dictionaries
        ff_init_configs.update(llm_configs)
        return ff_init_configs
    

# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    global llm

    # Initialize your LLM model configuration here
    configs_dict = get_configs()
    configs = SimpleNamespace(**configs_dict)
    ff.init(configs_dict)

    ff_data_type = (
        ff.DataType.DT_FLOAT if configs.full_precision else ff.DataType.DT_HALF
    )
    llm = ff.LLM(
        configs.llm_model,
        data_type=ff_data_type,
        cache_path=configs.cache_path,
        refresh_cache=configs.refresh_cache,
        output_file=configs.output_file,
    )

    generation_config = ff.GenerationConfig(
        do_sample=False, temperature=0.9, topp=0.8, topk=1
    )
    llm.compile(
        generation_config,
        max_requests_per_batch=16,
        max_seq_length=2048,
        max_tokens_per_batch=1024,
    )
    llm.start_server()

# API endpoint to generate response
@app.post("/generate/")
async def generate(prompt_request: PromptRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM model is not initialized.")
    
    # Call the model to generate a response
    full_output = llm.generate([prompt_request.prompt])[0].output_text.decode('utf-8')
    
    # Separate the prompt and response
    split_output = full_output.split('\n', 1)
    if len(split_output) > 1:
        response_text = split_output[1] 
    else:
        response_text = "" 
        
    # Return the prompt and the response in JSON format
    return {
        "prompt": prompt_request.prompt,
        "response": response_text
    }

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):

    if llm is None:
        raise HTTPException(status_code=503, detail="LLM model is not initialized.")
    
    print("received request:", request)
    result = llm.generate([message.dict() for message in request.messages], max_new_tokens=request.max_new_tokens)[0].output_text.decode('utf-8')
    print("returning response:", result)
    return {
        "response": result
    }
    return {
        "id": "1337",
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [{"message": Message(role="assistant", content=resp_content)}],
    }

# Shutdown event to stop the model server
@app.on_event("shutdown")
async def shutdown_event():
    global llm
    if llm is not None:
        llm.stop_server()

# Main function to run Uvicorn server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Running within the entrypoint folder:
# uvicorn fastapi_incr:app --reload --port

# Running within the python folder:
# uvicorn entrypoint.fastapi_incr:app --reload --port 3000
