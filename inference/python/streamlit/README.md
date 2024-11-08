# Streamlit demo

## Instructions

1. Build and install FlexFlow, or build and run `source ./set_python_envs.sh` from the build folder
2. Edit the FlexFlow/inference/python/streamlit/fastapi_incr.py to configure the model to run and the system configs (num gpus, amount of memory, etc)
3. In one terminal, launch the LLM engine with the commands below, and wait until the model's weights loading completes
```
cd FlexFlow/inference/python/streamlit
python fastapi_incr.py
```
4. In another terminal, launch the streamlit app:
```
cd FlexFlow/inference/python/streamlit
streamlit run app.py 
```
5. Open the URL printed to the terminal, e.g. `http://localhost:8501` and interact with the app via browser

