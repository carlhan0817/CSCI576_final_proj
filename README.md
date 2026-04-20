start file
usage of each folder:
    test:the unit test for the app and the testing video
    prompts:save all documentation markdown file here
    app: main program
    data: temp folder for the new generateed data


1. Set Up the Environment
Since you will likely be managing this in VSCode, setting up your Python environment and kernel correctly from day one will save your team a lot of headaches.

Initialize the Virtual Environment: Run python -m venv .venv in your root folder. Make sure your VSCode interpreter is pointed to this specific .venv.

**I run python -m venv .venv under src folder.**

Create pyproject.toml: Instead of a messy requirements.txt, define your dependencies here. You will need to install:

Core: numpy, pandas, pydantic

Media: opencv-python, librosa, ffmpeg-python (or imageio-ffmpeg)

AI Models: openai-whisper, open_clip_torch, sentence-transformers, torch

Backend: fastapi, uvicorn

**Now open terminal in VSCode and cd into CSCI576_final_proj folder, run the following command in order:**
python -m pip install --upgrade pip setuptools wheel
pip install -e .    



