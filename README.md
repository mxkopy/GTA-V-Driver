### GTA-RL

This (in progress!) repo is an attempt at making the streets of Los Santos safer from automotive injury. It implements Deep Deterministic Policy Gradient a la [OpenAI](https://spinningup.openai.com/en/latest/algorithms/ddpg.html) to train a model to avoid crashing a car. 

# How to run \[WIP\]
1. pip-install all of the python requirements. 
2. Copy everything in the `build` folder into the game directory (i.e. where the .exe is located). 
2. Install HidHide (https://github.com/nefarius/HidHide) and ViGEmBus (https://github.com/nefarius/ViGEmBus/releases)
3. Block all controller inputs to GTA using HidHide (at some point the goal will be to modify the player's inputs to avoid crashing, which is why we intercept the controller)
4. Run `python main.py controller` to start the controller intercept thread. 
5. Run `python main.py train` to start the model training thread. 
6. Launch GTA and load into a singleplayer free-roam save.
7. Press backspace.  

# Python requirements 
torch, torchvision, cupy
