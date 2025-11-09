### Making GTA V Safe for Drivers

This (in progress!) repo is an attempt at making the streets of Los Santos safer from automotive injury. It implements Deep Deterministic Policy Gradient a la [OpenAI](https://spinningup.openai.com/en/latest/algorithms/ddpg.html). 

The actor and critic models are extensions of MobileNetV3 lightly engineered to process the game state in a meaningful way. In this case, the state is:
* A screenshot of the game (IMG)
* Joystick and trigger input (INP)
* The camera vector (i.e. the direction the player is looking at) (CAM)
* The car's velocity projected onto the camera vector (VEL)
* The car's health-delta between the current and previous frame (DMG)

The actor outputs a vector that replaces INP, called ACT. 

The reward is 1 if DMG is 0, and -DMG otherwise.

The idea is that the model might learn some key things from this information: 
* Whether something is an obstacle
* The velocity of an obstacle relative to the player's vehicle 
* A mapping from controller input space to acceleration in game space 
* Whether or not the controller input will result in a crash

The model should run quite fast, so I've tried to encode as much of this information as possible into the architecture. The hope is to avoid 'LiDAR' methods (i.e. depth mapping and such). 

# Dependencies
* ScriptHookV.dll
* PyTorch
* VigEmBus
* HidHide

The rest of the dependencies can be installed using the .NET and python package managers. 

# Running the code
You'll want to build the C# project in Visual Studio, then copy the release files into the ScriptHookV scripts folder. 

**You'll also have to run HidHide for your controller on GTA V**, since the script works by simulating a virtual controller and the game won't recognize it unless it's the only one. 

Then, run `controller.py`, before starting GTA. This will launch the VigEmBus virtual controller and the event loop. Then you can launch GTA V and enter a vehicle, at which point training will start. 
