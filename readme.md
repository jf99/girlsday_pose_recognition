# How to run
The code has been tested on Ubuntu 24.04 LTS. It should work on other platforms as well.

## 1. Setup a virtual environment
Python and virtualenv must be installed first. Then run

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```
If you get an error during the last command, that may be caused by the specified versions of pip packages not being available for your platform. In that case, try removing the specific version numbers from the [requirements.txt](requirements.txt).

## 2. Run the code in your virtual environment
### Capture poses
```bash
python ./capture.py
```
The captured poses will be saved to `capturings/` (Make sure the folder exists). If you get an error that the camera could not be opened, try passing a different camera index to `cv2.VideoCapture`.

### Train the network
Make sure you have at least two subfolders in `training/` and `validation/`. Also make sure the naming is the same in both folders. Avoid special characters – they might cause problems later.
```bash
python ./train.py
```

### Run inference
```bash
python ./infer.py
```

### Control your mouse cursor by body gestures
Make sure the strings used in the comparisons at the end of [infer_mouse.py](infer_mouse.py) match the names of your classes. Then run
```bash
python ./infer_mouse.py
```
If the cursor does not move at all, you might be running a wayland session. pynput (the library used for mouse movement) [does not yet work with wayland](https://github.com/moses-palmer/pynput/issues/331). You can either login to your computer using X11 instead, or replace pynput with another library.
