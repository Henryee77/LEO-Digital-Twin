# Digital twin-MATD3 for low earth orbits
The MATD3 algorithm in LEO environment

## How to run this program
To run the code you need to execute the following command<br>
```
python main.py --max-ep-num 1000 --prefix "my_training"
```
The `ep-max-timesteps` is the variable specify the number of training episodes, and the `prefix` is the log name that you can change to whatever you want.
With different RL models
```
python main.py --max-ep-num 1000 --model DDPG --prefix "my_training_DDPG"
```
Or with different hyperparameters
```
python main.py --max-ep-num 1000 --actor-lr 0.001 --critic-lr 0.001 --prefix "my_training_lre-3"
```
Hyperparameters can either be set in `main.py` or assigning using the command.

## How to see the log file
### Method 1
If you're using vscode, you can just go to the `main.py` and click `Launch TensorBoard Session`<br>
![tensorboard_1](https://user-images.githubusercontent.com/16890671/227468519-274a6bf5-3422-45d0-a733-04bae1ed01a7.PNG)<br>
And then click `Select another folder`<br>
![tensorboard_2](https://user-images.githubusercontent.com/16890671/227468799-bce0431e-14c1-41a2-94c3-1a74c22a5e99.PNG)<br>
Finally choose the one you want to see.<br><br>

### Method 2
Use the following command
```
tensorboard --logdir="YOUR_LOG_DIR"
```
Or
```
py -m tensorboard.main --logdir="YOUR_LOG_DIR"
```
