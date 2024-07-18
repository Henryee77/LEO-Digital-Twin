# TD3 for low earth orbits
The TD3 algorithm in LEO environment

## How to run this program
To run the code you need to execute the following command<br>
```
python main.py --env-name "LEOSat-v0" --ep-max-timesteps 12000 --prefix "my_training"
```
The `ep-max-timesteps` is the variable specify the number of training episodes, and the `prefix` is the log name that you can change to whatever you want.

To test the training result
```
python main.py --env-name "LEOSat-v0" --ep-max-timesteps 1 --prefix "my_testing" --running-mode "testing"
```

Other hyper parameters can be set in `main.py`.

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
tensorboard --logdir=${YOUR_LOG_DIR}
```
Or
```
python3 -m tensorboard.main --logdir=${YOUR_LOG_DIR}
```
