# AI Frameworks
---
## Which result you achieved? In which computation time? On which engine?
---
### Engine

We have instanced a VM in Google Cloud Platform with the following caracteristics :
- Operating system : Ubuntu 20.04
- Serie : N1
- Number of virtual processors : 8
- RAM : 30Go
- GPU type : NVIDIA Tesla V100

### Results

Thanks to the engine rent to Google Cloud we achieve this performances :
- Loss on public train set : 0.368
- Accuracy on validation public set (10% of train set) : 0.874
- **f1 score on public test set : 0.799**
- **f1 score on public test set : 0.805** (without post processing improvements)

Learn and predict spend time : 
- Learning time (for 2 epochs with batch size of 16) : 02:54
- Prediction time (for the validation public set) : 00:02

## What do I have to install to be able to reproduce the code?
---
To be able to reproduce the code, you have to :
- Install Google Cloud SDK if you want to run this code on Google Cloud Platform.
- Install NVIDIA drivers in order to use the NVIDIA GPUS.
- Install Anaconda3 in order to get Python 3.8.5.
- Clone our github repository - https://github.com/tcotte/AI_Frameworks.git - to retrieve our code. 

We detail this commands in the next chapter.

## Which command do I have to run to reproduce the results?
---
1. Install Google cloud SDK --> follow this link : https://cloud.google.com/sdk/docs/install#deb

2. Connect to your instance (on Ubuntu 20.04) installed with a GPU V100 and install NVIDIA drivers
- To connect to the VM
```
$ gcloud compute ssh VMNAME
```

- To install NVIDIA drivers (exemple for Ubuntu 20.04) --> see documentation https://cloud.google.com/compute/docs/gpus/install-drivers-gpu
```
$ curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
$ sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
$ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
$ sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
```
Then update list of packages `sudo apt update` and install cuda `sudo apt install cuda`


3. Once connected to the instance, install Anaconda to get Python 3.8.5

- Run the following commands to update system packages and the core dependencies for Anaconda :
```
$ sudo apt-get update
$ sudo apt-get install bzip2 libxml2-dev
```

- Download a version of Anaconda :
```
$ wget https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh
```
It is not necessary to install VS Code

- Then run the shell script :
```
$ bash Anaconda3–2018.12-Linux-x86_64.sh
```

- After installation has completed, we can remove the shell script to save disk space.
```
$ rm Anaconda3–2018.12-Linux-x86_64.sh
```
Pass the conda environment variable to .bashrc file and reinitialize the shell to recognise the conda command.
```
$ echo ". /home/{surname}/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
$ export PATH="/home/{surname}/anaconda3/bin:$PATH"
$ source ~/.bashrc
```

4. Since Anaconda is installed, we can create an anaconda virtual environment with Python 3.8.5 :
```
$ conda create -n AIF python=3.8.5
```

And activate this environment :
```
$ conda activate AIF
```

When you are in the environment, you will see the environment name in the terminal prompt. Something like :
```
(AIF) ~ $
```

5. Now, you can download our project from github :
```
$ git clone https://github.com/tcotte/AI_Frameworks.git
```

6. Then, move in our project and create "*model*" directory. This directory will able us to save
our futur model later.
```
$ cd AI_Frameworks
$ mkdir model
```

7. Since you have activated your anaconda virtual environment, you can install, in this environment, libraries required for 
this project.
```
$ pip install -r requirements.txt
```
Libraries required are in your virtual environment, you can check this with `pip freeze` command. 

8. Unzip the "*data.zip*" folder to data/ following this command lines :
```
$ sudo apt-get install unzip
$ unzip data.zip -d data/
$ rm data.zip
``` 

8. You can train your model thanks to the command line : 
```
$ python script/learning.py
```
Python script "*learning.py*" uses argument parser. The previous command implied hard-coded arguments by default (epcohs --> 2 and
batch_size --> 16).
In fact, you can change the number of epochs, the batch size and directories where data, results and model are located. 
If you followed this tutorial, change the location of directories is not advised. 

If you want to change number of epochs and batch size, type :
```
$ python script/learning.py --epochs={epochs_nb} --batch_size={batch_size}
```
If the **-- results** value stayed by default, you can get your csv results file in :
"*results/results.csv*"


8. Once you have retrieved a model, you can make prediction on your "*test.json*" file located under the "*data*" folder.
 ```
$ python script/prediction.py --epochs={epochs_nb} --batch_size={batch_size}
```
This script uses argument parser also. The arguments and their default values are the same as the "*learning.py*" script. 
This script will retrieve the model under the **-- model** argument named **"bert_epochs_{epochs_nb}_batch_size_{batch_size}**".
You can change also the location of your results but this is not advised.
If the **-- results** value stayed by default, you can get your csv prediction result file in :
"*results/prediction_bert_epochs_{epochs_nb}_batch_size_{batch_size}.csv*"
