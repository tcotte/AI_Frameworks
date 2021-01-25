# AI Frameworks

## What do I have to install to be able to reproduce the code?

1. Install Google cloud SDK --> follow this link : https://cloud.google.com/sdk/docs/install#deb

2. Connect to your instance installed with a GPU V100 and NVIDIA drivers pre-installed

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

- Then run the shell script :
```
$ bash Anaconda3–2018.12-Linux-x86_64.sh
```

- After installation has completed, we can remove the shell script to save disk space.
```
$ bash Anaconda3–2018.12-Linux-x86_64.sh
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

6. Then, move in our project and create "*results*" and "*model*" directories. This directories will able us to save
our futur model and our results later.
```
$ cd AI_Frameworks
$ mkdir results
$ mkdir model
```

7. Since you have activated your anaconda virtual environment, you can install, in this environment, libraries required for 
this project.
```
$ pip install -r requirements.txt
```
Libraries required are in your virtual environment, you can check this with `pip freeze` command. 

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
