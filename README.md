# AI_Frameworks

# What do I have to install to be able to reproduce the code?

1. Install Google cloud SDK

2. Connect to your instance installed with a GPU P100 and NVIDIA drivers pre-installed

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
$ conda create -n AIF python=3.8
```

And activate this environment :
```
$ conda activate AIF
```

5. Now, we can download our project from github :
```
$ git clone https://github.com/tcotte/AI_Frameworks.git
```

6. Then, move in our project and create "results" and 
