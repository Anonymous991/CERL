################################################################# 
Guide to set up and run for CERL_ICML Experiments
################################################################# 

1. Setup Conda
    - Install Anaconda3
    - conda create -n $ENV_NAME$ python=3.6.1
    - source activate $ENV_NAME$

2. Install Pytorch version 1.0
    - Refer to https://pytorch.org/ for instructions
    - conda install pytorch torchvision -c pytorch [GPU-version]

3. Install Numpy, Cython and Scipy
    - pip install numpy==1.15.4
    - pip install cython==0.29.2
    - pip install scipy==1.1.0
    
4. Install Mujoco and OpenAI_Gym
    - Download mjpro150 from https://www.roboti.us/index.html
    - Unzip mjpro150 and place it + mjkey.txt (license file) in ~/.mujoco/ (create the .mujoco dir in you home folder)
    - pip install -U 'mujoco-py<1.50.2,>=1.50.1'
    - pip install 'gym[all]'
    

