# TasarimCalismasi
In this project VGG16 - Inception_v3 - ResNet50_v2 - Xception CNNs has been trained 1-10-25 epochs and compared with each other and ELM to obtain their differences on each train time, norm of time for each step, file size, accuracy, loss, validated accuracy, validated loss.

I've used conda for python environment. To clone this project follow these steps:  
1.Install conda.  
2.Create a conda environment from the file in the repository which is 'environment.yml'.To create environment from yml file use following code in your terminal or in your command prompt: conda env create -f environment.yml  
3.The created conda environment will be named py37 use this environment in your IDE as python interpreter.  
4.Download and unzip the dataset from: https://www.kaggle.com/c/digit-recognizer
4.Download and unzip the dataset from:  
5.Update the file path to your current filepath of dataset's in line 7 which is stored in variable base_dir.  
6.Run the code.
Note: Be aware of environment.yml is configured for Apple M Series Chipset. To use it on x86 processors (intel - amd) you may need to open yml file and install the python packages manually.
