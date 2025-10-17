1  ## Setup Environment
2  
3  We recommend using a conda environment to manage dependencies.
4  
5  ```bash
6  conda create -n braille_env python=3.10
7  conda activate braille_env
8  pip install -r requirements.txt
9  ```
10 
11 The installation of `pytorch` may vary depending on your system.  
12 Please refer to the [official website](https://pytorch.org) for more information.
13 
14 All the training and evaluation scripts use `accelerate` to speed up the training process.  
15 If you want to run the scripts without `accelerate`, you can remove the related code in the scripts.  
16 Remember to run `accelerate config` before you run our scripts, or you may encounter some errors.
