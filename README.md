# Bias and Fairness Detector

| **License** | ![APM](https://img.shields.io/apm/l/vim-mode?color=orange&label=License&logo=MIT) |
| ----- | ---- |
| **Libraries** |  ![APM](https://img.shields.io/badge/Pandas-1.2.5-green) ![APM](https://img.shields.io/badge/Numpy-1.21-green) ![APM](https://img.shields.io/badge/PrettyTable-2.1-green) 

<p align="center">
  <img src="img/bg.png" height="300px"/>
</p>

One of the biggest challenges in Machine Learning is dealing with biased datasets. Having autonomous decision making systems (ADM) trained with biased datasets might have strong negative consequences on the lives of their data subjects (see for example the [Compass case](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm)).

In this repository I provided an easy way for measuring the bias of your datasets. There are several biases measures, these are the ones that I have implemented:
* Gini Simpson diversity index
* Shannon diversity index
* Inverse Simpson diversity index
* Imabalance ratio

All these index will be in the range [0,1], and by construction, the higher their values will be, the "better" will be the dataset. 

Reference : 
* https://en.wikipedia.org/wiki/Diversity_index

### Table of Contents  
* [How can you run it?](#run)  
* [Do you want to contribute?](#contribute)  
* [What's next?](#next)  
* [Do you want to reach me out? ](#contacts)  



<a name="run"/>

## How can you run it?
Once you clone the repository, you can run the script with the default dataset and settings:
```
python main.py
```

The default [dataset](https://www.kaggle.com/danofer/compass) is the one used for the Compass case. You can add the following options:
* --dataset \<directory of the dataset\>
* --sensitive_attr \<list of sensitive attributes that you want to analyze\>

For example: 
```
python main.py --dataset data/propublica_data_cleaned.csv --sensitive_attr Ethnicity Age Female
```

The result will be: 

|  `Feature`  | `Gini` | `Shannon` | `Simpson` | `Imbalanced Ratio` |
|-----------|------|---------|---------|------------------|
| Ethnicity | 0.73 |   0.62  |   0.31  |       0.0        |
|    Age    | 0.87 |   0.89  |   0.69  |       0.37       |
|   Female  | 0.62 |   0.7   |   0.45  |       0.24       |




<a name="contribute"/>

## Do you want to contribute?

1. Fork the repository
2. Do the desired changes
3. Make a pull request

Et voila! 


<a name="next"/>

## What's next?
In my to do list you can find at the moment:
* streamlit interactive dashboard
* implementation of fairness measures

<a name="contacts"/>

## Do you want to reach me out? 
* [![Linkedin](https://img.shields.io/badge/-LinkedIn-blue?style=flat&logo=Linkedin&logoColor=white)](https://www.linkedin.com/in/francescodisalvo-pa/)
* [`francesco.disalvo99@gmail.com`](mailto:francesco.disalvo99@gmail.com)
