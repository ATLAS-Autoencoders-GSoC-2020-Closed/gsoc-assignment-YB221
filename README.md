# AUTOENCODER_ATLAS
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
Compressing 4 variable particle datasets using autoencoders, with the PyTorch and fastai python libraries.



## Setup:
#### Prerequisites:
Fastai v=1.60[different versions of fastai may cause problem in loading the pretrained weights if saved with fastai]<br />
Python 3.6<br />
jupyter notebook<br />
PyTorch v1<br />
Pandas<br />
Numpy<br />
matplotlib(optional:if you want to plot results)<br />

#### To run the project:

```
sudo apt-get install python3
pip3 install fastai
pip3 install pandas
pip3 install numpy
pip3 install matplotlib
pip3 install torch
pip3 install jupyter-core
git clone https://github.com/YB221/GSoC_Autoencoder_4var_to_3_CERN_HSF_ATLAS
```

Now run jupyter notebook and run the AE.ipynb in cloned folder

## Quick guide
**Network details:**
<br />AE(
  <br />(en1): Linear(in_features=4, out_features=200, bias=True)
  <br />(en2): Linear(in_features=200, out_features=200, bias=True)
  <br />(en3): Linear(in_features=200, out_features=100, bias=True)
  <br />(en4): Linear(in_features=100, out_features=100, bias=True)
  <br />(en5): Linear(in_features=100, out_features=50, bias=True)
  <br />(en6): Linear(in_features=50, out_features=25, bias=True)
  <br />(en7): Linear(in_features=25, out_features=3, bias=True)
  <br />(de1): Linear(in_features=3, out_features=25, bias=True)
  <br />(de2): Linear(in_features=25, out_features=50, bias=True)
  <br />(de3): Linear(in_features=50, out_features=100, bias=True)
  <br />(de4): Linear(in_features=100, out_features=100, bias=True)
  <br />(de5): Linear(in_features=100, out_features=200, bias=True)
  <br />(de6): Linear(in_features=200, out_features=200, bias=True)
  <br />(de7): Linear(in_features=200, out_features=4, bias=True)
  <br />(tanh): Tanh()<br />
)<br />
all the layers are linearly connected with tanh() activation function after each layer except last layer as the actual value range wider than(-1,1).

**Analysis:** <br />
![Alt Text](https://github.com/YB221/GSoC_Autoencoder_4var_to_3_CERN_HSF_ATLAS/blob/master/images/M.jpeg)
![Alt Text](https://github.com/YB221/GSoC_Autoencoder_4var_to_3_CERN_HSF_ATLAS/blob/master/images/eta.jpeg)
![Alt Text](https://github.com/YB221/GSoC_Autoencoder_4var_to_3_CERN_HSF_ATLAS/blob/master/images/phi.jpeg)
![Alt Text](https://github.com/YB221/GSoC_Autoencoder_4var_to_3_CERN_HSF_ATLAS/blob/master/images/pt.jpeg)
**RESIDUALS:**<br />
![alt Text](https://github.com/YB221/GSoC_Autoencoder_4var_to_3_CERN_HSF_ATLAS/blob/master/images/download%20(1).png)
![alt text](https://github.com/YB221/GSoC_Autoencoder_4var_to_3_CERN_HSF_ATLAS/blob/master/images/download%20(2).png)<br />
 <pre>                       M                                                  phi</pre><br />
![alt text](https://github.com/YB221/GSoC_Autoencoder_4var_to_3_CERN_HSF_ATLAS/blob/master/images/download%20(3).png)
![alt text](https://github.com/YB221/GSoC_Autoencoder_4var_to_3_CERN_HSF_ATLAS/blob/master/images/download%20(4).png)<br />
 <pre>                       eta                                                pt</pre><br />

**Code:** Model architecture is saved in Model.py and have following functions:<br />
1.AE(*inputs)-->input: a 4x1 numpy array, output: a 3x1 encoded numpy array and a 4x1 reconstructed output<br />
2.AE.describe()-->input:none, output:[input,200,200,100,50,3,50,100,200,200,Output]<br />
**saved models:**<br />
saved models are in saves folder and can be loaded with python interpreter or ipython into an AE(){model} objectby running:<br />
`torch.load_state_dict(*model,saves/AE.pth)`

## Data <br />
**training data**<br />
(https://drive.google.com/open?id=1-mujHxt2HEdmo6C4wAWW5k88d3YEV9XN)<br />
**testing data**<br />
(https://drive.google.com/open?id=1mhweZvAIELxWq522Q8QhSGDzkXzy_Wxq)<br />
## Training
Training process can be observed in AE.ipynb I used a batch size of 1024 elements with learning rate varying from .1 to 1e-8
model got its convergence in about 400 epochs with Ranger optimizer and no batch normalisation.

## TODO and ideas:
Make a fully abstract script for general purpose use not only for ML practitioners.<br />
Implement the project with variational autoencoder.<br />
Implement the architecture using Root
