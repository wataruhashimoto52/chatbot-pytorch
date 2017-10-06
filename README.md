# Japanese Chatbot (implementation by pytorch)

## Description
This is a sequence-to-sequence conversationa model with attention mechanism implemented by Pytorch. This model is optimized for Japanese. You may replace existing tokenizer with for your language. This implementation is based on official Pytorch Tutorials http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html .

## Difference
* Addition of a script (`collect_replies.py`) to collect conversation data  

* Change of preprocess scheme accompanying it  

## Contents


## Install
Before running these scripts, making a local python client using `pyenv` is reccomended, like:

```
$ pyenv install 3.5.0
$ pyenv virtualenv 3.5.0 example
$ pyenv shell example
# if OS X, python = 3.5, from pip and No CUDA
$ pip3 install http://download.pytorch.org/whl/torch-0.2.0.post3-cp35-cp35m-macosx_10_7_x86_64.whl   
$ pip3 install torchvision 
```

## How to use?

## Results

## Reference

* https://arxiv.org/abs/1506.05869  
* http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html   
* https://github.com/jinfagang/pytorch_chatbot  

## Author

[wataruhashimoto52] https://github.com/wataruhashimoto52 

## Contact
If you find an issue or have some questions, please contact Wataru Hashimoto.
- (at)TinyDrone on Twitter
- w.hashimoto (at) outlook.com