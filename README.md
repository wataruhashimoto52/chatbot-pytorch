# Japanese Chatbot (implementation by pytorch)

## Description
This is a sequence-to-sequence conversational model with attention mechanism implemented by Pytorch. This model is optimized for Japanese. You may replace existing tokenizer with for your language. This implementation is based on official Pytorch Tutorials http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html .

## Difference
* Addition of scripts (`collect_replies.py` & `settings.py`) to collect dialogue data  

* Change of preprocess scheme accompanying it  

## Contents  
* `settings.py` - Extract environment variables to use tweepy
* `collect_replies.py` - Obtain dialogue data  
* `preprocess.py` - Preprocess of dialogue data obtained  
* `model.py` - Encoder and Decoder with Attention
* `global_config.py` - Common variables and functions
* `train.py` - Script to run
* `loader.py` - Load previous model and save model

## Machine
* Ubuntu 16.04  
* GeForce GTX 1070
* Memory 16GB

## Requirements
* Anaconda3-4.2.0  
* tweepy  
* python-dotenv  
* MeCab

## Install
Please reference: http://pytorch.org/  

```
$ conda install pytorch torchvision -c soumith
```

## How to use?
1. Collect dialogue data.
```   
$ python collect_replies.py   
```   

2. Move collected data to data directory.
```
$ mv source.txt target.txt data/
```

3. Let's train.
```
$ python train.py   
```

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