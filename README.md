# Chinese Poem Generator

An ML-based classical Chinese poem generator.

**Update: This was migrated into Python3 and TensorFlow 1.8 in Jun 2018.**

**Update: This was migrated onto PyTorch from TensorFlow in Jan 2025.**


## Components

* vocab.py: A set of characters and their pre-trained embeddings.

* generator.py: A poem generator based on LSTM encoder-decoder.

* discriminator.py: An LSTM-based classifier that distinguishes real poems
from fake ones.

* train\_seq\_gan.py: A training program for a
[SeqGAN model](https://arxiv.org/pdf/1609.05473) implementation.



## To Install Dependencies

```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```


## Training

To train the word embedding model:

```
python3 vocab.py
```

To train the generator model:

```
python3 generator.py
```

To train the discriminator model:

```
python3 discriminator.py

```
To train the SeqGAN model:
```
python3 train_seq_gan.py
```

## Run Interactive Demos

```
python3 demo.py
```

## Preliminary Results

### Poem Generator

Generate a poem given the first character of each sentence (藏头诗):
```
>>> 风和日丽鸟语花香
风雨萧萧秋月明
和风吹雨湿衣裳
日暮东风吹柳絮
丽晴春色满天涯
鸟啼花落春风里
语语春风满袖风
花落花开不知处
香风吹落柳丝丝

>>> 精诚之至金石为开
精神禀气在天涯
诚是中原万事非
之子何人知此意
至今古道无人知
金丸玉骨不可识
石泉泉石无人知
为君一醉一杯酒
开尽桃花一万株

```

### Word Embedding

Search similar characters to a given character:
```
风: 飙飚吹飕飒
花: 蕊红葩杏桃
雪: 霜霰雨腊絮
月: 蟾影皎夜魄
鸟: 禽噪鸦鹂鸥
鱼: 鳞鲤鲈脍虾
虫: 蛩唧蟀啾蟋
```

