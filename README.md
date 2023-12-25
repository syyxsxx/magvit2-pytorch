## install
```
pip install -r requirements.txt

python setup.py install
```

## run
set device config
```
accelerate config
```
run train
```
accelerate launch main.py [dataset]
```
**参数**
* **--data-type** (str): 训练数据类型，可选值[videos,images]，默认videos
* **--bn** (int): 训练batchsize，默认1

