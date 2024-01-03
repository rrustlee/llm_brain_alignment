我们暂时使用llama-2-7b模型

## 代码
### extract_llm_act.py 

通过调用utils/activation.py里的custom_forward，从llm中提取需要的中间层activation，目前包含三种：attn（self-attention score）, ffn_gate（ffn的中间层）, ffn（ffn的输出）。我使用的llama是老版本tranformers的实现，可能会报错，如果报错，就用utils/modeling_llama.py加载模型。

### input_alignment.py

将llm的输入，即sequence of token和fMRI的音频输入对齐

### utils/activation.py
主要是用了custom_forward函数，用它代替了LLamaModel.forward(),通过inspect_acts参数传入需要提取的activations

这是我的史山代码，不用看，有问题问我

其它的文件暂时用不上

## 数据
### fMRI数据
#### narratives dataset
```
pip install datalad # 数据集管理软件
datalad install https://datasets.datalad.org/labs/hasson/narratives # 安装数据集，但不会直接下载数据
datalad get [TARGET_FILE] # 下载数据

pip install nibabel # 用于加载fMRI数据文件
```

我们目前主要使用的数据是`derivatives/afni-nosmooth/`文件夹下的fMRI数据以及`stimuli/`文件夹下的文本输入数据
```
cd narrative
datalad get derivatives/afni-nosmooth/*
datalad get stimuli/*
```
`derivatives/afni-nosmooth/`文件夹内以`sub-*`开头的文件夹代表一位被试的实验数据，以`sub-001`为例:
```
cd derivatives/afni-nosmooth/sub-001/func
```
可以看到一堆文件，我们只需要关注以`nii.gz`为后缀的文件，这些文件大抵遵循`sub-XXX_task-[TASK_NAME]_run-X_.....nii.gz`的命名方式，代表该被试在听`[TASK_NAME]`故事全程的fMRI信号记录

那么如何读取`nii.gz`文件呢，在python中：
```
import nibabel as nib

# 加载数据
fmri_imgs = nib.load('/data/gzhch/narratives/derivatives/afni-nosmooth/sub-001/func/sub-001_task-pieman_run-1_space-MNI152NLin2009cAsym_res-native_desc-clean_bold.nii.gz') 

# 显示数据形状
print(fmri_imgs.shape) # (65, 77, 49, 300) 前三维为空间维度，最后一维为时间维，时间单位是tr， 在这个数据集里 1 tr == 1.5 second

# 目前还没有拿到数据数组，需要get_fdata()加载具体数据
fmri_act = fmri_imgs.get_fdata() # fmri_act 是numpy数组
```

### llm数据

用extract_llm_act.py提取

由于llm activation的体积很大，所以对于ffn_gate，我目前用tensor.topk保留激活值偏大的信息，过滤了其他信息，然后存储为pickle文件。因为我认为ffn的neurons是稀疏激活的，从矩阵乘法计算的角度，只有较高的激活值所对应的行向量对ffn输出有较大贡献。但是我还没有验证这个猜想，以及这个性质应该只适用于ffn_gate，而不是适用于ffn和attn。所以如果存储空间足够大话，最好直接保存原始的tensor。



### Appendix
数据处理和对齐方法可以参考这个仓库 https://github.com/HuthLab/semantic-decoding

我在utils/data.py里写了一个数据处理类，也可以参考，但还没写完
