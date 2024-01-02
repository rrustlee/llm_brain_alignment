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

pip install datalad

datalad install https://datasets.datalad.org/labs/hasson/narratives

datalad get [TARGET_FILE]

### llm数据

用extract_llm_act.py提取