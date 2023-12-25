import os
import json
from transformers import AutoTokenizer
import pickle


model_dir = '/share/gzhch/resource/models/Llama-2-7b-hf/'
tokenizer = AutoTokenizer.from_pretrained(model_dir)

all_aligned_inputs = {}
wav_dir = '/data/gzhch/narratives/stimuli/gentle/'
for task_name in os.listdir(wav_dir):
    if task_name.startswith('.'):
        continue

    with open('/data/gzhch/narratives/stimuli/gentle/{task_name}/align.json'.format(task_name=task_name), 'r') as f:
        raw_input = json.loads(f.read())

    input_text = raw_input['transcript']
    words_wav = raw_input['words']
    words_llm = input_text.split()
    words_aligned = []
    enc = tokenizer(words_llm, is_split_into_words=True)

    print(len(words_llm), len(words_wav))

    j = 0
    for i in range(len(words_llm)):
        j0 = j
        word_llm = words_llm[i]
        word_wav = []
        start, end = 1000000, 0
        while j < len(words_wav) and words_wav[j]['word'] in word_llm:
            word_llm = word_llm.replace(words_wav[j]['word'], '', 1)
            word_wav.append(words_wav[j]['word'])
            if words_wav[j]['case'] == 'success':
                start = min(start, words_wav[j]['start'])
                end = max(end, words_wav[j]['end'])
            j += 1

        word = dict(word_to_token=enc.word_to_tokens(i), word_llm=words_llm[i], word_wav=word_wav, start=start, end=end)
        words_aligned.append(word)
        # if j - j0 > 1:
        #     print(i, words_llm[i], word_wav)

    all_aligned_inputs[task_name] = words_aligned


with open('aligned_input.pkl', 'wb') as f:
    pickle.dump(all_aligned_inputs, f)