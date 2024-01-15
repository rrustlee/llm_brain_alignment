# for layer in 1  4  7  10  13  16  19  22  25  28 31 
# do 
#     for layer2 in 1  4  7  10  13  16  19  22  25  28 31 
#     do
#         python decoding/align_llm.py --subject S1 --layer $layer --layer2 $layer2 --act_name ffn_gate --window 15
#         # python decoding/align_llm.py --subject S1 --layer $layer --layer2 $layer2 --act_name ffn_gate --window 240
#     done
# done


for window in 30 60 120
do
    python decoding/layerwise_alignment.py --window window --roi mean_most:1000
    python decoding/layerwise_alignment.py --window window --roi mean_least:1000
    python decoding/layerwise_alignment.py --window window --roi mean_th_most:0.1
    python decoding/layerwise_alignment.py --window window --roi std_most:1000
    python decoding/layerwise_alignment.py --window window --roi std_least:1000
    python decoding/layerwise_alignment.py --window window --roi std_th_most:0.2
    python decoding/layerwise_alignment.py --window window --roi all
done
