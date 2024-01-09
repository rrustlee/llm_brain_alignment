# python decoding/align.py --subject S1 --layer 1 --act_name layer_input --window 15
# python decoding/align.py --subject S1 --layer 5 --act_name layer_input --window 15
# python decoding/align.py --subject S1 --layer 9 --act_name layer_input --window 15
# python decoding/align.py --subject S1 --layer 13 --act_name layer_input --window 15
# python decoding/align.py --subject S1 --layer 17 --act_name layer_input --window 15
# python decoding/align.py --subject S1 --layer 21 --act_name layer_input --window 15
# python decoding/align.py --subject S1 --layer 25 --act_name layer_input --window 15
# python decoding/align.py --subject S1 --layer 29 --act_name layer_input --window 15

# python decoding/align.py --subject S1 --layer 1 --act_name ffn_gate --window 15
# python decoding/align.py --subject S1 --layer 5 --act_name ffn_gate --window 15
# python decoding/align.py --subject S1 --layer 9 --act_name ffn_gate --window 15
# python decoding/align.py --subject S1 --layer 13 --act_name ffn_gate --window 15
# python decoding/align.py --subject S1 --layer 17 --act_name ffn_gate --window 15
# python decoding/align.py --subject S1 --layer 21 --act_name ffn_gate --window 15
# python decoding/align.py --subject S1 --layer 25 --act_name ffn_gate --window 15
# python decoding/align.py --subject S1 --layer 29 --act_name ffn_gate --window 15

# python decoding/align.py --subject S1 --layer 1 --act_name ffn --window 15
# python decoding/align.py --subject S1 --layer 5 --act_name ffn --window 15
# python decoding/align.py --subject S1 --layer 9 --act_name ffn --window 15
# python decoding/align.py --subject S1 --layer 13 --act_name ffn --window 15
# python decoding/align.py --subject S1 --layer 17 --act_name ffn --window 15
# python decoding/align.py --subject S1 --layer 21 --act_name ffn --window 15
# python decoding/align.py --subject S1 --layer 25 --act_name ffn --window 15
# python decoding/align.py --subject S1 --layer 29 --act_name ffn --window 15

for layer in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 
do 
    for layer2 in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 
    do
        python decoding/align_llm.py --subject S1 --layer $layer --layer2 $layer2 --act_name ffn_gate --window 15
    done
done

# python decoding/align.py --subject S1 --layer 1 --act_name layer_input --window 10
# python decoding/align.py --subject S1 --layer 5 --act_name layer_input --window 10
# python decoding/align.py --subject S1 --layer 9 --act_name layer_input --window 10
# python decoding/align.py --subject S1 --layer 13 --act_name layer_input --window 10
# python decoding/align.py --subject S1 --layer 17 --act_name layer_input --window 10
# python decoding/align.py --subject S1 --layer 21 --act_name layer_input --window 10
# python decoding/align.py --subject S1 --layer 25 --act_name layer_input --window 10
# python decoding/align.py --subject S1 --layer 29 --act_name layer_input --window 10

# python decoding/align.py --subject S1 --layer 1 --act_name ffn_gate --window 10
# python decoding/align.py --subject S1 --layer 5 --act_name ffn_gate --window 10
# python decoding/align.py --subject S1 --layer 9 --act_name ffn_gate --window 10
# python decoding/align.py --subject S1 --layer 13 --act_name ffn_gate --window 10
# python decoding/align.py --subject S1 --layer 17 --act_name ffn_gate --window 10
# python decoding/align.py --subject S1 --layer 21 --act_name ffn_gate --window 10
# python decoding/align.py --subject S1 --layer 25 --act_name ffn_gate --window 10
# python decoding/align.py --subject S1 --layer 29 --act_name ffn_gate --window 10

# python decoding/align.py --subject S1 --layer 1 --act_name ffn --window 10
# python decoding/align.py --subject S1 --layer 5 --act_name ffn --window 10
# python decoding/align.py --subject S1 --layer 9 --act_name ffn --window 10
# python decoding/align.py --subject S1 --layer 13 --act_name ffn --window 10
# python decoding/align.py --subject S1 --layer 17 --act_name ffn --window 10
# python decoding/align.py --subject S1 --layer 21 --act_name ffn --window 10
# python decoding/align.py --subject S1 --layer 25 --act_name ffn --window 10
# python decoding/align.py --subject S1 --layer 29 --act_name ffn --window 10

