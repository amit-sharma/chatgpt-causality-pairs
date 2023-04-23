cut -f2 -d, prompts.csv > temp_v1.txt
cut -f3 -d, prompts.csv > temp_v2.txt

cat temp_v1.txt temp_v2.txt > swedish_names.txt

 grep -v "Radikulopati" swedish_names.txt | grep -v "cause" | grep -v "effect" > filt_swedish_names.txt
sort filt_swedish_names.txt >filt_swedish_names_sorted.txt 
uniq filt_swedish_names_sorted.txt > final_swedish_names.txt
