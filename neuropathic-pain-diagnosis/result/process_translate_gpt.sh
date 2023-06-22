sed 's/, \*//g' translated_gpt.csv > proc_translated_gpt.csv
sed 's/discomfort/pain/g' proc_translated_gpt.csv > proc_translated_gpt_pain.csv
sed -i  's/Discomfort/pain/g' proc_translated_gpt_pain.csv
sed -i  's/problems/pain/g' proc_translated_gpt_pain.csv
sed -i  's/Problems/pain/g' proc_translated_gpt_pain.csv
