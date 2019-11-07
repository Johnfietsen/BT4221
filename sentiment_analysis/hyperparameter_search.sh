for file_name in "sanders.csv" "semeval_balanced.csv" "semeval_sanders.csv"
do
for i in 64 128 256
do
for j in 64 128 256
do
echo $file_name embed_dim=$i lstm_out=$j 
python3 lstm.py --embed_dim=$i --lstm_out=$j --pretrain=True --data_file=$file_name --pretrained_folder=${i}_${j}
done
done
done
