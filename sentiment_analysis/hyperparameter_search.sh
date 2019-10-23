for i in 64 128 256
do
for j in 64 128 256
do
echo embed_dim=$i lstm_out=$j
python3 lstm.py --embed_dim=$i --lstm_out=$j
done
done
