python classifier/train.py --batch_size 24 > 1e-5_run.txt;
python classifier/train.py --batch_size 24 --learning_rate 1e-4 > 1e-4_run.txt;
python classifier/train.py --batch_size 24 --learning_rate 2e-5 > 2e-5_run.txt;
sudo shutdown -h now;