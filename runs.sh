python classifier/train.py --batch_size 24 > run1.txt;
python classifier/train.py --batch_size 24 --learning_rate 1e-4 > run2.txt;
python classifier/train.py --batch_size 24 --learning_rate 2e-5 > run3.txt;
sudo shutdown -h now;
