python main.py --local_ep 20 --rounds 40 --server_ep 20 --update dist --teacher_type SWAG --use_client --use_SWA --num_users 10 --model resnet32 --weight_decay 0.0002  --exp fedbe &
python main.py --local_ep 20 --rounds 40 --server_ep 20 --update dist --teacher_type clients --use_client --num_users 10 --model resnet32 --weight_decay 0.0002  --exp vanilla &
python main.py --local_ep 20 --rounds 40 --server_ep 20 --update FedAvg --teacher_type clients --num_users 10 --model resnet32 --weight_decay 0.0002 --exp fedavg


python main.py --local_ep 20 --rounds 40 --server_ep 20 --update dist --teacher_type SWAG --use_SWA --use_client --num_users 10 --model cnn --weight_decay 0.001 --exp fedbe  &
python main.py --local_ep 20 --rounds 40 --server_ep 20 --update dist --teacher_type clients --num_users 10 --model cnn --weight_decay 0.001 --use_client --exp vanilla &
python main.py --local_ep 20 --rounds 40 --server_ep 20 --update FedAvg --teacher_type clients --num_users 10 --model cnn --weight_decay 0.001 --exp fedavg




