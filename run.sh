for i in `seq 0 29`;
do
    python main.py configs/DAVIS16_oneshot $i
done
#python main.py configs/DAVIS16_oneshot 0
#python main.py configs/custom_oneshot 0
#python main.py configs/DAVIS_objectness 0
#python main.py configs/DAVIS16_oneshot 0


