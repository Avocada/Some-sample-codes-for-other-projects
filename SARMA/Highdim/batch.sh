for R in 5
do
    python3 sim_Time.py -s $R -n 500 -T 500 > /dev/null &
done
# for R in 6
# do
#     python3 sim_Time.py -r $R -n 500 -T 500 > /dev/null &
# done
wait
# python3 sim_Time.py -r 1 -N 5 -s 1 -n 500 -m 0 -T 1000 > /dev/null &
# python3 sim_Time.py -r 1 -N 8 -s 1 -n 500 -m 0 -T 1000 > /dev/null &
# python3 sim_Time.py -r 1 -N 3 -s 1 -n 500 -m 0 -T 1000 > /dev/null &
# wait

# for T in 1000
# do
#     for R in 1
#     do
#         python3 sim_Time.py -N 8 -s 1 -r $R -n 500 -m 0 -T $T > /dev/null &
#     done
# done 
# wait