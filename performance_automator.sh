#!/bin/bash

# Run the performance test 10x each, and write the values to a csv file
NUM_TESTS=10

cpu_scan_time_pot=0
cpu_scan_time_npot=0
cpu_compact_without_scan_time_pot=0
cpu_compact_without_scan_time_npot=0
cpu_compact_with_scan_time=0

naive_scan_time_pot=0
naive_scan_time_npot=0

efficient_scan_time_pot=0
efficient_scan_time_npot=0
efficient_compact_time_pot=0
efficient_compact_time_npot=0

thrust_scan_time_pot=0
thrust_scan_time_npot=0

for i in $(seq 1 $NUM_TESTS)
do
    echo -e "Test $i\n"
    echo -e "CPU Test:\n"

    result=$(./bin/cis5650_stream_compaction_test cpu)

    elapsed_time=$(echo "$result" | grep -A 1 "cpu scan, power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    cpu_scan_time_pot=$(echo "$cpu_scan_time_pot + $elapsed_time" | bc)

    elapsed_time=$(echo "$result" | grep -A 1 "cpu scan, non-power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    cpu_scan_time_npot=$(echo "$cpu_scan_time_npot + $elapsed_time" | bc)

    elapsed_time=$(echo "$result" | grep -A 1 "cpu compact without scan, power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    cpu_compact_without_scan_time_pot=$(echo "$cpu_compact_without_scan_time_pot + $elapsed_time" | bc)

    elapsed_time=$(echo "$result" | grep -A 1 "cpu compact without scan, non-power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    cpu_compact_without_scan_time_npot=$(echo "$cpu_compact_without_scan_time_npot + $elapsed_time" | bc)

    elapsed_time=$(echo "$result" | grep -A 1 "cpu compact with scan" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    cpu_compact_with_scan_time=$(echo "$cpu_compact_with_scan_time + $elapsed_time" | bc)

    echo -e "Naive Test:\n"

    result=$(./bin/cis5650_stream_compaction_test naive)

    elapsed_time=$(echo "$result" | grep -A 1 "naive scan, power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    naive_scan_time_pot=$(echo "$naive_scan_time_pot + $elapsed_time" | bc)

    elapsed_time=$(echo "$result" | grep -A 1 "naive scan, non-power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    naive_scan_time_npot=$(echo "$naive_scan_time_npot + $elapsed_time" | bc)

    echo -e "Efficient Test:\n"

    result=$(./bin/cis5650_stream_compaction_test efficient)

    elapsed_time=$(echo "$result" | grep -A 1 "efficient scan, power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    efficient_scan_time_pot=$(echo "$efficient_scan_time_pot + $elapsed_time" | bc)

    elapsed_time=$(echo "$result" | grep -A 1 "efficient scan, non-power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    efficient_scan_time_npot=$(echo "$efficient_scan_time_npot + $elapsed_time" | bc)

    elapsed_time=$(echo "$result" | grep -A 1 "efficient compact, power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    efficient_compact_time_pot=$(echo "$efficient_compact_time_pot + $elapsed_time" | bc)

    elapsed_time=$(echo "$result" | grep -A 1 "efficient compact, non-power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    efficient_compact_time_npot=$(echo "$efficient_compact_time_npot + $elapsed_time" | bc)

    echo -e "Thrust Test:\n"

    result=$(./bin/cis5650_stream_compaction_test thrust)

    elapsed_time=$(echo "$result" | grep -A 1 "thrust scan, power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    thrust_scan_time_pot=$(echo "$thrust_scan_time_pot + $elapsed_time" | bc)

    elapsed_time=$(echo "$result" | grep -A 1 "thrust scan, non-power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    thrust_scan_time_npot=$(echo "$thrust_scan_time_npot + $elapsed_time" | bc)

done

average_cpu_scan_time_pot=$(echo "scale=5; $cpu_scan_time_pot / $NUM_TESTS" | bc)
average_cpu_scan_time_npot=$(echo "scale=5; $cpu_scan_time_npot / $NUM_TESTS" | bc)
average_cpu_compact_without_scan_time_pot=$(echo "scale=5; $cpu_compact_without_scan_time_pot / $NUM_TESTS" | bc)
average_cpu_compact_without_scan_time_npot=$(echo "scale=5; $cpu_compact_without_scan_time_npot / $NUM_TESTS" | bc)
average_cpu_compact_with_scan_time=$(echo "scale=5; $cpu_compact_with_scan_time / $NUM_TESTS" | bc)

average_naive_scan_time_pot=$(echo "scale=5; $naive_scan_time_pot / $NUM_TESTS" | bc)
average_naive_scan_time_npot=$(echo "scale=5; $naive_scan_time_npot / $NUM_TESTS" | bc)

average_efficient_scan_time_pot=$(echo "scale=5; $efficient_scan_time_pot / $NUM_TESTS" | bc)
average_efficient_scan_time_npot=$(echo "scale=5; $efficient_scan_time_npot / $NUM_TESTS" | bc)
average_efficient_compact_time_pot=$(echo "scale=5; $efficient_compact_time_pot / $NUM_TESTS" | bc)
average_efficient_compact_time_npot=$(echo "scale=5; $efficient_compact_time_npot / $NUM_TESTS" | bc)

average_thrust_scan_time_pot=$(echo "scale=5; $thrust_scan_time_pot / $NUM_TESTS" | bc)
average_thrust_scan_time_npot=$(echo "scale=5; $thrust_scan_time_npot / $NUM_TESTS" | bc)

# Now I want to write the results to a csv file
echo -e ",CPU,Naive,Efficient,Thrust\n" > performance_results.csv
echo -e "Scan Time Power of Two,$average_cpu_scan_time_pot,$average_naive_scan_time_pot,$average_efficient_scan_time_pot,$average_thrust_scan_time_pot\n" >> performance_results.csv
echo -e "Scan Time Non-Power of Two,$average_cpu_scan_time_npot,$average_naive_scan_time_npot,$average_efficient_scan_time_npot,$average_thrust_scan_time_npot\n" >> performance_results.csv
echo -e "Compact Time Power of Two,$average_cpu_compact_without_scan_time_pot,,$average_efficient_compact_time_pot,\n" >> performance_results.csv
echo -e "Compact Time Non-Power of Two,$average_cpu_compact_without_scan_time_npot,,$average_efficient_compact_time_npot,\n" >> performance_results.csv
echo -e "(CPU) Compact Time With Scan,$average_cpu_compact_with_scan_time,,,\n" >> performance_results.csv    