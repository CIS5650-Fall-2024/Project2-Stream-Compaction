#!/bin/bash

# Run the performance test 10x each, and write the values to a csv file
NUM_TESTS=20

cpu_scan_time_pot=()
cpu_scan_time_npot=()
cpu_compact_without_scan_time_pot=()
cpu_compact_without_scan_time_npot=()
cpu_compact_with_scan_time=()

naive_scan_time_pot=()
naive_scan_time_npot=()

efficient_scan_time_pot=()
efficient_scan_time_npot=()
efficient_compact_time_pot=()
efficient_compact_time_npot=()

thrust_scan_time_pot=()
thrust_scan_time_npot=()

for i in $(seq 1 $NUM_TESTS)
do
    echo -e "Test $i\n"
    result=$(./bin/cis5650_stream_compaction_test cpu)

    elapsed_time=$(echo "$result" | grep -A 1 "cpu scan, power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    cpu_scan_time_pot+=($elapsed_time)

    elapsed_time=$(echo "$result" | grep -A 1 "cpu scan, non-power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    cpu_scan_time_npot+=($elapsed_time)

    elapsed_time=$(echo "$result" | grep -A 1 "cpu compact without scan, power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    cpu_compact_without_scan_time_pot+=($elapsed_time)

    elapsed_time=$(echo "$result" | grep -A 1 "cpu compact without scan, non-power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    cpu_compact_without_scan_time_npot+=($elapsed_time)

    elapsed_time=$(echo "$result" | grep -A 1 "cpu compact with scan" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    cpu_compact_with_scan_time+=($elapsed_time)

    result=$(./bin/cis5650_stream_compaction_test naive)

    elapsed_time=$(echo "$result" | grep -A 1 "naive scan, power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    naive_scan_time_pot+=($elapsed_time)

    elapsed_time=$(echo "$result" | grep -A 1 "naive scan, non-power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    naive_scan_time_npot+=($elapsed_time)

    result=$(./bin/cis5650_stream_compaction_test efficient)

    elapsed_time=$(echo "$result" | grep -A 1 "efficient scan, power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    efficient_scan_time_pot+=($elapsed_time)

    elapsed_time=$(echo "$result" | grep -A 1 "efficient scan, non-power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    efficient_scan_time_npot+=($elapsed_time)

    elapsed_time=$(echo "$result" | grep -A 1 "efficient compact, power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    efficient_compact_time_pot+=($elapsed_time)

    elapsed_time=$(echo "$result" | grep -A 1 "efficient compact, non-power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    efficient_compact_time_npot+=($elapsed_time)

    result=$(./bin/cis5650_stream_compaction_test thrust)

    elapsed_time=$(echo "$result" | grep -A 1 "thrust scan, power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    thrust_scan_time_pot+=($elapsed_time)

    elapsed_time=$(echo "$result" | grep -A 1 "thrust scan, non-power-of-two" | grep -oP 'elapsed time: \K[0-9]+\.[0-9]+')
    thrust_scan_time_npot+=($elapsed_time)

done

calculate_median() {
    arr=($(printf '%s\n' "${@}" | sort -n))
    len=${#arr[@]}
    if (( $len % 2 == 0 )); then
        echo "scale=5; (${arr[$len/2-1]} + ${arr[$len/2]}) / 2" | bc
    else
        echo "${arr[$len/2]}"
    fi
}

median_cpu_scan_time_pot=$(calculate_median "${cpu_scan_time_pot[@]}")
median_cpu_scan_time_npot=$(calculate_median "${cpu_scan_time_npot[@]}")
median_cpu_compact_without_scan_time_pot=$(calculate_median "${cpu_compact_without_scan_time_pot[@]}")
median_cpu_compact_without_scan_time_npot=$(calculate_median "${cpu_compact_without_scan_time_npot[@]}")
median_cpu_compact_with_scan_time=$(calculate_median "${cpu_compact_with_scan_time[@]}")

median_naive_scan_time_pot=$(calculate_median "${naive_scan_time_pot[@]}")
median_naive_scan_time_npot=$(calculate_median "${naive_scan_time_npot[@]}")

median_efficient_scan_time_pot=$(calculate_median "${efficient_scan_time_pot[@]}")
median_efficient_scan_time_npot=$(calculate_median "${efficient_scan_time_npot[@]}")
median_efficient_compact_time_pot=$(calculate_median "${efficient_compact_time_pot[@]}")
median_efficient_compact_time_npot=$(calculate_median "${efficient_compact_time_npot[@]}")

median_thrust_scan_time_pot=$(calculate_median "${thrust_scan_time_pot[@]}")
median_thrust_scan_time_npot=$(calculate_median "${thrust_scan_time_npot[@]}")

# Now write the results to a csv file
echo -e ",CPU,Naive,Efficient,Thrust\n" > performance_results.csv
echo -e "Scan Time Power of Two,$median_cpu_scan_time_pot,$median_naive_scan_time_pot,$median_efficient_scan_time_pot,$median_thrust_scan_time_pot" >> performance_results.csv
echo -e "Scan Time Non-Power of Two,$median_cpu_scan_time_npot,$median_naive_scan_time_npot,$median_efficient_scan_time_npot,$median_thrust_scan_time_npot" >> performance_results.csv
echo -e "Compact Time Power of Two,$median_cpu_compact_without_scan_time_pot,,$median_efficient_compact_time_pot," >> performance_results.csv
echo -e "Compact Time Non-Power of Two,$median_cpu_compact_without_scan_time_npot,,$median_efficient_compact_time_npot," >> performance_results.csv
echo -e "(CPU) Compact Time With Scan,$median_cpu_compact_with_scan_time,,," >> performance_results.csv

