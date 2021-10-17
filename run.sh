question=$1
train_dir=$2
test_dir=$3
output_dir=$4
if [[ ${question} == "1" ]];
    then python3 Q1.py $train_dir $test_dir $output_dir
fi
if [[ ${question} == "2" ]];
    then python3 Q2.py $train_dir $test_dir $output_dir
fi
if [[ ${question} == "3" ]];
    then python3 Q3.py $train_dir $test_dir $output_dir
fi
