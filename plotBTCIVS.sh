write_option_script="write_option_data.py" 

symbol_flag="--symbol=btc" 
side_flag="--side=ask" 
type_flag="--type=c" 

plot_python_script="plot_out.py"


python3 "$write_option_script" "$symbol_flag" "$side_flag" "$type_flag"

make

./compute_IV b a c

python3 "$plot_python_script"

make clean


