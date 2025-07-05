
from animate_collatz_conjecture_datas import (
    animate_each_value, 
    plot_all_value, 
    plot_log_value_of_one_number,
    animate_leading_digit,
    plot_all_leading_digit,
    plot_highest_reached_number
)
from collatz_conjecture_functions import generate_hailstone_sequences

num: int = 27
start_value: int = 50
end_value: int = 100
detrended: bool = True

# Call of the main function
hailstone_numbers, total_stoppage_time = generate_hailstone_sequences(start_value, end_value)

#plot_all_value(start_value, end_value, hailstone_numbers, total_stoppage_time)
#animate_each_value(start_value, hailstone_numbers, total_stoppage_time)
#log_value_of_one_number(num, hailstone_numbers, total_stoppage_time, detrended)
#animate_leading_digit(start_value, end_value, hailstone_numbers)
#plot_all_leading_digit(start_value, end_value, hailstone_numbers)
plot_highest_reached_number(start_value, end_value, hailstone_numbers)
