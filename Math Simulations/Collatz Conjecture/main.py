
#TODO: finish the plot all leading digit graph at once
#TODO: try yo do the general animated version, on the same figure. rn it is being closed / opened for every sequence. it should be smoother
#TODO: add the other data graphs from the video as noted on the notebook


from animate_collatz_conjecture_datas import (
    animate_each_value, 
    plot_all_value, 
    log_value_of_one_number,
    animate_leading_digit
)
from collatz_conjecture_functions import generate_hailstone_sequences


num: int = 27
start_value: int = 1
end_value: int = 10000
detrended: bool = True

# Call of the main function
hailstone_numbers, total_stoppage_time = generate_hailstone_sequences(start_value, end_value)

#plot_all_value(start_value, end_value, hailstone_numbers, total_stoppage_time)
#animate_each_value(start_value, hailstone_numbers, total_stoppage_time)
#log_value_of_one_number(num, hailstone_numbers, total_stoppage_time, detrended)
animate_leading_digit(start_value, end_value, hailstone_numbers)

