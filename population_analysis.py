import numpy as np
from sklearn.linear_model import LinearRegression

def setup():
    global countries_pop_data, year_row
    with open("populationbycountry19802010millions.csv") as population_record_file:
            year_row = list(map(int, population_record_file.readline().strip().split(',')[1:]))
            countries_pop_data = {}
            while (data := population_record_file.readline().strip()) != '':
                data = data.split(',')
                country_name = data[0]
                country_pop = data[1:]
                countries_pop_data[normalise_string(country_name)] = country_pop

def normalise_string(s):
    """Convert a string into another all-lowercase string with only it's letters and numbers, preserving order."""
    normalised = ''
    for letter in s:
        if letter.isalnum():
            normalised += letter
    return normalised.lower()

def get_population(countries_pop_data, country, year):
    country = normalise_string(country)
    try:
        year_idx = year_row.index(year)
    except ValueError:
        return "No data available"
    try:
        pop_value = countries_pop_data[country][year_idx]
    except KeyError:
        return "No data available"
    try:
        return int(float(pop_value) * 1000000)
    except ValueError:
        return "No data available"

def get_highest_population(countries_pop_data, country):
    """Returns a tuple (year, population) of the largest population count of that country."""
    year_with_largest_pop = 0
    largest_population = 0
    for year in year_row:
        pop_count = get_population(countries_pop_data, country, year)
        if pop_count == "No data available":
            continue
        if largest_population < pop_count:
            largest_population = pop_count
            year_with_largest_pop = year

    if year_with_largest_pop == 0:
        return "No data available"
    else:
        return (year_with_largest_pop, largest_population)

def filter_with_minimum_population_size(countries_pop_data, minimum_population, year_range=None):
    """Returns data with similar structure to the original `countries_pop_data`, but only with those countries which have a maximum
    population size >= `minimum_population` in a specified `year_range` (Default includes every year)"""
    if year_range is None:
        year_range = year_row
    
    filtered_pop_data = {}
    for country in countries_pop_data:
        maximum_pop_size = 0
        for year in year_range:
            pop_size = get_population(countries_pop_data, country, year)
            if pop_size != "No data available":
                maximum_pop_size = max(maximum_pop_size, pop_size)
        if maximum_pop_size >= minimum_population:
            filtered_pop_data[country] = countries_pop_data[country]
    return filtered_pop_data

def get_longest_nondecreasing_subsequnce_table(nums):
    memo = [1] * len(nums)
    for i in range(1, len(nums)):
        curr_num = nums[i]
        for j in range(0, i):
            if nums[j] <= curr_num:
                memo[i] = max(memo[i], memo[j] + 1)
    return memo

def longest_nondecreasing_subsequence(nums):
    if len(nums) == 0:
        return 0
    return max(get_longest_nondecreasing_subsequnce_table(nums))

def get_all_available_data(countries_pop_data, country):
    country_pop_data = []
    for year in year_row:
        data = get_population(countries_pop_data, country, year)
        if data != "No data available":
            country_pop_data.append(data)
    return country_pop_data

def is_population_trend_increasing(countries_pop_data, country, min_length_lnds=28):
    country_pop_data = get_all_available_data(countries_pop_data, country)
    if len(country_pop_data) == 0:
        raise ValueError("No data available to determine trend.")
    return longest_nondecreasing_subsequence(country_pop_data) >= min_length_lnds

def is_population_trend_bitonic(countries_pop_data, country, min_length_bitonic=28):
    country_pop_data = get_all_available_data(countries_pop_data, country)
    if len(country_pop_data) == 0:
        raise ValueError("No data available to determine trend.")
    forward_lis = get_longest_nondecreasing_subsequnce_table(country_pop_data)
    backward_lis = get_longest_nondecreasing_subsequnce_table(country_pop_data[::-1])[::-1]
    longest_bitonic_subsequence_length = max([a + b - 1 for a, b in zip(forward_lis, backward_lis)])
    return longest_bitonic_subsequence_length >= 28 and get_highest_population(countries_pop_data, country)[0] != year_row[-1] # If the highest is the last element, it's strictly increasing, and not bitonic.

def predict_next_value_linearly(vals):
    X = np.array(range(0, len(vals))).reshape(-1, 1)
    Y = np.array(vals).reshape(-1, 1)
    regsr = LinearRegression()
    regsr.fit(X, Y)
    to_predict = np.array([len(vals)]).reshape(-1, 1)
    predicted_y = regsr.predict(to_predict)
    return predicted_y[0][0]

def predict_next_year_population(countries_pop_data, country):
    country_pop_data = get_all_available_data(countries_pop_data, country)
    if len(country_pop_data) == 0:
        raise ValueError("No data available to predict any values.")
    if is_population_trend_increasing(countries_pop_data, country):
        return int(round(predict_next_value_linearly(country_pop_data)))
    else:
        # Probably bitonic
        year, highest_pop = get_highest_population(countries_pop_data, country)
        return int(round(predict_next_value_linearly(country_pop_data[year-year_row[0]:])))
        
def main():
    setup()

    print("Singapore's population in 2002 is %d." % get_population(countries_pop_data, 'Singapore', 2002))  # 4.19778 million
    print("%s for Aruba's population in 1985." % get_population(countries_pop_data, 'Aruba', 1985)) # No data available
    print()

    print("Singapore's largest population is in %d with %d people." % get_highest_population(countries_pop_data, 'Singapore')) # (2010, 4701070)
    print("Saint Pierre and Miquelon's largest population is in %d with %d people." % get_highest_population(countries_pop_data, 'Saint Pierre and Miquelon')) # (1999, 6430)
    print("%s for the largest population in Antarctica." % get_highest_population(countries_pop_data, 'Antarctica')) # No data available
    print()

    large_population_countries = filter_with_minimum_population_size(countries_pop_data, 100000000, range(1980, 1982))
    print("After filtering the data such that only countries with maximum population above or equal to 100 million is left, %s for Singapore's population in 2002." % get_population(large_population_countries, 'Singapore', 2002)) # No data available
    print()

    print("Is population trend for Singapore increasing? %r" % is_population_trend_increasing(countries_pop_data, 'Singapore')) # True
    print("Is population trend for Japan increasing? %r" % is_population_trend_increasing(large_population_countries, 'Japan')) # False

    print("Is population trend for Japan bitonic (ie. increasing then decreasing)? %r" % is_population_trend_bitonic(countries_pop_data, 'Japan')) # True
    print("Is population trend for Singapore bitonic (ie. increasing then decreasing)? %r" % is_population_trend_bitonic(countries_pop_data, 'Singapore')) # False
    
    try:
        print("Trying to search if Singapore's population trend is increasing in the filtered large population dataset...")
        print(is_population_trend_increasing(large_population_countries, 'Singapore'))
    except ValueError as err:
        print(err) # This should execute
    print()

    print("Singapore's population in 2011 should be %d." % predict_next_year_population(countries_pop_data, 'Singapore'))
    print("Saint Pierre and Miquelon's population in 2011 should be %d." % predict_next_year_population(countries_pop_data, 'Saint Pierre and Miquelon'))
    print("Japan's population in 2011 should be %d." % predict_next_year_population(countries_pop_data, 'Japan'))

    try:
        print("Trying to predict the population in Hawaiian Trade Zone (which had no data) for 2011...")
        print(predict_next_year_population(countries_pop_data, 'Hawaiian Trade Zone'))
    except ValueError as err:
        print(err) # This should execute

if __name__ == '__main__':
    main()
