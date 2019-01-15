# Lists and Strings
##############################################################################
def unnest_list_of_lists(LOL):
    """unnest list of lists"""
    import itertools
    return list(itertools.chain.from_iterable(LOL))

def rem_multiple_substr(your_string, your_removals):
    """remove list of substrings ('your_removals') from a string ('your_string')"""
    your_new_string = your_string
    replace_dict = dict(zip(your_removals, ['' for yr in your_removals]))
    for removal, blank in replace_dict.items():
        your_new_string = your_new_string.replace(removal, blank)
    return your_new_string

def remove_http_https(some_string):
    """remove URLs starting with 'http' or 'https'"""
    import re
    return re.sub(r'http\S+', '', some_string, flags = re.MULTILINE)

def remove_low_char_count(string_list, min_alpha_char = 10):
    """remove strings from list with less than 'min_alpha_char' alphabetic characters"""
    import re, tqdm
    new_string_list = []
    rmv_string_count = []
    for s in tqdm(string_list):
        if len(re.sub("[^a-zA-Z]+", "", s)) >= min_alpha_char:
            new_string_list.append(s)
        else:
            rmv_string_count.append(1)
    n_rmv = str(int(sum(rmv_string_count)))
    print("removed " + n_rmv  + " observations with less than " + str(min_alpha_char) + " alphabetic characters")
    return new_string_list

# Dates and Times
##############################################################################
def date_to_unix_ts(dt_str, dt_format = '%Y-%m-%d'):
    """date to unix timestamp"""
    import time, datetime
    return int(time.mktime(datetime.datetime.strptime(dt_str, dt_format).timetuple()))

def date_range_str(start_dt, end_dt, dt_format = '%Y-%m-%d'):
    """return range of dates in str format"""
    import datetime
    start = datetime.datetime.strptime(start_dt, dt_format)
    end = datetime.datetime.strptime(end_dt, dt_format)
    dt_array = (start + datetime.timedelta(days = i) for i in range(0, (end - start).days))
    return [str(i.date()) for i in dt_array]

def dt_add_days(date_str, add_n, dt_format = '%Y-%m-%d'):
    """add (or subtract) days from date"""
    import datetime
    return str((datetime.datetime.strptime(date_str, dt_format) + datetime.timedelta(days = add_n)).date())

def seconds_to_time(sec):
    """convert seconds (integer or float) to time in 'hh:mm:ss' format"""
    import numpy as np
    if (sec // 3600) == 0:
        HH = '00'
    elif (sec // 3600) < 10:
        HH = '0' + str(int(sec // 3600))
    else:
        HH = str(int(sec // 3600))
    min_raw = (np.float64(sec) - (np.float64(sec // 3600) * 3600)) // 60
    if min_raw < 10:
        MM = '0' + str(int(min_raw))
    else:
        MM = str(int(min_raw))
    sec_raw = (sec - (np.float64(sec // 60) * 60))
    if sec_raw < 10:
        SS = '0' + str(int(sec_raw))
    else:
        SS = str(int(sec_raw))
    return HH + ':' + MM + ':' + SS + ' (hh:mm:ss)'

def sec_to_time_elapsed(end_tm, start_tm, return_time = False):
    """apply seconds_to_time function to start and end times"""
    import numpy as np
    sec_elapsed = (np.float64(end_tm) - np.float64(start_tm))
    if return_time:
        return seconds_to_time(sec_elapsed)
    else:
        print('Execution Time: ' + seconds_to_time(sec_elapsed))