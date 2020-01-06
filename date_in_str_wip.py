### Config
######################################################################
import datetime


### Define Functions
######################################################################
def date_range_str(min_date = '1900-01-01', max_date = '2100-01-01', date_format = '%Y-%m-%d'):
    """Generate list of strings with dates in a given range"""
    d1 = datetime.datetime.strptime(min_date, date_format)
    d2 = datetime.datetime.strptime(max_date, date_format)
    dtime_range = [d1 + datetime.timedelta(days=x) for x in range((d2-d1).days + 1)]
    dt_range = [datetime.datetime.strftime(d, date_format) for d in dtime_range]
    return dt_range


def convert_date_str(in_str, in_format, out_format):
    """Convert date string from one format to another"""
    in_date = datetime.datetime.strptime(in_str, in_format)
    return datetime.datetime.strftime(in_date, out_format)


def date_in_str(in_str, date_length = 9, date_format = '%d%b%Y'):
    """
    Returns datestring if date of length <date_length>
    and in format <date_format> is found within <in_str>
    else returns None
    """
    date_found = None
    for i in range(len(in_str)):
        substr = in_str[i:(i + date_length)]
        try:
            date_found = datetime.datetime.strptime(substr, date_format)
        except:
            pass
    return datetime.datetime.strftime(date_found, date_format)


def date_range_in_str(in_str, min_date, max_date, date_length = 9, date_format = '%d%b%Y'):
    """
    Returns datestring if date of length <date_length>
    and in format <date_format> is found within <in_str>
    *and* the date is within the <min_date> - <max_date> range (inclusive)
    else returns None
    """
    date_found = date_in_str(in_str, date_length, date_format)
    if date_found is not None:
        check_range = date_range_str(min_date, max_date, date_format)
        if date_found not in check_range:
            return_date = None
        else:
            return_date = date_found
    return return_date


### Testing (unfinished)
######################################################################
# Should return None
test_i = date_range_in_str(in_str = 'tps_report_01APR2019.csv',
                           min_date = '15APR2019',
                           max_date = '25APR2019',
                           date_length = 9,
                           date_format = '%d%b%Y')

# Should return '01APR2019'
test_ii = date_range_in_str(in_str = 'tps_report_01APR2019.csv',
                           min_date = '01APR2019',
                           max_date = '25APR2019',
                           date_length = 9,
                           date_format = '%d%b%Y')

# Should return '20191219'
test_iii = date_in_str(in_str = 'tps_report_20191219.csv', date_length = 8, date_format = '%Y%m%d')

# Should return '2019_01_01'
test_iv = date_in_str(in_str = 'tps_report_2019_01_01.csv', date_length = 10, date_format = '%Y_%m_%d')










