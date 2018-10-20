# Scrape Trump Text from Github Repo to Use as Inputs to Text Generation Algorithm
# Citation: https://github.com/rtlee9

# Packages
###############################################################################
import numpy as np, pandas as pd, bs4 as bs, matplotlib.pyplot as plt, seaborn as sns
import nltk, urllib.parse, urllib.request, string
from nltk.corpus import stopwords
from urllib.error import URLError
from bs4 import BeautifulSoup
from selenium import webdriver
import time, re, string
from time import sleep
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

# URLs with Text
###############################################################################
url_list = ['https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/20160622-election-stakes.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/20160808-detroit-economics.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/20160809-second-amendment.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/20160809-second-amendment.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/20160820-lincoln.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/20160831-charlotte.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/20160831-immigration_phoenix.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/20160906-emails.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/20160907-security-philadelphia.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/20160909-florida.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/20160915-jobs.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/20161013-accusation-response.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/20161014-accusers.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/20161020-Al_Smith_dinner.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/debate_1.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/debate_2.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/debate_3.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/speech0.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/speech1.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/speech2.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/speech3.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/speech4.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/speech5.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/speech6.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/speech7.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/speech8.txt',
            'https://github.com/rtlee9/Trump-bot/blob/master/data/trump/speeches/speech9.txt']

xpath_list = ['//*[contains(concat( " ", @class, " " ), concat( " ", "js-file-line", " " ))]'] * len(url_list)


ph_path = 'C:/tmp/phantomjs.exe'

# Define Functions
###############################################################################
def semi_rand_intervals(max_time, n_nums):
    """random intervals of time between requests"""
    return np.random.choice(np.linspace(0, max_time, 1000), n_nums)

def phantom_scrape(phan_path, web_url, x_path):
    """uses phantomJS to scrap url for a given x path"""
    driver = webdriver.PhantomJS(executable_path = phan_path)
    driver.get(web_url)
    time.sleep(semi_rand_intervals(2,1))
    tmp_list = []
    for i in driver.find_elements_by_xpath(x_path):
        tmp_list.append(i.text)
        time.sleep(semi_rand_intervals(.35,1))
    return tmp_list

def phantom_scrape_multiple(phan_path, url_list, xpath_list):
    """applies 'phantom_scrape' function to list of URLs"""
    plholder = []
    for i, x in tqdm(enumerate(url_list)):
        imp_txt = phantom_scrape(phan_path, url_list[i], xpath_list[i])
        plholder.append(imp_txt)
    return plholder

def concat_nested_strings(nest_str_lst):
    """returns single string from nested lists of strings"""
    plholder = []
    for i in nest_str_lst:
        for j in i:
            plholder.append(j)
    output = ' '.join(plholder)
    return output

# Execute Functions
###############################################################################
pres_txt = phantom_scrape_multiple(ph_path, url_list, xpath_list)

pres_txt_unnested = concat_nested_strings(pres_txt)

pres_txt_nopunct = ''.join([w.lower() for w in re.sub(r'[^\w\s]', ' ', re.sub('['+string.punctuation+']', ' ', pres_txt_unnested))])