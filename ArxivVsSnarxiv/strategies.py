from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait
from abc import ABC, abstractmethod
from random import random, choice
import multiprocessing as mp
import os
import csv
from functools import partial
from copy import copy
from tqdm import tqdm
from selenium.webdriver.firefox.options import Options


class element_has_text(object):
    """
    An expectation for checking that an element has any text contained within it.

    locator: used to find the element
    returns the WebElement once it has any text.
    """

    def __init__(self, locator):
        self.locator = locator

    def __call__(self, driver):
        element = driver.find_element(*self.locator)
        if element.text:
            return element
        else:
            return False


def start_driver(driver_type='Firefox', headless=True):
    """
    Starts a webdriver of the type driver_type

    :param bool headless: Minimizes the driver immediately, currently only works for firefox.
    :param str driver_type: Which browser to use.
    :return: WebDriver
    """
    options = Options()
    if headless:
        options.headless = True
    driver = {
        'Chrome': webdriver.Chrome,
        'Safari': webdriver.Safari,
        'Edge': webdriver.Edge,
        'Firefox': webdriver.Firefox
    }[driver_type]
    driver = driver(options=options)
    driver.get('http://snarxiv.org/vs-arxiv/')
    return driver


def get_options(driver: webdriver.Chrome.__class__):
    """
    This takes the driver which has been already attached to ArXiv vs. SnarXiV and returns the two paper
    titles as web-elements.
    :param driver: WebDriver
    :return: WebElement, WebElement
    """
    WebDriverWait(driver, 20).until(element_has_text((By.CLASS_NAME, 'corner-0')))
    lhs = driver.find_element_by_class_name('corner-0')
    rhs = driver.find_element_by_class_name('corner-1')
    return lhs, rhs


class Strategy(ABC):
    """
    This is the generic strategy class. It implements the play function, and requires an optimize function to work.
    """
    class TextObject:
        def __init__(self, text: str):
            self.text = text

    def __init__(self):
        super(Strategy, self).__init__()

    @abstractmethod
    def optimize(self, lhs, rhs):
        """
        This should take in two web-elements that are of the ArXiV vs SnarXiV question format.
        It returns whichever of them satisfies the strategy that is dictated by this function.

        param WebElement: lhs This is the left hand element.
        param WebElement: rhs This is the right hand element.
        :return: WebElement
        """
        raise NotImplementedError

    def text_opti(self, lhs, rhs):
        """
        A wrapper of the optimize method which works on strings instead of WebElements.

        param str: lhs Left Hand Side
        param str: rhs Right Hand Side
        :return: int 0 for left, 1 for right.
        """
        best = self.optimize(self.TextObject(lhs), self.TextObject(rhs))
        if best == lhs:
            return 0
        else:
            return 1

    def play(self, driver_type: str = 'Firefox', n=1000):
        driver = start_driver(driver_type=driver_type)
        record = []
        count = 0
        while count < n:
            lhs, rhs = get_options(driver)
            winner = self.optimize(lhs, rhs)
            winner.click()
            response = str(lhs.get_attribute('class'))
            if response[-5:] == 'right':
                record.append(True)
            elif response[-5:] == 'wrong':
                record.append(False)
            count += 1
        driver.close()
        return record

    def play_mp(self, driver_type: str = 'Firefox', n=250, threads=4):
        if mp.cpu_count() < threads:
            print('More threads requested than processors available, are you sure?')
        player = partial(self.play, driver_type=driver_type, n=n)
        pool = mp.Pool(threads)
        records = [pool.apply_async(player) for _ in range(threads)]
        records = [f.get() for f in records]
        full_record = []
        for r in records:
            full_record += r
        return full_record


class Random(Strategy):
    def __init__(self, p=.5):
        super(Random, self).__init__()
        self.p = p

    def optimize(self, lhs, rhs):
        choiceseed = random()
        if choiceseed > self.p:
            return rhs
        elif choiceseed < self.p:
            return lhs
        else:
            return choice([lhs, rhs])


class Shortest(Strategy):
    def __init__(self):
        super(Shortest, self).__init__()

    def optimize(self, lhs, rhs):
        if len(lhs.text) < len(rhs.text):
            return lhs
        elif len(lhs.text) == len(rhs.text):
            return choice([lhs, rhs])
        else:
            return rhs


class Longest(Strategy):
    def __init__(self):
        super(Longest, self).__init__()

    def optimize(self, lhs, rhs):
        if len(lhs.text) > len(rhs.text):
            return lhs
        elif len(lhs.text) == len(rhs.text):
            return choice([lhs, rhs])
        else:
            return rhs


def add_to_dataset(filename='avs.csv'):
    dr = start_driver()
    dr.minimize_window()
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(['Right', 'Wrong'])
    lhs, rhs = get_options(dr)
    ltext = copy(lhs.text)
    rtext = copy(rhs.text)
    lhs.click()
    response = str(lhs.get_attribute('class'))
    with open(filename, 'a') as file:
        writer = csv.writer(file, delimiter=',')
        if response[-5:] == 'right':
            writer.writerow([ltext, rtext])
            dr.close()
            return [ltext, rtext]
        elif response[-5:] == 'wrong':
            writer.writerow([rtext, ltext])
            dr.close()
            return [rtext, ltext]


def make_data(n, fname='avs.csv'):
    p = mp.Pool()
    _ = p.map(add_to_dataset, tqdm([fname] * n), chunksize=1)


def main():
    make_data(10000)


if __name__ == '__main__':
    main()
