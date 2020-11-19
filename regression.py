import requests
import pandas as pd
import scipy
import numpy 
import sys

TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    s = response.content.decode('utf-8').splitlines()
    data = {"area":list(map(float,s[0].split(',')[1:])),"price":list(map(float,s[1].split(',')[1:]))}
    w = 0
    b = 0
    am = numpy.mean(data["area"])
    pm = numpy.mean(data["price"])
    w = [0,0]
    for a,p in zip(data["area"],data["price"]):
        w[0]+= (a-am)*(p-pm)
        w[1]+= (a-am)**2
    w1 = w[0]/w[1]
    b = pm - (w1*am)
    print(w1,b)
    hyp =[ w1*a + b for a in area]
    return hyp


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
