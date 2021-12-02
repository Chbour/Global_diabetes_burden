"""
    Utils concerning MongoDB

    Author: Adrian Ahne

    Date: 01.06.2018

"""

import sys
from pymongo import MongoClient


def connect_to_database(host='localhost', port=27017):
    """
        Connect to MongoDB database
    """

    try:
        client = MongoClient(host, port)
    except ConnectionFailure as e:
        sys.stderr.write("Could not connect to MongoDB for given host {} and port {}: {}".format(host, port, e))
        sys.exit(1)

    return client
