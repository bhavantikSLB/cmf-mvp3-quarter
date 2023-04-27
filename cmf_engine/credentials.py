# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:52:43 2023

@author: BGondaliya
"""

import os
from configparser import ConfigParser

import mysql.connector
import sqlalchemy as sql


class credentials:
    def __init__(self, path):
        self.path = path

    def read_cred(self):
        parser = ConfigParser()  ## use??
        _ = parser.read(os.path.join(self.path, "credentials.cfg"))

        POND_HOST = parser.get("pond", "CMF_POND_SERVER")
        POND_USER = parser.get("pond", "CMF_POND_USER")
        POND_PASSWORD = parser.get("pond", "CMF_POND_PASSWORD")

        uri_ods = "mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(
            POND_USER, POND_PASSWORD, POND_HOST, "3307", "ODS"
        )
        # uri_6438 = "mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(POND_USER,POND_PASSWORD,POND_HOST,"3307","EAR-AA-6438")

        cxn_ODS = sql.create_engine(uri_ods)
        # cxn_6438 = sql.create_engine(uri_6438)
        cxn_6438 = mysql.connector.connect(
            host=POND_HOST,
            user=POND_USER,
            password=POND_PASSWORD,
            port="3307",
            database="EAR-AA-6438",
        )

        return cxn_ODS, cxn_6438
