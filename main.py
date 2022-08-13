# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
from analytics import jaccard_score_dataframe

pd.options.display.width = 0
df = pd.read_csv("C://Users/SDietzel/Documents/EinfachEAuto/Website/API/wallboxassistent.csv", sep="\t", index_col=0)

needed_values = [
    "ausseneinsatz",
    "key",
    "rfid",
    "app_steuerung",
    "mid_or_eich",
    "ladeleistung_11kw",
    "ladeleistung_22kw",
    "umts",
    "lan",
    "wlan",
    "foerderung",
    "dcfs_integrateded",
    "cable_length_no_cable",
    "cable_length_4m",
    "cable_length_6m",
    "cable_length_7m_plus",
]
# all_values = [x.name for x in self.model._meta.get_fields()] + ["ausseneinsatz", "key", "rfid"]
all_values = ["id"] + needed_values

user_input = np.array(
    (
        1, # ausseneinsatz,
        0, # key,
        0, # rfid,
        1, # app_steuerung,
        0, # mid_eich_zaehler,
        1, # ladeleistung_11kw,
        0, # ladeleistung_22kw,
        0, # umts,
        0, # lan,
        0, # wlan,
        0, # foerderung,
        1, # dcfs_integrateded,
        0, # cable 0
        0, # cable 4
        0, # cable 6
        1, # cable 7
    )
).astype(int)

score = jaccard_score_dataframe(
    user_input,
    df,
    include=needed_values,
    # construct the weights according to the user_input array/vector
    weights=[2, 2, 2, 2, 5, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1],
    algorithm="jaccard",
    drop_na_input=True,
)
score1 = jaccard_score_dataframe(
    user_input,
    df,
    include=needed_values,
    # construct the weights according to the user_input array/vector
    weights=[2, 2, 2, 2, 5, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1],
    algorithm="jaccard",
    drop_na_input=False,
)

sorted_score = score.loc["user input"].sort_values(ascending=False)
sorted_score_1 = score1.loc["user input"].sort_values(ascending=False)

print("cosing drop Na", sorted_score)
print("cosing drop not Na",sorted_score_1)

