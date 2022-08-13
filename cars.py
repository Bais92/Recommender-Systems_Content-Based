# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
from analytics import jaccard_score_dataframe

pd.options.display.width = 0
df = pd.read_csv("C://Users/SDietzel/Documents/EinfachEAuto/Website/API/carassistent.csv", sep="\t", index_col=0)
prices = pd.read_csv("C://Users/SDietzel/Documents/EinfachEAuto/Website/API/carprices.csv", sep="\t", index_col=0)

df = df.reset_index().merge(prices, how="inner", on="minpreis").set_index('index')

needed_values = [
    'class_sitzplaetze_2',
    'class_sitzplaetze_4',
    'class_sitzplaetze_5',
    'class_sitzplaetze_7',
    'class_stauraum',
    'class_langstrecke',
    'class_tag_park_easy',
    'class_tag_sporty',
    'class_tag_luxury_high',
    'class_tag_cargo_high',
    'class_tag_seat_high',
    'class_tag_tech_high',
    'minpreis',
    'class_reichweite',
    'reichweite_epa',
    'class_form_city',
    'class_form_small',
    'class_form_compact',
    'class_form_suv',
    'class_form_limousine',
    'class_form_combi',
    'class_form_convertible',
    'class_form_sports_car',
    'class_form_van'
]

user_input = np.array(
    (
        0,  # 'class_sitzplaetze_2',
        1,  # 'class_sitzplaetze_4',
        0,  # 'class_sitzplaetze_5',
        0,  # 'class_sitzplaetze_7',
        0,  # 'class_stauraum',
        0,  # 'class_langstrecke',
        0,  # 'class_tag_park_easy',
        0,  # 'class_tag_sporty',
        0,  # 'class_tag_luxury_high',
        0,  # 'class_tag_cargo_high',
        0,  # 'class_tag_seat_high',
        0,  # 'class_tag_tech_high',
        300,  # 'minpreis',
        0,  # 'class_reichweite',
        0,  # 'reichweite_epa',
        1,  # 'class_form_city',
        0,  # 'class_form_small',
        0,  # 'class_form_compact',
        0,  # 'class_form_suv',
        0,  # 'class_form_limousine',
        0,  # 'class_form_combi',
        0,  # 'class_form_convertible',
        0,  # 'class_form_sports_car',
        0,  # 'class_form_van'
    )
).astype(int)

score = jaccard_score_dataframe(
    user_input,
    df,
    include=needed_values,
    # construct the weights according to the user_input array/vector
    algorithm="jaccard",
    drop_na_input=True,
)
score1 = jaccard_score_dataframe(
    user_input,
    df,
    include=needed_values,
    # construct the weights according to the user_input array/vector
    algorithm="jaccard",
    drop_na_input=False,
)

sorted_score = score.loc["user input"].sort_values(ascending=False)
sorted_score_1 = score1.loc["user input"].sort_values(ascending=False)

print("cosing drop Na", sorted_score)
print("cosing drop not Na",sorted_score_1)

