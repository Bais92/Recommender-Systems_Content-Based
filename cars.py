# This is a sample Python script.

import numpy as np
# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd

from analytics import similarity_score_dataframe, similarity_matrix_continuous_categorical

pd.options.display.width = 0
df = pd.read_csv("cars_full.csv", sep="\t", index_col=0)
additionals = pd.read_csv("carassistent.csv", sep="\t", index_col=0)

df = df.merge(additionals, how="left", left_on=["slug"], right_index=True, suffixes=("", '_right')).set_index('slug')
df.drop(["Honda-E-Standard", "Hyundai-Ioniq-Elektro","Peugeot-e-2008-standard","Skoda-Citigo-e-iV-Standard"], inplace=True)

# df.drop(columns=['minpreis', 'class_reichweite', 'reichweite_epa'], axis=1)

continuous = [
    "release_year",
    "release_month",
    "basispreis",
    "reichweite_epa",
    "v_max",
    "power_ps",
    "lieferzeit",
    "dc_max",
    "ac_real",
    "beschleunigung",
    #"batteriegroesse_netto",
    "batteriegroesse_brutto",
    #"time_to_200km",
 #   "consumption_wltp_official",
  #  "consumption_nefz_official",
    "consumption_wltp_calulation"
]

categorical = [
    "langstrecke",
    "antriebsart",
    "heat_pump",
    "bi_directional_loading",
    "electrical_system_voltage",
    "staatliche_foerderung",
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

needed_values = continuous + categorical




score2 = similarity_matrix_continuous_categorical(
    dataset=df,
    continuous=continuous,
    # construct the weights according to the user_input array/vector
    categorical=categorical,
    algorithm_continuous="cosine",
    algorithm_categorical="cosine"
)

# sorted_score = score.loc["Opel-Corsa-E-Standard"].sort_values(ascending=False)
# sorted_score_1 = score1.loc["Opel-Corsa-E-Standard"].sort_values(ascending=False)
#
# print("cosing drop Na", sorted_score)
# print("cosing drop not Na", sorted_score_1)

sorted_score_2 = score2.loc["Mini-Cooper-SE"].sort_values()[:10]
print(score2.loc["Mini-Cooper-SE"].sort_values()[:10])
print(score2.loc["50-quattro-54"].sort_values()[:10])
print(score2.loc["mazda-mx-30"].sort_values()[:10])
print(score2.loc["audi-e-tron-gt-quattro"].sort_values()[:10])


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
       # 300,  # 'minpreis',
        #0,  # 'class_reichweite',
        #0,  # 'reichweite_epa',
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

df = pd.read_csv("carassistent.csv", sep="\t", index_col=0)
prices = pd.read_csv("carprices.csv", sep="\t", index_col=0)

df = df.reset_index().merge(prices, how="inner", on="minpreis").set_index('index')

df.drop(columns=['minpreis', 'class_reichweite', 'reichweite_epa'], axis=1)
print(df)
score1 = similarity_score_dataframe(
    user_input,
    df,
    include=needed_values,
    # construct the weights according to the user_input array/vector
    algorithm="cosine",
    drop_na_input=True,
)

print(score1.loc["user input"].sort_values(ascending=False)[:10])
