# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from analytics import similarity_matrix_continuous_categorical


pd.options.display.width = None
pd.options.display.max_columns = None

df = pd.read_csv("cars_full.csv", sep="\t", index_col=0)
additionals = pd.read_csv("carassistent.csv", sep="\t", index_col=0)
modelles = pd.read_csv("modells.csv", sep="\t", index_col=0)
prices = pd.read_csv("carprices.csv", sep="\t")
prices.index = prices["slug"]

df = df.merge(additionals, how="inner", left_on=["slug"], right_index=True, suffixes=("", '_right')).set_index('slug')
df = df.merge(modelles, how="inner", left_on=["slug"], right_index=True, suffixes=("", '_right')).set_index('slug')
df = df.merge(prices, how="inner", left_on=["slug"], right_index=True, suffixes=("_left", '')).set_index('slug')
# print(df["minpreis"])
# df.drop(["Honda-E-Standard", "Hyundai-Ioniq-Elektro", "Peugeot-e-2008-standard", "Skoda-Citigo-e-iV-Standard"],
#         inplace=True)
#
# # df.drop(columns=['minpreis', 'class_reichweite', 'reichweite_epa'], axis=1)
#
# continuous = [
#     "release_year",
#     "release_month",
#     "basispreis",
#     "reichweite_epa",
#     "v_max",
#     "power_ps",
#     "lieferzeit",
#     "dclass_max",
#     "aclass_real",
#     "beschleunigung",
#     # "batteriegroesse_netto",
#     "batteriegroesse_brutto",
#     # "time_to_200km",
#     #   "consumption_wltp_official",
#     #  "consumption_nefz_official",
#     "minpreis",
#     "consumption_wltp_calulation",
#     "modelle__hoehe",
#     "modelle__breite",
#     "modelle__laenge",
#     "modelle__kofferraumvolumen_back",
#     "modelle__kofferraumvolumen_front"
# ]
#
# categorical = [
#     "langstrecke",
#     "antriebsart",
#     "heat_pump",
#     "bi_directional_loading",
#     "electrical_system_voltage",
#     "staatliche_foerderung",
#     'class_sitzplaetze_2',
#     'class_sitzplaetze_4',
#     'class_sitzplaetze_5',
#     'class_sitzplaetze_7',
#     'class_stauraum',
#     'class_langstrecke',
#     'class_tag_park_easy',
#     'class_tag_sporty',
#     'class_tag_luxury_high',
#     'class_tag_cargo_high',
#     'class_tag_seat_high',
#     'class_tag_tech_high',
#     'class_form_city',
#     'class_form_small',
#     'class_form_compact',
#     'class_form_suv',
#     'class_form_limousine',
#     'class_form_combi',
#     'class_form_convertible',
#     'class_form_sports_car',
#     'class_form_van'
# ]
#
# needed_values = continuous + categorical
#
# score2 = similarity_matrix_continuous_categorical(
#     dataset=df,
#     continuous=continuous,
#     # construct the weights according to the user_input array/vector
#     categorical=categorical,
#     algorithm="euclidean"
# )
#
# # sorted_score = score.loc["Opel-Corsa-E-Standard"].sort_values(ascending=False)
# # sorted_score_1 = score1.loc["Opel-Corsa-E-Standard"].sort_values(ascending=False)
# #
# # print("cosing drop Na", sorted_score)
# # print("cosing drop not Na", sorted_score_1)
#
# print(score2.loc["Mini-Cooper-SE"].sort_values()[1:4])
# print(score2.loc["50-quattro-54"].sort_values()[1:4])
# print(score2.loc["bmw-i4-edrive40"].sort_values()[1:4])
# print(score2.loc["audi-e-tron-gt-quattro"].sort_values()[1:4])
# print(score2.loc["polestar-2-standard-range-single-motor"].sort_values()[1:4])

# df.drop(columns=['minpreis', 'class_reichweite', 'reichweite_epa'], axis=1)

user_input = {
    'class_sitzplaetze_2': 0,
    'class_sitzplaetze_4': 0,
    'class_sitzplaetze_5': 0,
    'class_sitzplaetze_7': 1,
    "reichweite_epa": 350,
    "minpreis": 1500,
    "class_minpreis": 1,
    "class_reichweite_epa": 1,
    "form": ["suv"],
    "langstrecke": 3,
    "class_langstrecke": 1,
    "antriebsart": "Allrad",
    "heat_pump": True,
    "bi_directional_loading": True,
    "electrical_system_voltage": True,
}

categorical = [
    "antriebsart",
    "heat_pump",
    "bi_directional_loading",
    "electrical_system_voltage",
    'class_sitzplaetze_2',
    'class_sitzplaetze_4',
    'class_sitzplaetze_5',
    'class_sitzplaetze_7',
    'class_stauraum',
    'form',
    'class_reichweite_epa',
    'class_minpreis',
    "class_langstrecke",
]

continuous = [
]
user_input_df = pd.DataFrame(user_input, index=["user input"])
df = pd.concat([df, user_input_df])

print(df.iloc[-100:-50]["form"])
df["class_langstrecke"] = df["langstrecke"] >= user_input["langstrecke"]
df["class_reichweite_epa"] = df["reichweite_epa"] >= user_input["reichweite_epa"]
df["class_minpreis"] = df["minpreis"] <= user_input["minpreis"]

print(df.applymap(type)["form"])


weights = {
    "class_reichweite_epa": 1,
    "class_minpreis": 3,
    "class_sitzplaetze_7": 10,
}

score1 = similarity_matrix_continuous_categorical(
    dataset=df,
    continuous=continuous,
    categorical=categorical,
    algorithm="cosine",
    weights=weights,
)

up = score1.loc["user input"].sort_values()

# print(df.loc["vw-id3-pro", "minpreis"])
# print(df.loc["hyundai-ioniq-5-58-kwh-2wd", "minpreis"])
# print(df.loc["vw-id5-gtx", "minpreis"])


df = df.reindex(up.index)
print(up[:20])

print(df.loc[:, ["reichweite_epa", "minpreis", "class_minpreis", "antriebsart", "form", "class_sitzplaetze_7"]][:20])