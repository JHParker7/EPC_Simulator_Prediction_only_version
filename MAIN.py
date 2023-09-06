import pandas as pd
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import metrics
import warnings
import pickle

warnings.filterwarnings("ignore")

def replacer(df, x):
        try:
            dictdf = pd.read_csv("Dict/" + x + ".csv")
            y = dictdf["0"].tolist()
        except:
            pd.DataFrame([[]]).to_csv("Dict/" + x + ".csv")
            y = []
        Q = df[x].unique()
        for xq in Q:
            if xq not in y:
                y.append(xq)
        pd.Series(y).to_csv("Dict/" + x + ".csv")
        for x1 in range(len(y)):
            df[x] = df[x].replace(y[x1], x1)
            df[x] = [x if not isinstance(x, str) else -1 for x in df[x].tolist()]
        return df

def replacerwall(df):
        y = df["walls-description"].tolist()
        list2 = []
        insulation = []
        wall_type = []
        for x in y:
            if isinstance(x, float):
                wall_type.append(np.nan)
                insulation.append(np.nan)
                list2.append(-1)
            else:
                x1 = x.split(" ")
                if x1[0] == "average" or x1[0] == "Average":
                    i1 = -1
                    for i in range(len(x1)):
                        try:
                            i1 = float(x1[i])
                        except:
                            pass
                    list2.append(i1)
                else:
                    list2.append(np.nan)
                x1 = x.split(",")
                if len(x1) == 1:
                    wall_type.append(np.nan)
                    insulation.append(np.nan)
                    pass
                else:
                    wall_type.append(x1[0])
                    insulation.append(x1[-1])
        df["insulation-wall"] = insulation
        df["wall type"] = wall_type
        df["Average thermal transmittance-wall"] = list2
        return df

def replacerroof(df):
        y = df["roof-description"].tolist()
        list2 = []
        insulation = []
        wall_type = []
        for x in y:
            if not isinstance(x, float) and x == "(another dwelling above)":
                x1 = x.split(" ")
                if x1[0] == "average" or x1[0] == "Average":
                    i1 = -1
                    for i in range(len(x1)):
                        try:
                            i1 = float(x1[i])
                        except:
                            pass
                    list2.append(i1)
                else:
                    list2.append(-1)
                x1 = x.split(",")
                if len(x1) == 1:
                    wall_type.append(-1)
                    insulation.append(-1)
                    pass
                else:
                    wall_type.append(x1[0])
                    insulation.append(x1[-1])
            else:
                wall_type.append(-1)
                list2.append(-1)
                insulation.append(-1)
        df["insulation-roof"] = insulation
        df["roof type"] = wall_type
        df["Average thermal transmittance-roof"] = list2
        return df

def preproccesing(df):
    df = df[
        [
            "property-type",
            "built-form",
            "local-authority",
            "transaction-type",
            "total-floor-area",
            "energy-tariff",
            "mains-gas-flag",
            "floor-level",
            "flat-top-storey",
            "multi-glaze-proportion",
            "glazed-type",
            "glazed-area",
            "extension-count",
            "number-habitable-rooms",
            "number-heated-rooms",
            "low-energy-lighting",
            "number-open-fireplaces",
            "hotwater-description",
            "floor-description",
            "windows-description",
            "walls-description",
            "roof-description",
            "mainheat-description",
            "mainheatcont-description",
            "main-fuel",
            "heat-loss-corridor",
            "unheated-corridor-length",
            "floor-height",
            "solar-water-heating-flag",
            "mechanical-ventilation",
            "construction-age-band",
            "fixed-lighting-outlets-count",
            "low-energy-fixed-light-count",
            "tenure",
            "secondheat-description",
        ]
    ]
    df = df.rename(columns={"Unnamed: 0": "temp_index"})
    df = df.reset_index(drop=True)
    df = (
        df.replace("Very Poor", 1)
        .replace("Poor", 2)
        .replace("Average", 3)
        .replace("Good", 4)
        .replace("Very Good", 5)
        .replace("G", 1)
        .replace("F", 2)
        .replace("E", 3)
        .replace("D", 4)
        .replace("col", 5)
        .replace("B", 6)
        .replace("A", 7)
    )
    df["property-type"] = (
        df["property-type"]
        .replace("Flat", 1)
        .replace("House", 2)
        .replace("Bungalow", 3)
        .replace("Maisonette", 4)
        .replace("Park home", 5)
    )
    df = df.replace(["NO DATA!", "INVALID!", "", "NODATA!"], np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)

    df["flat-top-storey"] = df["flat-top-storey"].replace("N", 0).replace("Y", 1)
    df = replacer(df, "floor-level")
    df = replacer(df, "built-form")
    df = replacer(df, "transaction-type")
    df = replacer(df, "energy-tariff")
    df = replacer(df, "mains-gas-flag")
    df = replacer(df, "glazed-type")
    df = replacer(df, "glazed-area")
    df = replacer(df, "windows-description")
    df = replacer(df, "solar-water-heating-flag")
    df = replacer(df, "mechanical-ventilation")
    df = replacer(df, "construction-age-band")
    df = replacer(df, "tenure")
    df = replacerwall(df)
    df = replacerroof(df)
    df = replacer(df, "insulation-wall")
    df = replacer(df, "insulation-roof")
    df = replacer(df, "wall type")
    df = replacer(df, "roof type")
    df = replacer(df, "hotwater-description")
    df = replacer(df, "secondheat-description")
    df = replacer(df, "mainheat-description")
    df = replacer(df, "mainheatcont-description")
    df = replacer(df, "main-fuel")
    df = replacer(df, "heat-loss-corridor")
    df = replacer(df, "floor-description")
    df["local-authority"] = [
        x.split("E")[-1].split("W")[-1] if isinstance(x, str) else x
        for x in df["local-authority"].tolist()
    ]
    df["local-authority"] = df["local-authority"].astype(float)
    df = df.replace("NO DATA!", np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[
        [
            "property-type",
            "built-form",
            "local-authority",
            "transaction-type",
            "total-floor-area",
            "energy-tariff",
            "mains-gas-flag",
            "floor-level",
            "flat-top-storey",
            "multi-glaze-proportion",
            "glazed-type",
            "glazed-area",
            "extension-count",
            "number-habitable-rooms",
            "number-heated-rooms",
            "low-energy-lighting",
            "number-open-fireplaces",
            "hotwater-description",
            "floor-description",
            "windows-description",
            "mainheat-description",
            "mainheatcont-description",
            "main-fuel",
            "heat-loss-corridor",
            "unheated-corridor-length",
            "floor-height",
            "solar-water-heating-flag",
            "mechanical-ventilation",
            "construction-age-band",
            "fixed-lighting-outlets-count",
            "low-energy-fixed-light-count",
            "tenure",
            "secondheat-description",
            "insulation-wall",
            "wall type",
            "Average thermal transmittance-wall",
            "insulation-roof",
            "roof type",
            "Average thermal transmittance-roof",
        ]
    ]
    warnings.filterwarnings("ignore")
    for x in df.columns:
        df[x] = df[x].astype("float32")
    df = pd.DataFrame(df)
    df.to_csv("temp.csv")
    df = pd.read_csv("temp.csv")
    df = df.drop(df.columns.tolist()[0], axis=1)
    df.columns = [
        "property-type",
        "built-form",
        "local-authority",
        "transaction-type",
        "total-floor-area",
        "energy-tariff",
        "mains-gas-flag",
        "floor-level",
        "flat-top-storey",
        "multi-glaze-proportion",
        "glazed-type",
        "glazed-area",
        "extension-count",
        "number-habitable-rooms",
        "number-heated-rooms",
        "low-energy-lighting",
        "number-open-fireplaces",
        "hotwater-description",
        "floor-description",
        "windows-description",
        "mainheat-description",
        "mainheatcont-description",
        "main-fuel",
        "heat-loss-corridor",
        "unheated-corridor-length",
        "floor-height",
        "solar-water-heating-flag",
        "mechanical-ventilation",
        "construction-age-band",
        "fixed-lighting-outlets-count",
        "low-energy-fixed-light-count",
        "tenure",
        "secondheat-description",
        "insulation-wall",
        "wall type",
        "Average thermal transmittance-wall",
        "insulation-roof",
        "roof type",
        "Average thermal transmittance-roof",
    ]
    df = df.astype("float32").replace([np.inf, -np.inf], np.nan)

    ((df.min()) - 1).to_csv("DICTMIN.csv")

    temp = pd.read_csv("DICTMIN.csv").set_index("Unnamed: 0")
    temp = temp.T
    for col in df.columns:
        df[col] = df[col].fillna(temp[col].to_list()[0])
    return df


def X_Y(df):
    if df.columns.to_list()[0] == "Unnamed: 0":
        df = df.drop("Unnamed: 0", axis=1)
    if df.columns.to_list()[-1] == "UPRN_SOURCE":
        df.columns = [s.lower().replace("_", "-") for s in df.columns.to_list()]
    df = preproccesing(df)
    #  df=df.dropna()
    # df=df.drop_duplicates(subset=['property-type', 'built-form', 'transaction-type', 'total-floor-area', 'energy-tariff', 'mains-gas-flag', 'floor-level', 'flat-top-storey', 'multi-glaze-proportion', 'glazed-type', 'glazed-area', 'extension-count', 'number-habitable-rooms', 'number-heated-rooms', 'low-energy-lighting', 'number-open-fireplaces', 'hotwater-description', 'floor-description', 'windows-description', 'mainheat-description', 'mainheatcont-description', 'main-fuel', 'heat-loss-corridor', 'unheated-corridor-length', 'floor-height', 'solar-water-heating-flag', 'mechanical-ventilation', 'construction-age-band', 'fixed-lighting-outlets-count', 'low-energy-fixed-light-count', 'tenure', 'secondheat-description', 'insulation-wall', 'wall type', 'Average thermal transmittance-wall', 'insulation-roof', 'roof type', 'Average thermal transmittance-roof'],keep="first")
    print(df.columns.to_list())
    with open("SCALEX.pkl", "rb") as file:
        scale_x = pickle.load(file)
    df = scale_x.transform(df)
    return df


def CORE(df):
    df = X_Y(df)
    print(df)
    with open("SCALEY.pkl", "rb") as file:
        scale_y = pickle.load(file)
    model = tf.keras.models.load_model("MODEL.h5")
    prediction = model.predict(df)
    print(prediction)
    prediction = scale_y.inverse_transform(prediction)
    prediction = pd.DataFrame(prediction)
    return prediction


def TEST(df, Y):
    prediction = CORE(df)
    print(Y)
    print(prediction)
    for y in [
        "HEATING_COST_CURRENT",
        "HOT_WATER_COST_CURRENT",
        "ENVIRONMENT_IMPACT_CURRENT",
        "ENERGY_CONSUMPTION_CURRENT",
        "CO2_EMISSIONS_CURRENT",
        "LIGHTING_COST_CURRENT",
    ]:
        print(y)
        print("MAPE:", metrics.mean_absolute_percentage_error(Y[y], prediction[y]))
        print("MSE:", metrics.mean_squared_error(Y[y], prediction[y]))
        print("MAE:", metrics.mean_absolute_error(Y[y], prediction[y]))
        print(" ")


df = pd.read_csv("DATA.csv")
df = df.reset_index(drop=True)
# Y=df[["HEATING_COST_CURRENT","HOT_WATER_COST_CURRENT","ENVIRONMENT_IMPACT_CURRENT","ENERGY_CONSUMPTION_CURRENT","CO2_EMISSIONS_CURRENT","LIGHTING_COST_CURRENT"]]
# TEST(df,Y)
df[
    [
        "PRE_HEATING_COST_CURRENT",
        "PRE_HOT_WATER_COST_CURRENT",
        "PRE_ENVIRONMENT_IMPACT_CURRENT",
        "PRE_ENERGY_CONSUMPTION_CURRENT",
        "PRE_CO2_EMISSIONS_CURRENT",
        "PRE_LIGHTING_COST_CURRENT",
    ]
] = CORE(df)
df.to_csv("OUTPUT.csv")
print(df)