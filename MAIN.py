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

def preproccesing(Data):
    Data = Data[
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
    Data = Data.rename(columns={"Unnamed: 0": "temp_index"})
    Data = Data.reset_index(drop=True)
    Data = (
        Data.replace("Very Poor", 1)
        .replace("Poor", 2)
        .replace("Average", 3)
        .replace("Good", 4)
        .replace("Very Good", 5)
        .replace("G", 1)
        .replace("F", 2)
        .replace("E", 3)
        .replace("D", 4)
        .replace("C", 5)
        .replace("B", 6)
        .replace("A", 7)
    )
    Data["property-type"] = (
        Data["property-type"]
        .replace("Flat", 1)
        .replace("House", 2)
        .replace("Bungalow", 3)
        .replace("Maisonette", 4)
        .replace("Park home", 5)
    )
    Data = Data.replace(["NO DATA!", "INVALID!", "", "NODATA!"], np.nan)
    Data = Data.replace([np.inf, -np.inf], np.nan)

    Data["flat-top-storey"] = Data["flat-top-storey"].replace("N", 0).replace("Y", 1)
    Data = replacer(Data, "floor-level")
    Data = replacer(Data, "built-form")
    Data = replacer(Data, "transaction-type")
    Data = replacer(Data, "energy-tariff")
    Data = replacer(Data, "mains-gas-flag")
    Data = replacer(Data, "glazed-type")
    Data = replacer(Data, "glazed-area")
    Data = replacer(Data, "windows-description")
    Data = replacer(Data, "solar-water-heating-flag")
    Data = replacer(Data, "mechanical-ventilation")
    Data = replacer(Data, "construction-age-band")
    Data = replacer(Data, "tenure")
    Data = replacerwall(Data)
    Data = replacerroof(Data)
    Data = replacer(Data, "insulation-wall")
    Data = replacer(Data, "insulation-roof")
    Data = replacer(Data, "wall type")
    Data = replacer(Data, "roof type")
    Data = replacer(Data, "hotwater-description")
    Data = replacer(Data, "secondheat-description")
    Data = replacer(Data, "mainheat-description")
    Data = replacer(Data, "mainheatcont-description")
    Data = replacer(Data, "main-fuel")
    Data = replacer(Data, "heat-loss-corridor")
    Data = replacer(Data, "floor-description")
    Data["local-authority"] = [
        x.split("E")[-1].split("W")[-1] if isinstance(x, str) else x
        for x in Data["local-authority"].tolist()
    ]
    Data["local-authority"] = Data["local-authority"].astype(float)
    Data = Data.replace("NO DATA!", np.nan)
    Data = Data.replace([np.inf, -np.inf], np.nan)
    Data = Data[
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
    for x in Data.columns:
        Data[x] = Data[x].astype("float32")
    Data = pd.DataFrame(Data)
    Data.to_csv("temp.csv")
    Data = pd.read_csv("temp.csv")
    Data = Data.drop(Data.columns.tolist()[0], axis=1)
    Data.columns = [
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
    Data = Data.astype("float32").replace([np.inf, -np.inf], np.nan)

    ((Data.min()) - 1).to_csv("DICTMIN.csv")

    TEMP = pd.read_csv("DICTMIN.csv").set_index("Unnamed: 0")
    TEMP = TEMP.T
    for C in Data.columns:
        Data[C] = Data[C].fillna(TEMP[C].to_list()[0])
    return Data


def X_Y(Data):
    if Data.columns.to_list()[0] == "Unnamed: 0":
        Data = Data.drop("Unnamed: 0", axis=1)
    if Data.columns.to_list()[-1] == "UPRN_SOURCE":
        Data.columns = [s.lower().replace("_", "-") for s in Data.columns.to_list()]
    Data = preproccesing(Data)
    #  Data=Data.dropna()
    # Data=Data.drop_duplicates(subset=['property-type', 'built-form', 'transaction-type', 'total-floor-area', 'energy-tariff', 'mains-gas-flag', 'floor-level', 'flat-top-storey', 'multi-glaze-proportion', 'glazed-type', 'glazed-area', 'extension-count', 'number-habitable-rooms', 'number-heated-rooms', 'low-energy-lighting', 'number-open-fireplaces', 'hotwater-description', 'floor-description', 'windows-description', 'mainheat-description', 'mainheatcont-description', 'main-fuel', 'heat-loss-corridor', 'unheated-corridor-length', 'floor-height', 'solar-water-heating-flag', 'mechanical-ventilation', 'construction-age-band', 'fixed-lighting-outlets-count', 'low-energy-fixed-light-count', 'tenure', 'secondheat-description', 'insulation-wall', 'wall type', 'Average thermal transmittance-wall', 'insulation-roof', 'roof type', 'Average thermal transmittance-roof'],keep="first")
    print(Data.columns.to_list())
    with open("SCALEX.pkl", "rb") as file:
        SCALEX = pickle.load(file)
    Data = SCALEX.transform(Data)
    return Data


def CORE(Data):
    Data = X_Y(Data)
    print(Data)
    with open("SCALEY.pkl", "rb") as file:
        SCALEY = pickle.load(file)
    model = tf.keras.models.load_model("MODEL.h5")
    PRE = model.predict(Data)
    print(PRE)
    PRE = SCALEY.inverse_transform(PRE)
    PRE = pd.DataFrame(PRE)
    return PRE


def TEST(Data, Y):
    PRE = CORE(Data)
    print(Y)
    print(PRE)
    for y in [
        "HEATING_COST_CURRENT",
        "HOT_WATER_COST_CURRENT",
        "ENVIRONMENT_IMPACT_CURRENT",
        "ENERGY_CONSUMPTION_CURRENT",
        "CO2_EMISSIONS_CURRENT",
        "LIGHTING_COST_CURRENT",
    ]:
        print(y)
        print("MAPE:", metrics.mean_absolute_percentage_error(Y[y], PRE[y]))
        print("MSE:", metrics.mean_squared_error(Y[y], PRE[y]))
        print("MAE:", metrics.mean_absolute_error(Y[y], PRE[y]))
        print(" ")


DF = pd.read_csv("DATA.csv")
DF = DF.reset_index(drop=True)
# Y=DF[["HEATING_COST_CURRENT","HOT_WATER_COST_CURRENT","ENVIRONMENT_IMPACT_CURRENT","ENERGY_CONSUMPTION_CURRENT","CO2_EMISSIONS_CURRENT","LIGHTING_COST_CURRENT"]]
# TEST(DF,Y)
DF[
    [
        "PRE_HEATING_COST_CURRENT",
        "PRE_HOT_WATER_COST_CURRENT",
        "PRE_ENVIRONMENT_IMPACT_CURRENT",
        "PRE_ENERGY_CONSUMPTION_CURRENT",
        "PRE_CO2_EMISSIONS_CURRENT",
        "PRE_LIGHTING_COST_CURRENT",
    ]
] = CORE(DF)
DF.to_csv("OUTPUT.csv")
print(DF)
