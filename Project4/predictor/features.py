def prepare_features(df):
    df = df.copy()
    df = df.dropna()
    df["rooms_per_household"] - df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] - df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] - df["population"] / df["households"]
    x - df.drop("median_house_value", axis-1)
    y - df["median_house_value"]
    return x, y 


