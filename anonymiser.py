import pandas as pd
import humanreadable, MDAVGeneric, microaggregation


class customAnonymiser:
    def __init__(self, df, k, quasi_identifiers, generalisation_strategy):
        self.algorithm = MDAVGeneric.MDAVGeneric(k, quasi_identifiers)
        self.df = df
        self.quasi_identifiers = quasi_identifiers
        self.generalisation_strategy_type = generalisation_strategy

    def anonymise(self):
        partitions = self.algorithm.partition(self.df)
        generalisation = self.generalisation_strategy_type.create_for_data(self.df, self.quasi_identifiers)
        return generalisation.generalise(self.df, partitions)

if __name__ == "__main__":
    df = pd.read_csv("./health_ai_mdav_demo.csv")
    quasi = ["Age","Sex","ZIP"]
    for column in ("Sex", "Diagnosis"):
        df[column] = df[column].astype("category")
            
    # alternative more readable generaliser humanreadable.HumanReadable
    anonymiser = customAnonymiser(df, k=5, quasi_identifiers= quasi, generalisation_strategy=microaggregation.Microaggregation)
    anonymised_df= anonymiser.anonymise()
    print(anonymiser.algorithm.clusters)
    print(anonymised_df.head(5))
    anonymised_df.to_csv("anonymized_df_final.csv", sep=',')
    