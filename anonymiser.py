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

    # convert col Sex to numeric -> 0 for M and 1 for F
    df['Sex'] = df['Sex'].map({'M': 0, 'F': 1})

    # Set quasi-identifiers
    quasi = ["Age","Sex","ZIP"]

    # set Diagnosis as category
    df["Diagnosis"] = df["Diagnosis"].astype("category")
            
    # # alternative more readable generaliser humanreadable.HumanReadable
    # anonymiser = customAnonymiser(df, k=5, quasi_identifiers= quasi, generalisation_strategy=microaggregation.Microaggregation)
    # anonymised_df= anonymiser.anonymise()
    # print(anonymiser.algorithm.clusters)
    # print(anonymised_df.head(5))
    # anonymised_df.to_csv("anonymized_df_final.csv", sep=',')
    
    # LOUIS: did something to try diff K values and print some SSE
    # not sure about the calculation of SSE tho
    sse_results = []
    for k in range(1, 9):
        anonymiser = customAnonymiser(df, k=k, quasi_identifiers=quasi, generalisation_strategy=microaggregation.Microaggregation)
        anonymised_df = anonymiser.anonymise()

        # Compute SSE on numeric quasi-identifiers only
        numeric_quasi = df[quasi].columns
        diff = df[numeric_quasi] - anonymised_df[numeric_quasi]

        sse = (diff ** 2).sum().sum()
        sse = round(sse, 1)
        sse_results.append((k, sse))


    for k, sse in sse_results:
        print(f"k={k}, SSE={sse}")
    