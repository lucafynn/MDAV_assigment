from anonypyx.generalisation.schema import GeneralisedSchema, build_column_groups

class Microaggregation(GeneralisedSchema):
    '''
    Generalisation schema applying microaggregation.
    Integer quasi-identifiers are generalised by computing the mean over
    each partition.
    Categorical quasi-identifiers are not supported for the time being.
    '''
    @classmethod
    def create_for_data(cls, df, quasi_identifiers):
        _, integer, unaltered = build_column_groups(df, quasi_identifiers)
        return Microaggregation(integer, unaltered)

    @classmethod
    def from_json_dict(cls, json_dict):
        return Microaggregation(json_dict['integer'], json_dict['quasi_identifier'])

    def __init__(self, integer, unaltered):
        '''
        Constructor.

        Parameters
        ----------
        integer : list of str
            List of column names which are integer quasi-identifiers.
        unaltered : list of str
            List of column names which are not quasi-identifiers.
        '''
        super().__init__(unaltered)
        self._integer = integer

    def to_json_dict(self):
        return {'integer': self._integer, 'unaltered': self._unaltered}

    def quasi_identifier(self):
        return self._integer

    def _generalise_partition(self, df):
        row = []
        for col in self._integer:
            row.append(df[col].astype(float).mean())  # Convert to float before computing mean
        return row
