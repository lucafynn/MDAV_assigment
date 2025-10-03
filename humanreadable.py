import pandas as pd
import re

from anonypyx.generalisation.schema import GeneralisedSchema, build_column_groups


class HumanReadable(GeneralisedSchema):
    """
    Generalisation schema with improved readability for humans.
    Does not support most operations.
    The generalised values are strings.
    """

    @classmethod
    def create_for_data(cls, df, quasi_identifiers):
        categorical, integer, unaltered = build_column_groups(df, quasi_identifiers)
        return HumanReadable(categorical, integer, unaltered)

    @classmethod
    def from_json_dict(cls, json_dict):
        return HumanReadable(json_dict['categorical'], json_dict['integer'], json_dict['unaltered'])

    def __init__(self, categorical, integer, unaltered):
        """
        Constructor.

        Parameters
        ----------
        categorical : list of str
            List of column names which are categorical quasi-identifiers.
        integer : list of str
            List of column names which are integer quasi-identifiers.
        unaltered : list of str
            List of column names which are not quasi-identifiers.
        """
        super().__init__(unaltered)
        self._categorical = categorical
        self._integer = integer

    def to_json_dict(self):
        return {'categorical': self._categorical, 'integer': self._integer, 'unaltered': self._unaltered}

    def quasi_identifier(self):
        return self._integer + self._categorical

    def select(self, df, query):
        matching = [i for i, row in df.iterrows() if self._matches_query(row, query)]
        return pd.Index(matching)

    def match(self, df, record, on):
        matching_indices = []

        for idx, row in df.iterrows():
            abort = False
            for col in on:
                predicate = None
                if col in self._integer:
                    low, high = self._parse_interval(record[col])
                    predicate = (low, high)
                elif col in self._categorical:
                    predicate = self._parse_set(record[col])
                else:
                    predicate = record[col]

                if not self._is_match(row, predicate, col):
                    abort = True
                    break

            if not abort:
                matching_indices.append(idx)

        return pd.Index(matching_indices)

    def intersect(self, record_a, record_b, on, take_left, take_right):
        result = {}
        for col in on:
            if col in self._integer:
                a_low, a_high = self._parse_interval(record_a[col])
                b_low, b_high = self._parse_interval(record_b[col])
                new_low = max(a_low, b_low)
                new_high = min(a_high, b_high)
                if new_low > new_high:
                    return None
                result[col] = (
                    str(new_low) if new_low == new_high else f"{new_low}-{new_high}"
                )
            elif col in self._categorical:
                set_a = self._parse_set(record_a[col])
                set_b = self._parse_set(record_b[col])
                intersect_set = set_a.intersection(set_b)
                if not intersect_set:
                    return None
                result[col] = ",".join(sorted(intersect_set))
            elif col in self._unaltered:
                if record_a[col] != record_b[col]:
                    return None
                result[col] = record_a[col]
        for col in take_left:
            result[col] = record_a[col]
        for col in take_right:
            result[col] = record_b[col]
        return result

    def values_for(self, record, column):
        if column in self._integer:
            low, high = self._parse_interval(record[column])
            return set(range(low, high + 1))
        elif column in self._categorical:
            return self._parse_set(record[column])
        else:
            return {record[column]} if column in record else set()

    def set_cardinality(self, record, on):
        total = 1
        for col in on:
            total *= len(self.values_for(record, col))
        return total

    def query_overlap(self, record, query):
        total = 1
        for col, pred in query.items():
            if col in self._integer:
                total *= self._integer_overlap_size(record[col], pred)
            elif col in self._categorical:
                total *= self._categorical_overlap_size(record[col], pred)
            else:
                if isinstance(pred, tuple):
                    low, high = pred
                    if not (low <= record[col] <= high):
                        return 0
                elif isinstance(pred, (set, list, frozenset)):
                    if record[col] not in pred:
                        return 0
                else:
                    if record[col] != pred:
                        return 0
        return total

    def _integer_overlap_size(self, record_col_value, query_predicate):
        r_low, r_high = self._parse_interval(record_col_value)
        q_low, q_high = query_predicate
        overlap_low = max(r_low, q_low)
        overlap_high = min(r_high, q_high)
        if overlap_low > overlap_high:
            return 0
        return overlap_high - overlap_low + 1

    def _categorical_overlap_size(self, record_col_value, query_predicate):
        r_set = self._parse_set(record_col_value)
        overlap = r_set.intersection(query_predicate)
        return len(overlap)

    def _generalise_partition(self, df):
        row = []
        for col in self._integer:
            row.append(self._to_string_interval(df[col]))

        for col in self._categorical:
            row.append(self._to_string_set(df[col]))
        return row

    def _to_string_interval(self, series):
        minimum = series.min()
        maximum = series.max()
        if maximum == minimum:
            return str(maximum)
        return f"{minimum}-{maximum}"

    def _to_string_set(self, series):
        l = [str(value) for value in series.unique()]
        l.sort()
        return ",".join(l)

    def _parse_interval(self, interval_str):
        match = re.match(r"^(-?\d+)(?:-(-?\d+))?$", interval_str)
        if not match:
            raise ValueError(f"Invalid interval format: {interval_str}")
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) is not None else start
        return start, end

    def _parse_set(self, s):
        if s == "":
            return set()
        return set(s.split(","))

    def _is_match(self, row, predicate, col):
        cell = row[col]

        if isinstance(predicate, tuple) and len(predicate) == 2:
            low, high = predicate
            if col in self._integer:
                c_low, c_high = self._parse_interval(str(cell))
            else:
                c_low = c_high = float(cell)
            return not (c_high < low or c_low > high)

        if isinstance(predicate, (set, list, frozenset)):
            if col in self._categorical:
                cell_set = self._parse_set(str(cell))
            else:
                cell_set = {cell}
            return not cell_set.isdisjoint(predicate)
        return cell == predicate

    def _matches_query(self, record, query):
        return all(self._is_match(record, pred, col) for col, pred in query.items())
