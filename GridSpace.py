
class GridSpace:
    def __init__(self, num_rows=4, num_cols=4, inaccessible=[(1,1)], success=[(0,3)], fail=[(1,3)]):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.inaccessible = inaccessible
        self.success = success
        self.fail = fail
    
    def map_observation_to_repr(self, observation):
        return observation