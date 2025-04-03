from pandas import DataFrame

class Cities:
    def __init__(self ,path="./src/travelplanner/database/background/citySet_with_states.txt") -> None:
        self.path = path
        self.load_data()
        print("Cities loaded.")

    def load_data(self):
        self.data = {}
        cityStateMapping = None
        with open(self.path, "r") as f:
            cityStateMapping = f.read().strip().split("\n")
        for unit in cityStateMapping:
            city, state = unit.split("\t")
            if state not in self.data:
                self.data[state] = [city]
            else:
                self.data[state].append(city)
    
    def run(self, state) -> dict:
        if state not in self.data:
            return ValueError("Invalid State")
        else:
            return self.data[state]