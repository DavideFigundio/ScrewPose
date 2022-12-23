import math

class BinaryDistributionGraph:

    def __init__(self, interval=1, minimum=0, maximum=100):
        self.minimum = minimum
        self.maximum = maximum
        self.interval = interval
        self.categories = self.create_categories()

    def create_categories(self):
        num_categories = math.ceil((self.maximum - self.minimum)/self.interval)
        categories = {}
        self.maximum = self.minimum + num_categories*self.interval
        for i in range(num_categories):
            value = self.minimum + self.interval * i
            categories[value] = {"true": 0, "total": 0}
        categories[self.maximum] = {"true": 0, "total": 0}
        
        return categories

    def insert_value(self, value, classification):
        if value >= self.maximum:
            self.categories[self.maximum]["total"] += 1
            if classification:
                self.categories[self.maximum]["true"] += 1
        
        elif value <= self.minimum:
            self.categories[self.minimum]["total"] += 1
            if classification:
                self.categories[self.minimum]["true"] += 1

        else:
            category = math.floor((value - self.minimum)/self.interval)*self.interval + self.minimum
            self.categories[category]["total"] += 1
            if classification:
                self.categories[category]["true"] += 1

    def save_csv(self, filepath):
        with open(filepath, 'w') as csv:
            for category in self.categories.keys():
                csv.write(str(category) + "," + str(self.categories[category]['true']) + "," + str(self.categories[category]['total']) + "\n")