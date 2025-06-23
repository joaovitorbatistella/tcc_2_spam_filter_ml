from enum import Enum

class Algorithm(Enum):
    LOGISTICREGRESSION = "logistic-regression"
    NAIVEBAYES = "naive-bayes"
    
    def get_name(self):
        algorithms = {
            Algorithm.LOGISTICREGRESSION: "Logistic Regression",
            Algorithm.NAIVEBAYES: "Naïve Bayes",
        }
        return algorithms.get(self, "Algorithm unknown")