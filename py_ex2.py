from typing import List, Tuple
from abc import ABC, abstractmethod
from math import log2
from collections import Counter


def main():
    train_file_path = "train.txt"
    test_file_path = "test.txt"
    output_file_path = "output.txt"
    attributes, rows = parse_file(train_file_path)
    test_attributes, test_rows = parse_file(test_file_path)

    id3_classifier = ID3()
    id3_classifier.fit(attributes, rows)
    id3_classifier.output_to_file()
    id3_Predictions = id3_classifier.predict(test_rows)
    id3_accuracy = id3_classifier.score(test_rows)

    naive_bayes_classifier = NaiveBayes()
    naive_bayes_classifier.fit(attributes, rows)
    naiveBayes_Predictions = naive_bayes_classifier.predict(test_rows)
    naiveBayes_accuracy = naive_bayes_classifier.score(test_rows)

    with open(output_file_path, "w") as f:
        f.write("ID3\tNaiveBayes\n")
        for id3_prediction, naiveBayes_prediction in zip(id3_Predictions, naiveBayes_Predictions):
            f.write(f"{id3_prediction}\t{naiveBayes_prediction}\n")
        f.write(f"{id3_accuracy}\t{naiveBayes_accuracy}")



def parse_file(file_path: str) -> Tuple[List[str], List[List[str]]]:
    """
    Parse the file and return the attributes (first line) and rows (rest of the file)
    Args:
        file_path: The path to the file to parse
    Returns:
        A tuple containing the attributes and rows
    """
    with open(file_path) as file:
        lines = file.readlines()
        attributes = lines[0].strip('\n').split('\t')
        rows = [text.strip('\n').split('\t') for text in lines[1:]]
    return attributes, rows

class Classifier(ABC):
    @abstractmethod
    def fit(self, attributes: List[str], rows: List[List[str]]):
        """Train the classifier on the given data.
        Args:
            attributes: The attributes of the data
            rows: The rows of the data
        """
        pass

    @abstractmethod
    def predict(self, rows: List[List[str]]) -> list:
        """Predict the classification for each row in rows.
        Args:
            rows: The rows of the data
        Returns:
            A list of predictions
        """
        pass

    @abstractmethod
    def score(self, rows: List[List[str]]) -> float:
        """Compute the accuracy of the classifier on the given data.
        Args:
            rows: The rows of the data
        Returns:
            The accuracy of the classifier
        """
        pass

class ID3(Classifier):
    def __init__(self):
        """Initialize the ID3 classifier. The tree and attributes are set after fitting.
        Args:
            attributes: The attributes of the data
            rows: The rows of the data
        """
        self.tree = None
        self.attributes = None

    def fit(self, attributes, rows):
        """Build the decision tree using the ID3 algorithm.
        Args:
            attributes: The attributes of the data
            rows: The rows of the data
        """
        self.attributes = attributes
        self.tree = self._id3(attributes, rows)


    def _id3(self, attributes, rows):
        """
        Build the decision tree using the ID3 algorithm.
        Args:
            attributes: The attributes of the data
            rows: The rows of the data
        """
        if not rows:
            return self._most_common_classification(rows)
        if self._all_same_classification(rows):
            return rows[0][-1]
        if len(attributes) <= 1:
            return self._most_common_classification(rows)
        best_attribute = self._find_best_attribute(attributes, rows)
        best_index = attributes.index(best_attribute)
        tree = {best_attribute: {}}
        for val in {row[best_index] for row in rows}:
            sub_rows = [row for row in rows if row[best_index] == val]
            new_attrs = attributes[:best_index] + attributes[best_index + 1:]
            new_sub_rows = [row[:best_index] + row[best_index + 1:] for row in sub_rows] 
            tree[best_attribute][val] = self._id3(new_attrs, new_sub_rows)
        return tree

    def _most_common_classification(self, rows):
        """
        Find the most common classification in the rows.
        Args:
            rows: The rows of the data
        Returns:
            The most common classification
        """
        if not rows: return None
        counts = Counter(row[-1] for row in rows)
        max_count = max(counts.values())
        return min(k for k, v in counts.items() if v == max_count)

    def _all_same_classification(self, rows):
        """
        Check if all rows have the same classification.
        Args:
            rows: The rows of the data
        Returns:
            True if all rows have the same classification, False otherwise
        """
        return all(row[-1] == rows[0][-1] for row in rows) if rows else True

    def _find_best_attribute(self, attributes, rows):
        """
        Find the best attribute to split the rows on.
        Args:
            attributes: The attributes of the data
            rows: The rows of the data
        """
        if len(attributes) == 1: return attributes[0]
        base_entropy = self._entropy(rows)
        max_gain = 0
        best_attribute = None
        for i, attr in enumerate(attributes[:-1]):  # Exclude classification
            values = set(row[i] for row in rows) # Unique values of the attribute
            attr_entropy = 0.0
            for value in values:
                subset = [row for row in rows if row[i] == value]
                prob = len(subset) / len(rows)
                attr_entropy += prob * self._entropy(subset)
            gain = base_entropy - attr_entropy
            if gain > max_gain:
                max_gain = gain
                best_attribute = attr
        return best_attribute if best_attribute else attributes[0]


    def _entropy(self, rows):
        """
        Calculate the entropy of the rows.
        Args:
            rows: The rows of the data
        Returns:
            The entropy of the rows
        """
        if not rows: return 0.0
        counts = Counter(row[-1] for row in rows)
        total = len(rows)
        return -sum((c / total) * log2(c / total) for c in counts.values())
    
    def predict(self, rows):
        """
        Predict the classification for each row in rows.
        Args:
            rows: The rows of the data
        Returns:
            A list of predictions
        """
        return [self._predict_row(self.tree, row, self.attributes) for row in rows]

    def _predict_row(self, tree, row, attributes):
        """
        Predict the classification for a single row.
        Args:
            tree: The decision tree
            row: The row to predict
            attributes: The attributes of the data
        """
        if not isinstance(tree, dict):
            return tree
        attribute = next(iter(tree))
        index = attributes.index(attribute)
        subtree = tree[attribute].get(row[index])
        if subtree is None:
            return None
        new_attributes = attributes[:index] + attributes[index + 1:]
        new_row = row[:index] + row[index + 1:]
        return self._predict_row(subtree, new_row, new_attributes)

    def score(self, rows):
        """
        Compute the accuracy of the classifier on the given data.
        Args:
            rows: The rows of the data
        Returns:
            The accuracy of the classifier
        """
        predictions = self.predict(rows)
        correct = sum(1 for pred, row in zip(predictions, rows) if pred == row[-1])
        return correct / len(rows) if rows else 0.0

    def output_to_file(self, filename='output_tree.txt'):
        """
        Write the formatted tree to a file.
        Args:
            filename: The name of the file to write to
        """
        def format_tree(tree=None, indent=''):
            if tree is None: tree = self.tree
            lines = []
            for attr, values in tree.items():
                for val, subtree in values.items():
                    if isinstance(subtree, dict):
                        lines.append(f'{indent}{attr} = {val}')
                        lines.append(format_tree(subtree, indent + '| '))
                    else:
                        lines.append(f'{indent}{attr} = {val}: {subtree}')
            return '\n'.join(lines)
        with open(filename, 'w') as f:
            f.write(format_tree())

class NaiveBayes(Classifier):
    def __init__(self):
        """Initialize the Naive Bayes classifier.
        Args:
            attributes: The attributes of the data
            rows: The rows of the data
        """
        self.attributes = None
        self.classes = None
        self.priors = None
        self.conditionals = None
        self.training_label_counts = None

    def fit(self, attributes: List[str], rows: List[List[str]]):
        """Train the Naive Bayes classifier on the given data.
        Args:
            attributes: The attributes of the data
            rows: The rows of the data
        """
        self.attributes = attributes
        self.classes = set(row[-1] for row in rows)
        self.priors = self._calculate_priors(rows)
        self.conditionals = self._calculate_conditionals(attributes, rows)
        self.training_label_counts = Counter(row[-1] for row in rows)

    def _calculate_priors(self, rows: List[List[str]]):
        """Calculate the prior probabilities for each class.
        Args:
            rows: The rows of the data
        Returns:
            The prior probabilities
        """
        priors = Counter(row[-1] for row in rows)
        for label in priors:
            priors[label] /= len(rows)
        return priors

    def _calculate_conditionals(self, attributes: List[str], rows: List[List[str]]):
        """Calculate the conditional probabilities for each attribute with Laplace smoothing.
        Args:
            attributes: The attributes of the data
            rows: The rows of the data
        Returns:
            The conditional probabilities
        """
        conditionals = {attribute: {} for attribute in attributes[:-1]}
        label_counts = {label: 0 for label in self.classes}
        
        # Count occurrences of each attribute value for each class
        for row in rows:
            label = row[-1]
            label_counts[label] += 1
            for i, attribute in enumerate(attributes[:-1]):
                value = row[i]
                if value not in conditionals[attribute]:
                    conditionals[attribute][value] = {class_label: 0 for class_label in self.classes}
                conditionals[attribute][value][label] += 1
        
        for attribute in conditionals:
            k = len(conditionals[attribute])  # Number of unique values for this attribute 
            for value in conditionals[attribute]:
                for label in self.classes:
                    count = conditionals[attribute][value][label]
                    total = label_counts[label]
                    conditionals[attribute][value][label] = (count + 1) / (total + k) # Using laplace smoothing
        
        return conditionals

    def predict(self, test_rows: List[List[str]]):
        """Predict the classification for each row in rows.
        Args:
            test_rows: The rows of the data
        Returns:
            A list of predictions
        """
        predictions = []

        for row in test_rows:
            label_probs = {}
            for label in self.classes:
                prob = self._calculate_probability_for_label(row, label)
                label_probs[label] = prob
            
            # Pick the label with the highest probability
            best_label = max(label_probs.keys(), key=lambda x: label_probs[x])
            predictions.append(best_label)
        return predictions

    def _calculate_probability_for_label(self, row: List[str], label: str):
        """Calculate the probability of the row for a given label.
        Args:
            row: The row to calculate the probability for
            label: The label to calculate the probability for
        Returns:
            The probability of the row for the given label
        """
        prob = self.priors[label]
        for i, attribute in enumerate(self.attributes[:-1]):
            value = row[i]
            if value in self.conditionals[attribute]:
                prob *= self.conditionals[attribute][value][label]
            else:
                # Unseen value: use laplace smoothing
                k = len(self.conditionals[attribute])
                training_label_count = self.training_label_counts[label]
                prob *= 1 / (training_label_count + k)
        return prob

    def score(self, rows: List[List[str]]):
        """Compute the accuracy of the classifier on the given data (how many rows are predicted correctly)
        Args:
            rows: The rows of the data
        Returns:
            The accuracy of the classifier
        """
        if not rows: return 0.0
        predictions = self.predict(rows)
        correct = sum(1 for pred, row in zip(predictions, rows) if pred == row[-1])
        return correct / len(rows)

if __name__ == "__main__":
    main()