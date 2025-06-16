from typing import List, Set, Tuple
import math

def main():
    attributes, rows = parse_file("train.txt")
    tree = id3(attributes, rows)
    write_tree_to_file(tree)
    print("Tree has been written to output_tree.txt")
    attributes, rows = parse_file("test.txt")
    test_tree(tree, attributes, rows)
    #TODO: make the testing in the format requested in the assignment


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

def id3(attributes: List[str], rows: List[List[str]]) -> dict:
    """
    Implement the ID3 algorithm
    Args:
        attributes: The attributes of the data
        rows: The rows of the data
    Returns:
        The decision tree as a nested dictionary
    """

    if not rows:
        return most_common_classification(rows)  # TODO: check if this is correct, SHOULD BE default classification based on the presentation
    if all_same_classification(rows):
        return rows[0][-1]
    if len(attributes) <= 1:
        return most_common_classification(rows)  # TODO: check if this is correct, SHOULD BE MODE(examples) based on the presentation
    best_attribute = find_best_attribute(attributes, rows)
    best_attribute_index = attributes.index(best_attribute)
    tree = {best_attribute: {}}
    for value in unique_values_of_attribute(rows, best_attribute_index):
        sub_rows = [row for row in rows if row[best_attribute_index] == value]
        new_attributes = attributes.copy()
        new_attributes.remove(best_attribute)
        new_sub_rows = [row[:best_attribute_index] + row[best_attribute_index+1:] for row in sub_rows]
        subtree = id3(new_attributes, new_sub_rows)
        tree[best_attribute][value] = subtree
    return tree

def most_common_classification(rows: List[List[str]]) -> str:
    """
    Find the most common classification in the rows
    Args:
        rows: The rows of the data
    Returns:
        The most common classification value
    """
    if not rows:
        return None
    
    counts = {}
    for row in rows:
        classification = row[-1]
        counts[classification] = counts.get(classification, 0) + 1

    max_count = max(counts.values())
    return min(k for k, v in counts.items() if v == max_count)

def all_same_classification(rows: List[List[str]]) -> bool:
    """
    Check if all the classifications in the rows are the same
    Args:
        rows: The rows of the data
    """
    first_classification = rows[0][-1]
    return all(row[-1] == first_classification for row in rows)

def find_best_attribute(attributes: List[str], rows: List[List[str]]) -> str:
    """
    Find the best attribute to split the rows (using information gain)
    Args:
        attributes: The attributes of the data
        rows: The rows of the data
    Returns:
        The attribute with the highest information gain
    """
    if len(attributes) == 1:
        return attributes[0]
    base_entropy = entropy(rows)
    max_gain = 0
    best_attribute = None
    for i, attr in enumerate(attributes[:-1]):  # Exclude classification
        values = unique_values_of_attribute(rows, i)
        attr_entropy = 0.0
        for value in values:
            subset = [row for row in rows if row[i] == value]
            prob = len(subset) / len(rows)
            attr_entropy += prob * entropy(subset)
        gain = base_entropy - attr_entropy
        if gain > max_gain:
            max_gain = gain
            best_attribute = attr
    return best_attribute if best_attribute else attributes[0]

def unique_values_of_attribute(rows: List[List[str]], attribute_index: int) -> Set[str]:
    """
    Find the unique values of the attribute in the rows
    Args:
        rows: The rows of the data
        attribute_index: The index of the attribute to find the unique values of
    Returns:
        Set of unique values for the attribute
    """
    return set(row[attribute_index] for row in rows)

def entropy(rows: List[List[str]]) -> float:
    """
    Calculate the entropy of the class classifications in the rows
    Args:
        rows: The rows of the data
    Returns:
        The entropy value
    """
    if not rows:
        return 0.0
    classification_counts = {}
    for row in rows:
        classification = row[-1]
        classification_counts[classification] = classification_counts.get(classification, 0) + 1
    total = len(rows)
    ent = 0.0
    for count in classification_counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent

def format_tree(tree: dict, indent: str = "") -> str:
    """
    Format the tree according to the specified format
    Args:
        tree: The decision tree
        indent: Current indentation level
    Returns:
        Formatted tree string
    """
    result = []
    for attribute, values in tree.items():
        for value in values:
            subtree = values[value]
            if isinstance(subtree, dict):
                result.append(f"{indent}{attribute} = {value}")
                result.append(format_tree(subtree, indent + "| "))
            else:
                result.append(f"{indent}{attribute} = {value}: {subtree}")
    return "\n".join(result)

def write_tree_to_file(tree: dict, filename: str = "output_tree.txt"):
    """
    Write the formatted tree to a file
    Args:
        tree: The decision tree
        filename: Name of the output file
    """
    formatted_tree = format_tree(tree)
    with open(filename, 'w') as f:
        f.write(formatted_tree)

def predict_row(tree: dict, row: List[str], attributes: List[str]) -> str:
    """
    Predict the classification of a row using the tree
    Args:
        tree: The decision tree
        row: The row to predict
        attributes: The attributes of the data
    Returns:
        The predicted classification
    """
    if not isinstance(tree, dict):
        return tree
    attribute = next(iter(tree))
    values = tree[attribute]
    attr_index = attributes.index(attribute)
    row_value = row[attr_index]
    subtree = values.get(row_value)
    if subtree is None:
        return None
    new_attributes = attributes[:attr_index] + attributes[attr_index+1:]
    new_row = row[:attr_index] + row[attr_index+1:]
    return predict_row(subtree, new_row, new_attributes)

def test_tree(tree: dict, attributes: List[str], rows: List[List[str]]):
    """
    Test the tree on the test data
    Args:
        tree: The decision tree
        attributes: The attributes of the data
        rows: The rows of the data
    """
    print("Predictions:")
    correct = 0
    for row in rows:
        prediction = predict_row(tree, row, attributes)
        print(f"Input: {row} => Predicted: {prediction}")
        if prediction == row[-1]:
            correct += 1
    print(f"Accuracy: {correct / len(rows)}")


    
if __name__ == "__main__":
    main()