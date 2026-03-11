import math
from collections import Counter
from typing import Dict, List, Any, Union


def calculate_entropy(labels: List[Any]) -> float:
    total_count = len(labels)
    label_counts = Counter(labels)
    entropy = 0.0
    for count in label_counts.values():
        probability = count / total_count
        entropy -= probability * math.log2(probability)
    return entropy


def calculate_information_gain(
    examples: List[Dict[str, Any]], attr: str, target: str
) -> float:
    total_entropy = calculate_entropy([example[target] for example in examples])
    values = set(example[attr] for example in examples)
    weighted_entropy = 0.0
    for value in values:
        subset = [example for example in examples if example[attr] == value]
        weighted_entropy += (len(subset) / len(examples)) * calculate_entropy(
            [example[target] for example in subset]
        )
    return total_entropy - weighted_entropy


def learn_decision_tree(
    examples: List[Dict[str, Any]], attributes: List[str], target: str
) -> Union[Any, Dict[str, Any]]:
    if len(set(example[target] for example in examples)) == 1:
        return examples[0][target]

    if not attributes:
        return Counter(example[target] for example in examples).most_common(1)[0][0]

    best_attr = max(
        attributes, key=lambda attr: calculate_information_gain(examples, attr, target)
    )
    tree = {best_attr: {}}

    values = set(example[best_attr] for example in examples)
    for value in values:
        subset = [example for example in examples if example[best_attr] == value]
        if subset:
            subtree = learn_decision_tree(
                subset, [attr for attr in attributes if attr != best_attr], target
            )
            tree[best_attr][value] = subtree
        else:
            tree[best_attr][value] = Counter(
                example[target] for example in examples
            ).most_common(1)[0][0]

    return tree


examples = [
    {
        "Outlook": "Sunny",
        "Temperature": "Hot",
        "Humidity": "High",
        "Wind": "Weak",
        "PlayTennis": "No",
    },
    {
        "Outlook": "Sunny",
        "Temperature": "Hot",
        "Humidity": "High",
        "Wind": "Strong",
        "PlayTennis": "No",
    },
    {
        "Outlook": "Overcast",
        "Temperature": "Hot",
        "Humidity": "High",
        "Wind": "Weak",
        "PlayTennis": "Yes",
    },
    {
        "Outlook": "Rain",
        "Temperature": "Mild",
        "Humidity": "High",
        "Wind": "Weak",
        "PlayTennis": "Yes",
    },
]
attributes = ["Outlook", "Temperature", "Humidity", "Wind"]
target = "Temperature"

tree = learn_decision_tree(examples, attributes, target)
print(tree)
