import os
from prettytable import PrettyTable

# Define class labels
classes = {
    0: "mask worn incorrectly",
    1: "with mask",
    2: "without mask"
}

VERSION = 14


def count_classes(directory, title):
    # Initialize count dictionary
    class_counts = {key: 0 for key in classes}

    # Go through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_num = int(line.strip().split()[0])
                    if class_num in class_counts:
                        class_counts[class_num] += 1

    # Display the results using PrettyTable
    table = PrettyTable()
    table.title = title
    table.field_names = ["Class", "Description", "Count"]
    for key, value in class_counts.items():
        table.add_row([key, classes[key], value/3])

    print(table)

    total_samples = sum(class_counts.values())
    if class_counts[0] < total_samples * 0.15:
        print("\nSuggestion: Consider adding more samples of 'mask worn incorrectly' to make the dataset more balanced.")
    if class_counts[1] < total_samples * 0.15:
        print("\nSuggestion: Consider adding more samples of 'with mask' to make the dataset more balanced.")
    if class_counts[2] < total_samples * 0.15:
        print("\nSuggestion: Consider adding more samples of 'without mask' to make the dataset more balanced.")

    print("\n----------------------------------\n")


train_directory = f"../Mask-Detection-YOLOv8-{VERSION}/train/labels"
test_directory = f"../Mask-Detection-YOLOv8-{VERSION}/test/labels"
val_directory = f"../Mask-Detection-YOLOv8-{VERSION}/valid/labels"

count_classes(train_directory, "Train Dataset")
count_classes(test_directory, "Test Dataset")
count_classes(val_directory, "Validation Dataset")