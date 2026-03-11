import numpy as np

def stratified_train_test_split(X, y, test_size, random_seed=None):
    """
    Split data into train and test sets while maintaining class proportions.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label vector of shape (n_samples,)
        test_size: Proportion of data for test set (0 < test_size < 1)
        random_seed: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    unique_classes = np.unique(y)
    X_train_indices = []
    X_test_indices = []
    
    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        n_class_samples = len(class_indices)
        n_test = int(n_class_samples * test_size)

        np.random.shuffle(class_indices)

        test_indices = class_indices[:n_test]
        train_indices = class_indices[n_test:]

        X_test_indices.extend(test_indices)
        X_train_indices.extend(train_indices)

    X_train = X[X_train_indices]
    X_test = X[X_test_indices]
    y_train = y[X_train_indices]
    y_test = y[X_test_indices]

    return X_train, X_test, y_train, y_test

