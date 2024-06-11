from src.model import model
from src.dataset import dataset


def train(
    train_path,
    aug_train_path,
    val_path,
    save_path="src/models",
    seed=42,
    standard_size=(32, 32),
    label_size_factor=-1,
    max_iter=10000,
):
    # load the dataset
    train_data = dataset(
        img_dir=train_path,
        augment_path=aug_train_path,
        label_size_factor=label_size_factor,
        standard_size=standard_size,
    )
    val_data = dataset(val_path, train=False, standard_size=standard_size)
    my_model = model(seed)
    # chose one the models in src.model
    my_model.train_svm(train_data, verbose=1, max_iter=max_iter)
    # my_model.plot_learning_curve(train_data, verbose=1)
    print(
        f"Model trained with {my_model.name} with seed {seed},max_iter {max_iter}, standard_size {standard_size}, label_size_factor {label_size_factor}"
    )
    # evaluate the model
    accuracy = my_model.evaluate(val_data)
    print("Model accuracy with Logistic Regression: {}".format(accuracy))
    # save the model
    my_model.save(save_path)


"""
    my_model.train_svm(train_data)
    print("Model trained with SVM")
    # evaluate the model
    accuracy = my_model.evalutate(val_data)
    print("Model accuracy with SVM: {}".format(accuracy))
    return my_model
"""
