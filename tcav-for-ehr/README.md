# Concept-based model explanations for Electronic Health Records

This repository contains code to replicate the end-to-end experimental workflow
discussed in the eponymous paper ([arxiv]((https://arxiv.org/abs/2012.02308))).

Abstract: Recurrent Neural Networks (RNNs) are often used for sequential
modeling of adverse outcomes in electronic health records (EHRs) due to their
ability to encode past clinical states. These deep, recurrent architectures have
displayed increased performance compared to other modeling approaches in a
number of tasks, fueling the interest in deploying deep models in clinical
settings. One of the key elements in ensuring the safe model deployment and
building user trust is model explainability. Testing with Concept Activation
Vectors (TCAV) has recently been introduced as a way of providing
human-understandable explanations by comparing high-level concepts to the
network’s gradients. While the technique has shown promising results in
real-world imaging applications, it has not been applied to structured temporal
inputs. To enable an application of TCAV to sequential predictions in the EHR,
we propose an extension of the method to time series data. We evaluate the
proposed approach on an open EHR benchmark from the intensive care unit, as well
as synthetic data where we are able to better isolate individual effects.

This repository releases the code related to synthetic data generation, model
training, CAV building, and conceptual sensitivity analysis. The code is written
in Python 3 and uses Numpy, Scikit-Learn, TensorFlow and Sonnet. We provide
executables as well as Jupyter/Colaboratory notebooks.

## Setup

**Prerequisite:** `conda` distribution must be installed on your computer. The
code in this repository was tested with `conda 4.9.2`.

1.  Clone the repository to a local directory and set it as a working directory
    in the terminal.
2.  Run
    ```
    $ chmod +x init.sh
    $ ./init.sh
    ```
    to initialize the working environment, then activate the environment as per
    the `conda activate` command displayed in the terminal.
    (e.g. `conda activate /Users/admin/timeseries-informativeness/.conda_env`)
3.  Run `jupyter notebook` to start a local IPython kernel and open notebooks
    from this repository.

## Experimental Workflow

### Synthetic data generation:

Datasets can be generated using the `generate_dataset.py` script. Datasets are
generated using the causal graph specified by the user in the form of a config
structure. Dataset configs are constructed by providing specifications for
labels, features, and concepts in the form of \*Spec objects, located in
`dataset_specs.py`. The causal graph config used for the paper is provided in
`dataset_configs.py`, but the user is free to construct any graph they wish.

The script relies on `dataset_configs.py`, where `Features` (numerical or
binary) are defined as well as default settings for the dataset. To create the
dataset used in the paper, run:

```
$ python generate_dataset.py --datasets=main_paper
```

Additional command line argument information:

*   `datasets`: `List[str]` [required], dataset(s) to generate. The dataset
    specified must be defined in `dataset_configs.py`.
*   `export_dir`: `str` [optional], directory to export dataset to. Default is
    `‘datasets/’`.
*   `scaling_types`: `List[str]` [optional], what scaling method(s) to use on
    numerical features. Must be one or more of `“none”, “unitrange”, “meanstd”`.
    If multiple are provided, then multiple versions of each dataset are created
    for each scaling type, and the name of the scaling type is appended to the
    dataset file name. Default is `none`.
*   `num_train`: `int` [optional], number of samples to generate for the
    training split. Default is 100,000.
*   `num_test`: `int` [optional], number of samples to generate for the
    validation and test splits. Default is 10,000.
*   `suffix`: `str` [optional], identifier string to append to the end of the
    dataset filename. Default is "".

As mentioned above, multiple datasets and/or scaling types can be provided in
the same call to `generate_dataset.py`. All combinations of dataset and scaling
type will be created. Files are saved in `export_dir`, under the name
`<dataset>_<scaling_type>_v003_<date>_<time>_<suffix>.pkl`

**This needs updating with latest colab once it is finished** The Python
notebook `Synthetic_dataset_illustration.ipynb` allows exploring some of the
dataset generation parameters and illustrates the type of sequences that are
generated. The last cell also displays an example of a dataset being generated
and saved locally and leads to the same results as using the above script.

### Building and training models

The following command builds a RNN model and trains it using the previously
generated "train" split of the data. It regularly evaluates model performance on
the "test" split during training.

```
$ python train_model.py --dataset_path=datasets/<dataset_path.pkl>
```

Additional command line argument information:

*   `dataset_path`: `str` [required], path to the generated dataset .pkl file.
*   `model_cell_type`: `str` [optional], the RNN cell type, one of "LSTM,"
    "GRU," "SRU," or "UGRNN". Non-recurrent architectures can be created by
    using cell types of "MLP_tanh," "MLP_relu," or "MLP_sigmoid," where the
    second part of the name refers to the non-linearity used. Default is "LSTM".
    These architectures are defined in models.py and based on the Sonnet
    library.
*   `hidden_sizes`: `List[int]` [optional], the size of each hidden layer in the
    model architecture. This parameter also defines the number of layers.
    Default is `[64, 64, 64]`, i.e. 3 layers of 64 hidden units each. Custom
    values should be passed as a comma-separated list, e.g.
    `--hidden_sizes=32,32,32`.
*   `batch_size`: `int` [optional], the size of the batches sampled during
    training. Default is 32.
*   `learning_rate`: `float` [optional], the learning rate for gradient descent
    optimization. Default is 3e-4 because it is the best learning rate for Adam,
    hands down (<- joke).
*   `num_train_steps`: `int` [optional], the number of batches to sample and
    conduct backpropagation for. Default is 10000.
*   `model_seed`: `int` [optional], specific random seed to be used. Default
    is 1.
*   `logging_interval`: `int` [optional], the number of train steps between each
    evaluation on the test split and logging of the training progress to stdout.
    Default is 50.
*   `checkpoint_directory`: `str` [optional] , directory to save model
    checkpoints to. If no model checkpoint directory is provided, a base
    `model_checkpoints` directory is created. In the `checkpoint_directory`, an
    additional directory is added for the dataset being used if it doesn’t
    already exist, and model checkpoints are saved within a directory of the
    format `{model_cell_type}_{model_seed}`.Please note that to run multiple
    models from the same `dataset_path` with the same `model_cell_type` and
    `model_seed`, you must provide a unique training id (discussed below).
    Otherwise, the script will fail with a "File already exists" error.
*   `training_id`: `str` [optional], a string identifier to append to the end of
    the saved model checkpoint directory. Default is "".
*   `l1_regularization_scale`: `float` [optional], the weight attributed to the
    L1 loss for regularization.
*   `input_dropout_prob`: `float` [optional], the probability of setting inputs
    to hidden layers to zero.
*   `output_dropout_prob`: `float` [optional], the probability of setting
    outputs from hidden layers to zero.
*   `state_dropout_prob`: `float` [optional], the probability of setting hidden
    state values to zero.

Some additional notes:

*   Between each `logging_interval`, the script outputs loss, accuracy, prauc,
    and positive label incidence for train and test data.
*   The saved model checkpoint file ends with `/tfhub`. This is the file that is
    then passed to compute CAVs and conduct conceptual sensitivity analysis.

### Creating CAVs

The following command will build CAVs based on the generated dataset and trained
model:

```
$ python create_cavs.py --dataset_path=datasets/<dataset_path.pkl> \
  --model_path=model_checkpoints/<dataset_name>/<model_cell_type>_<model_seed>/
<model_checkpoint_path/tfhub> \
  --output_dir=<output_dir> \
  --synthetic_ntrains=100
```

Additional command line argument information:

*   dataset_path: `str` [required], path to the generated dataset .pkl file.
*   model_path: `str` [required], path to the model checkpoint (with /tfhub
    suffix).
*   output_dir: `str` [required], directory to save the pickled cav file and
    related metrics to.
*   synthetic_ntrains: `int` [required], number of data elements to include in
    the CAV training dataset for synthetic data. The first N time series that
    have obtained a classification accuracy of at least
    `synthetic_accuracy_threshold` (across time steps) and where the concept is
    present (resp. absent) will be considered as "positive" (resp. negative)
    examples to build the CAV on.
*   linear_model_type: `str` [optional], what kind of linear model should be
    used for training the CAVs. One of "logistic" or "linear". Default is
    "linear".
*   num_bootstrapped_datasets: `int` [optional], number of CAVs to be created.
    Each CAV is trained on a different bootstrapped dataset. Default is 100.
*   num_permutations: `int` [optional], number of datasets with permuted labels
    to be used for training random CAVs on each bootstrapped dataset. The total
    number of models with permuted labels will be `num_bootstrapped_datasets *
    num_permutations`. Default is 10.
*   cav_classifier_mode: `str` [optional], training data mode for CAV
    classifiers. Options are "t1_only", "t0_to_t1", & "t0_to_t1_diff". Default
    is "t0_to_t1".
*   cav_classifier_early_stopping: bool [optional] Whether to use early stopping
    when training CAV classifiers. Default is True.
*   synthetic_accuracy_threshold: `float` [optional], threshold model accuracy
    to filter CAV training examples for the standard synthetic dataset. Default
    is 0.0.
*   synthetic_changepoint_lookback: `int` [optional], the number of timesteps
    before the change point to select as t0 for standard synthetic CAV datasets.
    Default is 0.
*   synthetic_changepoint_lookahead: `int` [optional], the number of timesteps
    after the change point to select as t1 for standard synthetic CAV datasets.
    Default is 25.
*   metrics: `List[str]` [optional], list of metrics names to be used for
    evaluating CAVs. Options are "accuracy," "balanced_accuracy," "recall,"
    "precision," "rocauc," "prauc". Default is `["balanced_accuracy",
    "rocauc"]`.


### Computing Conceptual Sensitivity and Temporal Concept Alignment

With CAVs generated, we can now use these CAVs to compute conceptual sensitivity
(CS) and temporal concept alignment (tCA). Below is an example of a command that
will compute these metrics:

```
$ python run_cs_tca.py --dataset_path=datasets/<dataset_path.pkl> \
  --model_path=model_checkpoints/<dataset_name>/<model_cell_type>_<model_seed>/
<model_checkpoint_path/tfhub> \
  --cavs_path=<cavs_output_dir/cavs.pkl> \
  --output_dir=<output_dir> \
  --targets=0 --targets=1 \
  --layers=0 --layers=1 --layers=2 \
  --synthetic_n_eval=100
```

Additional command line argument information:

*   dataset_path: `str` [required], path to the generated dataset .pkl file.
*   model_path: `str` [required], path to the model checkpoint (with /tfhub
    suffix).
*   cavs_path: `str` [required], path to the cavs .pkl file.
*   output_dir: `str` [required], directory to save the pickled cs and tca file
    and related metrics to.
*   targets: `List[int]` [required], indices of the desired target label to
    compute cs and tca for. Starts with 0.
*   layers: `List[int]` [required], indices of the desired model layer to
    compute cs and tca for. Starts with 0.
*   synthetic_n_eval: `int` [required], the number of batch examples to compute
    and save cs and tca evaluation for.

## Notebooks

We provide two notebooks as a part of this repository:

*   verifying_synthetic_dataset.ipynb: This notebook helps with plotting
    examples from the synthetic dataset and verifying that the designated causal
    graph is adhered to. This is accomplished by computing conditional feature
    and label probabilities and confirming that they are as expected.
*   synthetic_dataset_figures.ipynb: This notebook provides all the code needed
    to regenerate most of the figures found in the main dataset. We provide this
    notebook so that others can generate comparable analyses while experimenting
    with their own results.
