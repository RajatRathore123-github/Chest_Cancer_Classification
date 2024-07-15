
# Chest Cancer Classification

A Deep Learning project to classify the Chest images as normal or Carcinogenic with complete workflow from loading data to prediction and using MLflow to track experiments to record and compare parameters and results.


## Getting Started

I have used the dataset from kaggle competition `https://www.kaggle.com/datasets`.

Broke the worklow into 4 Stages:

`01_data_ingestion`

`02_prepare_base_model`

`03_model_trainer`

`04_model_evaluation`

The complete set of packages I used are in the `requirements.txt` file in the root of the project.

The main endpoint of the project is the `main.py` file.
## Tech Stack

**Frameworks**: Flask, tensorflow

**Tools**: MLflow, DVC, Dagshub


## Installation

Install my-project with setting up the virtual environment using `conda` using the command `conda activate your_env` and then install all the dependencies in the `requirements.txt` file using the command `pip install -r requirements.txt`.
    
## Environment Variables

To run this project, you will need to add the following environment variables to add in your virtual environment.

`os.environ["MLFLOW_TRACKING_URI"] = "MLFLOW_TRACKING_URI"`

`os.environ["MLFLOW_TRACKING_USERNAME"] = "MLFLOW_USERNAME"`

`os.environ["MLFLOW_TRACKING_PASSWORD"] = "MLFLOW_PASSWORD"`




## Screenshots

Model info
![](/images/Model_info.png)

Model Metrics
![](/images/Model_loss_acc.png)


## Usage/Examples

## Configuring the data_ingestion_file

```python

class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir

        )

        return data_ingestion_config

```

## Configuring the base_model_file 

```python
     

class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
           
        )

        return prepare_base_model_config 
```


## Configuration for Model Training

```python
class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "Data")
        create_directories([
            Path(training.root_dir)
        ])    

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )

        return training_config
```


## Configuration of the Model Evaluation

```python

class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5",
            training_data="artifacts//data_ingestion/Data",
            mlflow_uri="https://dagshub.com/RajatRathore123-github/Chest_Cancer_Classification.mlflow",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )    

        return eval_config
```


## FAQ

#### Want to contribute to this project?

Create an issue with proper tag and make a PR regarding the same.

#### Something else?

Write to [rajat12350iam@gmail.com](rajat12350iam@gmail.com)


## Authors

- [@Rajat Rathore](https://github.com/RajatRathore123-github)

