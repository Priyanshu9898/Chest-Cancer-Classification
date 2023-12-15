from chestCancerClassification.constants import *
from chestCancerClassification.utils.common import read_yaml, create_directories
from chestCancerClassification.entity import DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig
class ConfigurationManager:
    def __init__(self, config_file_path = CONFIG_FILE_PATH, param_file_path = PARAMS_FILE_PATH) -> None:
       
        self.config = read_yaml(config_file_path)
        self.param = read_yaml(param_file_path)
        
        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_data=config.unzip_data 
        )
        
        return data_ingestion_config

    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        print(self.config.prepare_base_model)

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_vgg16=Path(config.base_model_vgg16),
            updated_base_model_vgg16=Path(config.updated_base_model_vgg16),
            base_model_vgg19=Path(config.base_model_vgg19),
            updated_base_model_vgg19=Path(config.updated_base_model_vgg19),
            base_model_inceptionv3=Path(config.base_model_inceptionv3),
            updated_base_model_inceptionv3=Path(
                config.updated_base_model_inceptionv3),
            base_model_exception=Path(config.base_model_exception),
            updated_base_model_exception=Path(
                config.updated_base_model_exception),
            all_model_params=self.param,
            params_classes= self.param.CLASSES,
            params_learning_rate = self.param.LEARNING_RATE
        )

        return prepare_base_model_config