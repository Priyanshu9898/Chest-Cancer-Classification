from chestCancerClassification.config.configuration import ConfigurationManager
from chestCancerClassification.components.prepare_base_model import Prepare_Base_Model
from chestCancerClassification import logger


STAGE_NAME = "Prepare Base Model Step"

class PrepareModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = Prepare_Base_Model(config=prepare_base_model_config)
        
        logger.info("Preparing base model VGG16")
        prepare_base_model.get_base_model_vgg16()
        prepare_base_model.update_base_model(prepare_base_model_config.updated_base_model_vgg16)
        
        logger.info("base model VGG16 updated successfully")
        
        logger.info("Preparing base model VGG19")
        prepare_base_model.get_base_model_vgg19()
        prepare_base_model.update_base_model(prepare_base_model_config.updated_base_model_vgg19)
        
        logger.info("base model VGG19 updated successfully")