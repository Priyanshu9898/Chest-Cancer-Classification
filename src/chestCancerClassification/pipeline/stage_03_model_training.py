from chestCancerClassification.config.configuration import ConfigurationManager
from chestCancerClassification.components.model_training import Training
from chestCancerClassification import logger


STAGE_NAME = "Model Training Stage"

class TrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
    
        logger.info("Training start for VGG16 model")
        
        training_config = config.get_training_config_vgg16()
        
        # print(training_config)
        training = Training(config=training_config, model_name = "VGG16")
        training.train()
        
        logger.info("Training ended successfully for VGG16 model")
        
        
        logger.info("Training start for VGG19 model")
        
        training_config = config.get_training_config_vgg19()
        
        print("VGG19 Training Config", training_config)
        logger.info("VGG19 Training Config", training_config)
        
        training = Training(config=training_config, model_name = "VGG19")
        training.train()
        
        logger.info("Training ended successfully for VGG19 model")
        
        
        logger.info("Training start for RESNET model")
        
        training_config = config.get_training_config_resnet()
        

        logger.info("RESNET Training Config", training_config)
        
        training = Training(config=training_config, model_name = "RESNET")
        training.train()
        
        logger.info("Training ended successfully for RESNET model")