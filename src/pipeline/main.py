from src.constant.constant import *
from src.pipeline.pipeline import TrainingPipeline

if __name__ == "__main__":
	pipeline = TrainingPipeline(
	    base_path=base_path,
	    categories=class_name,
	    class_map=class_map,
	    checkpoint_path=checkpoint_path,
	    epochs=epochs,
	    batch_size=batch_size,
	    lr=lr,
	    weight_decay=weight_decay,
	    in_channels=in_channels,
	    base_filters=base_filters,
	    input_shape=input_shape,
	    target_size=target_size,
	    output_path=output_path,
	)

	pipeline.run()