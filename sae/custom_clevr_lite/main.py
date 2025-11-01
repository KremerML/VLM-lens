from src.clevr_lite_data_generator import CLEVRLiteGenerator

generator = CLEVRLiteGenerator(
    output_dir='./data/',
    num_train=1000,
    num_val=100,
    held_out_ratio=0.4,
    seed=32,
)
generator.generate_dataset()