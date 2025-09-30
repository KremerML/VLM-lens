from src.clevr_lite_data_generator import CLEVRLiteGenerator

generator = CLEVRLiteGenerator(
    output_dir='./data/clevr_lite_dataset',
    num_train=200_000,
    num_val=22_000,
    held_out_ratio=0.5,
    seed=42,
)
generator.generate_dataset()