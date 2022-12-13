import os

ner_config = {
    "model_path": os.environ.get("NER_MODEL_PATH"),
    "examples_path": os.environ.get("NER_EXAMPLES_PATH", "ner_examples.txt"),
}

mgenre_config = {
    "num_beams": os.environ.get("MGENRE_NUM_BEAMS", 10),
    "num_return_sequences": os.environ.get("MGENRE_NUM_RETURN_SEQUENCES", 10),
    "examples_path": os.environ.get("NER_EXAMPLES_PATH", "mgenre_examples.txt"),
}
