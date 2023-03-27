import gradio as gr
import os
from src.entity_linking import EntityLinker
from config import ner_config, mgenre_config


if __name__ == "__main__":
    entity_linkier = EntityLinker(
        ner_model_path=ner_config["model_path"],
        ner_examples_path=ner_config["examples_path"],
        mgenre_examples_path=mgenre_config["examples_path"],
        mgenre_num_beams=mgenre_config["num_beams"],
        mgenre_num_return_sequences=mgenre_config["num_return_sequences"],
    )

    interface = gr.TabbedInterface(
        [
            entity_linkier.get_enities_linking_interface(),
            entity_linkier.get_ner_interface(),
            entity_linkier.get_mgenre_interface(),
        ],
        tab_names=["EntityLinking", "NER", "mGENRE"],
        analytics_enabled=True,
    )

    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        enable_queue=True,
    )
