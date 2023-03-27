for dataset in wdsq rubq mintaka
do  
    export SEQ2SEQ_DATASET=$dataset

    export SEQ2SEQ_RUN_NAME=wdsq_tunned
    export EL_NER_PATH=/mnt/raid/data/kbqa/ner/spacy_models/wdsq_tuned/model-best/
    python3 mgenre_entities_selection.py

    export SEQ2SEQ_RUN_NAME=mintaka_tunned
    export EL_NER_PATH=/mnt/raid/data/kbqa/ner/spacy_models/mintaka_tuned/model-best/
    python3 mgenre_entities_selection.py
done