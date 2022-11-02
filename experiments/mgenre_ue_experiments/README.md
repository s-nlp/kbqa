### How to replicate our results

- Run all cells in `setup_env.inpynb` notebook.
- Set beam_size in `mgenre_single_generate.py`.
- Run `nohup python mgenre_single_generate.py {ner_name} > mgenre_single_generate.log`, where `ner_name` is one of: `stanza`, `spacy`.
- Run `mgenre_eval.ipynb` notebook to obtain rejection results. Make sure to provide valid paths to archives with generation results.