from caches.base import CacheBase


class GENREWikidataEntityesCache(CacheBase):
    """GENREWikidataEntityesCache - Class for storing results of working GENRE or mGENRE model
    Just store results for provided string
    """

    def __init__(
        self,
        mgenre_model,
        reverse_vocab_wikidata,
        trie,
        lang_title_to_wikidata_id,
        cache_dir_path: str = "./cache_store",
    ) -> None:
        super().__init__(cache_dir_path, "genre_wikidata_entities_vocab.pkl")
        self.mgenre_model = mgenre_model
        self.reverse_vocab_wikidata = reverse_vocab_wikidata
        self.trie = trie
        self.lang_title_to_wikidata_id = lang_title_to_wikidata_id
        self.cache = {}
        self.load_from_cache()

    def sentences_batch_to_entities(self, sentences, beam=10):
        results = [None] * len(sentences)
        batch = []
        for idx, sent in enumerate(sentences):
            if sent in self.cache:
                results[idx] = self.cache[sent]
            else:
                batch.append(sent)

        if len(batch) > 0:
            batched_results = self._generate(batch, beam)
            for idx_from_batch, sent in enumerate(batch):
                global_idx = sentences.index(sent)
                results[global_idx] = batched_results[idx_from_batch]
                self.cache[sent] = batched_results[idx_from_batch]

                self.save_cache()

        return results

    def _generate(
        self,
        sentences,
        beam=10,
    ):
        generated_entities_batched = self.mgenre_model.sample(
            sentences,
            beam=beam,
            prefix_allowed_tokens_fn=lambda batch_id, sent: [
                e
                for e in self.trie.get(sent.tolist())
                if e < len(self.mgenre_model.task.target_dictionary)
            ],
            text_to_id=lambda x: max(
                self.lang_title_to_wikidata_id[tuple(reversed(x.split(" >> ")))],
                key=lambda y: int(y[1:]),
            ),
            marginalize=True,
            verbose=True,
        )
        entities_wikidata_from_mgenre = []

        for generated_entities in generated_entities_batched:
            entities_wikidata = {}
            for entity in generated_entities:
                entity_natural = entity["texts"][0].split(" >> ")[0]
                entities_wikidata[
                    self.reverse_vocab_wikidata.get(entity_natural)
                ] = entity_natural

            if None in entities_wikidata:
                del entities_wikidata[None]

            entities_wikidata_from_mgenre.append(entities_wikidata)

        return entities_wikidata_from_mgenre
