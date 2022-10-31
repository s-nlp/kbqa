from caches.base import CacheBase
import torch


class GENREWikidataEntityesCache(CacheBase):
    """GENREWikidataEntityesCache - Class for storing results of working GENRE or mGENRE model lang_title_to_wikidata_id
    Just store results for provided string
    """

    def __init__(
        self,
        mgenre_model,
        trie,
        lang_title_to_wikidata_id,
        cache_dir_path: str = "./cache_store",
    ) -> None:
        super().__init__(cache_dir_path, "genre_wikidata_entities_vocab.pkl")
        self.mgenre_model = mgenre_model
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
                res = self._mgenre_cache_formatter(batched_results[idx_from_batch])
                results[global_idx] = res
                self.cache[sent] = res

                self.save_cache()

        return results

    def _generate(
        self,
        sentences,
        beam=10,
    ):
        return self.mgenre_model.sample(
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

    def _mgenre_cache_formatter(self, cache):
        for item in cache:
            if isinstance(item["score"], torch.Tensor):
                item["score"] = item["score"].cpu().numpy().tolist()
            if isinstance(item["scores"], torch.Tensor):
                item["scores"] = item["scores"].cpu().numpy().tolist()
        return cache
