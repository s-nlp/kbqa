import re
from collections import defaultdict
from typing import List, Union, Optional, Tuple

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from kbqa.wikidata.utils import request_to_wikidata


class _WikiDataBase:
    @staticmethod
    def _entity_uri_to_id(uri):
        return uri.split("/")[-1]

    @staticmethod
    def _validate_entity_id(entity_id):
        return re.fullmatch(r"[P|Q][0-9]+", entity_id) is not None

    @staticmethod
    def _request_one_hop_neighbors(entity_id):
        query = """
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX schema: <http://schema.org/>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        SELECT DISTINCT ?property ?object WHERE {
            {?object ?property wd:<ENTITY>} UNION {wd:<ENTITY> ?property ?object} .
        }
        """.replace(
            "<ENTITY>", entity_id
        )

        responce = request_to_wikidata(query)
        return responce

    @staticmethod
    def _request_one_hop_neighbors_with_instance_of(entity_id):
        query = """
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        SELECT DISTINCT ?property ?object ?instance_of WHERE {
        {
            ?object ?property wd:<ENTITY> .
            ?object wdt:P31 ?instance_of
        } UNION {
            wd:<ENTITY> ?property ?object .
            ?object wdt:P31 ?instance_of
        }
        }
        """.replace(
            "<ENTITY>", entity_id
        )

        responce = request_to_wikidata(query)
        return responce

    @staticmethod
    def _request_instance_of(entity_id):
        query = """
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        SELECT DISTINCT ?instance_of WHERE {
            wd:<ENTITY> wdt:P31 ?instance_of
        }
        """.replace(
            "<ENTITY>", entity_id
        )

        return request_to_wikidata(query)

    @staticmethod
    def _request_label(entity_id):
        query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
        PREFIX wd: <http://www.wikidata.org/entity/> 
        SELECT DISTINCT ?label
        WHERE {
            wd:<ENTITY> rdfs:label ?label .
            FILTER (langMatches( lang(?label), "EN" ) )
        } 
        """.replace(
            "<ENTITY>", entity_id
        )

        responce = request_to_wikidata(query)
        return responce

    @staticmethod
    def _request_entity_by_label(label):
        query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
        SELECT * WHERE{
            ?item rdfs:label "<LABEL>"@en .
        }
        """.replace(
            "<LABEL>", label
        )

        responce = request_to_wikidata(query)
        return responce


class Entity(_WikiDataBase):
    __instances = {}

    def __new__(cls, entity_identifier, *args, **kwargs):
        entity_identifier = Entity.entity_identifier_to_id(entity_identifier)
        if entity_identifier not in cls.__instances:
            obj = super(Entity, cls).__new__(cls)
            cls.__instances[entity_identifier] = obj
        return cls.__instances[entity_identifier]

    def __init__(self, entity_identifier):
        entity_identifier = Entity.entity_identifier_to_id(entity_identifier)
        if self._validate_entity_id(entity_identifier):
            self.idx = entity_identifier
        else:
            raise ValueError(
                f"Wrong entity_identifier, can not be extracted Id for {entity_identifier}"
            )

        self._label = None
        self._one_hop_neighbors = None
        self._instance_of = None
        self.is_property = self.idx[0] == "P"

    @classmethod
    def from_label(cls, label):
        responce = cls._request_entity_by_label(label)
        if len(responce) > 0:
            return [Entity(r["item"]["value"]) for r in responce]
        else:
            raise ValueError(
                f"Wrong label, no one entity with label {label} was found. Attention: Supported only English labels"
            )

    @property
    def label(self):
        if self._label is None:
            responce = self._request_label(self.idx)
            if len(responce) > 0:
                self._label = responce[0]["label"]["value"]
        return self._label

    @property
    def one_hop_neighbors(self):
        if self._one_hop_neighbors is None:
            _one_hop_neighbors = []
            for r in self._request_one_hop_neighbors_with_instance_of(self.idx):
                property = Entity(r["property"]["value"])
                neighbor = Entity(r["object"]["value"])

                neighbor_instance_of = Entity(r["instance_of"]["value"])
                if neighbor._instance_of is None:
                    neighbor._instance_of = []
                if neighbor_instance_of not in neighbor._instance_of:
                    neighbor._instance_of.append(neighbor_instance_of)

                if (property, neighbor) not in _one_hop_neighbors:
                    _one_hop_neighbors.append((property, neighbor))

            self._one_hop_neighbors = _one_hop_neighbors
        return self._one_hop_neighbors

    # def get_one_hop_neighbors_filtered_by_instance_of(self, instance_of_entities: List[Entity]) -> List[Entity]:
    #     """
    #     Return one hop neighbors (in bouth directions) that isntance_of in instance_of_entities list
    #     """
    #     pass

    @property
    def instance_of(self):
        if self._instance_of is None:
            self._instance_of = [
                Entity(r["instance_of"]["value"])
                for r in self._request_instance_of(self.idx)
            ]
        return self._instance_of

    def __repr__(self):
        if self.is_property:
            return f"<Entity(Property): {self.idx}>"
        else:
            return f"<Entity: {self.idx}>"

    @staticmethod
    def entity_identifier_to_id(entity_identifier):
        if "http" in entity_identifier:
            return Entity._entity_uri_to_id(entity_identifier)
        return entity_identifier.upper()


INSTANCE_OF_IDX_BLACKLIST = [
    "Q4167410",  #   Wikimedia disambiguation page
    "Q14204246",  #  Wikimedia project page
    "Q35252665",  #  Wikimedia non-main namespace
    "Q11266439",  #  Wikimedia template
    "Q58494026",  #  Wikimedia page
    "Q17379835",  #  Wikimedia page outside the main knowledge tree
    "Q37152856",  #  Wikimedia page relating two or more distinct concepts
    "Q100510764",  # Wikibooks book
    "Q104696061",  # Wikibook page
    "Q114612576",  # Wikiversity course
    "Q115491908",  # Wikimedia subpage
    "Q115668764",  # Wiktionary rhymes page
    "Q15407973",  #  Wikimedia disambiguation category
    "Q22808320",  #  Wikimedia human name disambiguation page
    "Q61996773",  #  municipality name disambiguation page
    "Q66480449",  #  Wikimedia surname disambiguation page
    "Q15407973",  #  Wikimedia disambiguation category
]


class _QuestionToRankBase(_WikiDataBase):
    def __init__(
        self,
        question: str,
        question_entities: Union[List[str], List[Entity]],
        answers_candidates: Union[List[str], List[Entity]],
        target_entity: Optional[Entity] = None,
    ):
        self.question = question
        self.question_entities = [
            e if isinstance(e, Entity) else Entity(e) for e in question_entities
        ]
        self.answers_candidates = [
            e if isinstance(e, Entity) else Entity(e) for e in answers_candidates
        ]

        if target_entity is not None:
            self.target = target_entity
        else:
            self.target = None

        self._answer_instance_of_count = None
        self._answer_instance_of = None

    @property
    def answer_instance_of_count(self):
        if self._answer_instance_of is None or self._answer_instance_of_count is None:
            self._calculate_answer_instance_of()

        return self._answer_instance_of_count

    @property
    def answer_instance_of(self):
        if self._answer_instance_of is None or self._answer_instance_of_count is None:
            self._calculate_answer_instance_of()

        return self._answer_instance_of

    def final_answers(self) -> List[Entity]:
        raise NotImplementedError()

    def _calculate_answer_instance_of(self):
        if self._answer_instance_of is None or self._answer_instance_of_count is None:
            self._answer_instance_of_count = defaultdict(float)
            for answer_entity in self.answers_candidates:
                for instance_of_entity in answer_entity.instance_of:
                    self._answer_instance_of_count[instance_of_entity] += 1

            self._answer_instance_of_count = sorted(
                self._answer_instance_of_count.items(), key=lambda v: -v[1]
            )
            self._answer_instance_of_count = [
                (key, val)
                for key, val in self._answer_instance_of_count
                if key.idx not in INSTANCE_OF_IDX_BLACKLIST
            ]

            self._answer_instance_of = self._select_answer_instance_of(
                self._answer_instance_of_count
            )

    def _select_answer_instance_of(
        self, answer_instance_of_count: List[Tuple[Entity, int]]
    ) -> List[Entity]:
        raise NotImplementedError()

    def __repr__(self) -> str:
        if self.target is not None:
            return f"<QuestionRank: {self.question} | {self.target}>"
        else:
            return f"<QuestionRank: {self.question}>"


class QuestionToRank(_QuestionToRankBase):
    def _select_answer_instance_of(
        self, answer_instance_of_count: List[Tuple[Entity, int]]
    ) -> List[Entity]:
        initial_number = 2

        def split_toks(
            label, stopwords=stopwords.words("english"), stemmer=PorterStemmer()
        ):
            return [
                stemmer.stem(tok.lower())
                for tok in label.split()
                if stemmer.stem(tok.lower()) not in stopwords
            ]

        instance_of_label_splitted = [
            split_toks(str(entity.label)) for entity, _ in answer_instance_of_count
        ]
        initial_topN_freq_toks = instance_of_label_splitted[:initial_number]
        topN_freq_toks = initial_topN_freq_toks.copy()

        for toks in instance_of_label_splitted[initial_number:]:
            for check_toks in initial_topN_freq_toks:
                if (
                    toks not in topN_freq_toks
                    and len(set(check_toks).intersection(toks)) > 0
                ):
                    topN_freq_toks.append(toks)

        final_top_instance_of_ids = []
        for toks in topN_freq_toks:
            idx = instance_of_label_splitted.index(toks)
            entity, _ = answer_instance_of_count[idx]
            final_top_instance_of_ids.append(entity)

        return final_top_instance_of_ids

    def final_answers(self) -> List[Entity]:
        final_answers_list = set()
        for q_entity in self.question_entities:
            for _, entity in q_entity.one_hop_neighbors:
                if (
                    len(set(entity.instance_of).intersection(self.answer_instance_of))
                    > 0
                ):
                    final_answers_list.add(entity)
        return list(final_answers_list)
