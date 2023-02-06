import re
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
    def _request_forward_one_hop_neighbors_with_instance_of(entity_id):
        query = """
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        SELECT DISTINCT ?property ?object ?instance_of WHERE {
            wd:<ENTITY> ?property ?object .
            ?object wdt:P31 ?instance_of
        }
        """.replace(
            "<ENTITY>", entity_id
        )

        responce = request_to_wikidata(query)
        return responce

    @staticmethod
    def _request_backward_one_hop_neighbors_with_instance_of(entity_id):
        query = """
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        SELECT DISTINCT ?property ?object ?instance_of WHERE {
            ?object ?property wd:<ENTITY> .
            ?object wdt:P31 ?instance_of
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
        if not hasattr(self, 'idx'):
            if self._validate_entity_id(entity_identifier):
                self.idx = entity_identifier
            else:
                raise ValueError(
                    f"Wrong entity_identifier, can not be extracted Id for {entity_identifier}"
                )

            self._label = None
            self._forward_one_hop_neighbors = None
            self._backward_one_hop_neighbors = None
            self._instance_of = None
            self.is_property = self.idx[0] == "P"

    @classmethod
    def from_label(cls, label):
        responce = cls._request_entity_by_label(label)
        if len(responce) > 0:
            return [Entity(r["item"]["value"]) for r in responce]
        else:
            label = label[:1].lower() + label[1:]
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
    def forward_one_hop_neighbors(self):
        if self._forward_one_hop_neighbors is None:
            responce = self._request_forward_one_hop_neighbors_with_instance_of(self.idx)
            self._forward_one_hop_neighbors = Entity._process_one_hop_neighbots_with_instance_of(responce)
        return self._forward_one_hop_neighbors

    @property
    def backward_one_hop_neighbors(self):
        if self._backward_one_hop_neighbors is None:
            responce = self._request_backward_one_hop_neighbors_with_instance_of(self.idx)
            self._backward_one_hop_neighbors = Entity._process_one_hop_neighbots_with_instance_of(responce)
        return self._backward_one_hop_neighbors

    @property
    def one_hop_neighbors(self):
        return self.forward_one_hop_neighbors + self.backward_one_hop_neighbors

    @staticmethod
    def _process_one_hop_neighbots_with_instance_of(one_hop_neighbors_with_instance_of_responce):
        _one_hop_neighbors = []
        for r in one_hop_neighbors_with_instance_of_responce:
            try:
                property = Entity(r["property"]["value"])
                neighbor = Entity(r["object"]["value"])
            except ValueError:
                continue

            try:
                neighbor_instance_of = Entity(r["instance_of"]["value"])
                if neighbor._instance_of is None:
                    neighbor._instance_of = []
                if neighbor_instance_of not in neighbor._instance_of:
                    neighbor._instance_of.append(neighbor_instance_of)
            except ValueError:
                continue

            if (property, neighbor) not in _one_hop_neighbors:
                _one_hop_neighbors.append((property, neighbor))
        return _one_hop_neighbors

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

    def __json__(self):
        return {
            'idx': self._idx,
            '_label': self._label,
        }
