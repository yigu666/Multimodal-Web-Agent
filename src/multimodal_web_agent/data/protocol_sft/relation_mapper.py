from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class RelationMatch:
    relation: str
    query_prefix: str
    pattern_name: str
    matched: bool


RELATION_PATTERNS: Sequence[Tuple[re.Pattern[str], str, str, str]] = (
    # People and organizations responsible for the visible entity.
    (re.compile(r"\bwho\s+(?:is|was)\s+the\s+architect\b"), "architect", "architect of", "who_is_architect"),
    (re.compile(r"\bwho\b.*\barchitected\b"), "architect", "architect of", "who_architected"),
    (re.compile(r"\bwho\b.*\bcommissioned\b"), "commissioner", "commissioner of", "who_commissioned"),
    (re.compile(r"\bwho\s+(?:was\s+)?designed\b"), "designer", "designer of", "who_designed"),
    (re.compile(r"\bwho\b.*\bdesigned\b"), "designer", "designer of", "who_designed_object"),
    (re.compile(r"\bwho\s+is\s+the\s+designer\b"), "designer", "designer of", "who_is_designer"),
    (re.compile(r"\bwho\s+(?:was\s+)?built\b"), "builder", "builder of", "who_built"),
    (re.compile(r"\bwho\b.*\bbuilt\b"), "builder", "builder of", "who_built_object"),
    (re.compile(r"\bwho\s+(?:was\s+)?created\b"), "creator", "creator of", "who_created"),
    (re.compile(r"\bwho\b.*\bcreated\b"), "creator", "creator of", "who_created_object"),
    (re.compile(r"\bwho\s+(?:was\s+)?painted\b"), "artist", "artist of", "who_painted"),
    (re.compile(r"\bwho\s+is\s+the\s+artist\b"), "artist", "artist of", "who_is_artist"),
    (re.compile(r"\bwho\s+(?:was\s+)?sculpted\b"), "sculptor", "sculptor of", "who_sculpted"),
    (re.compile(r"\bwho\s+(?:was\s+)?manufactured\b"), "manufacturer", "manufacturer of", "who_manufactured"),
    (re.compile(r"\bwho\s+(?:was\s+)?invented\b"), "inventor", "inventor of", "who_invented"),
    (re.compile(r"\bwho\b.*\binvented\b"), "inventor", "inventor of", "who_invented_object"),
    (re.compile(r"\bwho\b.*\bdiscovered\b"), "discoverer", "discoverer of", "who_discovered"),
    (re.compile(r"\bwho\b.*\b(?:wrote|authored)\b"), "author", "author of", "who_wrote"),
    (re.compile(r"\bwho\b.*\b(?:directed|filmed)\b"), "director", "director of", "who_directed"),
    (re.compile(r"\bwho\b.*\b(?:composed|scored)\b"), "composer", "composer of", "who_composed"),
    (re.compile(r"\bwho\b.*\b(?:founded|established)\b"), "founder", "founder of", "who_founded"),
    (re.compile(r"\bwho\b.*\b(?:owns?|owned)\b"), "owner", "owner of", "who_owns"),
    (re.compile(r"\bwho\b.*\b(?:played|portrayed)\b"), "performer", "performer of", "who_played"),
    (re.compile(r"\bwho\b.*\bphotographed\b"), "photographer", "photographer of", "who_photographed"),
    (re.compile(r"\bwho\s+(?:is|was)\s+the\s+creator\b"), "creator", "creator of", "who_is_creator"),
    (re.compile(r"\bwho\s+(?:is|was)\s+the\s+author\b"), "author", "author of", "who_is_author"),
    (re.compile(r"\bwho\s+(?:is|was)\s+the\s+director\b"), "director", "director of", "who_is_director"),
    (re.compile(r"\bwho\s+(?:is|was)\s+the\s+composer\b"), "composer", "composer of", "who_is_composer"),
    (re.compile(r"\bwho\s+(?:is|was)\s+the\s+founder\b"), "founder", "founder of", "who_is_founder"),
    (re.compile(r"\bwho\s+(?:is|was)\s+the\s+owner\b"), "owner", "owner of", "who_is_owner"),
    (re.compile(r"\bwho\s+(?:is|was)\s+the\s+operator\b"), "operator", "operator of", "who_is_operator"),
    (re.compile(r"\bwho\s+(?:is|was)\s+the\s+manufacturer\b"), "manufacturer", "manufacturer of", "who_is_manufacturer"),
    (re.compile(r"\bwho\s+(?:is|was)\s+the\s+photographer\b"), "photographer", "photographer of", "who_is_photographer"),
    (re.compile(r"\bwhich\s+(?:company|manufacturer)\b.*\bmade\b"), "manufacturer", "manufacturer of", "company_made"),
    (re.compile(r"\b(?:which|what)\s+(?:company|manufacturer)\b.*\bmanufactured\b"), "manufacturer", "manufacturer of", "company_manufactured"),
    (re.compile(r"\b(?:which|what)\s+(?:company|brand)\b.*\b(?:made|makes|produced|produces|manufactured)\b"), "manufacturer", "manufacturer of", "company_produced"),
    (re.compile(r"\b(?:who|which\s+organization)\b.*\boperates?\b"), "operator", "operator of", "organization_operates"),

    # Places and origins.
    (re.compile(r"\bwhere\b.*\b(?:born|birthplace)\b"), "birthplace", "birthplace of", "where_born"),
    (re.compile(r"\bwhere\b.*\b(?:died|buried)\b"), "death_place", "death place of", "where_died"),
    (re.compile(r"\bwhere\b.*\b(?:made|manufactured|produced|originated)\b"), "origin", "origin of", "where_made"),
    (re.compile(r"\bwhere\b.*\bheadquartered\b"), "headquarters", "headquarters of", "where_headquartered"),
    (re.compile(r"\bwhere\b.*\blocated\b"), "location", "location of", "where_located"),
    (re.compile(r"\bwhere\s+is\b"), "location", "location of", "where_is"),
    (re.compile(r"\bwhich\s+(?:city|country|state|region)\b"), "location", "location of", "which_location"),
    (re.compile(r"\bwhat\s+(?:city|country|state|region)\b"), "location", "location of", "what_location"),

    # Dates and periods.
    (re.compile(r"\bwhen\b.*\b(?:built|constructed)\b"), "construction_date", "construction date of", "when_built"),
    (re.compile(r"\bwhat\s+year\b.*\b(?:built|constructed)\b"), "construction_date", "construction date of", "year_built"),
    (re.compile(r"\bwhen\b.*\bcreated\b"), "creation_date", "creation date of", "when_created"),
    (re.compile(r"\bwhat\s+year\b.*\bcreated\b"), "creation_date", "creation date of", "year_created"),
    (re.compile(r"\bwhen\b.*\b(?:founded|established)\b"), "founding_date", "founding date of", "when_founded"),
    (re.compile(r"\bwhen\b.*\bborn\b"), "birth_date", "birth date of", "when_born"),
    (re.compile(r"\bwhen\b.*\b(?:died|deceased)\b"), "death_date", "death date of", "when_died"),
    (re.compile(r"\b(?:when|what\s+year)\b.*\b(?:released|published|opened|introduced|launched|completed|invented|discovered|manufactured)\b"), "event_date", "date of", "when_event"),

    # Explicit attributes.  Deliberately omit generic identification prompts
    # such as "what is shown" or "what is the name".
    (re.compile(r"\bwhat\s+(?:is\s+the\s+)?nationality\b"), "nationality", "nationality of", "what_nationality"),
    (re.compile(r"\bwhat\s+(?:is\s+the\s+)?(?:occupation|profession)\b"), "occupation", "occupation of", "what_occupation"),
    (re.compile(r"\b(?:what|which)\s+(?:is\s+the\s+)?(?:genre|style|movement)\b"), "style", "genre style of", "what_style"),
    (re.compile(r"\b(?:what|which)\s+(?:is\s+the\s+)?language\b"), "language", "language of", "what_language"),
    (re.compile(r"\b(?:what|which)\s+(?:is\s+the\s+)?(?:breed|species)\b"), "classification", "breed species of", "what_species"),
    (re.compile(r"\b(?:how|what)\s+(?:tall|high|long|wide)\b"), "dimensions", "dimensions of", "what_dimensions"),
    (re.compile(r"\bwhat\s+(?:is\s+the\s+)?(?:height|length|width)\b"), "dimensions", "dimensions of", "what_dimension_noun"),
    (re.compile(r"\bwhat\s+(?:is\s+the\s+)?capital\b"), "capital", "capital of", "what_capital"),
    (re.compile(r"\b(?:what|which)\s+(?:is\s+the\s+)?(?:team|club|university|college)\b"), "affiliation", "affiliation of", "what_affiliation"),
    (re.compile(r"\bwhat\s+(?:organization|event)\b.*\bassociated\b"), "association", "organization associated with", "associated_organization"),
    (re.compile(r"\bwhat\s+material\b.*\bmade\b"), "material", "material of", "material_made"),
    (re.compile(r"\bwhich\s+material\b"), "material", "material of", "which_material"),
    (re.compile(r"\bwhat\s+is\s+the\s+material\b"), "material", "material of", "what_material"),
)


def map_question_relation(question: str) -> RelationMatch:
    normalized = " ".join(
        unicodedata.normalize("NFKC", str(question)).casefold().split()
    )
    for pattern, relation, query_prefix, pattern_name in RELATION_PATTERNS:
        if pattern.search(normalized):
            return RelationMatch(relation, query_prefix, pattern_name, True)
    return RelationMatch("", "", "none", False)


def relation_from_question(question: str) -> Optional[RelationMatch]:
    match = map_question_relation(question)
    return match if match.matched else None
