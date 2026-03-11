"""Moral Foundations Theory (MFT) and Schwartz value token lexicons.

Every token includes both bare and space-prefixed forms because GPT-2
tokenizes mid-sentence words with a leading space (e.g. " care" vs "care").
"""

MORAL_FOUNDATIONS = {
    "care_harm": {
        "positive": ["care", " care", "caring", " caring", "protect", " protect",
                     "nurture", " nurture", "compassion", " compassion", "empathy", " empathy",
                     "kindness", " kindness", "mercy", " mercy", "gentle", " gentle",
                     "heal", " heal", "welfare", " welfare", "support", " support"],
        "negative": ["harm", " harm", "hurt", " hurt", "cruel", " cruel",
                     "cruelty", " cruelty", "abuse", " abuse", "violence", " violence",
                     "suffering", " suffering", "pain", " pain", "damage", " damage",
                     "destroy", " destroy", "neglect", " neglect", "abandon", " abandon"],
    },
    "fairness_cheating": {
        "positive": ["fair", " fair", "fairness", " fairness", "justice", " justice",
                     "equal", " equal", "rights", " rights", "honest", " honest",
                     "impartial", " impartial", "equitable", " equitable"],
        "negative": ["cheat", " cheat", "unfair", " unfair", "bias", " bias",
                     "corrupt", " corrupt", "fraud", " fraud", "deceive", " deceive",
                     "injustice", " injustice", "inequality", " inequality", "manipulate", " manipulate"],
    },
    "loyalty_betrayal": {
        "positive": ["loyal", " loyal", "loyalty", " loyalty", "faithful", " faithful",
                     "solidarity", " solidarity", "allegiance", " allegiance",
                     "devoted", " devoted", "unity", " unity"],
        "negative": ["betray", " betray", "betrayal", " betrayal", "traitor", " traitor",
                     "treason", " treason", "disloyal", " disloyal", "treacherous", " treacherous"],
    },
    "authority_subversion": {
        "positive": ["authority", " authority", "respect", " respect", "obey", " obey",
                     "duty", " duty", "order", " order", "tradition", " tradition",
                     "discipline", " discipline", "hierarchy", " hierarchy"],
        "negative": ["rebel", " rebel", "subvert", " subvert", "disobey", " disobey",
                     "anarchy", " anarchy", "defiance", " defiance", "chaos", " chaos",
                     "undermine", " undermine", "overthrow", " overthrow"],
    },
    "purity_degradation": {
        "positive": ["pure", " pure", "purity", " purity", "sacred", " sacred",
                     "holy", " holy", "clean", " clean", "virtue", " virtue",
                     "divine", " divine", "noble", " noble"],
        "negative": ["impure", " impure", "degrade", " degrade", "disgust", " disgust",
                     "corrupt", " corrupt", "filth", " filth", "profane", " profane",
                     "obscene", " obscene", "tainted", " tainted"],
    },
}

SCHWARTZ_VALUES = {
    "power": ["power", " power", "control", " control", "dominant", " dominant",
              "wealth", " wealth", "status", " status", "prestige", " prestige", "influence", " influence"],
    "achievement": ["success", " success", "achieve", " achieve", "capable", " capable",
                    "ambitious", " ambitious", "competent", " competent", "excel", " excel"],
    "hedonism": ["pleasure", " pleasure", "enjoy", " enjoy", "fun", " fun",
                 "indulge", " indulge", "desire", " desire", "leisure", " leisure", "luxury", " luxury"],
    "stimulation": ["adventure", " adventure", "exciting", " exciting", "novelty", " novelty",
                    "challenge", " challenge", "risk", " risk", "thrill", " thrill", "daring", " daring"],
    "self_direction": ["freedom", " freedom", "independent", " independent", "creative", " creative",
                       "curious", " curious", "autonomy", " autonomy", "choice", " choice"],
    "universalism": ["equality", " equality", "justice", " justice", "peace", " peace",
                     "tolerance", " tolerance", "wisdom", " wisdom", "welfare", " welfare"],
    "benevolence": ["helpful", " helpful", "honest", " honest", "forgiving", " forgiving",
                    "loyal", " loyal", "responsible", " responsible", "caring", " caring",
                    "kindness", " kindness", "generous", " generous"],
    "tradition": ["tradition", " tradition", "custom", " custom", "religious", " religious",
                  "humble", " humble", "moderate", " moderate", "heritage", " heritage"],
    "conformity": ["obedient", " obedient", "polite", " polite", "discipline", " discipline",
                   "honor", " honor", "respect", " respect", "rules", " rules"],
    "security": ["safe", " safe", "security", " security", "stable", " stable",
                 "order", " order", "protection", " protection", "healthy", " healthy"],
}


def get_all_tokens(framework: str = "both") -> list[str]:
    """Return deduplicated flat list of all tokens for 'moral_foundations', 'schwartz', or 'both'."""
    tokens: set[str] = set()
    if framework in ("moral_foundations", "both"):
        for foundation in MORAL_FOUNDATIONS.values():
            for pole_tokens in foundation.values():
                tokens.update(pole_tokens)
    if framework in ("schwartz", "both"):
        for category_tokens in SCHWARTZ_VALUES.values():
            tokens.update(category_tokens)
    return sorted(tokens)


def get_mft_pole_tokens(foundation: str, pole: str) -> list[str]:
    """Get tokens for a specific MFT foundation and pole.

    Args:
        foundation: One of the MORAL_FOUNDATIONS keys, e.g. 'care_harm'.
        pole: 'positive' or 'negative'.

    Returns:
        List of token strings for that foundation/pole.

    Raises:
        KeyError: If foundation or pole not found.
    """
    return list(MORAL_FOUNDATIONS[foundation][pole])


def get_schwartz_tokens(category: str) -> list[str]:
    """Get tokens for a Schwartz value category.

    Args:
        category: One of the SCHWARTZ_VALUES keys, e.g. 'power'.

    Returns:
        List of token strings for that category.

    Raises:
        KeyError: If category not found.
    """
    return list(SCHWARTZ_VALUES[category])
