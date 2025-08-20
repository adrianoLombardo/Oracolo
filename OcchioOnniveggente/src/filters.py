# src/filters.py
# ------------------------------------------------------------
# Filtro per parolacce / offese (IT + EN) con:
# - normalizzazione (minuscole, rimozione accenti, leetspeak)
# - wildcard "*" nelle blacklist (es. "cazz*", "stronz*")
# - frasi multi-parola (es. "porco dio") con spazi/punteggiatura in mezzo
# - integrazione con better_profanity per l'inglese
# ------------------------------------------------------------

from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable, List
from unidecode import unidecode
from better_profanity import profanity

# Mappatura semplice per varianti "leet"
LEET = str.maketrans({
    "@": "a", "4": "a",
    "3": "e",
    "1": "i", "!": "i", "|": "i",
    "0": "o",
    "$": "s", "5": "s",
    "7": "t"
})

def normalize(text: str) -> str:
    """
    - toglie accenti
    - converte in minuscolo
    - sostituisce leet
    - compatta spazi
    """
    t = unidecode(text).lower()
    t = t.translate(LEET)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def load_blacklist(path: Path) -> List[str]:
    """
    Carica una blacklist (una voce per riga).
    Supporta wildcard "*" e frasi multi-parola.
    Le righe vuote o che iniziano con # vengono ignorate.
    """
    if not path.exists():
        return []
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        w = line.strip()
        if not w or w.startswith("#"):
            continue
        out.append(w.lower())
    return out

def _token_regex_from_phrase(phrase: str) -> re.Pattern:
    """
    Converte una voce di blacklist in un pattern robusto:
    - gestisce wildcard '*' -> [a-z]*   (dopo normalizzazione)
    - consente spazi/punteggiatura tra token -> [\\s\\W_]*
    - ancora ai confini di parola per ridurre i falsi positivi
    Esempio: "porco dio" => (?<!\\w)porco[\\s\\W_]*dio(?!\\w)
    """
    # spezza per spazi nella blacklist (supporto frasi)
    tokens = re.split(r"\s+", phrase.strip())
    # escape sicuro + wildcard
    esc_tokens = [re.escape(tok).replace(r"\*", r"[a-z]*") for tok in tokens]
    # consenti spazi/punteggiatura in mezzo
    body = r"[\s\W_]*".join(esc_tokens)
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)

class ProfanityFilter:
    """
    Uso:
        f = ProfanityFilter(Path("data/filters/it_blacklist.txt"),
                            Path("data/filters/en_blacklist.txt"))
        if f.contains_profanity(text): ...
        clean = f.mask(text)

    Puoi passare liste extra (runtime) per estendere i dizionari.
    """

    def __init__(
        self,
        it_path: Path,
        en_path: Path,
        extra_it: Iterable[str] = (),
        extra_en: Iterable[str] = (),
        mask_char: str = "*",
    ):
        self.mask_char = mask_char

        # carica liste
        it_words = set(load_blacklist(it_path)) | {w.lower() for w in extra_it}
        en_words = set(load_blacklist(en_path)) | {w.lower() for w in extra_en}

        # inizializza better_profanity per l'inglese (parole singole)
        profanity.load_censor_words()

        # compila pattern robusti (frasi + wildcard) per IT/EN
        self.it_patterns = [_token_regex_from_phrase(w) for w in it_words]
        self.en_patterns = [_token_regex_from_phrase(w) for w in en_words]

    # -------------------------
    # Check
    # -------------------------
    def contains_profanity(self, text: str) -> bool:
        n = normalize(text)

        # 1) rapido: better_profanity per inglese (parole singole comuni)
        if profanity.contains_profanity(n):
            return True

        # 2) robusto: nostre regex (IT + EN, frasi incluse)
        for p in self.it_patterns:
            if p.search(n):
                return True
        for p in self.en_patterns:
            if p.search(n):
                return True

        return False

    # -------------------------
    # Mask
    # -------------------------
    def mask(self, text: str) -> str:
        """
        Censura in due passaggi:
        1) better_profanity (inglese)
        2) regex nostre (IT/EN, frasi)
        Mantiene la lunghezza della sequenza censurata con asterischi.
        """
        masked = profanity.censor(text, self.mask_char)

        def repl(m: re.Match) -> str:
            # sostituisci intera sequenza con asterischi
            return self.mask_char * len(m.group(0))

        # Applichiamo sui testi normalizzando le posizioni?
        # Poiché normalizzare cambia gli indici, operiamo direttamente su 'masked'
        # con pattern case-insensitive su testo non normalizzato.
        # Per intercettare varianti con accenti/leet abbiamo già normalizzato
        # in fase di contains_profanity; qui applichiamo comunque le stesse regex
        # al testo originale (case-insensitive) per coprire i casi più comuni.

        for p in self.it_patterns:
            masked = p.sub(repl, masked)
        for p in self.en_patterns:
            masked = p.sub(repl, masked)

        return masked
