from enum import Enum


class Categories(Enum):
    BARENTNAHME = 'barentnahme'
    FINANZEN = 'finanzen'
    FREIZEITLIFESTYLE = 'freizeitlifestyle'
    LEBENSHALTUNG = 'lebenshaltung'
    MOBILITAETVERKEHR = 'mobilitaetverkehrsmittel'
    VERSICHERUNGEN = 'versicherungen'
    WOHNENHAUSHALT = 'wohnenhaushalt'


class FallbackCategorie(Enum):
    SONSTIGES = 'sonstiges'
