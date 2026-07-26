# Victor Hugo retrieval bridges

This file contains bilingual anchor patterns that help retrieval land on the right Hugo material even when the corpus language and the user language differ.

## Exile bridge

Turkish query forms:
- `Victor Hugo neden sürgüne gitti?`
- `Victor Hugo niçin Fransa'dan ayrıldı?`
- `sürgün yılları`
- `Guernsey dönemi`

English retrieval anchors:
- `Victor Hugo exile`
- `opposed Louis-Napoléon's coup`
- `Guernsey exile`
- `Jersey and Guernsey`

Likely relevant corpus areas:
- `victor_hugo_timeline.json`
- `victor_hugo_biography.md`
- `history_of_a_crime_en.txt`
- `napoleon_the_little_en.txt`

## Death penalty bridge

Turkish query forms:
- `Victor Hugo idam cezasına neden karşıydı?`
- `ölüm cezası hakkındaki görüşü`
- `mahkûmlar ve merhamet`

English retrieval anchors:
- `death penalty`
- `capital punishment`
- `condemned man`
- `justice and mercy`
- `In Defense of His Son`

Likely relevant corpus areas:
- `in_defense_of_his_son_notes.md`
- `letter_john_brown_notes.md`
- `Le Dernier jour d'un condamné`

## Les Misérables bridge

Turkish query forms:
- `Sefiller ne anlatır?`
- `Sefiller ana temaları`
- `Jean Valjean kimdir?`
- `Sefiller yoksulluk ve adalet`

English retrieval anchors:
- `Les Misérables themes`
- `Jean Valjean`
- `poverty justice mercy`
- `redemption and law`

Likely relevant corpus areas:
- `key_texts_and_retrieval_notes.md`
- `les_miserables_en.txt`
- `complete_bibliography.md`

## Notre-Dame bridge

Turkish query forms:
- `Notre Dame'ın Kamburu`
- `Notre-Dame de Paris ne anlatır?`
- `Quasimodo kimdir?`
- `katedral ve mimari teması`

English retrieval anchors:
- `The Hunchback of Notre-Dame`
- `Notre-Dame de Paris`
- `Quasimodo Esmeralda Frollo`
- `cathedral architecture`

Likely relevant corpus areas:
- `notre_dame_de_paris_en.txt`
- `key_texts_and_retrieval_notes.md`

## Style bridge

Turkish query forms:
- `Victor Hugo'nun üslubu`
- `nasıl bir dil kullanır`
- `retoriği nasıldır`
- `şiirsel ama politik tonu`

English retrieval anchors:
- `Victor Hugo style`
- `prophetic rhetoric`
- `moral grandeur`
- `epic contrast`
- `public oratory cadence`

Likely relevant corpus areas:
- `victor_hugo_persona_guide.md`
- `key_texts_and_retrieval_notes.md`
- poetry collections

## Retrieval rule

When the user asks in Turkish:

1. map Turkish title aliases and theme terms to canonical French work names
2. map canonical French names to English translation titles where needed
3. search both theme vocabulary and title aliases
4. prefer metadata files first for direct factual questions
5. prefer primary texts for style, tone, and quoted explanation
