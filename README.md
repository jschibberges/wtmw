# Wer talkt mit wem?

Politische Talkshows prägen, worüber in Deutschland gestritten wird und wer dabei zu Wort kommt.
Aber wer sitzt dort eigentlich Woche für Woche in den Sesseln? Welche Parteien, Berufe und
Stimmen sind häufig zu sehen, welche kaum? Welche Themen bestimmen die Sendungen, und wie
verschieben sie sich über die Jahre?

**Wer talkt mit wem?** sammelt die Gästelisten und Sendungsbeschreibungen der großen
politischen Talkshows und macht sie in einem Dashboard durchsuchbar.

---

## Welche Sendungen?

| Sendung | Sender |
|---|---|
| Anne Will | Das Erste (bis 2023) |
| Caren Miosga | Das Erste (seit 2024) |
| hart aber fair | Das Erste |
| maischberger | Das Erste |
| Markus Lanz | ZDF |
| Maybrit Illner | ZDF |

Ausgewertet werden alle Ausgaben seit Januar 2015, aktuell rund 2.650 Sendungen mit knapp
3.000 verschiedenen Gästen (Stand: Oktober 2026). Die Daten werden jeden Montag automatisch
um die Sendungen der Vorwoche ergänzt.

---

## Was lässt sich herausfinden?

- **Wer ist Dauergast?** Ranglisten der häufigsten Gäste, insgesamt oder für einen
  bestimmten Zeitraum und eine bestimmte Sendung.
- **Aus welcher Partei?** Politikerinnen und Politiker sind in Ranglisten, Gästelisten und
  Profilen mit ihrer Partei ausgewiesen.
- **Wer kommt aus Politik, Medien, Wissenschaft oder Wirtschaft?** Jeder Gast hat eine
  Kategorie (siehe unten). So wird sichtbar, wie viel Raum etwa Fachleute oder Betroffene bekommen.
- **Worüber wird gesprochen?** Jede Sendung ist einem Thema zugeordnet, zum Beispiel
  „Ukraine-Krieg & Russland“, „Corona-Pandemie“ oder „Flüchtlingskrise & Migration“. Zeitreihen
  zeigen, wann ein Thema die Talkshows beherrscht hat und wann es wieder verschwand.
- **Wer sitzt mit wem zusammen?** Das Gäste-Netzwerk zeigt, welche Personen besonders oft
  gemeinsam eingeladen werden. Das Themen-Netzwerk zeigt, welche Themen sich dieselben Gäste teilen.
- **Wer ist Allrounder, wer Spezialist?** Für jeden Gast lässt sich ablesen, zu wie vielen
  verschiedenen Themen er oder sie eingeladen wurde.
- **Personenprofile:** Alle Auftritte einer Person, ihre Themen und die Gäste, mit denen sie
  am häufigsten in einer Runde saß.

---

## Wer sitzt da?

Jeder Gast wird einer von zehn Gruppen zugeordnet:

| Gruppe | Zum Beispiel |
|---|---|
| Politik & Regierung | Abgeordnete, Ministerinnen, Botschafter |
| Medien & Kommunikation | Journalistinnen, Moderatoren, Chefredakteure |
| Wissenschaft & Expertise | Professorinnen, Forscher, Gutachter |
| Wirtschaft | Unternehmerinnen, Managerinnen, Ökonomen |
| Zivilgesellschaft | Aktivistinnen, Verbände, Gewerkschaften |
| Kunst & Kultur | Schauspieler, Autorinnen, Musiker |
| Sport | Sportlerinnen, Trainer |
| Religion | Theologen, Kirchenvertreterinnen |
| Bürgerinnen & Bürger | Betroffene, Privatpersonen |
| Influencer & Digitales | YouTuberinnen, Podcaster |

Grundlage ist die Beschreibung, mit der ein Gast in der Sendung vorgestellt wird
(z. B. „Bundesminister für Wirtschaft“ oder „Virologin an der Charité“), und eine
eventuelle Parteizugehörigkeit. Wo die Beschreibung nicht eindeutig ist, ordnet ein
Sprachmodell den Gast der ähnlichsten Gruppe zu.

---

## Wie entstehen die Themen?

Die Themen sind nicht vorgegeben, sie werden aus den Sendungsbeschreibungen gewonnen.
Ein Sprachmodell erkennt, welche Beschreibungen inhaltlich ähnlich sind, und fasst sie zu
Gruppen zusammen. Jede Gruppe erhält anschließend einen von Hand vergebenen Namen.

Das hat Vor- und Nachteile: Die Themen spiegeln wider, worüber tatsächlich gesprochen wurde,
und nicht ein vorab festgelegtes Raster. Dafür ist die Zuordnung nicht perfekt, und einzelne
Gruppen beschreiben eher das Sendungsformat (z. B. „Politische Runde (allgemein)“) als ein Thema.
Solche Format-Gruppen sind im Dashboard gekennzeichnet und zählen nicht zu den Themen-Ranglisten.

---

## Grenzen der Daten

- **Quelle:** Alle Angaben stammen von [fernsehserien.de](https://www.fernsehserien.de).
  Fehlt dort ein Gast oder eine Beschreibung, fehlt er auch hier.
- **Nicht jede Sendung hat ein Thema:** Knapp jede fünfte Sendung ist „nicht klassifiziert“,
  weil die Beschreibung fehlt oder keinem Thema klar zugeordnet werden kann.
- **Parteizugehörigkeit:** Angezeigt wird die zuletzt bekannte Partei, bei Sahra Wagenknecht
  also das BSW, auch für ihre früheren Auftritte als Politikerin der Linken.
- **Namen:** Schreibweisen werden automatisch zusammengeführt und bei Bedarf von Hand
  korrigiert. Vereinzelt können Personen trotzdem doppelt oder falsch zugeordnet sein.
- **Kategorien** sind eine Annäherung: Eine Person, die Politikerin und Autorin ist, landet in
  genau einer Gruppe.

Hinweise auf Fehler sind willkommen, gern als [Issue](https://github.com/jschibberges/wtmw/issues).

---

## Selbst ausführen

```bash
conda env create -f environment.yml && conda activate wtmw

python scrape_talkshows.py    # Sendungsdaten aktualisieren
python analyze_talkshows.py   # Gäste, Kategorien, Themen und Netzwerke berechnen
streamlit run app.py          # Dashboard öffnen (http://localhost:8501)
```

Installation, Projektstruktur, Konfiguration und alle weiteren technischen Details stehen in der
[Entwicklerdokumentation](docs/ENTWICKLUNG.md).

---

## Lizenz

MIT
