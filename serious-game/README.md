# Resilienz Stadt: Daten, KI und Klima

Ein webbasiertes **Serious Game**, in dem du die Rolle einer kommunalen
Entscheidungsstelle übernimmst. Ziel ist es, eine Stadt klimaresilienter,
datenbasierter und nachhaltiger zu machen – und dabei kommunale Datenkompetenz
zu vermitteln.

> **Zentrale Botschaft:** KI allein ist keine Lösung. Erst das fachliche Problem
> verstehen, dann Datenqualität, Governance, Datenschutz und offene Standards
> sichern – und danach KI sinnvoll einsetzen.

Das Spiel basiert inhaltlich auf dem Referenzdokument
*„Digitale Klimaresilienz – Meetup Juni 2026“* und greift reale kommunale
Use Cases auf (Bamberg, Jena, Wuppertal/Dresden, Mannheim, Duisburg).

## Zielgruppe

Kommunale Verwaltung, Civic-Tech-Community, Open-Data-Interessierte, Studierende,
Stadtplanung, Smart-City-Teams – auch ohne tiefes technisches Vorwissen.

## Spielprinzip

- Du verwaltest eine fiktive Stadt mit **begrenztem Budget**.
- Du wählst eine von **fünf Missionen** und triffst in mehreren Runden
  Entscheidungen über **Entscheidungskarten**.
- Jede Entscheidung verändert deine **Spielwerte** (alle starten bei `50`,
  bleiben im Bereich `0–100`).
- Nach jeder Mission gibt es eine kurze **Auswertung**, am Spielende einen
  **Resilienz-Score** mit Einordnung.
- Optionaler **Lernmodus**: Nach jeder Wahl erscheint eine kurze Erklärung,
  warum die Entscheidung gut oder riskant war.

### Missionen

1. **Hitzeinseln erkennen** – Mikroklima, Hitzesensoren, LoRaWAN (Mannheim)
2. **Starkregen vorbereiten** – Hochwasser-Simulation & Kanalsteuerung (Wuppertal/Dresden, Jena)
3. **Stadtbäume schützen** – Drohnen & KI gegen Baumschäden (Bamberg)
4. **Datenschutzkonforme KI einsetzen** – Edge, Anonymisierung, Human Oversight (Duisburg)
5. **Urbane Datenplattform aufbauen** – offene Standards statt Vendor Lock-in

### Spielwerte

Klimaresilienz · Budget · Datenqualität · Bürgerzufriedenheit ·
Datenschutz-Vertrauen · Verwaltungsfähigkeit · Open-Source-Reife ·
Interkommunale Kooperation · Technische Nachhaltigkeit

### Mögliche Ergebnisse

- Datengetriebene Vorreiterstadt
- Solide Smart-City-Verwaltung
- Technisch aktiv, aber organisatorisch schwach
- KI gekauft, Problem nicht gelöst
- Datenchaos und Vendor Lock-in

## Technik

- [React](https://react.dev/) + [TypeScript](https://www.typescriptlang.org/)
- [Vite](https://vitejs.dev/) als Build-Tool und Dev-Server
- [Tailwind CSS](https://tailwindcss.com/) für das Styling
- Lokales **JSON-Datenmodell** (`src/data/missions.json`) für Missionen,
  Karten, Optionen und Effekte
- Responsive Design für Desktop und Smartphone
- **Keine** externe Datenbank, **kein** Login – alles läuft lokal im Browser

## Installation & Start

Voraussetzung: [Node.js](https://nodejs.org/) (Version 18 oder neuer) und npm.

```bash
# In das Projektverzeichnis wechseln
cd serious-game

# Abhängigkeiten installieren
npm install

# Entwicklungsserver starten
npm run dev
```

Anschließend die angezeigte lokale URL (standardmäßig
`http://localhost:5173`) im Browser öffnen.

### Weitere Befehle

```bash
npm run build      # Produktions-Build nach dist/
npm run preview    # Produktions-Build lokal testen
npm run typecheck  # TypeScript-Typprüfung ohne Build
```

## Projektstruktur

```
serious-game/
├── index.html
├── package.json
├── tailwind.config.js
├── postcss.config.js
├── vite.config.ts
├── tsconfig.json
└── src/
    ├── main.tsx                 # Einstiegspunkt
    ├── App.tsx                  # Spiel-State & Ablaufsteuerung
    ├── index.css                # Tailwind-Einbindung & Basisstil
    ├── types/
    │   └── index.ts             # Zentrale TypeScript-Typen
    ├── data/
    │   ├── missions.json        # JSON-Datenmodell (5 Missionen × 4 Karten)
    │   ├── missions … index.ts  # Typisierter Daten-Zugriff
    │   ├── stats.ts             # Definition der Spielwerte
    │   └── resultTiers.ts       # Ergebnis-Stufen
    ├── logic/
    │   └── gameLogic.ts         # Wertelogik, Score & Auswertung
    └── components/
        ├── StartScreen.tsx
        ├── MissionSelect.tsx
        ├── GameDashboard.tsx
        ├── DecisionCard.tsx
        ├── ScoreBar.tsx
        ├── ResultScreen.tsx
        └── InfoModal.tsx
```

## Inhaltliche Erweiterung

Neue Missionen oder Karten lassen sich ohne Code-Änderung in
`src/data/missions.json` ergänzen. Jede Option besteht aus einem `label`,
einer `consequence` (Folge), den `effects` (Wertveränderungen) und einer
`learn`-Erklärung für den Lernmodus.
