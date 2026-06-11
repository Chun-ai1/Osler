# Oslian·Rx — Mobile Demo (Expo / React Native)

Mobile companion to the **Osler·Rx Drug Recommendation Agent** web demo
(`demo/` on the `demo-rx` branch). Same flow, same backend, same design language:

> import a patient case → symbolic engine ranks drug candidates →
> reasoning mind-map → chat to ask why.

The symbolic engine makes every recommendation (deterministic, explainable).
The LLM only parses free-text cases and explains results — it never makes a
medical decision. **Decision support demo only — not a clinically validated tool.**

## Branch layout (important)

- **`demo-rx`** — the runnable demo: `demo/` (Flask app + UI), `engine/` (pharmacology
  engine), `data/`. This `mobile/` app lives here too.
- **`main`** — a data-slice export ("Disease Expansion v1", organ-state JSON patch).
  It does **not** contain the runnable demo. Don't look for the app there.

## Run — two independent commands

### 1. Backend (Flask, port 5000)

From the **repo root**, on the `demo-rx` branch (or any branch containing `demo/`):

```bash
git switch demo-rx
pip install -r requirements.txt
python demo/demo_app.py        # → http://127.0.0.1:5000
```

Sanity check: `curl http://127.0.0.1:5000/api/cases` should return 5 preset cases.

### 2. Mobile app (Expo)

```bash
cd mobile
npm install
npx expo start                 # press i for the iOS simulator
```

The iOS **simulator shares the Mac's localhost**, so the app talks to
`http://127.0.0.1:5000` out of the box — no configuration needed.
(The app deliberately uses `127.0.0.1`, not `localhost`: macOS AirPlay Receiver
listens on port 5000 over IPv6, and `localhost` resolves to `::1` first.)

**Physical device:** set the backend host to your Mac's LAN IP before starting:

```bash
EXPO_PUBLIC_API_BASE=http://192.168.x.x:5000 npx expo start
```

(iOS ATS exceptions for local networking are pre-configured in `app.json`.)

## LLM key (optional)

Chat answers and free-text case parsing use an OpenAI or Gemini key — add it in
the app's **Settings** tab (stored on-device with `expo-secure-store`). Without a
key everything still works: preset cases use rule-based parsing and the symbolic
engine; only chat returns a fallback notice. For a reliable live demo, drive it
from the preset cases.

## App structure

```
src/
  app/            # expo-router tabs: Patients / Analyze / Reasoning / Chat / Settings
  api/osler.ts    # client for /api/cases, /api/analyze, /api/chat
  state/          # AppContext: analyzed patients, current case, LLM settings
  theme/tokens.ts # design tokens ported from demo/case_demo.html
  components/     # DrugCard, GraphWebView, TraceCard, ...
assets/vendor/    # vis-network standalone build (bundled — Graph renders offline)
```

The reasoning mind-map uses the same **vis-network** library as the web demo,
rendered in a WebView from a locally bundled build — no CDN, works offline.
